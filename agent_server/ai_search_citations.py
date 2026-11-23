"""Citations for AI Search (vector search) tool results.

The AI Search managed MCP server answers with the rows of the index's source
table, each row a JSON object keyed by column name, plus a similarity ``score``
(verified against a live index). There is no default citation column: a row
carries a URL only when the source table has a column such as ``source_url``
or ``doc_uri``, so the columns to read are configurable through
``AI_SEARCH_CITATION_COLUMNS`` (and ``AI_SEARCH_CITATION_TITLE_COLUMNS`` for
titles), with a fallback to any column whose whole value is an http(s) URL.

Web search citations reach the chat UI as ``response.output_text.annotation.added``
events with a ``url_citation`` annotation, which the Databricks AI SDK provider
turns into ``source-url`` parts — the pills and Sources list the UI already
renders. The helpers here produce the same annotations from AI Search rows, so
both sources look identical in the transcript.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable, List

# Column names checked, in order, for a clickable document URL. Override with
# AI_SEARCH_CITATION_COLUMNS when an index uses a different name.
DEFAULT_CITATION_COLUMNS = (
    "citation,source_url,doc_uri,url,uri,link,source,source_uri,document_url,doc_link"
)
# Column names checked, in order, for the label shown on a citation.
DEFAULT_CITATION_TITLE_COLUMNS = "citation_title,title,doc_title,source_title,name"

# Rows repeat a URL once per chunk, and a query can span indexes, so the
# citations are deduped by URL and capped to keep the Sources list readable.
MAX_CITATIONS = 12

_URL = re.compile(r"^https?://\S+$", re.IGNORECASE)
_MAX_URL_LENGTH = 2048


def _columns(env_name: str, default: str) -> List[str]:
    configured = os.getenv(env_name, default)
    return [column.strip().lower() for column in configured.split(",") if column.strip()]


def citation_columns() -> List[str]:
    return _columns("AI_SEARCH_CITATION_COLUMNS", DEFAULT_CITATION_COLUMNS)


def title_columns() -> List[str]:
    return _columns("AI_SEARCH_CITATION_TITLE_COLUMNS", DEFAULT_CITATION_TITLE_COLUMNS)


def _is_link(value: Any) -> bool:
    """Only http(s) URLs are clickable; a doc_uri like s3:// or /Volumes/ is not."""
    return (
        isinstance(value, str)
        and len(value) <= _MAX_URL_LENGTH
        and bool(_URL.match(value.strip()))
    )


def citations_from_rows(rows: Iterable[Any]) -> List[dict]:
    """``url_citation`` annotations for AI Search rows, deduped by URL."""
    url_columns = citation_columns()
    title_columns_ = title_columns()
    citations: List[dict] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        lowered = {str(key).lower(): value for key, value in row.items()}
        url = next(
            (lowered[column] for column in url_columns if _is_link(lowered.get(column))),
            None,
        )
        if url is None:
            # No configured column held a URL; accept any column whose whole
            # value is one (a passage's text is never exactly a URL).
            url = next((value for value in lowered.values() if _is_link(value)), None)
        if url is None or url.strip() in seen:
            continue
        title = next(
            (
                str(lowered[column]).strip()
                for column in title_columns_
                if isinstance(lowered.get(column), str) and lowered[column].strip()
            ),
            "",
        )
        seen.add(url.strip())
        citations.append(
            {"type": "url_citation", "url": url.strip(), "title": title}
        )
        if len(citations) >= MAX_CITATIONS:
            break
    return citations


def _parse_json(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def _is_text_part(part: Any) -> bool:
    return isinstance(part, dict) and isinstance(part.get("text"), str)


def rows_from_tool_output(output: Any) -> List[Any]:
    """The rows an AI Search tool returned, whichever wrapper they arrive in.

    The managed MCP server answers with one text part holding the row array
    as JSON. The SDK's tool items carry that as a bare string or as a list of
    text parts (``input_text``/``output_text``/``text``), so all of those
    shapes are read.
    """
    if isinstance(output, str):
        output = _parse_json(output)
    if isinstance(output, dict):
        if _is_text_part(output):
            return rows_from_tool_output(_parse_json(output.get("text")))
        # A JSON object answer; look for the rows under a common key before
        # treating the object itself as a single row.
        for key in ("chunks", "results", "rows", "data", "data_array", "docs"):
            value = output.get(key)
            if isinstance(value, list):
                return value
        return [output]
    if isinstance(output, list):
        if output and all(_is_text_part(part) for part in output):
            rows: List[Any] = []
            for part in output:
                rows.extend(rows_from_tool_output(_parse_json(part.get("text"))))
            return rows
        return output
    return []


def citations_from_tool_output(output: Any) -> List[dict]:
    """``url_citation`` annotations for one AI Search tool result."""
    return citations_from_rows(rows_from_tool_output(output))


def ai_search_tool_names(indexes: Iterable[str]) -> set[str]:
    """Function names of the AI Search MCP tools for the configured indexes.

    The managed MCP server names its one tool after the index, joining the
    three-level name's dots as double underscores:
    ``catalog.schema.index`` -> ``catalog__schema__index``.
    """
    return {index.replace(".", "__") for index in indexes}


def attach_citations(output_items: List[dict], tool_names: set[str]) -> None:
    """Attach AI Search citations to the last assistant message, in place.

    Streaming sends citations as annotation events (see
    ``process_agent_stream_events``). Invoke callers only see output items,
    so the citations collected from the tool results are attached to the final
    assistant message the same way hosted web search annotations arrive on
    that path.
    """
    names: dict[str, str] = {}
    citations: List[dict] = []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            call_id, name = item.get("call_id"), item.get("name")
            if isinstance(call_id, str) and isinstance(name, str):
                names[call_id] = name
        elif item.get("type") == "function_call_output":
            if names.get(item.get("call_id") or "") in tool_names:
                citations.extend(citations_from_tool_output(item.get("output")))

    if not citations:
        return
    for item in reversed(output_items):
        if not (isinstance(item, dict) and item.get("type") == "message"):
            continue
        if item.get("role") != "assistant":
            break
        content = item.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            part = content[0]
            annotations = part.get("annotations")
            if not isinstance(annotations, list):
                annotations = part["annotations"] = []
            existing = {
                annotation.get("url")
                for annotation in annotations
                if isinstance(annotation, dict)
            }
            annotations.extend(
                citation
                for citation in citations
                if citation["url"] not in existing
            )
        break
