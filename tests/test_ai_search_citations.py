"""AI Search rows become the url_citation annotations web search already uses."""

import asyncio
import json

from agent_server.ai_search_citations import (
    ai_search_tool_names,
    attach_citations,
    citations_from_rows,
    citations_from_tool_output,
)
from agent_server.utils import process_agent_stream_events


ROW = {
    "chunk_id": "DOE O 426.2B__15",
    "title": "PERSONNEL SELECTION AND TRAINING",
    "chunk_text": "Continuing training cycles may not exceed 5 years.",
    "score": 0.64,
}


def test_rows_with_a_source_url_column_become_citations():
    citations = citations_from_rows(
        [
            {**ROW, "source_url": "https://www.energy.gov/media/364621"},
            {**ROW, "chunk_id": "other", "source_url": "https://example.gov/other"},
        ]
    )
    assert citations == [
        {
            "type": "url_citation",
            "url": "https://www.energy.gov/media/364621",
            "title": "PERSONNEL SELECTION AND TRAINING",
        },
        {
            "type": "url_citation",
            "url": "https://example.gov/other",
            "title": "PERSONNEL SELECTION AND TRAINING",
        },
    ]


def test_doc_uri_column_is_used_and_non_url_values_are_skipped():
    citations = citations_from_rows(
        [
            {**ROW, "doc_uri": "s3://bucket/parsed/doc.pdf"},  # not clickable
            {**ROW, "doc_uri": "https://example.gov/parsed/doc.pdf"},
        ]
    )
    assert [c["url"] for c in citations] == ["https://example.gov/parsed/doc.pdf"]


def test_bare_url_column_is_found_without_being_configured():
    citations = citations_from_rows(
        [{**ROW, "document_link": "https://example.gov/directive"}]
    )
    assert [c["url"] for c in citations] == ["https://example.gov/directive"]


def test_citation_column_preference_is_configurable(monkeypatch):
    monkeypatch.setenv("AI_SEARCH_CITATION_COLUMNS", "citation,reference")
    monkeypatch.setenv("AI_SEARCH_CITATION_TITLE_COLUMNS", "heading")
    citations = citations_from_rows(
        [
            {
                "citation": "https://example.gov/cited",
                "heading": "Cited document",
                "url": "https://example.gov/ignored",
            }
        ]
    )
    assert citations == [
        {"type": "url_citation", "url": "https://example.gov/cited", "title": "Cited document"}
    ]


def test_repeated_urls_across_chunks_are_deduped():
    rows = [
        {**ROW, "chunk_id": f"chunk_{i}", "source_url": "https://example.gov/one"}
        for i in range(3)
    ]
    assert citations_from_rows(rows) == [
        {
            "type": "url_citation",
            "url": "https://example.gov/one",
            "title": "PERSONNEL SELECTION AND TRAINING",
        }
    ]


def test_title_falls_back_to_empty_and_rows_without_urls_are_skipped():
    citations = citations_from_rows([{"content": "just text", "id": "1"}])
    assert citations == []


def test_tool_output_shapes_all_yield_the_same_rows():
    rows = [{**ROW, "source_url": "https://example.gov/doc"}]
    as_text = json.dumps(rows)
    assert citations_from_tool_output(as_text) == citations_from_rows(rows)
    # The SDK's tool items wrap the MCP text part as input_text/output_text.
    assert citations_from_tool_output(
        [{"type": "input_text", "text": as_text}]
    ) == citations_from_rows(rows)
    assert citations_from_tool_output({"type": "output_text", "text": as_text}) == (
        citations_from_rows(rows)
    )
    assert citations_from_tool_output(
        [{"type": "input_text", "text": as_text}, {"type": "input_text", "text": "trailing"}]
    ) == citations_from_rows(rows)


def test_index_names_map_to_their_managed_mcp_tool_names():
    # Verified against a live index: the tool is named by joining the
    # three-level name's dots as double underscores.
    assert ai_search_tool_names(["catalog.schema.index_name"]) == {
        "catalog__schema__index_name"
    }


def test_attach_citations_appends_to_the_last_assistant_message():
    output = [
        {"type": "function_call", "call_id": "call_1", "name": "catalog__schema__index"},
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [{"type": "input_text", "text": json.dumps(
                [{**ROW, "source_url": "https://example.gov/doc"}]
            )}],
        },
        {"type": "function_call", "call_id": "call_2", "name": "unrelated_tool"},
        {"type": "function_call_output", "call_id": "call_2", "output": "no urls"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Answer"}],
        },
    ]
    attach_citations(output, ai_search_tool_names(["catalog.schema.index"]))
    annotations = output[-1]["content"][0]["annotations"]
    assert [a["url"] for a in annotations] == ["https://example.gov/doc"]
    # Unrelated tools contribute nothing.
    assert len(annotations) == 1


def test_attach_citations_leaves_output_without_citations_alone():
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Answer"}],
        }
    ]
    attach_citations(output, ai_search_tool_names(["catalog.schema.index"]))
    assert "annotations" not in output[0]["content"][0]


class _RawEvent:
    """The raw provider events carry pydantic models with a model_dump()."""

    def __init__(self, data):
        self.type = "raw_response_event"
        self.data = self
        self._data = data

    def model_dump(self):
        return self._data


class _RunItem:
    def __init__(self, item_type, to_input_item):
        self.type = item_type
        self._to_input_item = to_input_item

    def to_input_item(self):
        return self._to_input_item


class _RunItemEvent:
    def __init__(self, item):
        self.type = "run_item_stream_event"
        self.item = item


def _kind(event):
    """Raw events are yielded as dicts, the rest as ResponsesAgentStreamEvent."""
    return event["type"] if isinstance(event, dict) else event.type


def _collect(events, **kwargs):
    """Run process_agent_stream_events over a plain list of stream events."""

    async def _stream():
        for event in events:
            yield event

    async def run():
        return [
            event
            async for event in process_agent_stream_events(_stream(), **kwargs)
        ]

    return asyncio.run(run())


def test_stream_emits_web_search_style_annotation_events_for_ai_search_tools():
    rows = [{**ROW, "source_url": "https://example.gov/doc"}]
    events = _collect(
        [
            _RawEvent(
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "catalog__schema__index",
                    },
                }
            ),
            _RunItemEvent(
                _RunItem(
                    "tool_call_output_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "id": "item_1",
                        "output": [{"type": "input_text", "text": json.dumps(rows)}],
                    },
                )
            ),
        ],
        tool_annotations=lambda name, output: (
            citations_from_tool_output(output)
            if name == "catalog__schema__index"
            else []
        ),
    )

    assert [_kind(event) for event in events] == [
        "response.output_item.added",
        "response.output_item.done",
        "response.output_text.annotation.added",
    ]
    annotation_event = events[-1].model_dump()
    assert annotation_event["item_id"] == "item_1"
    assert annotation_event["annotation_index"] == 0
    assert annotation_event["annotation"] == {
        "type": "url_citation",
        "url": "https://example.gov/doc",
        "title": "PERSONNEL SELECTION AND TRAINING",
    }


def test_stream_emits_no_annotations_for_other_tools():
    events = _collect(
        [
            _RawEvent(
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "unrelated_tool",
                    },
                }
            ),
            _RunItemEvent(
                _RunItem(
                    "tool_call_output_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "plain result",
                    },
                )
            ),
        ],
        tool_annotations=lambda name, output: [],
    )
    assert [_kind(event) for event in events] == [
        "response.output_item.added",
        "response.output_item.done",
    ]


def test_stream_without_annotations_behaves_as_before():
    events = _collect(
        [
            _RunItemEvent(
                _RunItem(
                    "tool_call_output_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "plain result",
                    },
                )
            )
        ]
    )
    assert [_kind(event) for event in events] == ["response.output_item.done"]
