import asyncio
import json
import logging
from time import monotonic
from typing import Any, AsyncGenerator, AsyncIterator, Optional
from uuid import uuid4

from agents.models.fake_id import FAKE_RESPONSES_ID
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.result import StreamEvent
from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from mcp.types import CallToolResult
from mlflow.genai.agent_server import get_request_headers
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentStreamEvent


def get_session_id(request: ResponsesAgentRequest) -> str | None:
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    if request.custom_inputs and isinstance(request.custom_inputs, dict):
        return request.custom_inputs.get("session_id")
    return None


def get_databricks_host(workspace_client: WorkspaceClient | None = None) -> Optional[str]:
    workspace_client = workspace_client or WorkspaceClient()
    try:
        return workspace_client.config.host
    except Exception as e:
        logging.exception(f"Error getting databricks host from env: {e}")
        return None


def build_mcp_url(path: str, workspace_client: WorkspaceClient | None = None) -> str:
    if not path.startswith("/"):
        return path
    hostname = get_databricks_host(workspace_client)
    return f"{hostname}{path}"


def get_user_workspace_client() -> WorkspaceClient:
    token = get_request_headers().get("x-forwarded-access-token")
    return WorkspaceClient(token=token, auth_type="pat")


# Titles only change when a space is renamed, so one lookup per process is plenty.
_genie_space_names: dict[str, str] = {}


def genie_space_display_name(space_id: str, workspace_client: WorkspaceClient) -> str:
    """Name for a Genie space's MCP server, resolved from the space's title.

    Only ids are configured, so the title is looked up to keep the tool listing the
    model sees readable. A space whose title cannot be read is still attached under
    a name built from its id: the agent queries Genie as the signed-in user, who may
    well have access even when this lookup does not.
    """
    cached = _genie_space_names.get(space_id)
    if cached:
        return cached

    try:
        title = workspace_client.genie.get_space(space_id).title
    except Exception:
        logging.warning("Could not resolve title for Genie space %s", space_id, exc_info=True)
        return f"Genie Space {space_id}"

    name = f"Genie Space: {title}" if title else f"Genie Space {space_id}"
    _genie_space_names[space_id] = name
    return name


# Genie's MCP tools come in pairs per space: query_space_<id> starts a message and
# poll_response_<id> reads it once it has finished.
_GENIE_QUERY_TOOL_PREFIX = "query_space_"
_GENIE_POLL_TOOL_PREFIX = "poll_response_"
# Statuses a Genie message stops at. Any other status means it is still working:
# SUBMITTED, ASKING_AI, PENDING_WAREHOUSE and EXECUTING_QUERY all appear on the way.
_GENIE_FINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"})
_GENIE_POLL_INTERVAL_SECONDS = 3.0
# A ceiling, not an expectation: a warehouse that has to start costs the most time here.
# It stays well inside the chat proxy's own request timeout (CHAT_PROXY_TIMEOUT_SECONDS).
_GENIE_POLL_TIMEOUT_SECONDS = 180.0


def _genie_payload(result: CallToolResult) -> dict:
    """The JSON object a Genie tool answered with, or an empty one if it isn't JSON."""
    text = "".join(getattr(block, "text", "") or "" for block in result.content or [])
    try:
        payload = json.loads(text)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _genie_poll_arguments(result: CallToolResult) -> dict[str, str] | None:
    """Arguments for polling the message this result describes, or None if it finished.

    Failures and unrecognized answers read as finished: the model is better served by
    the message Genie sent than by waiting out the timeout on something that will not
    change.
    """
    payload = _genie_payload(result)
    status = payload.get("status")
    if not isinstance(status, str) or status in _GENIE_FINAL_STATUSES:
        return None
    conversation_id = payload.get("conversationId") or payload.get("conversation_id")
    message_id = payload.get("messageId") or payload.get("message_id")
    if not conversation_id or not message_id:
        return None
    return {"conversation_id": conversation_id, "message_id": message_id}


class GenieMcpServer(McpServer):
    """Genie MCP server that does its own polling.

    A question that outlives Genie's own wait comes back as a message id and a status
    such as EXECUTING_QUERY, leaving the caller to poll for the result. Both tool
    descriptions ask the model to do that, and small models tend instead to pass the
    "still processing" note to the user as though it were the answer, ending the turn
    with no data in it. Polling here means the query tool returns once, with the result,
    whichever model is driving it — and spends no model turns on the wait.
    """

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None, **kwargs: Any
    ) -> CallToolResult:
        result = await super().call_tool(tool_name, arguments, **kwargs)
        if not tool_name.startswith(_GENIE_QUERY_TOOL_PREFIX):
            return result

        space_id = tool_name[len(_GENIE_QUERY_TOOL_PREFIX) :]
        poll_tool = f"{_GENIE_POLL_TOOL_PREFIX}{space_id}"
        deadline = monotonic() + _GENIE_POLL_TIMEOUT_SECONDS
        while (poll_arguments := _genie_poll_arguments(result)) is not None:
            if monotonic() >= deadline:
                logging.warning(
                    "Genie message %s has not finished after %ss; handing its status to "
                    "the model to poll for itself",
                    poll_arguments["message_id"],
                    _GENIE_POLL_TIMEOUT_SECONDS,
                )
                return result
            await asyncio.sleep(_GENIE_POLL_INTERVAL_SECONDS)
            result = await super().call_tool(poll_tool, poll_arguments, **kwargs)
        return result


def adapt_input_for_chat_completions(items: list[dict]) -> list[dict]:
    """Give echoed assistant turns the id the chat completions converter looks for.

    The chat client sends each earlier assistant turn back as a Responses-shaped message
    carrying no id, which the Responses API accepts. The agents SDK recognizes an
    assistant message on the chat completions path only when it has one, and rejects the
    whole conversation with "Unhandled item type or structure" otherwise, so every
    question after the first fails. The id is used for nothing but that recognition here:
    it never reaches the provider, which is why the SDK's own placeholder fits.
    """
    for item in items:
        if item.get("type") == "message" and item.get("role") == "assistant":
            item.setdefault("id", FAKE_RESPONSES_ID)
    return items


def _part_field(part: Any, field: str) -> Any:
    return part.get(field) if isinstance(part, dict) else getattr(part, field, None)


def _flatten_content_parts(target: Any) -> None:
    """Rewrite a list-valued `content` into the fields chat completions specifies.

    Google models reached through the gateway answer with a list of typed parts —
    `text` alongside `reasoning` — where the chat completions API specifies a plain
    string, and the agents SDK raises a validation error on the list. Text is joined
    back into `content`, and reasoning summaries move to `reasoning_content`, the field
    the SDK already reads for models that report their thinking separately.
    """
    content = getattr(target, "content", None)
    if not isinstance(content, list):
        return

    text: list[str] = []
    reasoning: list[str] = []
    for part in content:
        part_type = _part_field(part, "type")
        if part_type == "text":
            text.append(_part_field(part, "text") or "")
        elif part_type == "reasoning":
            for summary in _part_field(part, "summary") or []:
                reasoning.append(_part_field(summary, "text") or "")

    target.content = "".join(text) or None
    if reasoning:
        existing = getattr(target, "reasoning_content", None) or ""
        target.reasoning_content = existing + "\n\n".join(reasoning)


def _relabel_thought_signatures(target: Any) -> None:
    """Put Google's thought signature where the agents SDK looks for it.

    Gemini rejects a conversation whose function calls come back without the signature
    it issued with them. The SDK already carries signatures across turns, but reads them
    from `extra_content.google.thought_signature`, and the gateway returns them as
    `thoughtSignature` on the tool call itself.
    """
    for tool_call in getattr(target, "tool_calls", None) or []:
        signature = getattr(tool_call, "thoughtSignature", None)
        if not signature or getattr(tool_call, "extra_content", None):
            continue
        tool_call.extra_content = {"google": {"thought_signature": signature}}


def normalize_gateway_message(target: Any) -> None:
    """Reshape one gateway message or streaming delta into what the SDK expects."""
    if target is None:
        return
    _flatten_content_parts(target)
    _relabel_thought_signatures(target)


def _restore_thought_signatures(message: dict) -> None:
    """Spell thought signatures the way the gateway reads them on the way back out.

    The other half of `_relabel_thought_signatures`: the SDK returns the signature it
    carried as `extra_content.google.thought_signature`, which the gateway ignores —
    leaving Gemini to reject the turn — so it is moved back onto the tool call.
    """
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        google = (tool_call.pop("extra_content", None) or {}).get("google") or {}
        signature = google.get("thought_signature")
        if signature:
            tool_call["thoughtSignature"] = signature


def _collapse_tool_output(message: dict) -> None:
    """Join a tool result's text parts into the single string Gemini insists on.

    MCP tools answer with a list of content blocks, which the SDK passes through as a
    list of text parts. Most providers accept that; Google's translation rejects it.
    Anything other than plain text is left alone rather than flattened away.
    """
    if message.get("role") != "tool":
        return
    parts = message.get("content")
    if not isinstance(parts, list):
        return
    if not all(isinstance(part, dict) and part.get("type") == "text" for part in parts):
        return
    message["content"] = "\n".join(part.get("text") or "" for part in parts)


def adapt_outbound_messages(messages: Any) -> None:
    """Rewrite a request's messages into the shapes the gateway accepts."""
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        _restore_thought_signatures(message)
        _collapse_tool_output(message)


class _RewritingCompletions:
    """Completions resource that fixes outbound requests, then delegates."""

    def __init__(self, completions: Any) -> None:
        self._completions = completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)

    async def create(self, **kwargs: Any) -> Any:
        adapt_outbound_messages(kwargs.get("messages"))
        return await self._completions.create(**kwargs)


class _NormalizingStream:
    """Chat completions stream that reshapes each chunk in passing."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def __aiter__(self) -> "_NormalizingStream":
        return self

    async def __anext__(self) -> Any:
        chunk = await self._stream.__anext__()
        for choice in getattr(chunk, "choices", None) or []:
            normalize_gateway_message(getattr(choice, "delta", None))
        return chunk

    async def aclose(self) -> None:
        await self._stream.close()


class GatewayOpenAI(AsyncDatabricksOpenAI):
    """Databricks client that also speaks the gateway's spelling of provider extras.

    Only the outbound direction needs a client of its own; inbound reshaping happens in
    `GatewayChatCompletionsModel`, which sees responses before the SDK parses them.
    """

    @property
    def chat(self) -> Any:
        chat = super().chat  # keeps the Databricks fixes for tool schemas and content
        chat.completions = _RewritingCompletions(chat.completions)
        return chat


class GatewayChatCompletionsModel(OpenAIChatCompletionsModel):
    """Chat completions model that tolerates the gateway's non-OpenAI response shapes.

    Providers reached through the gateway's OpenAI-compatible API do not all answer in
    exactly the shape the API specifies. Normalizing here, rather than in the agent,
    keeps every family on the SDK's ordinary chat completions path. It is a no-op for
    providers that already answer with plain strings.
    """

    async def _fetch_response(self, *args: Any, **kwargs: Any) -> Any:
        result = await super()._fetch_response(*args, **kwargs)
        if isinstance(result, tuple):  # streaming: (Response, AsyncStream)
            response, stream = result
            return response, _NormalizingStream(stream)
        for choice in getattr(result, "choices", None) or []:
            normalize_gateway_message(getattr(choice, "message", None))
        return result


def _replace_placeholder_id(event_data: dict, item_id: str) -> None:
    """Give each streamed item a unique id when the model API doesn't supply one.

    The chat completions API reuses a single placeholder id for every item, which the
    client can't tell apart. Responses API ids must be left alone: they are echoed back
    as input on later turns and are rejected unless the original prefix is preserved.
    """
    item = event_data.get("item")
    if item is not None and item.get("id") == FAKE_RESPONSES_ID:
        item["id"] = item_id
    elif event_data.get("item_id") == FAKE_RESPONSES_ID:
        event_data["item_id"] = item_id


async def process_agent_stream_events(
    async_stream: AsyncIterator[StreamEvent],
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    curr_item_id = str(uuid4())
    async for event in async_stream:
        if event.type == "raw_response_event":
            event_data = event.data.model_dump()
            if event_data["type"] == "response.output_item.added":
                curr_item_id = str(uuid4())
            _replace_placeholder_id(event_data, curr_item_id)
            yield event_data
        elif event.type == "run_item_stream_event" and event.item.type == "tool_call_output_item":
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=event.item.to_input_item(),
            )
