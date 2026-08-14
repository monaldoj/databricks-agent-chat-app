import logging
from typing import Any, AsyncGenerator, AsyncIterator, Optional
from uuid import uuid4

from agents.models.fake_id import FAKE_RESPONSES_ID
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.result import StreamEvent
from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI
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
