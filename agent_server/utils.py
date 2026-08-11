import logging
from typing import AsyncGenerator, AsyncIterator, Optional
from uuid import uuid4

from agents.models.fake_id import FAKE_RESPONSES_ID
from agents.result import StreamEvent
from databricks.sdk import WorkspaceClient
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
