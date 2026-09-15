
import logging
import os
import re
from contextlib import asynccontextmanager
from agents.mcp import MCPServer, MCPServerManager
from typing import AsyncGenerator, List

import mlflow
from agents import (
    Agent,
    Model,
    ModelSettings,
    OpenAIResponsesModel,
    Runner,
    WebSearchTool,
    set_default_openai_client,
)
from agents.tracing import set_trace_processors
from databricks.sdk import WorkspaceClient
from databricks_openai.agents import McpServer
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.date_tools import get_todays_date
from agent_server.model_profile import ModelProfile, model_family, model_profile
from agent_server.utils import (
    GatewayChatCompletionsModel,
    GatewayOpenAI,
    GenieMcpServer,
    adapt_input_for_chat_completions,
    build_mcp_url,
    genie_space_display_name,
    get_user_workspace_client,
    process_agent_stream_events,
)
from agent_server.web_search import (
    WEB_SEARCH_MCP_NAME,
    WEB_SEARCH_MCP_PATH,
    WebSearchMode,
    effective_web_search_mode,
    extra_body_for,
    is_web_search_mcp_server,
    resolved_web_search_mode,
    uses_web_search_mcp,
    web_search_mcp_spec,
)

# Every model is reached through the AI Gateway's OpenAI-compatible API at
# <workspace host>/ai-gateway/openai/v1, so a single client serves every provider and
# models can be named by their three-level Unity Catalog name (system.ai.claude-opus-5)
# as well as by serving endpoint name (databricks-claude-opus-5).
GATEWAY_CLIENT = GatewayOpenAI(use_ai_gateway_native_api=True)
set_default_openai_client(GATEWAY_CLIENT)
set_trace_processors([])  # only use mlflow for trace processing
mlflow.openai.autolog()

# GENERATED

NAME = 'agent-web-search-genie'
SYSTEM_PROMPT = """\
You are a helpful assistant. Answer clearly and accurately.
Use web search for current public information and Genie (when available) for workspace data.
Cite sources; say when a tool is missing or a result is incomplete."""
MODEL = 'system.ai.gemini-3-8-flash'
MCP_SERVERS = []

# END GENERATED

# Reasoning tokens are drawn from the same `max_tokens` budget as the visible
# answer (Gemini and Claude especially), so a high effort can exhaust the cap
# mid-response. "low" leaves the most room for the reply; raise it only if answer
# quality needs it. One of: none, low, medium, high.
REASONING_EFFORT = "low"

# Unity Gateway MCP Services are slower to initialize than managed Genie MCP.
# The agents SDK manager defaults to 10s and then *drops* a timed-out server
# from the tool list, which looks like "I have no web search tool".
MCP_CONNECT_TIMEOUT_SECONDS = 30.0

GENIE_MCP_PATH_PREFIX = "/api/2.0/mcp/genie/"

# UI renders Genie rows on the tool card; ```chart blocks render in the answer.
GENIE_VISUALIZATION_INSTRUCTIONS = """\
If Genie returns a chart, redraw it as a ```chart JSON block (same type, real numbers).
If it returns only rows, do not repeat them as a table or chart — describe the insight.
For other numbers, use a markdown table unless it is a time series, 8+ categories, or the user asked for a chart.
Chart JSON: {"type":"bar|horizontalBar|line|area|pie","title":"...","xKey":"...","series":[{"key":"...","label":"..."}],"data":[{...}]}.
Mermaid is for diagrams, not plots."""

# GenieMcpServer already waits; poll only if a query is still processing.
GENIE_PENDING_INSTRUCTIONS = """\
If Genie is still processing, poll with the returned ids until it completes. Do not tell the user to wait."""

NO_WEB_SEARCH_INSTRUCTIONS = """\
No web search. Use other tools and your knowledge; say when current facts are needed."""

# Gemini hosted search is a request parameter, not a function tool.
GEMINI_WEB_SEARCH_INSTRUCTIONS = """\
Live Google Search is available. Use it for current information. Put source links once at the end under Sources."""

MCP_WEB_SEARCH_INSTRUCTIONS = """\
You have a web search tool. Use it for current information. Put source links once at the end under Sources."""

DATE_CONTEXT_INSTRUCTIONS = """\
Before each web search, call get_todays_date and include that date in the query."""


def configured_model() -> str:
    """Name of the model to run, as the gateway should be asked for it.

    `AGENT_MODEL` keeps model choice a configuration change: set it in `.env` locally
    or in the app's env for a deployment to run `system.ai.claude-opus-5` or
    `system.ai.gemini-3-5-flash` without editing code.
    """
    return os.getenv("AGENT_MODEL", "").strip() or MODEL


def configured_reasoning_effort() -> str:
    """Reasoning effort for the selected model, overridable without code edits.

    Reasoning tokens count against the model's output-token budget, so a lower
    effort leaves more of `max_tokens` for the visible answer — the usual fix when
    responses cut off mid-stream. Set `AGENT_REASONING_EFFORT` in `.env` locally or
    the app env for a deployment. `model_profile` validates the value per family.
    """
    return os.getenv("AGENT_REASONING_EFFORT", "").strip() or REASONING_EFFORT


SELECTED_MODEL = configured_model()
SELECTED_REASONING_EFFORT = configured_reasoning_effort()
MODEL_PROFILE = model_profile(SELECTED_MODEL, SELECTED_REASONING_EFFORT)

_WEB_SEARCH_INSTRUCTIONS = {
    "google": [GEMINI_WEB_SEARCH_INSTRUCTIONS],
    "openai": [],
    "mcp": [MCP_WEB_SEARCH_INSTRUCTIONS],
    "off": [NO_WEB_SEARCH_INSTRUCTIONS],
}


def instructions_for(web_search: WebSearchMode) -> str:
    parts = [SYSTEM_PROMPT]
    if genie_space_ids():
        parts += [GENIE_VISUALIZATION_INSTRUCTIONS, GENIE_PENDING_INSTRUCTIONS]
    if web_search != "off":
        parts.append(DATE_CONTEXT_INSTRUCTIONS)
    parts.extend(_WEB_SEARCH_INSTRUCTIONS[web_search])
    return "\n\n".join(part.strip() for part in parts if part.strip())


logging.info(
    "Agent model %s (%s) via %s, reasoning effort %s, hosted web search %s, max_tokens %s",
    SELECTED_MODEL,
    model_family(SELECTED_MODEL),
    MODEL_PROFILE.api,
    SELECTED_REASONING_EFFORT,
    MODEL_PROFILE.web_search or "none",
    MODEL_PROFILE.max_tokens,
)


def build_model(profile: ModelProfile) -> Model:
    if profile.api == "responses":
        return OpenAIResponsesModel(model=SELECTED_MODEL, openai_client=GATEWAY_CLIENT)
    return GatewayChatCompletionsModel(model=SELECTED_MODEL, openai_client=GATEWAY_CLIENT)


def get_mcp_user_workspace_client():
    return get_user_workspace_client()


def genie_space_ids() -> List[str]:
    """Ids of the Genie spaces to attach as tools.

    `GENIE_SPACE_IDS` makes the agent portable: local runs read it from `.env`,
    while bundle deployments receive it from `BUNDLE_VAR_genie_space_ids`.
    Leaving it unset or empty runs the agent with web search only.
    """
    configured = os.getenv("GENIE_SPACE_IDS")
    if configured is None:
        return [
            url[len(GENIE_MCP_PATH_PREFIX) :]
            for _, url in MCP_SERVERS
            if url.startswith(GENIE_MCP_PATH_PREFIX)
        ]
    return [
        space_id.strip() for space_id in re.split(r"[,\s]+", configured) if space_id.strip()
    ]


def init_mcp_servers(web_search: WebSearchMode):
    user_workspace_client = get_mcp_user_workspace_client()

    def server(name: str, url: str, kind: type[McpServer] = McpServer) -> McpServer:
        return kind(
            name=name,
            url=build_mcp_url(url, user_workspace_client),
            workspace_client=user_workspace_client,
        )

    configured = [
        (name, url)
        for (name, url) in MCP_SERVERS
        if not url.startswith(GENIE_MCP_PATH_PREFIX)
    ]
    spec = web_search_mcp_spec(web_search, configured)
    if spec:
        configured.append(spec)

    servers = [server(name, url) for (name, url) in configured]
    servers.extend(
        server(
            genie_space_display_name(space_id, user_workspace_client),
            f"{GENIE_MCP_PATH_PREFIX}{space_id}",
            GenieMcpServer,  # polls Genie's own asynchronous queries to completion
        )
        for space_id in genie_space_ids()
    )
    return servers


def _mcp_server_url(server: MCPServer) -> str:
    params = getattr(server, "params", None) or {}
    return params.get("url") or ""


def _web_search_mcp_connected(servers: List[MCPServer]) -> bool:
    return any(
        is_web_search_mcp_server(getattr(server, "name", None), _mcp_server_url(server))
        for server in servers
    )


def _log_mcp_failures(manager: MCPServerManager) -> None:
    for server in manager.failed_servers:
        logging.error(
            "Failed to connect MCP server %s at %s: %s",
            getattr(server, "name", "?"),
            _mcp_server_url(server),
            manager.errors.get(server),
        )


def _web_search_mcp_server(workspace_client: WorkspaceClient) -> McpServer:
    return McpServer(
        name=WEB_SEARCH_MCP_NAME,
        url=build_mcp_url(WEB_SEARCH_MCP_PATH, workspace_client),
        workspace_client=workspace_client,
    )


@asynccontextmanager
async def connected_agent(web_search: WebSearchMode):
    """Connect MCP servers for one turn.

    ``MCPServerManager`` drops servers that fail to connect (timeout, 401 from
    a user token missing ``ai-gateway``, workspace where ``system.ai.web_search``
    is unavailable). If the Unity Gateway web-search server was requested and
    the user's token could not open it, retry once as the app service principal
    — public web search is not user-specific. If that also fails, run without
    search and tell the model so, instead of claiming a tool that is not there.
    """
    servers = init_mcp_servers(web_search)
    async with MCPServerManager(
        servers=servers,
        connect_in_parallel=True,
        connect_timeout_seconds=MCP_CONNECT_TIMEOUT_SECONDS,
    ) as manager:
        _log_mcp_failures(manager)
        active = list(manager.active_servers)
        if uses_web_search_mcp(web_search) and not _web_search_mcp_connected(active):
            logging.warning(
                "OBO connect to %s failed; retrying as the app service principal",
                WEB_SEARCH_MCP_PATH,
            )
            app_server = _web_search_mcp_server(WorkspaceClient())
            async with MCPServerManager(
                servers=[app_server],
                connect_timeout_seconds=MCP_CONNECT_TIMEOUT_SECONDS,
            ) as app_manager:
                _log_mcp_failures(app_manager)
                if app_manager.active_servers:
                    active = active + list(app_manager.active_servers)
                mode = effective_web_search_mode(
                    web_search,
                    web_search_mcp_connected=_web_search_mcp_connected(active),
                )
                if mode == "off":
                    logging.error(
                        "system.ai.web_search is not reachable with the signed-in "
                        "user token or the app identity; continuing without web search"
                    )
                yield create_agent(active, mode)
            return
        yield create_agent(active, web_search)


def create_agent(mcp_servers: List[MCPServer], web_search: WebSearchMode) -> Agent:
    return Agent(
        name=NAME,
        instructions=instructions_for(web_search),
        model=build_model(MODEL_PROFILE),
        mcp_servers=mcp_servers,
        # Genie and other MCP tools reach every model, since they are sent as
        # plain function tools. OpenAI hosted web search is a Responses tool;
        # Gemini's is the google_search extra-body field on chat completions.
        # Both are omitted when this workspace rejected them, in favour of
        # system.ai.web_search (see init_mcp_servers).
        tools=[
            get_todays_date,
            *([WebSearchTool()] if web_search == "openai" else []),
        ],
        model_settings=ModelSettings(
            reasoning=MODEL_PROFILE.reasoning,
            extra_body=extra_body_for(MODEL_PROFILE, web_search),
            # Per-model Databricks output cap. A value above the cap is a 400;
            # unset, some models (Claude Sonnet 4, GPT-5) stop far too early.
            max_tokens=MODEL_PROFILE.max_tokens,
        ),
    )


def conversation_items(request: ResponsesAgentRequest) -> List[dict]:
    """The conversation so far, in the shape the selected model's API accepts."""
    items = [item.model_dump() for item in request.input]
    if MODEL_PROFILE.api == "chat_completions":
        adapt_input_for_chat_completions(items)
    return items


@invoke()
async def invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    web_search = await resolved_web_search_mode(
        SELECTED_MODEL, MODEL_PROFILE, GATEWAY_CLIENT
    )
    async with connected_agent(web_search) as agent:
        messages = conversation_items(request)
        result = await Runner.run(agent, messages)
        return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])


@stream()
async def stream(request: dict) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    web_search = await resolved_web_search_mode(
        SELECTED_MODEL, MODEL_PROFILE, GATEWAY_CLIENT
    )
    async with connected_agent(web_search) as agent:
        messages = conversation_items(request)
        result = Runner.run_streamed(agent, input=messages)

        async for event in process_agent_stream_events(result.stream_events()):
            yield event
