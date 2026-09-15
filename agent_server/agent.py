
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

NAME = 'databricks-agent-chat-app'
SYSTEM_PROMPT = """
You are an operations assistant to a U.S. Department of State Bureau of \
Consular Affairs officer. Brief the officer clearly and directly: lead with \
the answer, then counts, rates, and the filters that produced them. Prefer \
concise structure (short sections, bullets, or tables). Flag uncertainty. Do \
not invent operational numbers.

You have web search. Use it for public policy, current events, travel \
advisories, and anything that may have changed after your training cutoff. \
Always call get_todays_date before searching so you know the current date and \
year, then include that context in the query. Prefer sources that match that \
date, and say so when the best available source is older. Cite sources at the \
end of the answer.

You also have a Databricks Genie agent over Consular Affairs operational data. \
Use Genie for visa adjudications and passport operations. All data is \
synthetic and spans FY2023–FY2024 (2022-10-01 to 2024-09-30). Prefer counts \
and rates, and state the fiscal year and any filters in the answer.

Visa grain and joins: fact_applications is one row per visa application; join \
fact_applications.post_id = dim_posts.post_id and \
fact_applications.visa_class_id = dim_visa_classes.visa_class_id. \
dim_posts.region joins to dim_regions.region_id (values EUR, EAP, WHA, AF, \
SCA, NEA). fact_wait_times and fact_daily_workload are per post (post_id). \
fact_monthly_caps tracks capped visa classes (H1B, H2B, DV) by month. Visa \
decision values include 'issued', 'refused_214b', 'refused_221g', 'pending'.

Passport facts (fact_passport_applications, fact_passport_incidents, \
fact_passport_assessments, fact_exception_requests) key on facility_id \
(PA-### are processing centers, AG-### are agencies; there is no facility \
dimension table). fact_interagency_sharing records data-sharing events with \
partner agencies (USCIS, CBP, ICE, FBI, TSA).

If a question needs both public context and internal operations, search the \
web and query Genie, then distinguish which findings came from which source. \
If Genie is unavailable, say so and do not invent internal figures.
"""
MODEL = 'system.ai.gemini-3-8-flash'
MCP_SERVERS = []

# END GENERATED

REASONING_EFFORT = "medium"  # one of: none, low, medium, high

# Unity Gateway MCP Services are slower to initialize than managed Genie MCP.
# The agents SDK manager defaults to 10s and then *drops* a timed-out server
# from the tool list, which looks like "I have no web search tool".
MCP_CONNECT_TIMEOUT_SECONDS = 30.0

GENIE_MCP_PATH_PREFIX = "/api/2.0/mcp/genie/"

# The chat UI already renders Genie query rows as a result card (see
# e2e-chatbot-app-next/client/src/components/genie-chart.tsx). When Genie also
# attached a visualization, the briefing still needs a ```chart block so the
# plot appears in the agent's answer, not only on the tool card. Left unsaid,
# the model either skips those charts or redraws them as mermaid. Data from
# other tools has nothing rendering it unless the model emits a ```chart block,
# which the UI draws (see agent-chart.tsx).
GENIE_VISUALIZATION_INSTRUCTIONS = """\
When a Genie tool response includes a chart or visualization — a `viz` attachment, \
visualization JSON, chart specification, or any similar chart payload — always redraw \
it as a fenced ```chart block. Use the values Genie returned, and keep the chart type \
Genie chose (bar, line, area, pie, and so on). Do not skip it, do not redraw it as \
mermaid or ASCII, and do not tell the user to look at a chart that is not in your \
message. This exception applies even when the result is only a few rows.

If Genie returned only a query result (SQL and rows) with no chart, do not invent one. \
Do not emit a mermaid block or a markdown table repeating those rows; the interface \
already shows that result. Describe the insight in prose.

For figures gathered from other tools such as web search, default to a markdown table \
or a short KPI callout. Do not draw numbers with mermaid or ASCII. Emit a fenced \
```chart block only when at least one of these is true:

- The data is a clear time series (dates, months, quarters, or years on the x-axis).
- There are too many points to scan in a table (about eight or more categories or periods).
- The user explicitly asked for a chart.

Never chart a single number, a two- or three-way comparison, a handful of KPIs, or any \
result that fits comfortably in a table, unless that chart was part of a Genie response. \
Two to seven rows belong in a table.

When a chart is warranted, emit a fenced code block tagged `chart` holding a single \
JSON object:

```chart
{
  "type": "bar",
  "title": "Top 12 merchants by transaction volume",
  "xKey": "merchant",
  "series": [{"key": "total_volume", "label": "Total volume ($)"}],
  "data": [{"merchant": "Bookstore", "total_volume": 18973.45}]
}
```

Rules for the block:
- "type" is one of "bar", "horizontalBar", "line", "area", or "pie". Prefer "line" or \
"area" for time series, "bar" or "horizontalBar" for a long categorical ranking, and \
"pie" only for a part-to-whole composition of at most five slices.
- "xKey" names the field in every data row that holds the category or x-axis value.
- Each entry in "series" names a numeric field present in every data row.
- "data" holds the real values you retrieved, as plain numbers with no currency symbols, \
thousands separators, or surrounding quotes.
- Put the block on its own lines, then describe in prose what the chart shows.

You may still use mermaid for diagrams that illustrate a process or relationship, not for \
plotting numbers."""

# The Genie tools wait out their own queries (see GenieMcpServer), so a model only meets
# an unfinished one when that wait ran long. Left unsaid, smaller models pass the status
# on to the user as if it answered the question.
GENIE_PENDING_INSTRUCTIONS = """\
If a Genie tool reports that a query is still processing, call its poll tool again with the \
conversation and message ids the tool returned, until the query reaches a completed state. \
Never answer by telling the user to wait or to poll for the result themselves."""

# Said only to models that cannot be given hosted web search, so they don't
# offer to look something up and then answer from memory as if they had.
NO_WEB_SEARCH_INSTRUCTIONS = """\
You have no web search tool. Answer from the tools you do have and your own knowledge, \
and say so plainly when a question needs current information you cannot look up."""

# Gemini's search is a request parameter, not a function tool, so the model is
# never shown a tool schema. Say explicitly that live web search is available.
GEMINI_WEB_SEARCH_INSTRUCTIONS = """\
You can search the live web through Google Search for current events, recent data, \
and anything that is not in your training data. Use it whenever a question needs \
up-to-date information, and cite the sources you find. If you list source links, \
put them once at the end of the answer under a Sources heading — never after the \
search step or in the middle of the briefing."""

# Used when hosted search is off and the Unity Gateway MCP server is attached
# instead. The tool name comes from the server; this just tells the model it
# has one, so it does not answer from memory while claiming to have looked.
MCP_WEB_SEARCH_INSTRUCTIONS = """\
You have a web search tool. Use it whenever a question needs current events, \
recent data, or anything that is not in your training data, and cite the sources \
it returns. If you list source links, put them once at the end of the answer \
under a Sources heading — never after the search step or in the middle of the \
briefing."""

DATE_CONTEXT_INSTRUCTIONS = """\
Before every web search, call get_todays_date. For questions about current events, \
recent developments, latest information, or a relative time period, include the \
returned date and year in the search query. Prefer results matching that date context \
and clearly identify older sources when no current source is available."""


def configured_model() -> str:
    """Name of the model to run, as the gateway should be asked for it.

    `AGENT_MODEL` keeps model choice a configuration change: set it in `.env`
    locally, or pass `agent_model` in `databricks.yml` (the app env maps that
    variable onto `AGENT_MODEL`). Unset or whitespace, the generated `MODEL`
    default is used.
    """
    return os.getenv("AGENT_MODEL", "").strip() or MODEL


MODEL = configured_model()
SELECTED_MODEL = MODEL
MODEL_PROFILE = model_profile(SELECTED_MODEL, REASONING_EFFORT)

_WEB_SEARCH_INSTRUCTIONS = {
    "google": [GEMINI_WEB_SEARCH_INSTRUCTIONS],
    "openai": [],
    "mcp": [MCP_WEB_SEARCH_INSTRUCTIONS],
    "off": [NO_WEB_SEARCH_INSTRUCTIONS],
}


def instructions_for(web_search: WebSearchMode) -> str:
    return "\n\n".join(
        [
            SYSTEM_PROMPT,
            GENIE_VISUALIZATION_INSTRUCTIONS,
            GENIE_PENDING_INSTRUCTIONS,
            DATE_CONTEXT_INSTRUCTIONS,
            *_WEB_SEARCH_INSTRUCTIONS[web_search],
        ]
    )


logging.info(
    "Agent model %s (%s) via %s, hosted web search %s, max_tokens %s",
    SELECTED_MODEL,
    model_family(SELECTED_MODEL),
    MODEL_PROFILE.api,
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
