
import logging
import os
import re
from dataclasses import dataclass
from agents.mcp import MCPServer, MCPServerManager
from typing import Any, AsyncGenerator, List, Literal

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
from databricks_openai.agents import McpServer
from openai.types.shared import Reasoning
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

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
SYSTEM_PROMPT = 'You are a helpful assistant.'
MODEL = 'system.ai.gpt-5-6-terra'
MCP_SERVERS = []

# END GENERATED

REASONING_EFFORT = "medium"  # one of: none, low, medium, high

GENIE_MCP_PATH_PREFIX = "/api/2.0/mcp/genie/"

# The chat UI already charts Genie query results from the rows Genie returned,
# tooltips and all (see e2e-chatbot-app-next/client/src/components/genie-chart.tsx).
# Left to itself the model also draws a mermaid xychart of the same figures, so the
# reader gets the same answer twice, and the redrawn copy is the one that can be
# wrong. Data from other tools has nothing rendering it unless the model emits a
# ```chart block, which the UI draws (see agent-chart.tsx).
GENIE_VISUALIZATION_INSTRUCTIONS = """\
Results from a Genie space are charted automatically in the interface, directly from the \
rows Genie returned. Never redraw them — do not emit a mermaid block (xychart-beta, pie, or \
otherwise), a fenced ```chart block, an ASCII chart, or a markdown table repeating those \
rows. Describe what the data shows in prose instead, and refer to the chart as something \
the user can already see.

This applies only to Genie results. For figures gathered from other tools such as web \
search, do not draw them with mermaid or ASCII. Once you have the data, emit a fenced \
code block tagged `chart` holding a single JSON object:

```chart
{
  "type": "bar",
  "title": "Top 5 merchants by transaction volume",
  "xKey": "merchant",
  "series": [{"key": "total_volume", "label": "Total volume ($)"}],
  "data": [{"merchant": "Bookstore", "total_volume": 18973.45}]
}
```

Rules for the block:
- "type" is one of "bar", "horizontalBar", "line", "area", or "pie".
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

# Said only to models that cannot be given the hosted web search tool, so they don't
# offer to look something up and then answer from memory as if they had.
NO_WEB_SEARCH_INSTRUCTIONS = """\
You have no web search tool. Answer from the tools you do have and your own knowledge, \
and say so plainly when a question needs current information you cannot look up."""


@dataclass(frozen=True)
class ModelProfile:
    """How one model family has to be addressed through the gateway.

    The gateway speaks OpenAI's API for every model, but not every model accepts every
    part of it, and the differences are what break a bare model swap:

    * Only OpenAI GPT models accept the Responses API. Anthropic and Google models —
      and the open-weight endpoints, GPT-OSS included — answer any request to
      `/responses` with "Responses API passthrough is not supported for model ...",
      so they have to go through `/chat/completions`.
    * Reasoning is named differently per provider. GPT takes `reasoning.effort`,
      Gemini takes `reasoning_effort` (and rejects "none"), and Claude takes neither:
      it wants Anthropic's adaptive thinking plus a separate `output_config.effort`.
    * Hosted tools such as web search only exist on the Responses API. Chat
      completions rejects any tool that isn't a function.
    """

    api: Literal["responses", "chat_completions"]
    reasoning: Reasoning | None
    # Provider-native request fields the OpenAI SDK has no parameter for. They have to
    # ride in the request body: passed as keyword arguments the SDK rejects them.
    extra_body: dict[str, Any] | None
    hosted_tools: bool


# Efforts each family accepts, and what a value the family rejects is sent as instead.
# GPT has no "minimal" on the Responses API; Gemini has no "none", so the lightest
# thinking level it does take stands in for it.
_GPT_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})
_GEMINI_EFFORTS = {
    "none": "minimal",
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
}
_CLAUDE_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def configured_model() -> str:
    """Name of the model to run, as the gateway should be asked for it.

    `AGENT_MODEL` keeps model choice a configuration change: set it in `.env` locally
    or in the app's env for a deployment to run `system.ai.claude-opus-5` or
    `system.ai.gemini-3-5-flash` without editing code.
    """
    return os.getenv("AGENT_MODEL", "").strip() or MODEL


def model_family(model: str) -> Literal["gpt", "claude", "gemini", "other"]:
    """Provider family behind a gateway model name.

    Both naming schemes reach the same model and are reduced to the same bare name, so
    `system.ai.gpt-5-6-sol` and `databricks-gpt-5-6-sol` are read alike.
    """
    name = model.rsplit(".", 1)[-1].removeprefix("databricks-").lower()
    if name.startswith("gpt-oss"):
        return "other"  # open weights, and not on the Responses API
    if name.startswith("gpt-"):
        return "gpt"
    if "claude" in name:
        return "claude"
    if "gemini" in name:
        return "gemini"
    return "other"


def model_profile(model: str, effort: str) -> ModelProfile:
    family = model_family(model)

    if family == "gpt":
        if effort not in _GPT_EFFORTS:
            raise ValueError(
                f"Reasoning effort {effort!r} is not accepted for {model}; "
                f"use one of {sorted(_GPT_EFFORTS)}."
            )
        return ModelProfile(
            api="responses",
            reasoning=Reasoning(effort=effort),
            extra_body=None,
            hosted_tools=True,
        )

    if family == "claude":
        # Anthropic decides per turn whether to think, and takes the effort separately.
        if effort == "none":
            thinking: dict[str, Any] = {"thinking": {"type": "disabled"}}
        elif effort in _CLAUDE_EFFORTS:
            thinking = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
            }
        else:
            raise ValueError(
                f"Reasoning effort {effort!r} is not accepted for {model}; "
                f"use 'none' or one of {sorted(_CLAUDE_EFFORTS)}."
            )
        return ModelProfile(
            api="chat_completions",
            reasoning=None,
            extra_body=thinking,
            hosted_tools=False,
        )

    if family == "gemini":
        if effort not in _GEMINI_EFFORTS:
            raise ValueError(
                f"Reasoning effort {effort!r} is not accepted for {model}; "
                f"use one of {sorted(_GEMINI_EFFORTS)}."
            )
        # The agents SDK sends ModelSettings.reasoning.effort as `reasoning_effort` on
        # the chat completions path, which is the field Gemini wants.
        return ModelProfile(
            api="chat_completions",
            reasoning=Reasoning(effort=_GEMINI_EFFORTS[effort]),
            extra_body=None,
            hosted_tools=False,
        )

    # Llama, Kimi, GLM, Qwen, GPT-OSS and the like: chat completions, and no reasoning
    # control that is safe to assume across them.
    return ModelProfile(
        api="chat_completions",
        reasoning=None,
        extra_body=None,
        hosted_tools=False,
    )


SELECTED_MODEL = configured_model()
MODEL_PROFILE = model_profile(SELECTED_MODEL, REASONING_EFFORT)

INSTRUCTIONS = "\n\n".join(
    [SYSTEM_PROMPT, GENIE_VISUALIZATION_INSTRUCTIONS, GENIE_PENDING_INSTRUCTIONS]
    + ([] if MODEL_PROFILE.hosted_tools else [NO_WEB_SEARCH_INSTRUCTIONS])
)

logging.info(
    "Agent model %s (%s) via %s, web search %s",
    SELECTED_MODEL,
    model_family(SELECTED_MODEL),
    MODEL_PROFILE.api,
    "on" if MODEL_PROFILE.hosted_tools else "off",
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


def init_mcp_servers():
    user_workspace_client = get_mcp_user_workspace_client()

    def server(name: str, url: str, kind: type[McpServer] = McpServer) -> McpServer:
        return kind(
            name=name,
            url=build_mcp_url(url, user_workspace_client),
            workspace_client=user_workspace_client,
        )

    servers = [
        server(name, url)
        for (name, url) in MCP_SERVERS
        if not url.startswith(GENIE_MCP_PATH_PREFIX)
    ]
    servers.extend(
        server(
            genie_space_display_name(space_id, user_workspace_client),
            f"{GENIE_MCP_PATH_PREFIX}{space_id}",
            GenieMcpServer,  # polls Genie's own asynchronous queries to completion
        )
        for space_id in genie_space_ids()
    )
    return servers

def create_agent(mcp_servers: List[MCPServer]) -> Agent:
    return Agent(
        name=NAME,
        instructions=INSTRUCTIONS,
        model=build_model(MODEL_PROFILE),
        mcp_servers=mcp_servers,
        # Genie and other MCP tools reach every model, since they are sent as plain
        # function tools. Hosted web search only exists on the Responses API.
        tools=[WebSearchTool()] if MODEL_PROFILE.hosted_tools else [],
        model_settings=ModelSettings(
            reasoning=MODEL_PROFILE.reasoning,
            extra_body=MODEL_PROFILE.extra_body,
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
    mcp_servers = init_mcp_servers()
    async with MCPServerManager(servers = mcp_servers, connect_in_parallel=True) as manager:
        agent = create_agent(manager.active_servers)
        messages = conversation_items(request)
        result = await Runner.run(agent, messages)
        return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])


@stream()
async def stream(request: dict) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    mcp_servers = init_mcp_servers()
    async with MCPServerManager(servers = mcp_servers, connect_in_parallel=True) as manager:
        agent = create_agent(manager.active_servers)
        messages = conversation_items(request)
        result = Runner.run_streamed(agent, input=messages)

        async for event in process_agent_stream_events(result.stream_events()):
            yield event
