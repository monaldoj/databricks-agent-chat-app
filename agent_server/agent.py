
from agents.mcp import MCPServer, MCPServerManager
from typing import AsyncGenerator, List

import mlflow
from agents import (
    Agent,
    ModelSettings,
    Runner,
    WebSearchTool,
    set_default_openai_api,
    set_default_openai_client,
)
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from openai.types.shared import Reasoning
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.utils import (
    build_mcp_url,
    get_user_workspace_client,
    process_agent_stream_events,
)

# The Responses API is required for reasoning alongside tools, and for hosted tools
# such as web search. GPT-OSS models need "chat_completions" instead.
set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("responses")
set_trace_processors([])  # only use mlflow for trace processing
mlflow.openai.autolog()

# GENERATED

NAME = 'agent-web-search-genie'
SYSTEM_PROMPT = 'You are a helpful assistant.'
MODEL = 'databricks-gpt-5-6-terra'
MCP_SERVERS = [
    ('Genie Space: Campus Card Swipe Analytics', '/api/2.0/mcp/genie/01f169d6d34a1233bb6d5ab580b58495'),
]

# END GENERATED

REASONING_EFFORT = "medium"  # one of: none, low, medium, high

# The chat UI already charts Genie query results from the rows Genie returned,
# tooltips and all (see e2e-chatbot-app-next/client/src/components/genie-chart.tsx).
# Left to itself the model also draws a mermaid xychart of the same figures, so the
# reader gets the same answer twice, and the redrawn copy is the one that can be
# wrong. Data from other tools has nothing rendering it, so charts stay allowed there.
GENIE_VISUALIZATION_INSTRUCTIONS = """\
Results from a Genie space are charted automatically in the interface, directly from the \
rows Genie returned. Never redraw them — do not emit a mermaid block (xychart-beta, pie, or \
otherwise), an ASCII chart, or a markdown table repeating those rows. Describe what the \
data shows in prose instead, and refer to the chart as something the user can already see.

This applies only to Genie results. You may still use mermaid for diagrams that illustrate \
a process or relationship, and for charting figures gathered from other tools such as web \
search."""

INSTRUCTIONS = f"{SYSTEM_PROMPT}\n\n{GENIE_VISUALIZATION_INSTRUCTIONS}"


def get_mcp_user_workspace_client():
    return get_user_workspace_client()


def init_mcp_servers():
    user_workspace_client = get_mcp_user_workspace_client()
    return [
        McpServer(
            name=name,
            url=build_mcp_url(url, user_workspace_client),
            workspace_client=user_workspace_client,
        )
        for (name, url) in MCP_SERVERS
    ]

def create_agent(mcp_servers: List[MCPServer]) -> Agent:
    return Agent(
        name=NAME,
        instructions=INSTRUCTIONS,
        model=MODEL,
        mcp_servers=mcp_servers,
        tools=[WebSearchTool()],
        model_settings=ModelSettings(reasoning=Reasoning(effort=REASONING_EFFORT)),
    )


@invoke()
async def invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    mcp_servers = init_mcp_servers()
    async with MCPServerManager(servers = mcp_servers, connect_in_parallel=True) as manager:
        agent = create_agent(manager.active_servers)
        messages = [i.model_dump() for i in request.input]
        result = await Runner.run(agent, messages)
        return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])


@stream()
async def stream(request: dict) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    mcp_servers = init_mcp_servers()
    async with MCPServerManager(servers = mcp_servers, connect_in_parallel=True) as manager:
        agent = create_agent(manager.active_servers)
        messages = [i.model_dump() for i in request.input]
        result = Runner.run_streamed(agent, input=messages)

        async for event in process_agent_stream_events(result.stream_events()):
            yield event
