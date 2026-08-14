# Agent Development Guide

## MANDATORY First Actions

**Ask the user interactively:**

1. **App deployment target:**
   > "Do you have an existing Databricks app you want to deploy to, or should we create a new one? If existing, what's the app name?"

   *Note: New apps should use the `agent-*` prefix (e.g., `agent-data-analyst`) unless the user specifies otherwise.*

2. **If the user mentions memory, conversation history, or persistence:**
 > "For memory capabilities, do you have an existing Lakebase instance? If so, what's the instance name?"

3. **If deploying to a workspace this checkout has not been set up for:**
 > "What experiment name, Unity Catalog catalog, and schema should hold MLflow traces?"
 > "Which Genie space ids should the agent attach as tools?"

 *Both are configuration, not code — see "Workspace-specific configuration" below.*

**Then set up the experiment and environment:**

1. Run `databricks auth profiles` and choose a valid profile.
2. Create or reuse the UC-backed experiment:
   ```bash
   uv run setup-mlflow-experiment \
     --profile <profile-name> \
     --experiment-name <workspace-experiment-path> \
     --catalog <catalog> \
     --schema <schema>
   ```
   This writes `.env` for local runs and target-specific DAB overrides under
   `.databricks/bundle/<target>/`. It also parks deployment state that belongs to
   another workspace as `.databricks/bundle/<target>@<host>/`, restoring it if that
   workspace is targeted again. **Always re-run it when switching a target to a
   different workspace** — see "Switching workspaces" below.

**CRITICAL: All `databricks` CLI commands must include the profile from `.env`.** Either use `--profile` or set the env var:

```bash
databricks <command> --profile <profile>
# or
DATABRICKS_CONFIG_PROFILE=<profile> databricks <command>
```

> **Why this matters:** Without the profile, the CLI may target the wrong workspace, causing "not found" errors for experiments, apps, or other resources.

## Understanding User Goals

**Ask the user questions to understand what they're building:**

1. **What is the agent's purpose?** (e.g., data analyst assistant, customer support, code helper)
2. **What data or tools does it need access to?**
   - Databases/tables (Unity Catalog)
   - Documents for RAG (Vector Search)
   - Natural language data queries (Genie Spaces)
   - External APIs or services
3. **Any specific Databricks resources they want to connect?**

Use `uv run discover-tools` to show them available resources in their workspace, then help them select the right ones for their use case. **See the `add-tools` skill for how to connect tools and grant permissions.**

## Handling Deployment Errors

**If `databricks bundle deploy` fails with "An app with the same name already exists":**

Ask the user: "I see there's an existing app with the same name. Would you like me to bind it to this bundle so we can manage it, or delete it and create a new one?"

- **If they want to bind**: See the **deploy** skill for binding steps
- **If they want to delete**: Run `databricks apps delete <app-name>` then deploy again

**If `databricks bundle deploy` panics with a nil pointer dereference in
`dresources.(*ResourceApp).OverrideChangeDesc`:**

The target's deployment state names an app the workspace does not have, and CLI
v1.10.0 crashes instead of reporting it. This happens when a target is pointed at a
new workspace, or when the app was deleted outside the bundle. Run
`uv run setup-mlflow-experiment --profile <profile> ...` for the workspace being
deployed to — it swaps the target's state — then deploy again.

### Switching workspaces

DAB keys deployment state by target name alone, so one target cannot track two
workspaces at once. The setup script parks the outgoing workspace's state as
`.databricks/bundle/<target>@<host>/` and restores it on return. Never delete
`.databricks/bundle/<target>/` to work around a workspace switch: that orphans the
app and Lakebase project already deployed in the other workspace, which then have
to be re-adopted with `databricks bundle deployment bind`.

## Supervisor API (Offloading the Agent Loop)

The **Supervisor API** lets Databricks run the tool-selection and agent loop server-side. Declare hosted tools (Genie spaces, UC functions, Knowledge Assistants, UC connection MCP servers, Databricks App endpoints) and call `responses.create()` — Databricks handles the rest.

**Use when the user wants to** connect Genie spaces, UC functions, or other Databricks-hosted tools without managing the agent loop themselves.

**Limitations:**
- Tools run as the app's service principal (no user token forwarding) — grant permissions in `databricks.yml`
- Cannot mix hosted tools with client-side function tools in the same request
- Inference parameters (`temperature`, `top_p`, etc.) are not supported when tools are passed
- `stream` and `background` cannot both be `true` in the same request
- Background mode has a maximum execution time of 30 minutes

**Skills:**
- Use **supervisor-api** to set up the Supervisor API with hosted tools
- Use **supervisor-api-background-mode** for tasks that may exceed HTTP timeout limits (complex multi-tool workflows, large data analysis)

## Long-Term Memory (Managed)

> **Beta.** Managed memory uses the Databricks memory-store APIs, which are in beta — APIs and behavior may change.

This template is **stateless by default** (the client carries conversation history). **Managed memory** adds durable memory that persists *across* conversations, backed by a **Unity Catalog memory store** (`catalog.schema.name`, type `MEMORY_STORE`) — governed by UC, with no database for you to provision or run. Databricks manages the underlying quality logic — deduplication, scope isolation, and fact reconciliation. The agent gets tools to store, retrieve, and modify memories, and memories are scoped per end user through code (the model never chooses the scope).

**Use when the user wants to** have the agent remember a user's preferences, facts, or decisions across separate sessions — without standing up or operating any storage infrastructure.

> **Managed memory vs. Lakebase.** Managed memory is the simplest, governed, zero-ops path. The alternative is **Lakebase** — long-term memories in your own Postgres instance (billable) with direct SQL access and embedding-based semantic search; choose it when you need SQL access or it's already part of your stack.

**Skills:**
- Use **managed-memory** to add the memory-store tools and per-user scope wiring.

## Agent Evaluation

When the user asks about evaluating their agent (quality, metrics, scorers, datasets, or tracing), suggest installing the **MLflow Skills** from https://github.com/mlflow/skills. These provide expert guidance for evaluation workflows using MLflow's native APIs.

**Relevant skills:**
- **agent-evaluation** — end-to-end evaluation: dataset creation, scorer selection, execution, result analysis
- **instrumenting-with-mlflow-tracing** — set up automatic tracing for debugging and observability
- **analyze-mlflow-trace** — examine span data and assessments to identify issues

**Install command:**
```bash
npx skills add mlflow/skills
```

After installation, the skills will be available as slash commands (e.g., `/agent-evaluation`). This template also includes a built-in `evaluate_agent.py` script — run it with `uv run agent-evaluate` after starting the local server.

---

## Available Skills

**Before executing any task, read the relevant skill file in `.claude/skills/`** - they contain tested commands, patterns, and troubleshooting steps.

| Task | Skill | Path |
|------|-------|------|
| Find tools/resources | **discover-tools** | `.claude/skills/discover-tools/SKILL.md` |
| Create tool resources | **create-tools** | `.claude/skills/create-tools/SKILL.md` |
| Deploy to Databricks | **deploy** | `.claude/skills/deploy/SKILL.md` |
| Add tools & permissions | **add-tools** | `.claude/skills/add-tools/SKILL.md` |
| Run/test locally | **run-locally** | `.claude/skills/run-locally/SKILL.md` |
| Modify agent code | **modify-agent** | `.claude/skills/modify-agent/SKILL.md` |
| Add long-term (cross-session) memory | **managed-memory** | `.claude/skills/managed-memory/SKILL.md` |
| Offload agent loop to Databricks | **supervisor-api** | `.claude/skills/supervisor-api/SKILL.md` |
| Long-running background tasks | **supervisor-api-background-mode** | `.claude/skills/supervisor-api-background-mode/SKILL.md` |

**Note:** All agent skills are located in `.claude/skills/` directory.

---

## Quick Commands

| Task | Command |
|------|---------|
| Set up UC-backed experiment | `uv run setup-mlflow-experiment --profile <p> --experiment-name <name> --catalog <catalog> --schema <schema>` |
| Pin the agent's model | add `--agent-model system.ai.claude-opus-5` to the setup command |
| Deploy on a different model once | `BUNDLE_VAR_agent_model=system.ai.claude-opus-5 databricks bundle deploy` |
| Discover tools | `uv run discover-tools` |
| Run locally | `uv run start-app` |
| Deploy and start | `databricks bundle deploy` |
| View logs | `databricks apps logs <app-name> --follow` |

---

## Workspace-specific configuration

The agent has no workspace ids in its code. `setup-mlflow-experiment` writes
local-development settings to `.env` and target-specific DAB overrides. DAB
deploys the app and Lakebase Autoscaling project:

| Setting | Effect |
|---------|--------|
| `MLFLOW_EXPERIMENT_ID` | Local experiment id written by `setup-mlflow-experiment`; DAB receives the same id from its target override. |
| `AGENT_MODEL` | Optional three-level model name (`system.ai.claude-opus-5`) for local runs, written by `setup-mlflow-experiment --agent-model`. Deployments receive `BUNDLE_VAR_agent_model` instead. Unset, the agent runs the model named in `agent_server/agent.py`. See "Model selection" below. |
| `GENIE_SPACE_IDS` | Optional local comma-separated Genie ids. Deployments receive `BUNDLE_VAR_genie_space_ids`; each signed-in user must have access. |
| `MLFLOW_TRACING_SQL_WAREHOUSE_ID` | Warehouse selected by experiment setup for creating and querying UC trace tables. |
| `MLFLOW_TRACE_LOCATION` | `catalog.schema.table_prefix` written for local development. |
| `LAKEBASE_PROJECT_ID` | Local only: `start-app` resolves the project's `production/primary` endpoint through the CLI and runs in Persistent mode. Unset, or unresolvable, means Ephemeral mode (in-memory history). The deployed app gets its connection from the bound `postgres` resource instead. |
| `BUNDLE_VAR_app_name` | Overrides the app name pinned by setup (`--app-name`, default `agent-web-search-genie-<target>`) for one deploy. |
| `BUNDLE_VAR_agent_model` | Overrides the model pinned by setup (`--agent-model`) for one deploy. |
| `BUNDLE_VAR_genie_space_ids` | Optional comma-separated Genie ids for the deployed app. Pin them in `variable-overrides.json` to keep them across deploys. |
| `BUNDLE_VAR_lakebase_project_id` | Overrides the Lakebase project pinned by setup for one deploy. Unset and unpinned, the deploy uses `<app-name>-lakebase` and creates it if missing. |

The setup script creates the four UC OTEL trace tables. DAB declares each table
as an app resource with `MODIFY` (which includes `SELECT`), so no post-deploy
grant script is needed.

---

## Model selection

Every model runs through the AI Gateway's OpenAI-compatible API
(`<host>/ai-gateway/openai/v1`), so one client serves every provider and models are named
by their three-level Unity Catalog name — `system.ai.claude-opus-5`,
`system.ai.gemini-3-5-flash`, `system.ai.gpt-5-6-sol`. Serving endpoint names
(`databricks-claude-opus-5`) reach the same models. A name the workspace does not serve
fails with `NOT_FOUND` or `ENDPOINT_NOT_FOUND`; `system.ai` lists what is registered,
which is not always what is deployed.

**Choose the model through configuration, not by editing `agent_server/agent.py`.** The
model is a bundle variable, so a redeploy can change it:

```bash
# One deploy on a different model.
BUNDLE_VAR_agent_model=system.ai.claude-opus-5 \
  databricks bundle deploy --target dev --profile <profile>

# Or pin it for every later deploy, and for local runs via .env.
uv run setup-mlflow-experiment --profile <profile> --agent-model system.ai.claude-opus-5 \
  --experiment-name <path> --catalog <catalog> --schema <schema>
```

Unpinned, the variable's whitespace default means no override and the app runs the model
named in `agent_server/agent.py`. So a deploy that omits `BUNDLE_VAR_agent_model` returns
a previously overridden app to that default — pin the model with `--agent-model` if it
should survive redeploys. `databricks apps get <app-name>` shows the `AGENT_MODEL` the
app is running with.

**The gateway is OpenAI-compatible, not uniform**, and swapping a model without
accounting for that is what breaks. `model_profile()` in `agent_server/agent.py` holds the
differences, all verified against the gateway:

| | GPT | Claude | Gemini | Llama, Kimi, GLM, GPT-OSS |
|---|---|---|---|---|
| API | Responses | chat completions | chat completions | chat completions |
| Reasoning | `reasoning.effort` (no `minimal`) | `thinking.type: adaptive` + `output_config.effort` (no `none`; `xhigh`/`max` also valid) | `reasoning_effort` (no `none`) | none assumed |
| Hosted web search | yes | no | no | no |

Only GPT models accept the Responses API — everything else answers `/responses` with
"Responses API passthrough is not supported for model ...". Hosted tools such as web
search exist only there, so on any other model the agent runs with its MCP tools alone
and is told it has no web search. Reasoning effort is validated at import: a value the
selected family rejects raises rather than failing on the first request.

Two Gemini quirks are absorbed by `GatewayOpenAI` and `GatewayChatCompletionsModel` in
`agent_server/utils.py`, so nothing else has to know about them. Google returns `content`
as a list of typed parts where chat completions specifies a string, and it rejects any
turn whose function calls come back without the `thoughtSignature` it issued — which the
agents SDK carries under a different name than the gateway reads. Tool results are
collapsed to a single string for the same reason.

MLflow autologging sees Gemini responses before that normalization, so it logs pydantic
serializer warnings and cannot aggregate streamed chunks (the gateway sends `id: null`).
Traces are still recorded; the warnings are noise.

The chat client sends earlier assistant turns back as Responses-shaped messages with no
id. The Responses API accepts that, while the agents SDK recognizes an assistant message
on the chat completions path only when it carries one — otherwise it raises "Unhandled item
type or structure" and every question after the first fails. `conversation_items()` in
`agent_server/agent.py` fills the id in for that path; it serves only that recognition and
never reaches the provider.

---

## Genie's asynchronous queries

A Genie question that outlives the MCP server's own wait comes back as a message id and a
status such as `EXECUTING_QUERY`, and the caller is expected to call
`poll_response_<space id>` until the message completes. `GenieMcpServer` in
`agent_server/utils.py` does that polling, so the query tool returns once, with the result.
Both Genie tool descriptions ask the model to poll instead, which strong models do and
small ones do not — they hand the "still processing" note to the user as though it answered
the question. Polling stops at `COMPLETED`, `FAILED`, `CANCELLED` or
`QUERY_RESULT_EXPIRED`, and gives up after 180s; the model keeps the poll tool and is told
to use it in that case.

---

## Persistent Agent Memory (Lakebase)

`databricks.yml` declares a Lakebase Autoscaling project and stops there. The
project creates its production branch and primary endpoint implicitly, and the app's
`postgres` resource points at that branch's `databricks_postgres` database by path,
so Databricks injects `PGHOST`/`PGUSER`/`PGDATABASE`/`PGPORT`/`PGSSLMODE` and
conversation history survives restarts.

**Do not declare `postgres_branches`, `postgres_endpoints`, or the project's
`default_endpoint_settings`.** Compute is workspace-tier-dependent: Free Edition
pins endpoints at 1 CU and answers any write to the scale-to-zero timeout with
`Auto-suspend timeout cannot be modified for this workspace tier`, which fails the
deploy. Declaring the endpoint also makes DAB plan a `recreate` of the branch
whenever its parent reference is unresolved, and recreating the production branch
destroys the chat history. Tune compute per workspace with
`databricks postgres update-endpoint <endpoint> "spec.suspension"` instead.

Removing those declarations from a bundle that already deployed them is not enough
on its own: the deploy then deletes the orphaned nodes and PATCHes
`spec.default_endpoint_settings` with an empty body. `bundle deployment unbind`
cannot help, because it resolves keys against the current config. `prune_retired_state`
in the setup script edits the target's `resources.json` to forget them, leaving the
live branch and endpoint running.

**Never hard-code the project id.** Unpinned, it defaults to `<app-name>-lakebase`,
which the deploy creates if it does not exist. `setup-mlflow-experiment` pins
something better: it reuses the project the state or overrides already name while
that project is live, adopts the newest live `<app-name>-lakebase*` project
otherwise, and only then mints `<app-name>-lakebase-<UTC timestamp>`. That last case
matters because a deleted project holds its id until it is purged a week later, so
after any deletion the default name is unusable and only a re-run frees the deploy.
The script also binds an adopted app or project and unbinds state naming one that is
gone — the two situations that otherwise fail the deploy.

**`lifecycle.prevent_destroy: true` protects the declared project.** It makes
`databricks bundle destroy` refuse to run and blocks any deploy that would replace
the project. Do not remove that block unless the user explicitly asks to delete
their agent memory.

That protection covers only the node that declares it. Deployment state left over
from a resource type the config no longer declares — such as the pre-Autoscaling
`database_instances.chatbot_lakebase` — is destroyed on the next deploy, and it can
name the same underlying Lakebase. If a deploy prompts to delete Lakebase data,
stop and check which node it names before approving. A deleted project is
recoverable for seven days:

```bash
databricks postgres undelete-project projects/<project-id> --profile <profile>
databricks bundle deployment bind chatbot_lakebase_project projects/<project-id> \
  --target <target> --profile <profile>
```

Its slug also stays reserved until it is purged, so creating a replacement under
the same name fails with "project slug already exists" until then.

Do not rename a deployed Lakebase project casually: DAB treats a rename as
delete-then-create, and `prevent_destroy` blocks replacement. Adopting an existing
project requires an explicit one-time `bundle deployment bind`.

> **Managed memory vs. Lakebase.** Managed memory (see the **managed-memory** skill) needs no
> instance to provision or protect. Prefer Lakebase when direct SQL access to the history
> matters or Postgres is already in the stack.

---

## Key Files

| File | Purpose |
|------|---------|
| `e2e-chatbot-app-next/` | Chat UI, vendored from `databricks/app-templates` and **locally customized** — do not delete or re-clone it |
| `.../client/src/lib/genie-result.ts` | Parses Genie MCP query results (schema + rows) out of tool output |
| `.../client/src/components/genie-chart.tsx` | Renders those Genie results as charts and tables |
| `agent_server/agent.py` | Agent logic, model, instructions, MCP servers (Genie spaces come from `GENIE_SPACE_IDS`) |
| `agent_server/start_server.py` | FastAPI server + MLflow setup |
| `agent_server/evaluate_agent.py` | Agent evaluation with MLflow scorers |
| `databricks.yml` | Bundle config & resource permissions |
| `scripts/setup_mlflow_experiment.py` | Creates/reuses the UC-backed experiment and writes DAB overrides |
| `scripts/start_app.py` | Runtime entry point for the agent server and chat UI |
| `scripts/discover_tools.py` | Discovers available workspace resources |

---

## Agent Framework Capabilities

> **⚠️ IMPORTANT:** When adding any tool to the agent, you MUST also grant permissions in `databricks.yml`. See the **add-tools** skill for required steps and examples.

**Tool Types:**
1. **Unity Catalog Function Tools** - SQL UDFs managed in UC with built-in governance
2. **Agent Code Tools** - Defined directly in agent code for REST APIs and low-latency operations
3. **MCP Tools** - Interoperable tools via Model Context Protocol (Databricks-managed, external, or self-hosted)

**Built-in Tools:**
- **system.ai.python_exec** - Execute Python code dynamically within agent queries (code interpreter)

**Common Patterns:**
- **Structured data retrieval** - Query SQL tables/databases
- **Unstructured data retrieval** - Document search and RAG via Vector Search
- **Code interpreter** - Python execution for analysis via system.ai.python_exec
- **External connections** - Integrate services like Slack via HTTP connections

Reference: https://docs.databricks.com/aws/en/generative-ai/agent-framework/
