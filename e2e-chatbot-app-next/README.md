# Databricks Agent Chat App

Chat UI for a Databricks App that runs an OpenAI-compatible agent with web search, optional Genie spaces, MLflow traces in Unity Catalog, and persistent chat history in Lakebase.

This directory is the vendored chat UI. Clone and deploy from the **repository root**.

Repo: [https://github.com/monaldoj/databricks-agent-chat-app](https://github.com/monaldoj/databricks-agent-chat-app)

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Node.js 20+
- [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install)
- A workspace with a Unity Catalog catalog/schema, a SQL warehouse, Databricks Apps, and Lakebase Autoscaling

A Genie space is **not** required to get started. Attach one later with `BUNDLE_VAR_genie_space_ids` if you want natural-language SQL tools.

The recommended model is **`system.ai.gemini-3-6-flash`**.

## 1. Clone

```bash
git clone https://github.com/monaldoj/databricks-agent-chat-app.git
cd databricks-agent-chat-app
```

## 2. Log in to the workspace

```bash
databricks auth login --profile DEFAULT
export DATABRICKS_CONFIG_PROFILE=DEFAULT
```

Use `--profile DEFAULT` (or `DATABRICKS_CONFIG_PROFILE`) on every later CLI command so you target this workspace.

## 3. Set bundle variables

Export these before `databricks bundle deploy`. Replace the example values with IDs from **your** workspace.

```bash
export BUNDLE_VAR_app_name=agent-chat-app
export BUNDLE_VAR_lakebase_project_id=agent-chat-app-lakebase
export BUNDLE_VAR_experiment_id=<pull from experiment you created in MLflow>
export BUNDLE_VAR_sql_warehouse_id=<pull from compute section of databricks>
export BUNDLE_VAR_trace_catalog=<catalog you selected when setting up the MLflow experiment>
export BUNDLE_VAR_trace_schema=<schema you selected when setting up the MLflow experiment>
export BUNDLE_VAR_trace_table_prefix=<prefix you selected when setting up the MLflow experiment>
export BUNDLE_VAR_agent_model=system.ai.gemini-3-6-flash
```

| Variable | Required | Notes |
| --- | --- | --- |
| `BUNDLE_VAR_app_name` | Yes | Workspace-unique Databricks App name. Prefer the `agent-` prefix. |
| `BUNDLE_VAR_experiment_id` | Yes | Existing UC-backed MLflow experiment. |
| `BUNDLE_VAR_sql_warehouse_id` | Yes | Warehouse used to create and query UC trace tables. |
| `BUNDLE_VAR_trace_catalog` / `trace_schema` / `trace_table_prefix` | Yes | Catalog, schema, and prefix of the experiment's OTEL trace tables. |
| `BUNDLE_VAR_agent_model` | Recommended | Defaults to the model in `agent_server/agent.py` if unset. Use `system.ai.gemini-3-6-flash`. |
| `BUNDLE_VAR_lakebase_project_id` | No | Defaults to `<app-name>-lakebase`. The deploy creates the project if it does not exist. |
| `BUNDLE_VAR_genie_space_ids` | No | Comma-separated Genie space ids. Omit this to deploy without Genie. |

If you do not already have an MLflow experiment and trace tables, create them first:

```bash
uv run setup-mlflow-experiment \
  --profile DEFAULT \
  --experiment-name /Users/<user>/agent-chat-app \
  --catalog <catalog> \
  --schema <schema> \
  --app-name agent-chat-app \
  --agent-model system.ai.gemini-3-6-flash
```

That command writes `.env` and DAB overrides so you can skip the `BUNDLE_VAR_*` exports above. Re-run it whenever you point this checkout at a different workspace.

To add Genie on a later deploy (each signed-in user must have access to the space):

```bash
export BUNDLE_VAR_genie_space_ids=01f117dad52a14098f4f6b2153480c07
```

## 4. Deploy

From the repository root:

```bash
databricks bundle validate --profile DEFAULT
databricks bundle deploy --profile DEFAULT
```

That single deploy:

- Creates and starts the Databricks App
- Creates a Lakebase Autoscaling project (production branch and primary endpoint) if needed
- Binds the app to the project's `databricks_postgres` database
- Binds the UC-backed MLflow experiment, SQL warehouse, and trace tables

No `databricks bundle run` step is required. The app starts as part of the deploy.

Check status and logs:

```bash
databricks bundle summary --profile DEFAULT
databricks apps get agent-chat-app --profile DEFAULT
databricks apps logs agent-chat-app --follow --profile DEFAULT
```

To deploy to another bundle target (`test`, `prod`, and so on):

```bash
databricks bundle deploy --target prod --profile DEFAULT
```

## Local development

After the experiment setup command (or an equivalent `.env`):

```bash
uv run start-app
```

Open [http://localhost:8000](http://localhost:8000). Local chat history is ephemeral unless `.env` contains `LAKEBASE_PROJECT_ID`, `PGHOST`, or `POSTGRES_URL`.
