# Web Search + Genie Agent

A Databricks App that exposes an OpenAI Responses API agent with:

- Web search
- Optional Databricks Genie spaces
- MLflow traces stored in Unity Catalog
- A built-in chat UI
- Persistent chat history in Lakebase Autoscaling

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Node.js 20+
- [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install)
- A Databricks CLI profile with access to:
  - A Unity Catalog catalog and schema
  - A SQL warehouse
  - Databricks Apps and Lakebase Autoscaling

## Setup and deploy

### 1. Authenticate

```bash
databricks auth login --profile <profile>
export DATABRICKS_CONFIG_PROFILE=<profile>
```

### 2. Set up the MLflow experiment

```bash
uv run setup-mlflow-experiment \
  --profile "$DATABRICKS_CONFIG_PROFILE" \
  --experiment-name /Users/<user>/agent-web-search-genie \
  --catalog <catalog> \
  --schema <schema>
```

This command:

- Creates or reuses the named MLflow experiment
- Backs its traces with Unity Catalog tables
- Uses the normalized experiment leaf name as the table prefix
- Selects the first running accessible SQL warehouse
- Reuses the app's Lakebase project, or names a new one for the deploy to create
- Writes local settings to `.env`
- Writes DAB settings to `.databricks/bundle/dev/variable-overrides.json`

Use `--app-name <name>` to name the app (default `agent-web-search-genie-<target>`),
`--sql-warehouse-id <id>` to select a specific warehouse, or `--target prod` to
configure another bundle target. Re-running the command keeps the app, warehouse,
and Lakebase project it already resolved.

Re-run this command whenever you point a target at a different workspace. Its
ids are workspace-specific, and the command also swaps the target's deployment
state to match, so each workspace keeps its own (see [Deploying to a second
workspace](#deploying-to-a-second-workspace)).

### 3. Deploy

```bash
databricks bundle validate --profile "$DATABRICKS_CONFIG_PROFILE"
databricks bundle deploy --profile "$DATABRICKS_CONFIG_PROFILE"
```

That single deployment:

- Creates and starts the Databricks App
- Creates a Lakebase Autoscaling project, which brings its own production branch and endpoint
- Binds the app to the project's `databricks_postgres` database
- Binds the UC-backed MLflow experiment
- Grants the app access to the experiment, SQL warehouse, and trace tables

No post-deployment script or `databricks bundle run` is required.

## Deploying to a second workspace

Run the experiment setup command against the new profile before deploying:

```bash
uv run setup-mlflow-experiment --profile <new-profile> \
  --experiment-name /Users/<user>/agent-web-search-genie \
  --catalog <catalog> --schema <schema>
```

DAB keys deployment state by target name alone, under
`.databricks/bundle/<target>/`. The setup command parks the outgoing workspace's
state as `.databricks/bundle/<target>@<host>/` and restores it if you target that
workspace again, so both workspaces stay deployable from the same target.

Deploying with state from another workspace crashes the CLI (v1.10.0) with a nil
pointer panic in `ResourceApp.OverrideChangeDesc`: it plans against an app id the
workspace does not have. If you hit that, run the setup command above and deploy
again.

## Optional configuration

The setup command pins the app name and Lakebase project in
`.databricks/bundle/<target>/variable-overrides.json`; without it they default to
`agent-web-search-genie-<target>` and `<app-name>-lakebase`. Pass
`--genie-space-ids <space-id>,<space-id>` to pin Genie spaces in the same file, or
export them for a single deploy:

```bash
export BUNDLE_VAR_genie_space_ids=<space-id>,<space-id>
```

Genie calls use on-behalf-of user authentication. Each signed-in user must have
access to the configured Genie spaces.

## Lakebase compute settings

The bundle declares the Lakebase project and nothing below it, so each workspace
applies its own defaults to the production branch and its endpoint. Workspace tiers
disagree about compute — Free Edition pins its endpoints to 1 CU and rejects every
write to the scale-to-zero timeout, which fails any deploy that tries to set one.
Change the autoscaling range or the idle timeout per workspace instead:

```bash
databricks postgres update-endpoint \
  projects/<project-id>/branches/production/endpoints/primary \
  "spec.suspension" --json '{"spec": {"suspend_timeout_duration": "3600s"}}' \
  --profile <profile>
```

## Lakebase project names

By default the app uses a project named `<app-name>-lakebase`, created on the first
deploy. Deleting a project reserves its id until it is purged a week later, so that
default name stops working for a week after any deletion. The setup command avoids
the wait: it reuses the app's live project when there is one and otherwise pins a
timestamped id such as `agent-web-search-genie-dev-lakebase-202608131949`. It also
repairs the target's deployment state, binding an adopted app or project and
releasing state that names one which no longer exists.

To start over with an empty chat history, delete the project and re-run the setup
command — the next deploy creates a fresh one under a new id:

```bash
databricks postgres delete-project projects/<project-id> --profile <profile>
```

## Local development

After running the experiment setup command:

```bash
uv run start-app
```

Open <http://localhost:8000>.

Run only the API server:

```bash
uv run start-server --reload
```

Local chat history is ephemeral unless `.env` contains
`LAKEBASE_PROJECT_ID`, `PGHOST`, or `POSTGRES_URL`.

## Query the deployed app

```python
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI(workspace_client=WorkspaceClient())

response = client.responses.create(
    model="apps/<app-name>",
    input="What can you help me with?",
)
print(response)
```

## Key files

- `agent_server/agent.py` — agent, model, instructions, and tools
- `agent_server/start_server.py` — MLflow AgentServer entry point
- `scripts/setup_mlflow_experiment.py` — one-time workspace/target setup
- `scripts/start_app.py` — app runtime entry point
- `databricks.yml` — app, Lakebase, environment, and permissions
- `app.yaml` — non-DAB app configuration
- `e2e-chatbot-app-next/` — chat UI

## Resource lifecycle

The Lakebase Autoscaling project has `prevent_destroy: true` so normal bundle operations
cannot accidentally delete persistent chat history. Adopting an existing app or
Lakebase project requires an explicit one-time `databricks bundle deployment
bind`.
