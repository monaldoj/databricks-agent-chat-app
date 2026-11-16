import logging
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.entities.trace_location import UnityCatalog
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

# Load env vars from .env before importing the agent for proper auth
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
# catalog.schema only — cannot carry a table prefix, so it must not stay set.
os.environ.pop("MLFLOW_TRACING_DESTINATION", None)


def _configure_uc_trace_location() -> None:
    """Point tracing at the experiment's UC tables (catalog/schema/prefix).

    ``MLFLOW_TRACING_DESTINATION`` only accepts ``catalog.schema`` and cannot
    carry a table prefix, so the location is set here from env vars instead.

    Empty strings count as unset everywhere: when every trace variable is
    empty (the `nolakebase` target ships them as ""), the experiment is set
    WITHOUT a trace location so traces land in the experiment's own
    MLflow-managed storage instead of Unity Catalog.
    """
    catalog = os.getenv("TRACE_CATALOG", "").strip()
    schema = os.getenv("TRACE_SCHEMA", "").strip()
    prefix = os.getenv("TRACE_TABLE_PREFIX", "").strip()
    if not (catalog and schema and prefix):
        combined = os.getenv("MLFLOW_TRACE_LOCATION", "").strip()
        parts = [part.strip() for part in combined.split(".")]
        if len(parts) == 3 and all(parts):
            catalog, schema, prefix = parts
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "").strip()
    if not experiment_id:
        logging.warning(
            "MLFLOW_EXPERIMENT_ID is not set; tracing cannot be configured. "
            "Set MLFLOW_EXPERIMENT_ID plus TRACE_CATALOG, TRACE_SCHEMA, and "
            "TRACE_TABLE_PREFIX for Unity Catalog trace storage."
        )
        return
    if catalog and schema and prefix:
        mlflow.set_experiment(
            experiment_id=experiment_id,
            trace_location=UnityCatalog(
                catalog_name=catalog,
                schema_name=schema,
                table_prefix=prefix,
            ),
        )
    else:
        # No (or empty) UC trace configuration: use the experiment's own
        # MLflow-managed trace storage.
        logging.info(
            "UC trace location is not configured; tracing to the experiment's "
            "MLflow-managed storage."
        )
        mlflow.set_experiment(experiment_id=experiment_id)


_configure_uc_trace_location()

# Need to import the agent to register the functions with the server
import agent_server.agent  # noqa: E402
from agent_server.agent import GATEWAY_CLIENT, MODEL_PROFILE, SELECTED_MODEL  # noqa: E402
from agent_server.web_search import resolved_web_search_mode  # noqa: E402

agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
# Define the app as a module level variable to enable multiple workers
app = agent_server.app  # noqa: F841
setup_mlflow_git_based_version_tracking()


@app.on_event("startup")
async def prefetch_web_search_mode() -> None:
    """Resolve hosted vs MCP search before the first chat, so that request is not the probe."""
    try:
        await resolved_web_search_mode(SELECTED_MODEL, MODEL_PROFILE, GATEWAY_CLIENT)
    except Exception:
        logging.exception(
            "Web search mode probe failed; will retry on the first chat"
        )


def main():
    agent_server.run(app_import_string="agent_server.start_server:app")
