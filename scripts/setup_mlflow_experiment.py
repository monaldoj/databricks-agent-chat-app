#!/usr/bin/env python3
"""Create or reuse an MLflow experiment backed by Unity Catalog trace tables.

The script writes the resolved workspace ids to DAB's target-specific
``variable-overrides.json`` file. After it succeeds, deployment is simply:

    databricks bundle deploy --target <target> --profile <profile>
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.errors import NotFound
from mlflow.entities.trace_location import UnityCatalog

HOST_MARKER = ".workspace-host"

# Resource keys, as declared in databricks.yml.
APP_KEY = "responses_agent_chat_app"
PROJECT_KEY = "chatbot_lakebase_project"

# The bundle used to declare the project's branch and endpoint. State written then
# still names them, and a deploy deletes state whose node the config dropped, taking
# the live branch — and the chat history on it — with it.
RETIRED_NODES = (
    ("postgres_endpoints", "chatbot_lakebase_primary"),
    ("postgres_branches", "chatbot_lakebase_production"),
)


def table_prefix(experiment_name: str) -> str:
    """Convert the experiment's leaf name to a valid UC table prefix."""
    leaf_name = experiment_name.rstrip("/").rsplit("/", 1)[-1]
    prefix = re.sub(r"[^A-Za-z0-9_]", "_", leaf_name).strip("_")
    if not prefix:
        raise ValueError("The experiment name must contain at least one letter or digit.")
    if len(prefix) > 238:
        raise ValueError("The normalized experiment name exceeds the 238-character limit.")
    return prefix


def resolve_warehouse_id(
    workspace: WorkspaceClient, requested_id: str | None, pinned_id: str = ""
) -> str:
    if requested_id:
        workspace.warehouses.get(requested_id)
        return requested_id

    # Re-runs stay on the warehouse this target already deploys with, unless it is gone.
    if pinned_id:
        try:
            workspace.warehouses.get(pinned_id)
            return pinned_id
        except NotFound:
            pass

    warehouses = [warehouse for warehouse in workspace.warehouses.list() if warehouse.id]
    if not warehouses:
        raise RuntimeError(
            "No SQL warehouse is available. Create one or pass --sql-warehouse-id."
        )

    running = [
        warehouse
        for warehouse in warehouses
        if str(getattr(warehouse.state, "value", warehouse.state)).upper() == "RUNNING"
    ]
    selected = (running or warehouses)[0]
    print(f"Using SQL warehouse '{selected.name}' ({selected.id}).")
    return selected.id


def ensure_schema(workspace: WorkspaceClient, catalog: str, schema: str) -> None:
    workspace.catalogs.get(catalog)
    full_name = f"{catalog}.{schema}"
    try:
        workspace.schemas.get(full_name)
        print(f"Using existing schema '{full_name}'.")
    except Exception:
        workspace.schemas.create(name=schema, catalog_name=catalog)
        print(f"Created schema '{full_name}'.")


def configure_experiment(
    experiment_name: str,
    catalog: str,
    schema: str,
    prefix: str,
) -> str:
    location = UnityCatalog(
        catalog_name=catalog,
        schema_name=schema,
        table_prefix=prefix,
    )
    experiment = mlflow.set_experiment(
        experiment_name=experiment_name,
        trace_location=location,
    )
    if not experiment.experiment_id:
        raise RuntimeError("MLflow did not return an experiment id.")
    return experiment.experiment_id


def state_dir(target: str) -> Path:
    return Path(".databricks") / "bundle" / target


def workspace_slug(host: str) -> str:
    bare_host = host.removeprefix("https://").removeprefix("http://").rstrip("/")
    return re.sub(r"[^a-z0-9]+", "-", bare_host.lower()).strip("-")


def state_app_name(resources_path: Path) -> str:
    """Return the app name recorded in a target's deployment state, if any."""
    try:
        saved = json.loads(resources_path.read_text()).get("state", {})
    except (OSError, json.JSONDecodeError):
        return ""
    for node, entry in saved.items():
        if node.startswith("resources.apps."):
            return entry.get("__id__", "")
    return ""


def previous_host() -> str:
    """Best-effort host of the last setup run, for naming its parked state."""
    path = Path(".env")
    if not path.exists():
        return ""
    match = re.search(r"^\s*DATABRICKS_CONFIG_PROFILE=(.+)$", path.read_text(), re.MULTILINE)
    if not match:
        return ""
    try:
        return Config(profile=match.group(1).strip()).host or ""
    except Exception:
        return ""


def app_exists(workspace: WorkspaceClient, app_name: str) -> bool:
    if not app_name:
        return False
    try:
        workspace.apps.get(name=app_name)
    except NotFound:
        return False
    return True


def state_belongs_to_workspace(workspace: WorkspaceClient, resources_path: Path) -> bool:
    app_name = state_app_name(resources_path)
    return not app_name or app_exists(workspace, app_name)


def align_bundle_state(workspace: WorkspaceClient, target: str, host: str) -> None:
    """Keep one deployment-state directory per workspace for this bundle target.

    DAB keys deployment state by target name alone, so pointing a target at a
    second workspace leaves the CLI planning against resource ids that workspace
    never had. For apps that crashes the CLI outright (nil pointer in
    ResourceApp.OverrideChangeDesc), so foreign state is parked under
    <target>@<host> instead, and restored if that workspace is targeted again.
    """
    active = state_dir(target)
    marker = active / HOST_MARKER
    resources_path = active / "resources.json"
    recorded_host = marker.read_text().strip() if marker.exists() else ""
    if recorded_host == host:
        return

    # State written before this repo tracked hosts may still be this workspace's.
    if not recorded_host:
        attributed = workspace_slug(previous_host())
        if attributed == workspace_slug(host) or (
            resources_path.exists() and state_belongs_to_workspace(workspace, resources_path)
        ):
            active.mkdir(parents=True, exist_ok=True)
            marker.write_text(host + "\n")
            return

    if active.exists() and any(active.iterdir()):
        slug = workspace_slug(recorded_host or previous_host()) or "unknown"
        parked = active.parent / f"{target}@{slug}"
        # Newer state supersedes older state for the same host, but state that
        # cannot be attributed to one is kept under a free name.
        counter = 2
        while parked.exists() and slug == "unknown":
            parked = active.parent / f"{target}@unknown-{counter}"
            counter += 1
        if parked.exists():
            shutil.rmtree(parked)
        active.rename(parked)
        print(f"Parked deployment state for another workspace in '{parked}'.")

    restored = active.parent / f"{target}@{workspace_slug(host)}"
    if restored.exists():
        restored.rename(active)
        print(f"Restored this workspace's deployment state from '{restored}'.")

    active.mkdir(parents=True, exist_ok=True)
    (active / HOST_MARKER).write_text(host + "\n")


def overrides_path(target: str) -> Path:
    return state_dir(target) / "variable-overrides.json"


def read_bundle_overrides(target: str) -> dict[str, str]:
    path = overrides_path(target)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_bundle_overrides(target: str, values: dict[str, str]) -> Path:
    path = overrides_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    overrides = read_bundle_overrides(target)
    overrides.update(values)
    path.write_text(json.dumps(overrides, indent=2) + "\n")
    return path


def state_resource_id(target: str, node: str) -> str:
    """Return the workspace id the target's state records for a bundle resource."""
    resources_path = state_dir(target) / "resources.json"
    if not resources_path.exists():
        return ""
    try:
        saved = json.loads(resources_path.read_text()).get("state", {})
    except (OSError, json.JSONDecodeError):
        return ""
    return saved.get(node, {}).get("__id__", "")


def project_is_live(workspace: WorkspaceClient, project_id: str) -> bool:
    if not project_id:
        return False
    try:
        project = workspace.postgres.get_project(name=f"projects/{project_id}")
    except NotFound:
        return False
    return not project.delete_time


def newest_live_project(workspace: WorkspaceClient, base: str) -> str:
    """Most recently created live project this bundle would have named."""
    candidates = [
        project
        for project in workspace.postgres.list_projects()
        if not project.delete_time
        and project.project_id
        and (project.project_id == base or project.project_id.startswith(f"{base}-"))
    ]
    if not candidates:
        return ""
    candidates.sort(key=lambda project: project.create_time or "")
    return candidates[-1].project_id


def resolve_lakebase_project(workspace: WorkspaceClient, target: str, app_name: str) -> str:
    """Reuse the app's Lakebase project, or mint a timestamped id for a new one.

    A deleted project keeps its id reserved until it is purged a week later, so a
    fixed id makes the bundle undeployable for that week once anything deletes the
    project. Ids are therefore minted per project and pinned in the target's
    overrides, and only reused while the project they name is still live.
    """
    base = f"{app_name}-lakebase"
    tracked = state_resource_id(target, f"resources.postgres_projects.{PROJECT_KEY}")
    for candidate in (
        tracked.removeprefix("projects/"),
        read_bundle_overrides(target).get("lakebase_project_id", ""),
    ):
        if project_is_live(workspace, candidate):
            print(f"Reusing Lakebase project '{candidate}'.")
            return candidate

    discovered = newest_live_project(workspace, base)
    if discovered:
        print(f"Adopting existing Lakebase project '{discovered}'.")
        return discovered

    minted = f"{base[:48]}-{datetime.now(timezone.utc):%Y%m%d%H%M}"
    print(f"Deploy will create Lakebase project '{minted}'.")
    return minted


def run_bundle_command(
    arguments: list[str], target: str, profile: str, tolerate: tuple[str, ...] = ()
) -> bool:
    """Run a bundle command, reporting whether it changed anything.

    Deployment state also lives in the workspace, so a binding this checkout does not
    know about may already exist; `tolerate` covers the CLI's complaints about that.
    """
    command = ["databricks", "bundle", *arguments, "--target", target, "--profile", profile]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return True
    output = f"{result.stdout}\n{result.stderr}"
    if any(text in output for text in tolerate):
        return False
    raise RuntimeError(f"`{' '.join(command)}` failed:\n{result.stderr.strip()}")


ALREADY_BOUND = ("already managed", "already bound")
NOT_BOUND = ("not managed", "not bound", "does not exist in the bundle")
# Matches any output, for commands run only for a side effect worth attempting.
ANY_FAILURE = ("",)


def reconcile_app_binding(
    workspace: WorkspaceClient, target: str, profile: str, app_name: str
) -> None:
    """Point the target's state at the app it will deploy.

    Without this, a target whose state was cleared plans to create an app that
    already exists, and one whose state names a renamed or deleted app plans
    against an id the workspace cannot resolve.
    """
    tracked = state_resource_id(target, f"resources.apps.{APP_KEY}")
    exists = app_exists(workspace, app_name)

    if tracked and (tracked != app_name or not exists):
        if run_bundle_command(["deployment", "unbind", APP_KEY], target, profile, NOT_BOUND):
            print(f"Released stale app state for '{tracked}'.")
        tracked = ""

    if not tracked and exists:
        if run_bundle_command(
            ["deployment", "bind", APP_KEY, app_name, "--auto-approve"],
            target,
            profile,
            ALREADY_BOUND,
        ):
            print(f"Bound app '{app_name}' to this bundle.")


def reconcile_lakebase_binding(
    target: str, profile: str, project_id: str, project_exists: bool
) -> None:
    """Point the target's state at the resolved project.

    State naming a different project makes the deploy plan a rename, which
    `prevent_destroy` blocks; state naming no project makes it plan a create, which
    fails when the project already exists.
    """
    tracked = state_resource_id(target, f"resources.postgres_projects.{PROJECT_KEY}")
    if tracked == f"projects/{project_id}":
        return

    if tracked:
        run_bundle_command(["deployment", "unbind", PROJECT_KEY], target, profile, NOT_BOUND)
        print(f"Released stale Lakebase state for '{tracked}'.")

    if project_exists:
        if run_bundle_command(
            ["deployment", "bind", PROJECT_KEY, f"projects/{project_id}", "--auto-approve"],
            target,
            profile,
            ALREADY_BOUND,
        ):
            print(f"Bound Lakebase project '{project_id}' to this bundle.")


def prune_retired_state(target: str, profile: str) -> None:
    """Forget the Lakebase compute the bundle used to declare, leaving it running.

    A deploy deletes state whose node the config dropped and PATCHes fields it stopped
    declaring. Here both are damaging: deleting the production branch destroys the chat
    history, and writing endpoint compute settings fails outright on workspace tiers
    that pin them. `bundle deployment unbind` cannot reach either, since it resolves
    keys against the current config, so the state file is edited directly.
    """
    resources_path = state_dir(target) / "resources.json"
    if not resources_path.exists():
        # Hydrate from the workspace copy, which a fresh checkout has yet to download.
        run_bundle_command(["plan"], target, profile, ANY_FAILURE)
    try:
        document = json.loads(resources_path.read_text())
    except (OSError, json.JSONDecodeError):
        return

    saved = document.get("state", {})
    retired = [f"resources.{node_type}.{key}" for node_type, key in RETIRED_NODES]
    dropped = [node for node in retired if saved.pop(node, None) is not None]

    project = saved.get(f"resources.postgres_projects.{PROJECT_KEY}", {}).get("state", {})
    if project.pop("default_endpoint_settings", None) is not None:
        dropped.append("the project's default endpoint settings")

    if not dropped:
        return
    resources_path.write_text(json.dumps(document, indent=2) + "\n")
    print(f"Stopped tracking {', '.join(dropped)}; the live resources are untouched.")


def update_env_file(values: dict[str, str]) -> None:
    """Mirror resolved values to .env for local development."""
    path = Path(".env")
    lines = path.read_text().splitlines() if path.exists() else []

    for key, value in values.items():
        replacement = f"{key}={value}"
        for index, line in enumerate(lines):
            if re.match(rf"^\s*#?\s*{re.escape(key)}=", line):
                lines[index] = replacement
                break
        else:
            lines.append(replacement)

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up a UC-backed MLflow experiment for this bundle."
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument(
        "--profile",
        default="DEFAULT",
        help="Databricks CLI profile used for all workspace operations (default: DEFAULT).",
    )
    parser.add_argument("--target", default="dev")
    parser.add_argument(
        "--app-name",
        default="",
        help="Databricks app name; defaults to agent-web-search-genie-<target>.",
    )
    parser.add_argument(
        "--sql-warehouse-id",
        default="",
        help="Optional warehouse id; defaults to the first running accessible warehouse.",
    )
    parser.add_argument(
        "--genie-space-ids",
        default="",
        help="Optional comma-separated Genie space ids to pin for deployments.",
    )
    args = parser.parse_args()

    try:
        prefix = table_prefix(args.experiment_name)
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
        os.environ["MLFLOW_TRACKING_URI"] = "databricks"

        workspace = WorkspaceClient(profile=args.profile)
        align_bundle_state(workspace, args.target, workspace.config.host)
        warehouse_id = resolve_warehouse_id(
            workspace,
            args.sql_warehouse_id or None,
            read_bundle_overrides(args.target).get("sql_warehouse_id", ""),
        )
        os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = warehouse_id
        mlflow.set_tracking_uri("databricks")

        ensure_schema(workspace, args.catalog, args.schema)
        experiment_id = configure_experiment(
            args.experiment_name,
            args.catalog,
            args.schema,
            prefix,
        )
        trace_location = f"{args.catalog}.{args.schema}.{prefix}"

        app_name = args.app_name or f"agent-web-search-genie-{args.target}"
        project_id = resolve_lakebase_project(workspace, args.target, app_name)

        genie_space_ids = args.genie_space_ids or read_bundle_overrides(args.target).get(
            "genie_space_ids", ""
        )
        overrides = {
            "app_name": app_name,
            "experiment_id": experiment_id,
            "trace_catalog": args.catalog,
            "trace_schema": args.schema,
            "trace_table_prefix": prefix,
            "sql_warehouse_id": warehouse_id,
            "lakebase_project_id": project_id,
        }
        if genie_space_ids:
            overrides["genie_space_ids"] = genie_space_ids
        override_path = write_bundle_overrides(args.target, overrides)
        prune_retired_state(args.target, args.profile)
        reconcile_app_binding(workspace, args.target, args.profile, app_name)
        reconcile_lakebase_binding(
            args.target,
            args.profile,
            project_id,
            project_is_live(workspace, project_id),
        )
        local_values = {
            "DATABRICKS_CONFIG_PROFILE": args.profile,
            "MLFLOW_EXPERIMENT_ID": experiment_id,
            "MLFLOW_TRACE_LOCATION": trace_location,
            "MLFLOW_TRACING_DESTINATION": f"{args.catalog}.{args.schema}",
            "MLFLOW_TRACING_SQL_WAREHOUSE_ID": warehouse_id,
            "LAKEBASE_PROJECT_ID": project_id,
        }
        if genie_space_ids:
            local_values["GENIE_SPACE_IDS"] = genie_space_ids
        update_env_file(local_values)
    except Exception as error:
        print(f"Experiment setup failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(f"Workspace: {workspace.config.host} (profile '{args.profile}')")
    print(f"App: {app_name}")
    print(f"Experiment: {args.experiment_name} ({experiment_id})")
    print(f"Trace tables: {trace_location}_otel_*")
    print(f"Lakebase project: {project_id}")
    print(f"Bundle overrides: {override_path}")
    print(
        "Next: "
        f"databricks bundle deploy --target {args.target} --profile {args.profile}"
    )


if __name__ == "__main__":
    main()
