"""Hosted web search when the workspace allows it, else Unity Gateway MCP.

GPT and Gemini can take a provider-native search parameter through the AI
Gateway. That parameter is rejected in some workspaces — notably when
cross-region processing is disabled, or HIPAA/BAA is on — with
``INVALID_PARAMETER_VALUE``. Claude and open-weight models have no hosted
search at all.

In those cases the agent attaches Databricks' ``system.ai.web_search`` MCP
Service instead of sending a parameter the gateway will refuse.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Literal

from agent_server.model_profile import ModelProfile

WebSearchMode = Literal["openai", "google", "mcp", "off"]

WEB_SEARCH_MCP_PATH = "/ai-gateway/mcp-services/system.ai.web_search"
WEB_SEARCH_MCP_NAME = "Databricks Web Search"

# Override the probe: "native" (hosted, if the family has it), "mcp", or "off".
_BACKEND_ENV = "WEB_SEARCH_BACKEND"
_PROBE_TIMEOUT_SECONDS = 20.0

_UNSET: object = object()
_mode_cache: WebSearchMode | object = _UNSET
_mode_lock: asyncio.Lock | None = None


def native_web_search_rejected(error: BaseException) -> bool:
    """True when the gateway refused the hosted web-search parameter itself."""
    text = str(error).lower()
    body = getattr(error, "body", None)
    if body is not None:
        text = f"{text} {body}".lower()
    mentions_search = "web search" in text or "google_search" in text
    unavailable = (
        "not available" in text
        or "cross-region" in text
        or "cross-geo" in text
        or "hipaa" in text
    )
    return mentions_search and unavailable


def extra_body_for(profile: ModelProfile, mode: WebSearchMode) -> dict[str, Any] | None:
    """ModelSettings.extra_body with hosted Google search only when it is live."""
    body = None
    if profile.extra_body:
        body = {k: v for k, v in profile.extra_body.items() if k != "google_search"}
    if mode == "google":
        body = {**(body or {}), "google_search": {}}
    return body or None


def uses_web_search_mcp(mode: WebSearchMode) -> bool:
    return mode == "mcp"


def web_search_mcp_spec(
    mode: WebSearchMode, already: list[tuple[str, str]] | None = None
) -> tuple[str, str] | None:
    """The Unity Gateway web-search MCP server, unless it is already configured."""
    if not uses_web_search_mcp(mode):
        return None
    for _, url in already or []:
        if url.rstrip("/").endswith("system.ai.web_search"):
            return None
    return WEB_SEARCH_MCP_NAME, WEB_SEARCH_MCP_PATH


def backend_override() -> Literal["native", "mcp", "off"] | None:
    raw = os.getenv(_BACKEND_ENV, "").strip().lower()
    if raw in {"native", "mcp", "off"}:
        return raw  # type: ignore[return-value]
    return None


def mode_from_preference(
    profile: ModelProfile,
    *,
    backend: Literal["native", "mcp", "off"] | None = None,
    native_available: bool | None = None,
) -> WebSearchMode:
    """Resolve hosted vs MCP vs off from family support and a probe result.

    ``native_available`` is ignored when ``backend`` forces mcp/off, and when
    the family has no hosted search (those always fall through to MCP unless
    search is turned off).
    """
    chosen = backend if backend is not None else backend_override()
    if chosen == "off":
        return "off"
    if chosen == "mcp":
        return "mcp"
    if profile.web_search is None:
        return "mcp"
    if chosen == "native" or native_available:
        return profile.web_search
    return "mcp"


def reset_web_search_mode_cache() -> None:
    """Drop the process-wide probe result. Tests call this between cases."""
    global _mode_cache
    _mode_cache = _UNSET


def _mode_lock() -> asyncio.Lock:
    global _mode_lock
    if _mode_lock is None:
        _mode_lock = asyncio.Lock()
    return _mode_lock


async def resolved_web_search_mode(
    model: str, profile: ModelProfile, client: Any
) -> WebSearchMode:
    """Hosted search if a cheap probe succeeds, otherwise the MCP fallback.

    The first call in a process hits the gateway; later calls reuse that
    result. ``WEB_SEARCH_BACKEND`` skips the probe entirely.
    """
    global _mode_cache
    if _mode_cache is not _UNSET:
        return _mode_cache  # type: ignore[return-value]
    async with _mode_lock():
        if _mode_cache is not _UNSET:
            return _mode_cache  # type: ignore[return-value]
        mode = await detect_web_search_mode(model, profile, client)
        _mode_cache = mode
        logging.info("Web search mode %s for %s", mode, model)
        return mode


async def detect_web_search_mode(
    model: str, profile: ModelProfile, client: Any
) -> WebSearchMode:
    """Probe once (unless overridden) and map the result onto a mode."""
    backend = backend_override()
    if backend == "off":
        return "off"
    if backend == "mcp" or profile.web_search is None:
        return mode_from_preference(profile, backend=backend or "mcp")
    if backend == "native":
        return mode_from_preference(profile, backend="native")

    native_ok = await probe_native_web_search(model, profile, client)
    return mode_from_preference(profile, native_available=native_ok)


async def probe_native_web_search(
    model: str, profile: ModelProfile, client: Any
) -> bool:
    """True when this workspace accepts the family's hosted web-search parameter.

    The request is only large enough to exercise validation: a 400 naming web
    search means the parameter is forbidden here. Any other failure is treated
    as "assume hosted still works" so a flaky probe does not strip search from
    workspaces that already had it.
    """
    if profile.web_search is None:
        return False
    try:
        await asyncio.wait_for(
            _send_native_probe(model, profile, client),
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        if native_web_search_rejected(exc):
            logging.warning(
                "Hosted web search is not available for %s in this workspace; "
                "falling back to %s: %s",
                model,
                WEB_SEARCH_MCP_PATH,
                exc,
            )
            return False
        logging.warning(
            "Hosted web search probe failed for %s; keeping native search: %s",
            model,
            exc,
        )
        return True
    return True


async def _send_native_probe(model: str, profile: ModelProfile, client: Any) -> None:
    prompt = "Reply with the single word ok."
    if profile.web_search == "openai":
        await client.responses.create(
            model=model,
            input=prompt,
            tools=[{"type": "web_search"}],
            max_output_tokens=64,
        )
        return
    await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        extra_body={"google_search": {}},
    )
