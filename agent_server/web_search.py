"""Hosted web search when the workspace allows it, else Unity Gateway MCP.

GPT and Gemini can take a provider-native search parameter through the AI
Gateway. That parameter is rejected in some workspaces — notably when
cross-region processing is disabled, or HIPAA/BAA is on — with
``INVALID_PARAMETER_VALUE``. Claude and open-weight models have no hosted
search at all.

In those cases the agent attaches Databricks' ``system.ai.web_search`` MCP
Service instead of sending a parameter the gateway will refuse. The choice is
sticky for the process: a startup probe or a live turn that is refused pins
MCP, so later questions do not send hosted search again.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Literal

from agent_server.model_profile import ModelProfile

WebSearchMode = Literal["openai", "google", "mcp", "off"]

WEB_SEARCH_MCP_PATH = "/ai-gateway/mcp-services/system.ai.web_search"
WEB_SEARCH_MCP_NAME = "Databricks Web Search"

# Override the probe: "native" (hosted, if the family has it), "mcp", or "off".
_BACKEND_ENV = "WEB_SEARCH_BACKEND"
_PROBE_TIMEOUT_SECONDS = 20.0
_NATIVE_MODES = frozenset({"openai", "google"})
_HTTP_STATUS_RE = re.compile(r"error code:\s*(\d+)", re.I)

_UNSET: object = object()
_mode_cache: WebSearchMode | object = _UNSET
_MODE_LOCK = asyncio.Lock()


def _error_text(error: BaseException) -> str:
    text = str(error)
    body = getattr(error, "body", None)
    if body is not None:
        text = f"{text} {body}"
    return text.lower()


def _http_status(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    match = _HTTP_STATUS_RE.search(_error_text(error))
    return int(match.group(1)) if match else None


def native_web_search_rejected(error: BaseException) -> bool:
    """True when the gateway refused the hosted web-search parameter itself."""
    text = _error_text(error)
    mentions_search = (
        "web search" in text or "google_search" in text or "web_search" in text
    )
    unavailable = (
        "not available" in text
        or "unavailable" in text
        or "not supported" in text
        or "unsupported" in text
        or "does not support" in text
        or "not enabled" in text
        or "not allowed" in text
        or "not permitted" in text
        or "cross-region" in text
        or "cross-geo" in text
        or "hipaa" in text
    )
    if mentions_search and unavailable:
        return True
    # A 400 that names the search parameter is the same refusal, even when the
    # message does not use "unavailable" / "not supported".
    return mentions_search and _http_status(error) == 400


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


def is_web_search_mcp_server(name: str | None, url: str | None = None) -> bool:
    if name == WEB_SEARCH_MCP_NAME:
        return True
    if url and url.rstrip("/").endswith("system.ai.web_search"):
        return True
    return False


def effective_web_search_mode(
    requested: WebSearchMode, *, web_search_mcp_connected: bool
) -> WebSearchMode:
    """Don't tell the model it has MCP search if that server never connected."""
    if requested == "mcp" and not web_search_mcp_connected:
        return "off"
    return requested


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


def remember_native_web_search_failure(
    current: WebSearchMode, error: BaseException
) -> bool:
    """Pin MCP after a live native-search refusal.

    The startup probe can succeed (the parameter is accepted on a dummy
    request) and still lose on a real turn — Gemini ``google_search`` especially,
    when workspace settings only reject search once the model actually uses it.
    Without pinning, every later question sends hosted search again.

    Returns True when the caller should retry *this* turn with MCP.
    ``WEB_SEARCH_BACKEND=native`` still wins and is not overridden.
    """
    global _mode_cache
    if current not in _NATIVE_MODES:
        return False
    if not native_web_search_rejected(error):
        return False
    if backend_override() == "native":
        logging.warning(
            "Hosted web search was refused, but WEB_SEARCH_BACKEND=native; "
            "not falling back to %s: %s",
            WEB_SEARCH_MCP_PATH,
            error,
        )
        return False
    _mode_cache = "mcp"
    logging.warning(
        "Hosted web search was refused on a live request; using %s for this "
        "turn and every later one: %s",
        WEB_SEARCH_MCP_PATH,
        error,
    )
    return True


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
    async with _MODE_LOCK:
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

    The request is only large enough to exercise validation. A 400 — or any
    error that names hosted search as unavailable — means the parameter is
    forbidden here. Auth, timeout, and 5xx failures are treated as "assume
    hosted still works" so a flaky probe does not strip search from workspaces
    that already had it. A later live refusal still pins MCP; see
    ``remember_native_web_search_failure``.
    """
    if profile.web_search is None:
        return False
    try:
        await asyncio.wait_for(
            _send_native_probe(model, profile, client),
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        if native_web_search_rejected(exc) or _http_status(exc) == 400:
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
