"""Hosted web search vs Unity Gateway MCP fallback."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_server.model_profile import model_profile
from agent_server.utils import header_value
from agent_server.web_search import (
    WEB_SEARCH_MCP_NAME,
    WEB_SEARCH_MCP_PATH,
    extra_body_for,
    detect_web_search_mode,
    effective_web_search_mode,
    is_web_search_mcp_server,
    mode_from_preference,
    native_web_search_rejected,
    probe_native_web_search,
    reset_web_search_mode_cache,
    resolved_web_search_mode,
    web_search_mcp_spec,
)

GEMINI_CROSS_REGION_ERROR = (
    "Error code: 400 - {'error_code': 'INVALID_PARAMETER_VALUE', 'message': "
    "'INVALID_PARAMETER_VALUE: Web search for Gemini is not available when "
    "cross-region processing is disabled.'}"
)


@pytest.fixture(autouse=True)
def _clear_web_search_cache(monkeypatch):
    monkeypatch.delenv("WEB_SEARCH_BACKEND", raising=False)
    reset_web_search_mode_cache()
    yield
    reset_web_search_mode_cache()


def test_rejects_gemini_cross_region_error():
    assert native_web_search_rejected(Exception(GEMINI_CROSS_REGION_ERROR))


@pytest.mark.parametrize(
    "message",
    [
        "Web search is unavailable for this endpoint",
        "google_search is not supported by this model",
        "This model does not support web search",
    ],
)
def test_rejects_other_native_search_unavailable_errors(message):
    assert native_web_search_rejected(Exception(message))


def test_ignores_unrelated_gateway_errors():
    assert not native_web_search_rejected(Exception("Error code: 400 - max_tokens too large"))
    assert not native_web_search_rejected(Exception("INVALID_PARAMETER_VALUE: unknown field"))


def test_gemini_mcp_strips_google_search_extra_body():
    profile = model_profile("system.ai.gemini-3-8-flash", "medium")
    assert extra_body_for(profile, "google") == {"google_search": {}}
    assert extra_body_for(profile, "mcp") is None
    assert extra_body_for(profile, "off") is None


def test_claude_keeps_thinking_extra_body_when_falling_back_to_mcp():
    profile = model_profile("system.ai.claude-opus-5", "medium")
    assert extra_body_for(profile, "mcp") == profile.extra_body
    assert "google_search" not in (extra_body_for(profile, "mcp") or {})


def test_families_without_hosted_search_use_mcp():
    claude = model_profile("system.ai.claude-opus-5", "medium")
    llama = model_profile("system.ai.meta-llama-3-3-70b-instruct", "medium")
    assert mode_from_preference(claude) == "mcp"
    assert mode_from_preference(llama) == "mcp"


def test_hosted_search_when_probe_succeeds():
    gemini = model_profile("system.ai.gemini-3-8-flash", "medium")
    gpt = model_profile("system.ai.gpt-5-6-terra", "medium")
    assert mode_from_preference(gemini, native_available=True) == "google"
    assert mode_from_preference(gpt, native_available=True) == "openai"
    assert mode_from_preference(gemini, native_available=False) == "mcp"


def test_backend_env_overrides_probe(monkeypatch):
    gemini = model_profile("system.ai.gemini-3-8-flash", "medium")
    monkeypatch.setenv("WEB_SEARCH_BACKEND", "mcp")
    assert mode_from_preference(gemini, native_available=True) == "mcp"
    monkeypatch.setenv("WEB_SEARCH_BACKEND", "off")
    assert mode_from_preference(gemini, native_available=True) == "off"
    monkeypatch.setenv("WEB_SEARCH_BACKEND", "native")
    assert mode_from_preference(gemini, native_available=False) == "google"


def test_mcp_spec_only_when_falling_back():
    already = [("other", "/api/2.0/mcp/external/other")]
    assert web_search_mcp_spec("google") is None
    assert web_search_mcp_spec("openai") is None
    assert web_search_mcp_spec("off") is None
    assert web_search_mcp_spec("mcp", already) == (
        WEB_SEARCH_MCP_NAME,
        WEB_SEARCH_MCP_PATH,
    )
    assert (
        web_search_mcp_spec(
            "mcp",
            [("Databricks Web Search", WEB_SEARCH_MCP_PATH)],
        )
        is None
    )


def _gemini_client(side_effect: Exception | None = None) -> MagicMock:
    client = MagicMock()
    create = AsyncMock(side_effect=side_effect) if side_effect else AsyncMock()
    client.chat.completions.create = create
    client.responses.create = AsyncMock()
    return client


def test_probe_falls_back_when_gemini_rejects_hosted_search():
    profile = model_profile("system.ai.gemini-3-8-flash", "medium")
    client = _gemini_client(Exception(GEMINI_CROSS_REGION_ERROR))
    assert asyncio.run(probe_native_web_search("system.ai.gemini-3-8-flash", profile, client)) is False
    client.chat.completions.create.assert_awaited()
    extra_body = client.chat.completions.create.await_args.kwargs["extra_body"]
    assert extra_body == {"google_search": {}}


def test_probe_keeps_native_on_unrelated_failure():
    profile = model_profile("system.ai.gemini-3-8-flash", "medium")
    client = _gemini_client(Exception("Error code: 401 - unauthorized"))
    assert asyncio.run(probe_native_web_search("system.ai.gemini-3-8-flash", profile, client)) is True


def test_detect_uses_mcp_when_hosted_search_is_rejected():
    profile = model_profile("system.ai.gemini-3-8-flash", "medium")
    client = _gemini_client(Exception(GEMINI_CROSS_REGION_ERROR))
    assert (
        asyncio.run(detect_web_search_mode("system.ai.gemini-3-8-flash", profile, client))
        == "mcp"
    )


def test_detect_skips_probe_when_backend_is_mcp(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_BACKEND", "mcp")
    profile = model_profile("system.ai.gemini-3-8-flash", "medium")
    client = _gemini_client()
    assert (
        asyncio.run(detect_web_search_mode("system.ai.gemini-3-8-flash", profile, client))
        == "mcp"
    )
    client.chat.completions.create.assert_not_called()


def test_detect_skips_probe_for_models_without_hosted_search():
    profile = model_profile("system.ai.claude-opus-5", "medium")
    client = _gemini_client()
    assert asyncio.run(detect_web_search_mode("system.ai.claude-opus-5", profile, client)) == "mcp"
    client.chat.completions.create.assert_not_called()
    client.responses.create.assert_not_called()


def test_gpt_probe_uses_responses_web_search_tool():
    profile = model_profile("system.ai.gpt-5-6-terra", "medium")
    client = _gemini_client()
    assert asyncio.run(probe_native_web_search("system.ai.gpt-5-6-terra", profile, client)) is True
    client.responses.create.assert_awaited()
    assert client.responses.create.await_args.kwargs["tools"] == [{"type": "web_search"}]
    client.chat.completions.create.assert_not_called()


def test_resolved_mode_enters_the_lock_and_caches_mcp_fallback():
    profile = model_profile("system.ai.gemini-3-8-flash", "medium")
    client = _gemini_client(Exception(GEMINI_CROSS_REGION_ERROR))
    assert (
        asyncio.run(resolved_web_search_mode("system.ai.gemini-3-8-flash", profile, client))
        == "mcp"
    )
    client.chat.completions.create.reset_mock()
    assert (
        asyncio.run(resolved_web_search_mode("system.ai.gemini-3-8-flash", profile, client))
        == "mcp"
    )
    client.chat.completions.create.assert_not_called()


def test_effective_mode_turns_off_when_mcp_never_connected():
    assert effective_web_search_mode("mcp", web_search_mcp_connected=False) == "off"
    assert effective_web_search_mode("mcp", web_search_mcp_connected=True) == "mcp"
    assert effective_web_search_mode("google", web_search_mcp_connected=False) == "google"


def test_recognises_the_gateway_web_search_server():
    assert is_web_search_mcp_server(WEB_SEARCH_MCP_NAME)
    assert is_web_search_mcp_server(None, f"https://example.cloud.databricks.com{WEB_SEARCH_MCP_PATH}")
    assert not is_web_search_mcp_server("Genie Space: Finance")


def test_header_value_is_case_insensitive():
    assert header_value({"X-Forwarded-Access-Token": "abc"}, "x-forwarded-access-token") == "abc"
    assert header_value({"x-forwarded-access-token": "abc"}, "X-Forwarded-Access-Token") == "abc"
    assert header_value({}, "x-forwarded-access-token") is None
