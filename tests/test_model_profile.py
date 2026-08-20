"""model_profile() wiring, including hosted web search per family."""

from agent_server.model_profile import model_profile


def test_gpt_keeps_responses_web_search_tool():
    profile = model_profile("system.ai.gpt-5-6-terra", "medium")
    assert profile.api == "responses"
    assert profile.web_search == "openai"
    assert profile.extra_body is None


def test_gemini_uses_google_search_extra_body():
    profile = model_profile("system.ai.gemini-3-5-flash", "medium")
    assert profile.api == "chat_completions"
    assert profile.web_search == "google"
    assert profile.extra_body == {"google_search": {}}


def test_claude_and_open_weight_have_no_hosted_search():
    claude = model_profile("system.ai.claude-opus-5", "medium")
    assert claude.web_search is None
    assert claude.extra_body is not None
    assert "google_search" not in claude.extra_body

    llama = model_profile("system.ai.meta-llama-3-3-70b-instruct", "medium")
    assert llama.web_search is None
    assert llama.extra_body is None
