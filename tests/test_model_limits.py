from agent_server.model_limits import bare_model_name, max_output_tokens_for


def test_bare_name_strips_catalog_and_endpoint_prefix():
    assert bare_model_name("system.ai.gemma-3-12b") == "gemma-3-12b"
    assert bare_model_name("databricks-gemma-3-12b") == "gemma-3-12b"
    assert bare_model_name("gemma-3-12b") == "gemma-3-12b"


def test_gemma_uses_databricks_8k_cap():
    assert max_output_tokens_for("system.ai.gemma-3-12b") == 8192
    assert max_output_tokens_for("databricks-gemma-3-12b") == 8192


def test_llama_and_gpt_oss_use_published_caps():
    assert max_output_tokens_for("system.ai.llama-4-maverick") == 8192
    assert max_output_tokens_for("system.ai.meta-llama-3-3-70b-instruct") == 8192
    assert max_output_tokens_for("system.ai.gpt-oss-120b") == 25000


def test_frontier_models_keep_a_high_cap_under_provider_maxima():
    assert max_output_tokens_for("system.ai.gpt-5-6-terra") == 32768
    assert max_output_tokens_for("system.ai.claude-opus-5") == 32768
    assert max_output_tokens_for("system.ai.gemini-3-5-flash") == 65536


def test_unknown_open_weight_defaults_to_8k_not_32k():
    assert max_output_tokens_for("system.ai.unknown-open-model") == 8192
