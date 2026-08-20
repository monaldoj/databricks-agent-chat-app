"""Output-token caps for models reached through the Databricks AI Gateway.

Databricks Foundation Model APIs reject a request whose ``max_tokens`` exceeds
the model's output cap (Gemma 3 12B is 8,192; GPT OSS is 25,000). Frontier
models match their provider's published maximum, which is well above the
32,768 we ask for so GPT-5 reasoning still has room to finish.

Caps are keyed off the bare model name so ``system.ai.gemma-3-12b`` and
``databricks-gemma-3-12b`` resolve the same way. Unknown open-weight names
fall back to 8,192, the common Databricks cap, rather than a value that
would 400.
"""

from __future__ import annotations

# Databricks-published output caps for hosted open-weight endpoints.
# https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits
_EXACT_CAPS: dict[str, int] = {
    "gemma-3-12b": 8192,
    "llama-4-maverick": 8192,
    "meta-llama-3-3-70b-instruct": 8192,
    "meta-llama-3-1-405b-instruct": 8192,
    "meta-llama-3-1-70b-instruct": 8192,
    "meta-llama-3-1-8b-instruct": 8192,
    "gpt-oss-120b": 25000,
    "gpt-oss-20b": 25000,
}

# Longest prefix first so ``gpt-oss`` does not fall through to ``gpt``.
_PREFIX_CAPS: tuple[tuple[str, int], ...] = (
    ("gpt-oss", 25000),
    ("gemma", 8192),
    ("meta-llama", 8192),
    ("llama", 8192),
    ("qwen", 25000),
    ("gpt", 32768),
    ("claude", 32768),
    ("gemini", 65536),
)

_DEFAULT_CAP = 8192


def bare_model_name(model: str) -> str:
    """Last path segment, without a ``databricks-`` serving-endpoint prefix."""
    return model.rsplit(".", 1)[-1].removeprefix("databricks-").lower()


def max_output_tokens_for(model: str) -> int:
    """Largest ``max_tokens`` the gateway will accept for this model."""
    name = bare_model_name(model)
    if name in _EXACT_CAPS:
        return _EXACT_CAPS[name]
    for prefix, cap in _PREFIX_CAPS:
        if name == prefix or name.startswith(f"{prefix}-"):
            return cap
    return _DEFAULT_CAP
