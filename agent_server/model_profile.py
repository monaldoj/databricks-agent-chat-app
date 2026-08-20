"""How one model family has to be addressed through the Databricks AI Gateway.

The gateway speaks OpenAI's API for every model, but not every model accepts every
part of it. GPT uses the Responses API and its ``web_search`` tool. Gemini uses
chat completions plus a ``google_search`` extra-body field. Claude and open-weight
models use chat completions with no hosted search.

See https://docs.databricks.com/aws/en/machine-learning/model-serving/web-search
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from openai.types.shared import Reasoning

from agent_server.model_limits import max_output_tokens_for

# Efforts each family accepts, and what a value the family rejects is sent as instead.
# GPT has no "minimal" on the Responses API; Gemini has no "none", so the lightest
# thinking level it does take stands in for it.
_GPT_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})
_GEMINI_EFFORTS = {
    "none": "minimal",
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
}
_CLAUDE_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


@dataclass(frozen=True)
class ModelProfile:
    api: Literal["responses", "chat_completions"]
    reasoning: Reasoning | None
    # Provider-native request fields the OpenAI SDK has no parameter for.
    extra_body: dict[str, Any] | None
    # "openai" = Responses ``web_search`` tool; "google" = Chat Completions
    # ``google_search`` extra body; None = no hosted search.
    web_search: Literal["openai", "google"] | None
    max_tokens: int


def model_family(model: str) -> Literal["gpt", "claude", "gemini", "other"]:
    """Provider family behind a gateway model name.

    Both naming schemes reach the same model and are reduced to the same bare name, so
    ``system.ai.gpt-5-6-sol`` and ``databricks-gpt-5-6-sol`` are read alike.
    """
    name = model.rsplit(".", 1)[-1].removeprefix("databricks-").lower()
    if name.startswith("gpt-oss"):
        return "other"  # open weights, and not on the Responses API
    if name.startswith("gpt-"):
        return "gpt"
    if "claude" in name:
        return "claude"
    if "gemini" in name:
        return "gemini"
    return "other"


def model_profile(model: str, effort: str) -> ModelProfile:
    family = model_family(model)

    if family == "gpt":
        if effort not in _GPT_EFFORTS:
            raise ValueError(
                f"Reasoning effort {effort!r} is not accepted for {model}; "
                f"use one of {sorted(_GPT_EFFORTS)}."
            )
        return ModelProfile(
            api="responses",
            reasoning=Reasoning(effort=effort),
            extra_body=None,
            web_search="openai",
            max_tokens=max_output_tokens_for(model),
        )

    if family == "claude":
        # Anthropic decides per turn whether to think, and takes the effort separately.
        if effort == "none":
            thinking: dict[str, Any] = {"thinking": {"type": "disabled"}}
        elif effort in _CLAUDE_EFFORTS:
            thinking = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
            }
        else:
            raise ValueError(
                f"Reasoning effort {effort!r} is not accepted for {model}; "
                f"use 'none' or one of {sorted(_CLAUDE_EFFORTS)}."
            )
        return ModelProfile(
            api="chat_completions",
            reasoning=None,
            extra_body=thinking,
            web_search=None,
            max_tokens=max_output_tokens_for(model),
        )

    if family == "gemini":
        if effort not in _GEMINI_EFFORTS:
            raise ValueError(
                f"Reasoning effort {effort!r} is not accepted for {model}; "
                f"use one of {sorted(_GEMINI_EFFORTS)}."
            )
        # The agents SDK sends ModelSettings.reasoning.effort as `reasoning_effort` on
        # the chat completions path, which is the field Gemini wants. Hosted web
        # search is a top-level Chat Completions field, not a function tool.
        # https://docs.databricks.com/aws/en/machine-learning/model-serving/web-search
        return ModelProfile(
            api="chat_completions",
            reasoning=Reasoning(effort=_GEMINI_EFFORTS[effort]),
            extra_body={"google_search": {}},
            web_search="google",
            max_tokens=max_output_tokens_for(model),
        )

    # Llama, Kimi, GLM, Qwen, GPT-OSS and the like: chat completions, and no reasoning
    # control that is safe to assume across them.
    return ModelProfile(
        api="chat_completions",
        reasoning=None,
        extra_body=None,
        web_search=None,
        max_tokens=max_output_tokens_for(model),
    )
