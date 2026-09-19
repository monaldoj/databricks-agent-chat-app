"""Gemini / gateway tool-call id reuse is rewritten before the next request."""

from types import SimpleNamespace

from agents.models.fake_id import FAKE_RESPONSES_ID

from agent_server.utils import (
    adapt_input_for_chat_completions,
    adapt_outbound_messages,
    call_ids_from_input,
    collapse_system_messages,
    uniquify_outbound_tool_call_ids,
    uniquify_response_item_call_ids,
    uniquify_tool_call_ids,
)


def test_outbound_rewrites_completed_id_on_a_later_invocation():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "web_search"}}],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "results"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "web_search"}}],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "more results"},
    ]

    uniquify_outbound_tool_call_ids(messages)

    first_id = messages[0]["tool_calls"][0]["id"]
    second_id = messages[2]["tool_calls"][0]["id"]
    assert first_id == "call_0"
    assert messages[1]["tool_call_id"] == first_id
    assert second_id != first_id
    assert messages[3]["tool_call_id"] == second_id
    assert second_id.startswith("call_")


def test_outbound_replaces_placeholder_ids_and_keeps_pairs():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": FAKE_RESPONSES_ID, "type": "function", "function": {"name": "web_search"}}
            ],
        },
        {"role": "tool", "tool_call_id": FAKE_RESPONSES_ID, "content": "results"},
    ]

    uniquify_outbound_tool_call_ids(messages)

    call_id = messages[0]["tool_calls"][0]["id"]
    assert call_id != FAKE_RESPONSES_ID
    assert messages[1]["tool_call_id"] == call_id


def test_outbound_leaves_unique_ids_alone():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "a"}}],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "ok"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_def", "type": "function", "function": {"name": "b"}}],
        },
        {"role": "tool", "tool_call_id": "call_def", "content": "ok"},
    ]

    uniquify_outbound_tool_call_ids(messages)

    assert messages[0]["tool_calls"][0]["id"] == "call_abc"
    assert messages[1]["tool_call_id"] == "call_abc"
    assert messages[2]["tool_calls"][0]["id"] == "call_def"
    assert messages[3]["tool_call_id"] == "call_def"


def test_inbound_always_mints_and_stream_index_stays_stable():
    reserved: set[str] = set()
    index_ids: dict[int, str] = {}
    first = SimpleNamespace(
        tool_calls=[SimpleNamespace(id="call_0", index=0, thoughtSignature="sig")]
    )
    uniquify_tool_call_ids(first, reserved, index_ids)
    minted = first.tool_calls[0].id
    assert minted != "call_0"
    assert minted.startswith("call_")

    later = SimpleNamespace(tool_calls=[SimpleNamespace(id="", index=0)])
    uniquify_tool_call_ids(later, reserved, index_ids)
    assert later.tool_calls[0].id == minted


def test_inbound_does_not_reuse_a_completed_history_id():
    reserved = call_ids_from_input(
        [
            {"type": "function_call", "call_id": "call_0", "name": "web_search"},
            {"type": "function_call_output", "call_id": "call_0", "output": "done"},
        ]
    )
    assert "call_0" in reserved
    message = SimpleNamespace(tool_calls=[SimpleNamespace(id="call_0")])
    uniquify_tool_call_ids(message, reserved)
    assert message.tool_calls[0].id != "call_0"


def test_response_items_rewrite_reused_call_ids():
    items = [
        {"type": "function_call", "call_id": "call_0", "name": "web_search"},
        {"type": "function_call_output", "call_id": "call_0", "output": "first"},
        {"type": "function_call", "call_id": "call_0", "name": "web_search"},
        {"type": "function_call_output", "call_id": "call_0", "output": "second"},
    ]

    uniquify_response_item_call_ids(items)

    assert items[0]["call_id"] == "call_0"
    assert items[1]["call_id"] == "call_0"
    assert items[2]["call_id"] != "call_0"
    assert items[3]["call_id"] == items[2]["call_id"]


def test_adapt_input_still_fills_assistant_message_ids():
    items = [{"type": "message", "role": "assistant", "content": "hi"}]
    adapt_input_for_chat_completions(items)
    assert items[0]["id"] == FAKE_RESPONSES_ID


def test_adapt_outbound_messages_runs_id_rewrite():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "", "type": "function", "function": {"name": "web_search"}}],
        },
        {"role": "tool", "tool_call_id": "", "content": [{"type": "text", "text": "ok"}]},
    ]
    adapt_outbound_messages(messages)
    call_id = messages[0]["tool_calls"][0]["id"]
    assert call_id.startswith("call_")
    assert messages[1]["tool_call_id"] == call_id
    assert messages[1]["content"] == "ok"


def test_collapse_joins_system_and_developer_into_one_system_message():
    messages = [
        {"role": "system", "content": "You are a SOC analyst."},
        {"role": "developer", "content": "Use web search for public intel."},
        {"role": "user", "content": "What happened?"},
    ]
    collapse_system_messages(messages)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == (
        "You are a SOC analyst.\n\nUse web search for public intel."
    )


def test_collapse_leaves_a_single_system_message_alone():
    messages = [
        {"role": "system", "content": "Only one."},
        {"role": "user", "content": "Hi"},
    ]
    collapse_system_messages(messages)
    assert messages[0] == {"role": "system", "content": "Only one."}


def test_adapt_outbound_messages_collapses_system_prompts():
    messages = [
        {"role": "system", "content": "First"},
        {"role": "system", "content": "Second"},
        {"role": "user", "content": "Hi"},
    ]
    adapt_outbound_messages(messages)
    assert len([m for m in messages if m["role"] == "system"]) == 1
    assert messages[0]["content"] == "First\n\nSecond"
