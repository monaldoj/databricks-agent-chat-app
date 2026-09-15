"""Provider-private reasoning items do not cross the chat API boundary."""

import asyncio
from types import SimpleNamespace

from agent_server.utils import process_agent_stream_events, public_response_items


class _Dump:
    def __init__(self, payload: dict):
        self.payload = payload

    def model_dump(self) -> dict:
        return self.payload


class _InputItem:
    def __init__(self, payload: dict):
        self.payload = payload

    def to_input_item(self) -> dict:
        return self.payload


def test_public_response_items_remove_encrypted_reasoning():
    items = [
        _InputItem({"type": "reasoning", "id": "rs_1", "encrypted_content": "secret"}),
        _InputItem({"type": "message", "id": "msg_1", "content": "Complete answer"}),
        {"type": "function_call", "call_id": "call_1", "name": "search"},
    ]

    assert public_response_items(items) == [
        {"type": "message", "id": "msg_1", "content": "Complete answer"},
        {"type": "function_call", "call_id": "call_1", "name": "search"},
    ]


def test_stream_removes_reasoning_and_preserves_text_and_completion():
    events = [
        SimpleNamespace(
            type="raw_response_event",
            data=_Dump(
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "secret",
                    },
                }
            ),
        ),
        SimpleNamespace(
            type="raw_response_event",
            data=_Dump(
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": "rs_1",
                    "delta": "private summary",
                }
            ),
        ),
        SimpleNamespace(
            type="raw_response_event",
            data=_Dump(
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_1",
                    "delta": "Complete answer",
                }
            ),
        ),
        SimpleNamespace(
            type="raw_response_event",
            data=_Dump(
                {
                    "type": "response.completed",
                    "response": {
                        "output": [
                            {
                                "type": "reasoning",
                                "id": "rs_1",
                                "encrypted_content": "secret",
                            },
                            {
                                "type": "message",
                                "id": "msg_1",
                                "content": "Complete answer",
                            },
                        ]
                    },
                }
            ),
        ),
    ]

    async def event_stream():
        for event in events:
            yield event

    async def collect_events():
        return [event async for event in process_agent_stream_events(event_stream())]

    emitted = asyncio.run(collect_events())

    assert [event["type"] for event in emitted] == [
        "response.output_text.delta",
        "response.completed",
    ]
    assert emitted[-1]["response"]["output"] == [
        {"type": "message", "id": "msg_1", "content": "Complete answer"}
    ]
