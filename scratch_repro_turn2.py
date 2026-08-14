"""Throwaway repro: second turn of a conversation on a chat-completions model.

Replays the exact item shape the chat UI echoes back, per the deployed app's error.
Usage: python scratch_repro_turn2.py system.ai.claude-opus-5
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
os.environ["AGENT_MODEL"] = sys.argv[1]
os.environ["GENIE_SPACE_IDS"] = ""

from mlflow.types.responses import ResponsesAgentRequest  # noqa: E402

from agent_server import agent as agent_mod  # noqa: E402

agent_mod.get_mcp_user_workspace_client = lambda: WorkspaceClient(profile="DEFAULT")

TURN_2_INPUT = [
    {"role": "user", "content": "What is the capital of France?"},
    {
        "status": None,
        "content": [{"text": "The capital of France is Paris.", "type": "output_text"}],
        "role": "assistant",
        "type": "message",
    },
    {"role": "user", "content": "And what is its population?"},
]


async def main():
    print(f"model={agent_mod.SELECTED_MODEL} api={agent_mod.MODEL_PROFILE.api}\n")
    request = ResponsesAgentRequest(input=TURN_2_INPUT)
    print("items as the agent sees them:")
    for item in request.input:
        print("  ", json.dumps(item.model_dump()))
    print()
    try:
        response = await agent_mod.invoke(request)
        texts = [
            part.get("text")
            for item in response.output
            for part in (item.get("content") or [])
            if isinstance(part, dict)
        ]
        print("OK ->", " ".join(t for t in texts if t)[:300])
    except Exception as error:  # noqa: BLE001
        print(f"FAILED {type(error).__name__}: {str(error)[:400]}")


asyncio.run(main())
