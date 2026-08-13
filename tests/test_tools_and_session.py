from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pi_agent.session import JsonlSession
from pi_agent.providers.openai_compatible import _chat_completions_url, _to_openai_messages
from pi_agent.tools import BashTool, EditTool, ReadTool, WriteTool


class ToolsAndSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_bash_returns_output_and_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = await BashTool(directory).execute(
                "b",
                {"command": "python -c \"print('hello')\"", "timeout": 10},
            )
            self.assertIn("hello", result.content[0]["text"])
            self.assertEqual(result.details["exitCode"], 0)

    async def test_write_read_and_multi_edit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write = WriteTool(directory)
            read = ReadTool(directory)
            edit = EditTool(directory)

            await write.execute("w", {"path": "sample.txt", "content": "alpha\nbeta\ngamma\n"})
            await edit.execute(
                "e",
                {
                    "path": "sample.txt",
                    "edits": [
                        {"oldText": "alpha", "newText": "ALPHA"},
                        {"oldText": "gamma", "newText": "GAMMA"},
                    ],
                },
            )
            result = await read.execute("r", {"path": "sample.txt"})
            self.assertEqual(result.content[0]["text"], "ALPHA\nbeta\nGAMMA\n")

    async def test_edit_rejects_non_unique_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "sample.txt").write_text("same same", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "恰好出现一次"):
                await EditTool(directory).execute(
                    "e",
                    {"path": "sample.txt", "edits": [{"oldText": "same", "newText": "new"}]},
                )

    async def test_file_tools_reject_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PermissionError):
                await WriteTool(directory).execute("w", {"path": "../outside.txt", "content": "x"})

    async def test_jsonl_session_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "session.jsonl")
            session = JsonlSession(path, directory, create=True)
            first = session.append({"role": "user", "content": [{"type": "text", "text": "one"}]})
            old_leaf = session.append({"role": "assistant", "content": [{"type": "text", "text": "old"}]})
            session.branch_from(first)
            new_leaf = session.append({"role": "assistant", "content": [{"type": "text", "text": "new"}]})

            reopened = JsonlSession(path, directory)
            self.assertEqual(reopened.leaf_id, new_leaf)
            self.assertEqual([m["content"][0]["text"] for m in reopened.messages()], ["one", "new"])
            self.assertEqual([m["content"][0]["text"] for m in reopened.messages(old_leaf)], ["one", "old"])

    async def test_openai_message_conversion_keeps_tool_pairing(self) -> None:
        messages = _to_openai_messages(
            "system",
            [
                {"role": "user", "content": [{"type": "text", "text": "go"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "calling"},
                        {"type": "toolCall", "id": "c1", "name": "read", "arguments": {"path": "a.txt"}},
                    ],
                },
                {
                    "role": "toolResult",
                    "toolCallId": "c1",
                    "content": [{"type": "text", "text": "data"}],
                },
            ],
        )
        self.assertEqual(messages[0], {"role": "system", "content": "system"})
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["arguments"], '{"path": "a.txt"}')
        self.assertEqual(messages[3]["tool_call_id"], "c1")
        self.assertEqual(_chat_completions_url("http://localhost:8000/v1/"), "http://localhost:8000/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
