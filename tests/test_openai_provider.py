from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pi_agent import Agent, AgentOptions, AgentTool, Model, ToolResult
from pi_agent.providers import openai_compatible_stream
from pi_agent.types import AbortSignal, ToolUpdateCallback


class EchoTool(AgentTool):
    name = "echo"
    label = "echo"
    description = "Echo text"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        return ToolResult.text(params["text"])


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 规定的方法名
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append(payload)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if len(type(self).requests) == 1:
            chunks = [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {"name": "echo", "arguments": '{"text":'},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"hello"}'}}]},
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
        else:
            chunks = [
                {"choices": [{"delta": {"content": "完成"}, "finish_reason": None}]},
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                },
            ]
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        return


class OpenAIProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_sse_tool_call_round_trip(self) -> None:
        FakeOpenAIHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = Model(
                id="mock-model",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key="test-key",
            )
            agent = Agent(
                AgentOptions(
                    model=model,
                    stream_fn=openai_compatible_stream,
                    tools=[EchoTool()],
                )
            )
            await agent.prompt("call echo")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual([message["role"] for message in agent.state.messages], ["user", "assistant", "toolResult", "assistant"])
        self.assertEqual(agent.state.messages[2]["content"][0]["text"], "hello")
        self.assertEqual(agent.state.messages[-1]["content"][0]["text"], "完成")
        self.assertEqual(FakeOpenAIHandler.requests[0]["tools"][0]["function"]["name"], "echo")
        self.assertEqual(FakeOpenAIHandler.requests[1]["messages"][-1]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
