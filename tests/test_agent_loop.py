from __future__ import annotations

import asyncio
import time
import unittest
from typing import Any

from pi_agent import Agent, AgentContext, AgentOptions, AgentTool, Model, ToolResult
from pi_agent.loop import agent_loop, run_agent_loop
from pi_agent.stream import scripted_stream
from pi_agent.types import AbortSignal, ToolUpdateCallback


def assistant(*blocks: dict[str, Any], stop: str = "stop") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": list(blocks),
        "api": "test",
        "provider": "test",
        "model": "mock",
        "usage": {},
        "stopReason": stop,
        "timestamp": time.time(),
    }


def tool_call(call_id: str, name: str, arguments: Any) -> dict[str, Any]:
    return {"type": "toolCall", "id": call_id, "name": name, "arguments": arguments}


class DelayTool(AgentTool):
    name = "delay"
    label = "delay"
    description = "Return after a delay"
    parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "delay": {"type": "number", "minimum": 0},
            "terminate": {"type": "boolean"},
        },
        "required": ["value", "delay"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        if on_update:
            await on_update(ToolResult.text(f"starting {params['value']}"))
        await asyncio.sleep(params["delay"])
        return ToolResult.text(params["value"], terminate=params.get("terminate", False))


class SequentialDelayTool(DelayTool):
    execution_mode = "sequential"


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_low_level_event_stream_returns_new_messages(self) -> None:
        options = AgentOptions(
            model=Model(id="mock"),
            stream_fn=scripted_stream([assistant({"type": "text", "text": "ok"})]),
        )
        stream = agent_loop(
            [{"role": "user", "content": [{"type": "text", "text": "go"}]}],
            AgentContext(""),
            options,
        )
        events = [event async for event in stream]
        messages = await stream.result()
        self.assertEqual(events[-1]["type"], "agent_end")
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])

    async def test_plain_lifecycle_and_state(self) -> None:
        model = Model(id="mock", provider="test", api="test")
        agent = Agent(
            AgentOptions(
                model=model,
                stream_fn=scripted_stream([assistant({"type": "text", "text": "你好"})]),
            )
        )
        events: list[str] = []

        async def listener(event: dict[str, Any], signal: AbortSignal) -> None:
            await asyncio.sleep(0)
            events.append(event["type"])

        agent.subscribe(listener)
        await agent.prompt("测试")

        self.assertEqual(events[0:2], ["agent_start", "turn_start"])
        self.assertEqual(events[-2:], ["turn_end", "agent_end"])
        self.assertEqual([message["role"] for message in agent.state.messages], ["user", "assistant"])
        self.assertFalse(agent.state.is_streaming)
        self.assertEqual(agent.state.messages[-1]["content"][0]["text"], "你好")

    async def test_tool_loop_and_result(self) -> None:
        replies = [
            assistant(tool_call("a", "delay", {"value": "done", "delay": 0}), stop="toolUse"),
            assistant({"type": "text", "text": "工具完成"}),
        ]
        agent = Agent(
            AgentOptions(
                model=Model(id="mock"),
                stream_fn=scripted_stream(replies),
                tools=[DelayTool()],
            )
        )
        await agent.prompt("运行工具")
        self.assertEqual(
            [message["role"] for message in agent.state.messages],
            ["user", "assistant", "toolResult", "assistant"],
        )
        self.assertEqual(agent.state.messages[2]["content"][0]["text"], "done")

    async def test_parallel_completion_event_but_source_order_messages(self) -> None:
        first = assistant(
            tool_call("slow", "delay", {"value": "slow", "delay": 0.04}),
            tool_call("fast", "delay", {"value": "fast", "delay": 0.001}),
            stop="toolUse",
        )
        options = AgentOptions(
            model=Model(id="mock"),
            stream_fn=scripted_stream([first, assistant({"type": "text", "text": "ok"})]),
            tools=[DelayTool()],
        )
        events: list[dict[str, Any]] = []
        new_messages = await run_agent_loop(
            [{"role": "user", "content": [{"type": "text", "text": "go"}]}],
            AgentContext("", [], [DelayTool()]),
            options,
            lambda event: events.append(event),
        )
        completed = [event["toolCallId"] for event in events if event["type"] == "tool_execution_end"]
        results = [message["toolCallId"] for message in new_messages if message.get("role") == "toolResult"]
        self.assertEqual(completed, ["fast", "slow"])
        self.assertEqual(results, ["slow", "fast"])

    async def test_sequential_tool_override(self) -> None:
        running = 0
        maximum = 0

        class ProbeTool(SequentialDelayTool):
            async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
                nonlocal running, maximum
                running += 1
                maximum = max(maximum, running)
                await asyncio.sleep(0.01)
                running -= 1
                return ToolResult.text("ok")

        first = assistant(
            tool_call("1", "delay", {"value": "a", "delay": 0}),
            tool_call("2", "delay", {"value": "b", "delay": 0}),
            stop="toolUse",
        )
        agent = Agent(
            AgentOptions(
                model=Model(id="mock"),
                stream_fn=scripted_stream([first, assistant({"type": "text", "text": "ok"})]),
                tools=[ProbeTool()],
            )
        )
        await agent.prompt("go")
        self.assertEqual(maximum, 1)

    async def test_follow_up_queue(self) -> None:
        agent = Agent(
            AgentOptions(
                model=Model(id="mock"),
                stream_fn=scripted_stream(
                    [assistant({"type": "text", "text": "one"}), assistant({"type": "text", "text": "two"})]
                ),
            )
        )
        agent.follow_up("第二个问题")
        await agent.prompt("第一个问题")
        roles = [message["role"] for message in agent.state.messages]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])

    async def test_all_terminating_tools_skip_followup_llm_call(self) -> None:
        first = assistant(
            tool_call("1", "delay", {"value": "a", "delay": 0, "terminate": True}),
            tool_call("2", "delay", {"value": "b", "delay": 0, "terminate": True}),
            stop="toolUse",
        )
        agent = Agent(
            AgentOptions(
                model=Model(id="mock"),
                stream_fn=scripted_stream([first]),
                tools=[DelayTool()],
            )
        )
        await agent.prompt("go")
        self.assertIsNone(agent.state.error_message)
        self.assertEqual(agent.state.messages[-1]["role"], "toolResult")

    async def test_length_stop_never_executes_tool(self) -> None:
        executed = False

        class ProbeTool(DelayTool):
            async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
                nonlocal executed
                executed = True
                return ToolResult.text("bad")

        agent = Agent(
            AgentOptions(
                model=Model(id="mock"),
                stream_fn=scripted_stream(
                    [
                        assistant(tool_call("1", "delay", {"value": "x", "delay": 0}), stop="length"),
                        assistant({"type": "text", "text": "recovered"}),
                    ]
                ),
                tools=[ProbeTool()],
            )
        )
        await agent.prompt("go")
        self.assertFalse(executed)
        self.assertTrue(agent.state.messages[2]["isError"])

    async def test_message_end_listener_is_barrier_before_tool_hook(self) -> None:
        assistant_seen = False

        async def listener(event: dict[str, Any], signal: AbortSignal) -> None:
            nonlocal assistant_seen
            if event["type"] == "message_end" and event["message"].get("role") == "assistant":
                await asyncio.sleep(0.01)
                assistant_seen = True

        async def before_tool(context: dict[str, Any], signal: AbortSignal) -> None:
            self.assertTrue(assistant_seen)

        agent = Agent(
            AgentOptions(
                model=Model(id="mock"),
                stream_fn=scripted_stream(
                    [
                        assistant(tool_call("1", "delay", {"value": "x", "delay": 0}), stop="toolUse"),
                        assistant({"type": "text", "text": "ok"}),
                    ]
                ),
                tools=[DelayTool()],
                before_tool_call=before_tool,
            )
        )
        agent.subscribe(listener)
        await agent.prompt("go")


if __name__ == "__main__":
    unittest.main()
