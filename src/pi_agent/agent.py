"""带状态、消息队列和监听器屏障的 Agent 封装。"""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from collections import deque
from dataclasses import replace
from typing import Any

from .loop import run_agent_loop, run_agent_loop_continue
from .types import (
    AbortSignal,
    AgentContext,
    AgentEvent,
    AgentMessage,
    AgentOptions,
    AgentState,
    EventListener,
    QueueMode,
)


class _PendingMessageQueue:
    def __init__(self, mode: QueueMode) -> None:
        self.mode = mode
        self._messages: deque[AgentMessage] = deque()

    def enqueue(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def drain(self) -> list[AgentMessage]:
        if self.mode == "all":
            messages = list(self._messages)
            self._messages.clear()
            return messages
        return [self._messages.popleft()] if self._messages else []

    def clear(self) -> None:
        self._messages.clear()

    def __bool__(self) -> bool:
        return bool(self._messages)


class Agent:
    """Pi Agent Core 的状态化 Python 对应实现。"""

    def __init__(self, options: AgentOptions) -> None:
        self.options = options
        self.state = AgentState(
            system_prompt=options.system_prompt,
            model=options.model,
            thinking_level=options.thinking_level,
            tools=list(options.tools),
            messages=list(options.messages),
        )
        self._listeners: list[EventListener] = []
        self._steering = _PendingMessageQueue(options.steering_mode)
        self._follow_ups = _PendingMessageQueue(options.follow_up_mode)
        self._signal: AbortSignal | None = None
        self._idle: asyncio.Future[None] | None = None
        self._running_task: asyncio.Task[Any] | None = None

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_ups.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_ups.mode = mode

    @property
    def signal(self) -> AbortSignal | None:
        return self._signal

    def subscribe(self, listener: EventListener):
        """注册监听器并返回取消订阅函数。监听器按注册顺序等待。"""

        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def steer(self, message: AgentMessage | str) -> None:
        self._steering.enqueue(self._normalize_one(message))

    def follow_up(self, message: AgentMessage | str) -> None:
        self._follow_ups.enqueue(self._normalize_one(message))

    def clear_steering_queue(self) -> None:
        self._steering.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_ups.clear()

    def clear_all_queues(self) -> None:
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        return bool(self._steering or self._follow_ups)

    def abort(self) -> None:
        if self._signal:
            self._signal.abort()

    async def wait_for_idle(self) -> None:
        # shield 防止等待方被取消时连带取消 Agent 自己的完成信号。
        if self._idle:
            await asyncio.shield(self._idle)

    def reset(self) -> None:
        if self.state.is_streaming:
            raise RuntimeError("Agent 正在处理消息，请等待完成后再重置")
        self.state.messages = []
        self.state.streaming_message = None
        self.state.pending_tool_calls = set()
        self.state.error_message = None
        self.clear_all_queues()

    async def prompt(
        self,
        prompt: str | AgentMessage | list[AgentMessage],
        images: list[dict[str, Any]] | None = None,
    ) -> None:
        if self.state.is_streaming:
            raise RuntimeError("Agent 正在处理消息；请使用 steer()/follow_up() 排队，或等待完成")
        messages = self._normalize_prompt(prompt, images)
        await self._run(lambda options: run_agent_loop(messages, self._snapshot(), options, self._process_event, self._signal))

    async def continue_(self) -> None:
        """从现有 user/toolResult 尾部继续；名称加下划线以避开 Python 关键字。"""

        if self.state.is_streaming:
            raise RuntimeError("Agent 正在处理消息，请等待完成")
        if not self.state.messages:
            raise RuntimeError("没有可继续的消息")

        if self.state.messages[-1].get("role") == "assistant":
            steering = self._steering.drain()
            if steering:
                await self._run(
                    lambda options: run_agent_loop(
                        steering, self._snapshot(), options, self._process_event, self._signal
                    ),
                    skip_initial_steering=True,
                )
                return
            follow_ups = self._follow_ups.drain()
            if follow_ups:
                await self._run(
                    lambda options: run_agent_loop(
                        follow_ups, self._snapshot(), options, self._process_event, self._signal
                    )
                )
                return
            raise RuntimeError("无法从 assistant 消息继续")

        await self._run(
            lambda options: run_agent_loop_continue(
                self._snapshot(), options, self._process_event, self._signal
            )
        )

    async def _run(self, executor: Any, *, skip_initial_steering: bool = False) -> None:
        if self.state.is_streaming:
            raise RuntimeError("Agent 已在运行")

        loop = asyncio.get_running_loop()
        self._idle = loop.create_future()
        self._signal = AbortSignal()
        self._running_task = asyncio.current_task()
        self.state.is_streaming = True
        self.state.streaming_message = None
        self.state.error_message = None

        first_poll = skip_initial_steering

        async def steering_messages() -> list[AgentMessage]:
            nonlocal first_poll
            if first_poll:
                first_poll = False
                return []
            return self._steering.drain()

        # 每次运行都拍摄当前配置，避免运行中改配置造成半个 turn 使用新值。
        run_options = replace(
            self.options,
            model=self.state.model,
            system_prompt=self.state.system_prompt,
            tools=list(self.state.tools),
            messages=list(self.state.messages),
            thinking_level=self.state.thinking_level,
            get_steering_messages=steering_messages,
            get_follow_up_messages=lambda: self._follow_ups.drain(),
        )

        try:
            await executor(run_options)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            await self._handle_failure(error)
        finally:
            self.state.is_streaming = False
            self.state.streaming_message = None
            self.state.pending_tool_calls = set()
            if self._idle and not self._idle.done():
                self._idle.set_result(None)
            self._signal = None
            self._running_task = None

    async def _handle_failure(self, error: BaseException) -> None:
        aborted = (self._signal.aborted if self._signal else False) or isinstance(error, asyncio.CancelledError)
        message: AgentMessage = {
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
            "api": self.state.model.api,
            "provider": self.state.model.provider,
            "model": self.state.model.id,
            "usage": _empty_usage(),
            "stopReason": "aborted" if aborted else "error",
            "errorMessage": "操作已取消" if aborted else str(error),
            "timestamp": time.time(),
        }
        await self._process_event({"type": "message_start", "message": message})
        await self._process_event({"type": "message_end", "message": message})
        await self._process_event({"type": "turn_end", "message": message, "toolResults": []})
        await self._process_event({"type": "agent_end", "messages": [message]})

    async def _process_event(self, event: AgentEvent) -> None:
        event_type = event["type"]
        if event_type in {"message_start", "message_update"}:
            self.state.streaming_message = event["message"]
        elif event_type == "message_end":
            self.state.streaming_message = None
            self.state.messages.append(event["message"])
        elif event_type == "tool_execution_start":
            self.state.pending_tool_calls = {*self.state.pending_tool_calls, event["toolCallId"]}
        elif event_type == "tool_execution_end":
            pending = set(self.state.pending_tool_calls)
            pending.discard(event["toolCallId"])
            self.state.pending_tool_calls = pending
        elif event_type == "turn_end":
            message = event["message"]
            if message.get("role") == "assistant" and message.get("errorMessage"):
                self.state.error_message = message["errorMessage"]
        elif event_type == "agent_end":
            self.state.streaming_message = None

        # 这是高层 Agent 相对低层 EventStream 的关键差异：监听器是状态机屏障。
        if self._signal is None:
            raise RuntimeError("Agent 事件出现在活动运行之外")
        for listener in list(self._listeners):
            result = listener(event, self._signal)
            if inspect.isawaitable(result):
                await result

    def _snapshot(self) -> AgentContext:
        return AgentContext(self.state.system_prompt, list(self.state.messages), list(self.state.tools))

    @staticmethod
    def _normalize_prompt(
        prompt: str | AgentMessage | list[AgentMessage],
        images: list[dict[str, Any]] | None,
    ) -> list[AgentMessage]:
        if isinstance(prompt, list):
            return prompt
        if isinstance(prompt, dict):
            return [prompt]
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(images or [])
        return [{"role": "user", "content": content, "timestamp": time.time()}]

    @staticmethod
    def _normalize_one(message: AgentMessage | str) -> AgentMessage:
        if isinstance(message, dict):
            return message
        return {"role": "user", "content": [{"type": "text", "text": message}], "timestamp": time.time()}


def _empty_usage() -> dict[str, Any]:
    return {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }
