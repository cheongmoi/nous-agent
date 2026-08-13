"""可等待结果的异步事件流。"""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import Any, Generic, TypeVar

from .types import AgentEvent, AgentMessage

T = TypeVar("T")
R = TypeVar("R")
_END = object()


class EventStream(Generic[T, R]):
    """边生产边消费事件，并可在末尾取得完整结果。

    低层 ``agent_loop`` 会立即启动后台任务，行为与 Pi 的 EventStream 接近。
    消费事件只是观察行为；需要事件监听成为状态更新屏障时，应使用 ``Agent``。
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[T | object] = asyncio.Queue()
        self._result: asyncio.Future[R] = asyncio.get_running_loop().create_future()

    def push(self, event: T) -> None:
        self._queue.put_nowait(event)

    def end(self, result: R) -> None:
        if not self._result.done():
            self._result.set_result(result)
        self._queue.put_nowait(_END)

    def fail(self, error: BaseException) -> None:
        if not self._result.done():
            self._result.set_exception(error)
        self._queue.put_nowait(_END)

    def __aiter__(self) -> "EventStream[T, R]":
        return self

    async def __anext__(self) -> T:
        item = await self._queue.get()
        if item is _END:
            raise StopAsyncIteration
        return item  # type: ignore[return-value]

    async def result(self) -> R:
        return await self._result


class AssistantMessageStream:
    """模型适配器返回的事件流。"""

    def __init__(self) -> None:
        self._events: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue()
        self._result: asyncio.Future[AgentMessage] = asyncio.get_running_loop().create_future()

    def push(self, event: dict[str, Any]) -> None:
        self._events.put_nowait(event)

    def finish(self, message: AgentMessage, *, error: bool = False) -> None:
        if not self._result.done():
            self._result.set_result(message)
        self.push({"type": "error" if error else "done", "partial": copy.deepcopy(message)})
        self._events.put_nowait(_END)

    def __aiter__(self) -> "AssistantMessageStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        item = await self._events.get()
        if item is _END:
            raise StopAsyncIteration
        return item  # type: ignore[return-value]

    async def result(self) -> AgentMessage:
        return await self._result


def scripted_stream(messages: Iterable[AgentMessage]) -> Callable[..., Awaitable[AssistantMessageStream]]:
    """创建确定性的模型函数，供测试、示例与离线集成使用。"""

    remaining = iter(messages)

    async def stream_fn(*_: Any, **__: Any) -> AssistantMessageStream:
        try:
            message = copy.deepcopy(next(remaining))
        except StopIteration as error:
            raise RuntimeError("scripted_stream 没有更多响应") from error

        stream = AssistantMessageStream()

        async def produce() -> None:
            partial = {
                **message,
                "content": [],
                "stopReason": None,
                "timestamp": message.get("timestamp", time.time()),
            }
            stream.push({"type": "start", "partial": copy.deepcopy(partial)})
            for block in message.get("content", []):
                partial["content"].append(copy.deepcopy(block))
                event_type = "text_delta" if block.get("type") == "text" else "toolcall_end"
                stream.push(
                    {
                        "type": event_type,
                        "delta": block.get("text", ""),
                        "partial": copy.deepcopy(partial),
                    }
                )
            stream.finish(message, error=message.get("stopReason") == "error")

        asyncio.create_task(produce())
        return stream

    return stream_fn


async def maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


async def emit_to(listener: Callable[[AgentEvent], Any], event: AgentEvent) -> None:
    """按顺序等待事件接收方，确保状态更新先于下一阶段。"""

    result = listener(event)
    if inspect.isawaitable(result):
        await result

