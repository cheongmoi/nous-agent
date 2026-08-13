"""Agent 运行时使用的公共类型。

消息故意使用普通 ``dict``，以保持与 JSON、OpenAI-compatible API 以及
Pi 的可扩展 AgentMessage 设计之间的互操作性。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, TypeAlias

Message: TypeAlias = dict[str, Any]
AgentMessage: TypeAlias = dict[str, Any]
AgentEvent: TypeAlias = dict[str, Any]
ContentBlock: TypeAlias = dict[str, Any]
QueueMode: TypeAlias = Literal["all", "one-at-a-time"]
ToolExecutionMode: TypeAlias = Literal["parallel", "sequential"]
ThinkingLevel: TypeAlias = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]


@dataclass(slots=True)
class Model:
    """一次模型请求所需的稳定配置。"""

    id: str
    provider: str = "openai"
    api: str = "openai-completions"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    max_tokens: int | None = None
    context_window: int | None = None


class AbortSignal:
    """轻量级取消信号，作用相当于 TypeScript 的 AbortSignal。"""

    def __init__(self) -> None:
        import asyncio

        self._event = asyncio.Event()

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def abort(self) -> None:
        self._event.set()

    def raise_if_aborted(self) -> None:
        import asyncio

        if self.aborted:
            raise asyncio.CancelledError("操作已取消")


ToolUpdateCallback: TypeAlias = Callable[["ToolResult"], Awaitable[None]]


@dataclass(slots=True)
class ToolResult:
    """工具返回给模型的内容，以及仅供宿主/UI 使用的详情。"""

    content: list[ContentBlock]
    details: Any = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    added_tool_names: list[str] | None = None
    terminate: bool = False

    @classmethod
    def text(cls, text: str, **kwargs: Any) -> "ToolResult":
        return cls(content=[{"type": "text", "text": text}], **kwargs)


class AgentTool(ABC):
    """工具基类。

    ``parameters`` 使用 JSON Schema。工具失败时应抛出异常，不要把错误伪装成
    成功文本；循环会统一转换为 ``isError=true`` 的工具结果。
    """

    name: str
    label: str
    description: str
    parameters: dict[str, Any]
    execution_mode: ToolExecutionMode | None = None

    def prepare_arguments(self, arguments: Any) -> Any:
        return arguments

    @abstractmethod
    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        raise NotImplementedError

    def as_llm_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class AgentContext:
    system_prompt: str
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] = field(default_factory=list)

    def copy(self) -> "AgentContext":
        # 消息块内部不在循环中原地改写，复制顶层数组即可隔离调用方的追加操作。
        return AgentContext(self.system_prompt, list(self.messages), list(self.tools))


@dataclass(slots=True)
class AgentState:
    system_prompt: str = ""
    model: Model = field(default_factory=lambda: Model(id="unknown"))
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    streaming_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error_message: str | None = None


ConvertToLlm: TypeAlias = Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]
TransformContext: TypeAlias = Callable[
    [list[AgentMessage], AbortSignal | None],
    list[AgentMessage] | Awaitable[list[AgentMessage]],
]
EventListener: TypeAlias = Callable[[AgentEvent, AbortSignal], None | Awaitable[None]]


@dataclass(slots=True)
class AgentOptions:
    """Agent 配置；钩子参数与 Pi Agent Core 的概念一一对应。"""

    model: Model
    stream_fn: Any
    system_prompt: str = ""
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    thinking_level: ThinkingLevel = "off"
    convert_to_llm: ConvertToLlm | None = None
    transform_context: TransformContext | None = None
    get_api_key: Callable[[str], str | None | Awaitable[str | None]] | None = None
    before_tool_call: Callable[..., Any] | None = None
    after_tool_call: Callable[..., Any] | None = None
    should_stop_after_turn: Callable[..., Any] | None = None
    prepare_next_turn: Callable[..., Any] | None = None
    # 这两个回调通常由 Agent 内部的消息队列注入；低层调用者也可直接提供。
    get_steering_messages: Callable[..., Any] | None = None
    get_follow_up_messages: Callable[..., Any] | None = None
    steering_mode: QueueMode = "one-at-a-time"
    follow_up_mode: QueueMode = "one-at-a-time"
    session_id: str | None = None
    tool_execution: ToolExecutionMode = "parallel"
    temperature: float | None = None


def text_message(role: str, text: str, *, timestamp: float | None = None) -> AgentMessage:
    """创建 Pi 风格的文本消息。"""

    import time

    return {
        "role": role,
        "content": [{"type": "text", "text": text}],
        "timestamp": timestamp if timestamp is not None else time.time(),
    }
