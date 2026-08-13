"""Pi 风格的模型—工具循环。"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
from typing import Any, Callable

from .stream import EventStream, emit_to, maybe_await
from .types import (
    AbortSignal,
    AgentContext,
    AgentEvent,
    AgentMessage,
    AgentOptions,
    AgentTool,
    Model,
    ToolResult,
)
from .validation import validate_arguments

EventSink = Callable[[AgentEvent], Any]


def _default_convert(messages: list[AgentMessage]) -> list[AgentMessage]:
    return [message for message in messages if message.get("role") in {"user", "assistant", "toolResult"}]


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    options: AgentOptions,
    emit: EventSink,
    signal: AbortSignal | None = None,
) -> list[AgentMessage]:
    new_messages = list(prompts)
    current = context.copy()
    current.messages.extend(prompts)

    await emit_to(emit, {"type": "agent_start"})
    await emit_to(emit, {"type": "turn_start"})
    for prompt in prompts:
        await emit_to(emit, {"type": "message_start", "message": prompt})
        await emit_to(emit, {"type": "message_end", "message": prompt})

    await _run_loop(current, new_messages, options, signal, emit)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    options: AgentOptions,
    emit: EventSink,
    signal: AbortSignal | None = None,
) -> list[AgentMessage]:
    if not context.messages:
        raise ValueError("无法继续：上下文中没有消息")
    if context.messages[-1].get("role") == "assistant":
        raise ValueError("无法从 assistant 消息继续")

    current = context.copy()
    new_messages: list[AgentMessage] = []
    await emit_to(emit, {"type": "agent_start"})
    await emit_to(emit, {"type": "turn_start"})
    await _run_loop(current, new_messages, options, signal, emit)
    return new_messages


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    options: AgentOptions,
    signal: AbortSignal | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    stream: EventStream[AgentEvent, list[AgentMessage]] = EventStream()

    async def produce() -> None:
        try:
            result = await run_agent_loop(prompts, context, options, stream.push, signal)
            stream.end(result)
        except BaseException as error:
            stream.fail(error)

    asyncio.create_task(produce())
    return stream


def agent_loop_continue(
    context: AgentContext,
    options: AgentOptions,
    signal: AbortSignal | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    stream: EventStream[AgentEvent, list[AgentMessage]] = EventStream()

    async def produce() -> None:
        try:
            result = await run_agent_loop_continue(context, options, stream.push, signal)
            stream.end(result)
        except BaseException as error:
            stream.fail(error)

    asyncio.create_task(produce())
    return stream


async def _run_loop(
    context: AgentContext,
    new_messages: list[AgentMessage],
    options: AgentOptions,
    signal: AbortSignal | None,
    emit: EventSink,
) -> None:
    first_turn = True
    pending_messages = await _poll_messages(getattr(options, "get_steering_messages", None))

    # 外层循环只负责“本来要停止时”接收 follow-up；内层循环负责工具链和 steer。
    while True:
        has_more_tool_calls = True
        while has_more_tool_calls or pending_messages:
            if not first_turn:
                await emit_to(emit, {"type": "turn_start"})
            else:
                first_turn = False

            for message in pending_messages:
                await emit_to(emit, {"type": "message_start", "message": message})
                await emit_to(emit, {"type": "message_end", "message": message})
                context.messages.append(message)
                new_messages.append(message)
            pending_messages = []

            message = await _stream_assistant_response(context, options, signal, emit)
            new_messages.append(message)
            stop_reason = message.get("stopReason")
            if stop_reason in {"error", "aborted"}:
                await emit_to(emit, {"type": "turn_end", "message": message, "toolResults": []})
                await emit_to(emit, {"type": "agent_end", "messages": new_messages})
                return

            tool_calls = [block for block in message.get("content", []) if block.get("type") == "toolCall"]
            tool_results: list[AgentMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                if stop_reason == "length":
                    batch = await _fail_truncated_tool_calls(tool_calls, emit)
                else:
                    batch = await _execute_tool_calls(context, message, tool_calls, options, signal, emit)
                tool_results.extend(batch.messages)
                has_more_tool_calls = not batch.terminate
                context.messages.extend(tool_results)
                new_messages.extend(tool_results)

            await emit_to(emit, {"type": "turn_end", "message": message, "toolResults": tool_results})
            turn_context = {
                "message": message,
                "toolResults": tool_results,
                "context": context,
                "newMessages": new_messages,
            }

            if options.prepare_next_turn:
                update = await maybe_await(options.prepare_next_turn(turn_context, signal))
                if update:
                    context = update.get("context", context)
                    options.model = update.get("model", options.model)
                    if "thinkingLevel" in update:
                        options.thinking_level = update["thinkingLevel"]

            if options.should_stop_after_turn and await maybe_await(
                options.should_stop_after_turn(turn_context, signal)
            ):
                await emit_to(emit, {"type": "agent_end", "messages": new_messages})
                return

            pending_messages = await _poll_messages(getattr(options, "get_steering_messages", None))

        follow_ups = await _poll_messages(getattr(options, "get_follow_up_messages", None))
        if follow_ups:
            pending_messages = follow_ups
            continue
        break

    await emit_to(emit, {"type": "agent_end", "messages": new_messages})


async def _poll_messages(callback: Callable[..., Any] | None) -> list[AgentMessage]:
    if callback is None:
        return []
    return list(await maybe_await(callback()) or [])


async def _stream_assistant_response(
    context: AgentContext,
    options: AgentOptions,
    signal: AbortSignal | None,
    emit: EventSink,
) -> AgentMessage:
    messages = list(context.messages)
    if options.transform_context:
        messages = await maybe_await(options.transform_context(messages, signal))
    converter = options.convert_to_llm or _default_convert
    llm_messages = await maybe_await(converter(messages))

    api_key = options.model.api_key
    if options.get_api_key:
        api_key = await maybe_await(options.get_api_key(options.model.provider)) or api_key

    stream_options = {
        "api_key": api_key,
        "signal": signal,
        "session_id": options.session_id,
        "reasoning": None if options.thinking_level == "off" else options.thinking_level,
        "temperature": options.temperature,
    }
    llm_context = {
        "systemPrompt": context.system_prompt,
        "messages": llm_messages,
        "tools": context.tools,
    }
    response = await maybe_await(options.stream_fn(options.model, llm_context, stream_options))

    partial: AgentMessage | None = None
    added_partial = False
    async for event in response:
        event_type = event.get("type")
        if event_type == "start":
            partial = event["partial"]
            context.messages.append(partial)
            added_partial = True
            await emit_to(emit, {"type": "message_start", "message": copy.deepcopy(partial)})
        elif event_type in {
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        }:
            if partial is not None:
                partial = event["partial"]
                context.messages[-1] = partial
                await emit_to(
                    emit,
                    {
                        "type": "message_update",
                        "message": copy.deepcopy(partial),
                        "assistantMessageEvent": event,
                    },
                )
        elif event_type in {"done", "error"}:
            final = await response.result()
            if added_partial:
                context.messages[-1] = final
            else:
                context.messages.append(final)
                await emit_to(emit, {"type": "message_start", "message": copy.deepcopy(final)})
            await emit_to(emit, {"type": "message_end", "message": final})
            return final

    final = await response.result()
    if added_partial:
        context.messages[-1] = final
    else:
        context.messages.append(final)
        await emit_to(emit, {"type": "message_start", "message": copy.deepcopy(final)})
    await emit_to(emit, {"type": "message_end", "message": final})
    return final


@dataclass(slots=True)
class _Batch:
    messages: list[AgentMessage]
    terminate: bool


@dataclass(slots=True)
class _Prepared:
    tool_call: dict[str, Any]
    tool: AgentTool
    arguments: dict[str, Any]


@dataclass(slots=True)
class _Finalized:
    tool_call: dict[str, Any]
    result: ToolResult
    is_error: bool


async def _fail_truncated_tool_calls(tool_calls: list[dict[str, Any]], emit: EventSink) -> _Batch:
    messages: list[AgentMessage] = []
    for call in tool_calls:
        await _emit_tool_start(call, emit)
        outcome = _Finalized(
            call,
            _error_result(
                f'工具调用 "{call.get("name")}" 未执行：模型输出达到长度上限，参数可能被截断，请重新发起完整调用。'
            ),
            True,
        )
        await _emit_tool_end(outcome, emit)
        message = _tool_result_message(outcome)
        await _emit_tool_message(message, emit)
        messages.append(message)
    return _Batch(messages, False)


async def _execute_tool_calls(
    context: AgentContext,
    assistant: AgentMessage,
    tool_calls: list[dict[str, Any]],
    options: AgentOptions,
    signal: AbortSignal | None,
    emit: EventSink,
) -> _Batch:
    sequential_override = any(
        tool.execution_mode == "sequential"
        for call in tool_calls
        for tool in context.tools
        if tool.name == call.get("name")
    )
    if options.tool_execution == "sequential" or sequential_override:
        return await _execute_sequential(context, assistant, tool_calls, options, signal, emit)
    return await _execute_parallel(context, assistant, tool_calls, options, signal, emit)


async def _execute_sequential(
    context: AgentContext,
    assistant: AgentMessage,
    tool_calls: list[dict[str, Any]],
    options: AgentOptions,
    signal: AbortSignal | None,
    emit: EventSink,
) -> _Batch:
    finalized: list[_Finalized] = []
    messages: list[AgentMessage] = []
    for call in tool_calls:
        await _emit_tool_start(call, emit)
        prepared = await _prepare_tool_call(context, assistant, call, options, signal)
        if isinstance(prepared, _Finalized):
            outcome = prepared
        else:
            executed = await _execute_prepared(prepared, signal, emit)
            outcome = await _finalize(context, assistant, prepared, executed, options, signal)
        await _emit_tool_end(outcome, emit)
        message = _tool_result_message(outcome)
        await _emit_tool_message(message, emit)
        finalized.append(outcome)
        messages.append(message)
        if signal and signal.aborted:
            break
    return _Batch(messages, bool(finalized) and all(item.result.terminate for item in finalized))


async def _execute_parallel(
    context: AgentContext,
    assistant: AgentMessage,
    tool_calls: list[dict[str, Any]],
    options: AgentOptions,
    signal: AbortSignal | None,
    emit: EventSink,
) -> _Batch:
    # 预检严格串行，确保审批/UI 顺序稳定；只有真正的 execute 阶段并发。
    entries: list[_Finalized | _Prepared] = []
    for call in tool_calls:
        await _emit_tool_start(call, emit)
        prepared = await _prepare_tool_call(context, assistant, call, options, signal)
        if isinstance(prepared, _Finalized):
            await _emit_tool_end(prepared, emit)
        entries.append(prepared)
        if signal and signal.aborted:
            break

    async def run(prepared: _Prepared) -> _Finalized:
        executed = await _execute_prepared(prepared, signal, emit)
        outcome = await _finalize(context, assistant, prepared, executed, options, signal)
        # 完成事件按实际完成顺序发出，而最终消息仍由 gather 的源顺序保证。
        await _emit_tool_end(outcome, emit)
        return outcome

    tasks: list[asyncio.Task[_Finalized] | None] = []
    for entry in entries:
        tasks.append(None if isinstance(entry, _Finalized) else asyncio.create_task(run(entry)))

    finalized: list[_Finalized] = []
    for entry, task in zip(entries, tasks, strict=True):
        finalized.append(entry if isinstance(entry, _Finalized) else await task)  # type: ignore[arg-type]

    messages: list[AgentMessage] = []
    for outcome in finalized:
        message = _tool_result_message(outcome)
        await _emit_tool_message(message, emit)
        messages.append(message)
    return _Batch(messages, bool(finalized) and all(item.result.terminate for item in finalized))


async def _prepare_tool_call(
    context: AgentContext,
    assistant: AgentMessage,
    call: dict[str, Any],
    options: AgentOptions,
    signal: AbortSignal | None,
) -> _Prepared | _Finalized:
    tool = next((item for item in context.tools if item.name == call.get("name")), None)
    if tool is None:
        return _Finalized(call, _error_result(f"未找到工具 {call.get('name')}"), True)
    try:
        arguments = tool.prepare_arguments(call.get("arguments", {}))
        arguments = validate_arguments(tool.parameters, arguments)
        if options.before_tool_call:
            hook_context = {
                "assistantMessage": assistant,
                "toolCall": call,
                "args": arguments,
                "context": context,
            }
            decision = await maybe_await(options.before_tool_call(hook_context, signal))
            if signal and signal.aborted:
                return _Finalized(call, _error_result("操作已取消"), True)
            if decision and decision.get("block"):
                result = _error_result(decision.get("reason") or "工具执行已被阻止")
                result.terminate = decision.get("terminate") is True
                return _Finalized(call, result, True)
        if signal and signal.aborted:
            return _Finalized(call, _error_result("操作已取消"), True)
        return _Prepared(call, tool, arguments)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return _Finalized(call, _error_result(str(error)), True)


async def _execute_prepared(
    prepared: _Prepared,
    signal: AbortSignal | None,
    emit: EventSink,
) -> tuple[ToolResult, bool]:
    accepting_updates = True

    async def on_update(partial: ToolResult) -> None:
        if not accepting_updates:
            return
        await emit_to(
            emit,
            {
                "type": "tool_execution_update",
                "toolCallId": prepared.tool_call.get("id"),
                "toolName": prepared.tool_call.get("name"),
                "args": prepared.tool_call.get("arguments", {}),
                "partialResult": partial,
            },
        )

    try:
        result = await prepared.tool.execute(
            str(prepared.tool_call.get("id")), prepared.arguments, signal, on_update
        )
        accepting_updates = False
        return result, False
    except asyncio.CancelledError:
        accepting_updates = False
        return _error_result("操作已取消"), True
    except BaseException as error:
        accepting_updates = False
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return _error_result(str(error)), True


async def _finalize(
    context: AgentContext,
    assistant: AgentMessage,
    prepared: _Prepared,
    executed: tuple[ToolResult, bool],
    options: AgentOptions,
    signal: AbortSignal | None,
) -> _Finalized:
    result, is_error = executed
    if options.after_tool_call:
        try:
            hook_context = {
                "assistantMessage": assistant,
                "toolCall": prepared.tool_call,
                "args": prepared.arguments,
                "result": result,
                "isError": is_error,
                "context": context,
            }
            override = await maybe_await(options.after_tool_call(hook_context, signal))
            if override:
                # 与上游一致：字段级替换，不对 details/content 做深合并。
                if "content" in override:
                    result.content = override["content"]
                if "details" in override:
                    result.details = override["details"]
                if "usage" in override:
                    result.usage = override["usage"]
                if "terminate" in override:
                    result.terminate = override["terminate"]
                if "isError" in override:
                    is_error = override["isError"]
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            result, is_error = _error_result(str(error)), True
    return _Finalized(prepared.tool_call, result, is_error)


def _error_result(message: str) -> ToolResult:
    return ToolResult.text(message)


async def _emit_tool_start(call: dict[str, Any], emit: EventSink) -> None:
    await emit_to(
        emit,
        {
            "type": "tool_execution_start",
            "toolCallId": call.get("id"),
            "toolName": call.get("name"),
            "args": call.get("arguments", {}),
        },
    )


async def _emit_tool_end(outcome: _Finalized, emit: EventSink) -> None:
    await emit_to(
        emit,
        {
            "type": "tool_execution_end",
            "toolCallId": outcome.tool_call.get("id"),
            "toolName": outcome.tool_call.get("name"),
            "result": outcome.result,
            "isError": outcome.is_error,
        },
    )


def _tool_result_message(outcome: _Finalized) -> AgentMessage:
    message: AgentMessage = {
        "role": "toolResult",
        "toolCallId": outcome.tool_call.get("id"),
        "toolName": outcome.tool_call.get("name"),
        "content": outcome.result.content or [],
        "details": outcome.result.details,
        "isError": outcome.is_error,
        "timestamp": time.time(),
    }
    if outcome.result.usage is not None:
        message["usage"] = outcome.result.usage
    if outcome.result.added_tool_names:
        message["addedToolNames"] = outcome.result.added_tool_names
    return message


async def _emit_tool_message(message: AgentMessage, emit: EventSink) -> None:
    await emit_to(emit, {"type": "message_start", "message": message})
    await emit_to(emit, {"type": "message_end", "message": message})
