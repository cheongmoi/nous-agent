"""基于标准库实现的 OpenAI Chat Completions 流式适配器。"""

from __future__ import annotations

import asyncio
import copy
import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from ..stream import AssistantMessageStream
from ..types import AbortSignal, AgentMessage, Model


async def openai_compatible_stream(
    model: Model,
    context: dict[str, Any],
    options: dict[str, Any] | None = None,
) -> AssistantMessageStream:
    """调用 OpenAI-compatible ``/chat/completions`` SSE 接口。

    网络、鉴权和协议错误会编码为正常的 error 消息流，满足 Agent Loop 的
    “模型函数不抛业务错误”契约。
    """

    options = options or {}
    stream = AssistantMessageStream()
    loop = asyncio.get_running_loop()

    def notify(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(stream.push, event)

    async def produce() -> None:
        try:
            message = await asyncio.to_thread(_request, model, context, options, notify)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            message = _error_message(model, str(error), aborted=isinstance(error, asyncio.CancelledError))
        stream.finish(message, error=message.get("stopReason") in {"error", "aborted"})

    asyncio.create_task(produce())
    return stream


def _request(
    model: Model,
    context: dict[str, Any],
    options: dict[str, Any],
    notify: Callable[[dict[str, Any]], None],
) -> AgentMessage:
    signal: AbortSignal | None = options.get("signal")
    api_key = options.get("api_key") or model.api_key
    payload: dict[str, Any] = {
        "model": model.id,
        "messages": _to_openai_messages(context.get("systemPrompt", ""), context.get("messages", [])),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    tools = context.get("tools") or []
    if tools:
        payload["tools"] = [tool.as_llm_tool() for tool in tools]
    if model.max_tokens:
        payload["max_tokens"] = model.max_tokens
    if options.get("temperature") is not None:
        payload["temperature"] = options["temperature"]
    if options.get("reasoning"):
        payload["reasoning_effort"] = options["reasoning"]

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream", **model.headers}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        _chat_completions_url(model.base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    message: AgentMessage = {
        "role": "assistant",
        "content": [],
        "api": model.api,
        "provider": model.provider,
        "model": model.id,
        "usage": _usage(None),
        "stopReason": None,
        "timestamp": time.time(),
    }
    notify({"type": "start", "partial": copy.deepcopy(message)})
    text_index: int | None = None
    thinking_index: int | None = None
    tool_indexes: dict[int, int] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw_line in response:
                if signal and signal.aborted:
                    return _finish_aborted(message)
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]

                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        if thinking_index is None:
                            thinking_index = len(message["content"])
                            message["content"].append({"type": "thinking", "thinking": ""})
                            event_type = "thinking_start"
                        else:
                            event_type = "thinking_delta"
                        message["content"][thinking_index]["thinking"] += reasoning
                        notify({"type": event_type, "delta": reasoning, "partial": copy.deepcopy(message)})

                    content = delta.get("content")
                    if content:
                        if text_index is None:
                            text_index = len(message["content"])
                            message["content"].append({"type": "text", "text": ""})
                            event_type = "text_start"
                        else:
                            event_type = "text_delta"
                        message["content"][text_index]["text"] += content
                        notify({"type": event_type, "delta": content, "partial": copy.deepcopy(message)})

                    for tool_delta in delta.get("tool_calls") or []:
                        source_index = int(tool_delta.get("index", 0))
                        if source_index not in tool_indexes:
                            tool_indexes[source_index] = len(message["content"])
                            message["content"].append(
                                {
                                    "type": "toolCall",
                                    "id": tool_delta.get("id") or f"call_{source_index}",
                                    # 名称与参数统一在下方按 delta 追加，首包不能预填后再重复拼接。
                                    "name": "",
                                    "arguments": "",
                                }
                            )
                            event_type = "toolcall_start"
                        else:
                            event_type = "toolcall_delta"
                        block = message["content"][tool_indexes[source_index]]
                        if tool_delta.get("id"):
                            block["id"] = tool_delta["id"]
                        function = tool_delta.get("function") or {}
                        block["name"] += function.get("name") or ""
                        block["arguments"] += function.get("arguments") or ""
                        notify({"type": event_type, "partial": copy.deepcopy(message)})
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return _error_message(model, f"HTTP {error.code}: {body[:2000]}")
    except urllib.error.URLError as error:
        return _error_message(model, f"网络请求失败：{error.reason}")
    except TimeoutError:
        return _error_message(model, "模型请求超时")

    for index in tool_indexes.values():
        block = message["content"][index]
        raw_arguments = block["arguments"]
        try:
            block["arguments"] = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            # 保留原字符串，让工具参数校验明确报错，绝不猜测并执行残缺参数。
            block["arguments"] = raw_arguments
        notify({"type": "toolcall_end", "partial": copy.deepcopy(message)})

    message["usage"] = _usage(usage)
    message["stopReason"] = {
        "stop": "stop",
        "tool_calls": "toolUse",
        "length": "length",
        "content_filter": "error",
    }.get(finish_reason, finish_reason or "stop")
    if finish_reason == "content_filter":
        message["errorMessage"] = "模型响应被内容过滤器阻止"
    return message


def _to_openai_messages(system_prompt: str, messages: list[AgentMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})
    for message in messages:
        role = message.get("role")
        if role == "user":
            result.append({"role": "user", "content": _user_content(message.get("content", []))})
        elif role == "assistant":
            item: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(
                    block.get("text", "") for block in message.get("content", []) if block.get("type") == "text"
                )
                or None,
            }
            calls = []
            for block in message.get("content", []):
                if block.get("type") == "toolCall":
                    arguments = block.get("arguments", {})
                    calls.append(
                        {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": arguments
                                if isinstance(arguments, str)
                                else json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    )
            if calls:
                item["tool_calls"] = calls
            result.append(item)
        elif role == "toolResult":
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("toolCallId"),
                    "content": _flatten_text(message.get("content", [])),
                }
            )
    return result


def _user_content(blocks: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if all(block.get("type") == "text" for block in blocks):
        return "".join(block.get("text", "") for block in blocks)
    content: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("type") == "text":
            content.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            mime = block.get("mimeType", "image/png")
            data = block.get("data", "")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    return content


def _flatten_text(blocks: list[dict[str, Any]]) -> str:
    parts = [block.get("text", "") for block in blocks if block.get("type") == "text"]
    image_count = sum(block.get("type") == "image" for block in blocks)
    if image_count:
        parts.append(f"[{image_count} 张图片已由工具返回，但 Chat Completions 工具消息无法直接附带图片]")
    return "\n".join(filter(None, parts))


def _chat_completions_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url if url.endswith("/chat/completions") else f"{url}/chat/completions"


def _usage(value: dict[str, Any] | None) -> dict[str, Any]:
    value = value or {}
    prompt = value.get("prompt_tokens", 0)
    completion = value.get("completion_tokens", 0)
    details = value.get("prompt_tokens_details") or {}
    return {
        "input": prompt,
        "output": completion,
        "cacheRead": details.get("cached_tokens", 0),
        "cacheWrite": 0,
        "totalTokens": value.get("total_tokens", prompt + completion),
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }


def _error_message(model: Model, reason: str, *, aborted: bool = False) -> AgentMessage:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "api": model.api,
        "provider": model.provider,
        "model": model.id,
        "usage": _usage(None),
        "stopReason": "aborted" if aborted else "error",
        "errorMessage": reason,
        "timestamp": time.time(),
    }


def _finish_aborted(message: AgentMessage) -> AgentMessage:
    message["stopReason"] = "aborted"
    message["errorMessage"] = "操作已取消"
    return message
