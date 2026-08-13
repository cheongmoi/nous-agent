"""Pi Agent Python 的命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

from .agent import Agent
from .context import build_system_prompt
from .providers import openai_compatible_stream
from .session import JsonlSession, SessionStore
from .tools import coding_tools
from .types import AgentEvent, AgentOptions, Model

VERSION = "0.1.0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pi-agent-py",
        description="参考 Pi Agent 源码实现的 Python 编码 Agent",
    )
    parser.add_argument("prompt", nargs="*", help="初始提示词；省略时进入交互模式")
    parser.add_argument("-p", "--print", action="store_true", dest="print_mode", help="回复后退出")
    parser.add_argument("--mode", choices=["text", "json"], default="text", help="输出文本或 JSONL 事件")
    parser.add_argument("--model", default=os.getenv("PI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini")
    parser.add_argument(
        "--base-url",
        default=os.getenv("PI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
    )
    parser.add_argument("--provider", default=os.getenv("PI_PROVIDER", "openai"))
    parser.add_argument("--api-key", default=None, help="API Key；更推荐使用 PI_API_KEY/OPENAI_API_KEY")
    parser.add_argument("--cwd", default=os.getcwd(), help="Agent 工作目录")
    parser.add_argument("--system", help="完全替换默认系统提示词")
    parser.add_argument("--append-system", help="附加到系统提示词")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--no-tools", action="store_true", help="禁用内置编码工具")
    parser.add_argument("--sequential-tools", action="store_true", help="工具调用全部串行执行")
    parser.add_argument("--allow-outside-cwd", action="store_true", help="允许文件工具访问工作目录外部")
    parser.add_argument("--no-session", action="store_true", help="不保存 JSONL 会话")
    parser.add_argument("--session", type=Path, help="使用指定 JSONL 会话文件")
    parser.add_argument("-c", "--continue", action="store_true", dest="continue_session", help="恢复当前项目最近会话")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser


class _Renderer:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self._assistant_streamed = False

    async def __call__(self, event: AgentEvent, _signal: Any) -> None:
        if self.mode == "json":
            print(json.dumps(event, ensure_ascii=False, default=_json_default), flush=True)
            return
        event_type = event["type"]
        if event_type == "message_start" and event.get("message", {}).get("role") == "assistant":
            self._assistant_streamed = False
        elif event_type == "message_update":
            update = event.get("assistantMessageEvent", {})
            if update.get("type") in {"text_start", "text_delta"}:
                sys.stdout.write(update.get("delta", ""))
                sys.stdout.flush()
                self._assistant_streamed = True
        elif event_type == "message_end" and event.get("message", {}).get("role") == "assistant":
            message = event["message"]
            if not self._assistant_streamed:
                sys.stdout.write(_message_text(message))
            if message.get("errorMessage"):
                sys.stderr.write(f"\n错误：{message['errorMessage']}")
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif event_type == "tool_execution_start":
            args = json.dumps(event.get("args", {}), ensure_ascii=False)
            print(f"\n→ {event.get('toolName')} {args}", file=sys.stderr, flush=True)
        elif event_type == "tool_execution_end":
            marker = "失败" if event.get("isError") else "完成"
            print(f"← {event.get('toolName')} {marker}", file=sys.stderr, flush=True)


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value).__name__}")


def _message_text(message: dict[str, Any]) -> str:
    return "".join(
        block.get("text", "") for block in message.get("content", []) if block.get("type") == "text"
    )


def _open_session(args: argparse.Namespace, cwd: str) -> JsonlSession | None:
    if args.no_session:
        return None
    if args.session:
        return JsonlSession(args.session, cwd, create=not args.session.exists())
    store = SessionStore()
    if args.continue_session:
        return store.latest(cwd)
    return store.create(cwd)


async def _run(args: argparse.Namespace) -> int:
    cwd_path = Path(args.cwd).resolve()
    if not cwd_path.is_dir():
        print(f"工作目录不存在：{cwd_path}", file=sys.stderr)
        return 2

    tools = [] if args.no_tools else coding_tools(str(cwd_path), restrict_to_cwd=not args.allow_outside_cwd)
    system_prompt = build_system_prompt(
        str(cwd_path),
        tools,
        custom_prompt=args.system,
        append_prompt=args.append_system,
    )
    session = _open_session(args, str(cwd_path))
    previous_messages = session.messages() if session else []
    api_key = args.api_key or os.getenv("PI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = Model(
        id=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=api_key,
    )
    options = AgentOptions(
        model=model,
        stream_fn=openai_compatible_stream,
        system_prompt=system_prompt,
        tools=tools,
        messages=previous_messages,
        tool_execution="sequential" if args.sequential_tools else "parallel",
        session_id=session.session_id if session else None,
        temperature=args.temperature,
    )
    agent = Agent(options)
    agent.subscribe(_Renderer(args.mode))

    if session:
        async def persist(event: AgentEvent, _signal: Any) -> None:
            if event["type"] == "message_end":
                session.append(event["message"])

        agent.subscribe(persist)

    piped = ""
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
    prompt_parts = [*args.prompt, *([piped] if piped else [])]
    initial_prompt = "\n\n".join(prompt_parts).strip()
    if initial_prompt:
        await agent.prompt(initial_prompt)
        if args.print_mode or not sys.stdin.isatty():
            return 1 if agent.state.error_message else 0
    elif args.print_mode:
        print("print 模式需要提示词或管道输入", file=sys.stderr)
        return 2

    if session and args.mode == "text":
        print(f"会话：{session.path}", file=sys.stderr)
    print(f"Pi Agent Python · {model.provider}/{model.id} · 输入 /help 查看命令", file=sys.stderr)
    while True:
        try:
            user_input = (await asyncio.to_thread(input, "> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            break
        if user_input == "/help":
            print("/help  /session  /clear  /exit", file=sys.stderr)
            continue
        if user_input == "/session":
            print(session.path if session else "会话保存已禁用", file=sys.stderr)
            continue
        if user_input == "/clear":
            agent.reset()
            if session:
                # 新消息从根节点开始，确保下次恢复时不会把已清空的旧上下文接回来。
                session.branch_from(None)
            print("上下文已清空", file=sys.stderr)
            continue
        await agent.prompt(user_input)
    return 0


def main() -> None:
    # Windows 终端可能继承 CP950 等区域代码页，无法表示简体中文帮助与模型输出。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
