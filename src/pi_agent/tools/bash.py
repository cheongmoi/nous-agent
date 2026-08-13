"""bash/shell 命令工具。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..types import AbortSignal, AgentTool, ToolResult, ToolUpdateCallback
from .path_utils import truncate_output


class BashTool(AgentTool):
    name = "bash"
    label = "bash"
    description = (
        "Execute a shell command in the working directory. Returns stdout, stderr and exit code; "
        "optional timeout is measured in seconds."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command"},
            "timeout": {"type": "number", "minimum": 0.1, "description": "Optional timeout in seconds"},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, cwd: str) -> None:
        self.cwd = Path(cwd).resolve()

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        if signal:
            signal.raise_if_aborted()
        process = await asyncio.create_subprocess_shell(
            params["command"],
            cwd=str(self.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        chunks: list[str] = []

        async def pump(reader: asyncio.StreamReader | None, label: str) -> None:
            if reader is None:
                return
            while chunk := await reader.read(4096):
                text = chunk.decode("utf-8", errors="replace")
                chunks.append(text)
                if on_update:
                    visible, _ = truncate_output("".join(chunks))
                    await on_update(ToolResult.text(visible, details={"stream": label}))

        stdout_task = asyncio.create_task(pump(process.stdout, "stdout"))
        stderr_task = asyncio.create_task(pump(process.stderr, "stderr"))
        wait_task = asyncio.create_task(process.wait())
        abort_task = asyncio.create_task(signal.wait()) if signal else None
        timeout_task = asyncio.create_task(asyncio.sleep(float(params["timeout"]))) if params.get("timeout") else None
        watchers = [wait_task, *([abort_task] if abort_task else []), *([timeout_task] if timeout_task else [])]

        done, _ = await asyncio.wait(watchers, return_when=asyncio.FIRST_COMPLETED)
        timed_out = timeout_task in done if timeout_task else False
        aborted = abort_task in done if abort_task else False
        if (timed_out or aborted) and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        for watcher in watchers:
            if not watcher.done():
                watcher.cancel()
        await asyncio.gather(*watchers, return_exceptions=True)
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        output, truncated = truncate_output("".join(chunks))
        if timed_out:
            output += "\n[命令执行超时]"
        elif aborted:
            output += "\n[命令已取消]"
        if not output:
            output = "(无输出)"
        return ToolResult.text(
            output,
            details={
                "command": params["command"],
                "cwd": str(self.cwd),
                "exitCode": process.returncode,
                "timedOut": timed_out,
                "aborted": aborted,
                "truncated": truncated,
            },
        )
