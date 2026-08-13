"""write 工具。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..types import AbortSignal, AgentTool, ToolResult, ToolUpdateCallback
from .path_utils import resolve_path


class WriteTool(AgentTool):
    name = "write"
    label = "write"
    description = "Write UTF-8 text to a file, creating parent directories and replacing existing content."
    execution_mode = "sequential"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative or absolute path"},
            "content": {"type": "string", "description": "Complete UTF-8 file content"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def __init__(self, cwd: str, *, restrict_to_cwd: bool = True) -> None:
        self.cwd = Path(cwd).resolve()
        self.restrict_to_cwd = restrict_to_cwd

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        if signal:
            signal.raise_if_aborted()
        path = resolve_path(params["path"], self.cwd, self.restrict_to_cwd)

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(params["content"], encoding="utf-8", newline="")

        # 文件写入放在线程中，避免大文件阻塞事件循环；结束后再次观察取消信号。
        await asyncio.to_thread(write)
        if signal:
            signal.raise_if_aborted()
        size = len(params["content"].encode("utf-8"))
        return ToolResult.text(
            f"已写入 {path}（{size} 字节）",
            details={"path": str(path), "bytes": size},
        )

