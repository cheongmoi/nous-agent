"""read 工具。"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Any

from ..types import AbortSignal, AgentTool, ToolResult, ToolUpdateCallback
from .path_utils import resolve_path


class ReadTool(AgentTool):
    name = "read"
    label = "read"
    description = (
        "Read a text file or image. Text can be paged with 1-based offset and limit; "
        "large output is truncated."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative or absolute path"},
            "offset": {"type": "integer", "minimum": 1, "description": "First line, 1-based"},
            "limit": {"type": "integer", "minimum": 1, "description": "Maximum lines"},
        },
        "required": ["path"],
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
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")

        mime, _ = mimetypes.guess_type(path.name)
        if mime and mime.startswith("image/"):
            data = await asyncio.to_thread(path.read_bytes)
            if len(data) > 20 * 1024 * 1024:
                raise ValueError("图片超过 20 MiB，拒绝读取")
            return ToolResult(
                content=[{"type": "image", "data": base64.b64encode(data).decode("ascii"), "mimeType": mime}],
                details={"path": str(path), "bytes": len(data), "mimeType": mime},
            )

        raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
        lines = raw.splitlines(keepends=True)
        offset = params.get("offset", 1)
        requested_limit = params.get("limit")
        start = min(offset - 1, len(lines))
        limit = min(requested_limit or 2000, 2000)
        selected = lines[start : start + limit]
        text = "".join(selected)
        encoded = text.encode("utf-8")
        byte_truncated = len(encoded) > 50 * 1024
        if byte_truncated:
            text = encoded[: 50 * 1024].decode("utf-8", errors="ignore")

        shown_lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        has_more = start + len(selected) < len(lines) or byte_truncated
        if has_more:
            next_offset = start + max(shown_lines, 1) + 1
            text += f"\n[内容已截断，请使用 offset={next_offset} 继续读取]"
        return ToolResult(
            content=[{"type": "text", "text": text}],
            details={
                "path": str(path),
                "offset": offset,
                "lines": shown_lines,
                "totalLines": len(lines),
                "truncated": has_more,
            },
        )

