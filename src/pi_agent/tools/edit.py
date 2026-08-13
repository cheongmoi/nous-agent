"""edit 工具：一次应用多个互不重叠的精确替换。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..types import AbortSignal, AgentTool, ToolResult, ToolUpdateCallback
from .path_utils import resolve_path


class EditTool(AgentTool):
    name = "edit"
    label = "edit"
    description = (
        "Apply one or more exact, non-overlapping text replacements to a UTF-8 file. "
        "Every oldText must occur exactly once in the original file."
    )
    execution_mode = "sequential"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}},
                    "required": ["oldText", "newText"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["path", "edits"],
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
        edits = params["edits"]
        if not edits:
            raise ValueError("edits 至少需要一项")
        def read_exact() -> str:
            # newline="" 禁用通用换行转换，避免一次局部编辑改写整份 CRLF 文件。
            with path.open("r", encoding="utf-8", newline="") as handle:
                return handle.read()

        original = await asyncio.to_thread(read_exact)

        spans: list[tuple[int, int, str]] = []
        for index, edit in enumerate(edits):
            old = edit["oldText"]
            if not old:
                raise ValueError(f"edits[{index}].oldText 不能为空")
            count = original.count(old)
            if count != 1:
                raise ValueError(f"edits[{index}].oldText 应在原文件中恰好出现一次，实际出现 {count} 次")
            start = original.index(old)
            spans.append((start, start + len(old), edit["newText"]))

        ordered = sorted(spans)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1]:
                raise ValueError("edits 中存在重叠或嵌套的 oldText")

        # 所有位置都相对原文件计算；逆序替换避免前面的长度变化污染后续下标。
        updated = original
        for start, end, replacement in reversed(ordered):
            updated = updated[:start] + replacement + updated[end:]
        await asyncio.to_thread(path.write_text, updated, encoding="utf-8", newline="")
        if signal:
            signal.raise_if_aborted()
        return ToolResult.text(
            f"已对 {path} 应用 {len(edits)} 处修改",
            details={
                "path": str(path),
                "edits": len(edits),
                "beforeBytes": len(original.encode("utf-8")),
                "afterBytes": len(updated.encode("utf-8")),
            },
        )
