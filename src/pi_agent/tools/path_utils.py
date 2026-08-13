"""编码工具共享的路径安全逻辑。"""

from __future__ import annotations

from pathlib import Path


def resolve_path(path: str, cwd: Path, restrict_to_cwd: bool) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    resolved = candidate.resolve()
    if restrict_to_cwd:
        try:
            resolved.relative_to(cwd)
        except ValueError as error:
            raise PermissionError(f"路径超出工作目录：{resolved}") from error
    return resolved


def truncate_output(text: str, max_lines: int = 2000, max_bytes: int = 50 * 1024) -> tuple[str, bool]:
    """保留尾部输出；命令错误通常位于末尾。"""

    encoded = text.encode("utf-8", errors="replace")
    truncated = len(encoded) > max_bytes
    if truncated:
        encoded = encoded[-max_bytes:]
        text = encoded.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > max_lines:
        truncated = True
        lines = lines[-max_lines:]
        text = "\n".join(lines)
    if truncated:
        text = f"[输出已截断，仅显示末尾]\n{text}"
    return text, truncated

