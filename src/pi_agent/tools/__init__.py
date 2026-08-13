"""内置编码工具。"""

from .bash import BashTool
from .edit import EditTool
from .read import ReadTool
from .write import WriteTool

__all__ = ["BashTool", "EditTool", "ReadTool", "WriteTool", "coding_tools"]


def coding_tools(cwd: str, *, restrict_to_cwd: bool = True):
    """按 Pi 默认顺序创建 read、bash、edit、write。"""

    return [
        ReadTool(cwd, restrict_to_cwd=restrict_to_cwd),
        BashTool(cwd),
        EditTool(cwd, restrict_to_cwd=restrict_to_cwd),
        WriteTool(cwd, restrict_to_cwd=restrict_to_cwd),
    ]

