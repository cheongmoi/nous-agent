"""系统提示词、AGENTS.md 与技能发现。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .types import AgentTool


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    path: Path


def load_project_context(cwd: str) -> list[tuple[Path, str]]:
    """从文件系统根到 cwd 依次加载 AGENTS.md（缺失时尝试 CLAUDE.md）。"""

    current = Path(cwd).resolve()
    directories = list(reversed([current, *current.parents]))
    loaded: list[tuple[Path, str]] = []
    for directory in directories:
        for name in ("AGENTS.md", "CLAUDE.md"):
            candidate = directory / name
            if candidate.is_file():
                loaded.append((candidate, candidate.read_text(encoding="utf-8")))
                break
    global_context = Path.home() / ".pi" / "agent" / "AGENTS.md"
    if global_context.is_file() and all(path != global_context for path, _ in loaded):
        loaded.insert(0, (global_context, global_context.read_text(encoding="utf-8")))
    return loaded


def discover_skills(cwd: str) -> list[Skill]:
    """发现 Agent Skills 目录，只读取 frontmatter 中的名称与描述。"""

    current = Path(cwd).resolve()
    roots = [Path.home() / ".pi" / "agent" / "skills", Path.home() / ".agents" / "skills"]
    for directory in reversed([current, *current.parents]):
        roots.extend([directory / ".pi" / "skills", directory / ".agents" / "skills"])

    by_name: dict[str, Skill] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/SKILL.md")):
            text = path.read_text(encoding="utf-8")
            metadata = _frontmatter(text)
            name = metadata.get("name") or path.parent.name
            description = metadata.get("description") or _first_paragraph(text)
            by_name[name] = Skill(name, description, path)
    return list(by_name.values())


def build_system_prompt(
    cwd: str,
    tools: list[AgentTool],
    *,
    custom_prompt: str | None = None,
    append_prompt: str | None = None,
    context_files: list[tuple[Path, str]] | None = None,
    skills: list[Skill] | None = None,
) -> str:
    """构造与 Pi coding-agent 结构相近的系统提示词。"""

    path = Path(cwd).resolve()
    local_custom = path / ".pi" / "SYSTEM.md"
    local_append = path / ".pi" / "APPEND_SYSTEM.md"
    if custom_prompt is None and local_custom.is_file():
        custom_prompt = local_custom.read_text(encoding="utf-8")
    if append_prompt is None and local_append.is_file():
        append_prompt = local_append.read_text(encoding="utf-8")

    if custom_prompt:
        prompt = custom_prompt.rstrip()
    else:
        tool_lines = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools) or "(none)"
        prompt = f"""You are an expert coding assistant operating inside a Python reimplementation of Pi Agent.
You help users by reading files, executing commands, editing code, and writing files.

Available tools:
{tool_lines}

Guidelines:
- Be concise in your responses.
- Show file paths clearly when working with files.
- Inspect relevant code before editing it.
- Use tools to verify non-trivial changes.
- Never claim a command or edit succeeded unless its tool result confirms it."""

    if append_prompt:
        prompt += f"\n\n{append_prompt.strip()}"

    files = context_files if context_files is not None else load_project_context(cwd)
    if files:
        prompt += "\n\n<project_context>\nProject-specific instructions and guidelines:\n"
        for file_path, content in files:
            prompt += f'\n<project_instructions path="{file_path}">\n{content}\n</project_instructions>\n'
        prompt += "</project_context>"

    found_skills = skills if skills is not None else discover_skills(cwd)
    if any(tool.name == "read" for tool in tools) and found_skills:
        prompt += "\n\n<available_skills>\nRead the matching SKILL.md before using a skill.\n"
        for skill in found_skills:
            prompt += f'- {skill.name}: {skill.description} (file: {skill.path})\n'
        prompt += "</available_skills>"

    return f"{prompt}\n\nCurrent working directory: {path.as_posix()}"


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    _, separator, rest = text.partition("\n---")
    if not separator:
        return {}
    header = text[3 : text.index("\n---")]
    result: dict[str, str] = {}
    for line in header.splitlines():
        key, colon, value = line.partition(":")
        if colon and key.strip() in {"name", "description"}:
            result[key.strip()] = value.strip().strip("\"'")
    return result


def _first_paragraph(text: str) -> str:
    body = text
    if text.startswith("---") and "\n---" in text[3:]:
        body = text.split("\n---", 1)[1]
    paragraphs = [part.strip() for part in body.split("\n\n") if part.strip() and not part.lstrip().startswith("#")]
    return " ".join(paragraphs[0].splitlines()) if paragraphs else "No description"

