"""追加式 JSONL 会话树。"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from .types import AgentMessage


class JsonlSession:
    """每条消息保存 parentId，因此一个文件可容纳多条分支。"""

    VERSION = 1

    def __init__(self, path: Path, cwd: str, *, create: bool = False) -> None:
        self.path = path.resolve()
        self.cwd = str(Path(cwd).resolve())
        self.session_id = uuid.uuid4().hex
        self.leaf_id: str | None = None
        self._records: dict[str, dict[str, Any]] = {}
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            header = {
                "type": "session",
                "version": self.VERSION,
                "id": self.session_id,
                "cwd": self.cwd,
                "createdAt": time.time(),
            }
            self.path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"会话文件不存在：{self.path}")
        last_id: str | None = None
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"会话文件第 {number} 行不是合法 JSON") from error
                if record.get("type") == "session":
                    self.session_id = record["id"]
                    self.cwd = record.get("cwd", self.cwd)
                elif record.get("type") == "message":
                    self._records[record["id"]] = record
                    last_id = record["id"]
        self.leaf_id = last_id

    def append(self, message: AgentMessage) -> str:
        record_id = uuid.uuid4().hex
        record = {
            "type": "message",
            "id": record_id,
            "parentId": self.leaf_id,
            "timestamp": time.time(),
            "message": message,
        }
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._records[record_id] = record
        self.leaf_id = record_id
        return record_id

    def messages(self, leaf_id: str | None = None) -> list[AgentMessage]:
        current = leaf_id if leaf_id is not None else self.leaf_id
        branch: list[AgentMessage] = []
        seen: set[str] = set()
        while current:
            if current in seen:
                raise ValueError("会话 parentId 形成循环")
            seen.add(current)
            record = self._records.get(current)
            if record is None:
                raise ValueError(f"会话分支引用了不存在的消息：{current}")
            branch.append(record["message"])
            current = record.get("parentId")
        branch.reverse()
        return branch

    def branch_from(self, message_id: str | None) -> None:
        if message_id is not None and message_id not in self._records:
            raise KeyError(f"未知消息 ID：{message_id}")
        self.leaf_id = message_id


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or Path.home() / ".pi-agent-python" / "sessions").resolve()

    def project_dir(self, cwd: str) -> Path:
        resolved = str(Path(cwd).resolve())
        slug = Path(resolved).name or "root"
        digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:10]
        return self.root / f"{slug}-{digest}"

    def create(self, cwd: str) -> JsonlSession:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.project_dir(cwd) / f"{stamp}-{uuid.uuid4().hex[:8]}.jsonl"
        return JsonlSession(path, cwd, create=True)

    def latest(self, cwd: str) -> JsonlSession:
        files = sorted(self.project_dir(cwd).glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not files:
            raise FileNotFoundError("当前项目没有历史会话")
        return JsonlSession(files[0], cwd)

