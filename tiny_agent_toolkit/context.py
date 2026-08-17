"""
Scratchpad — a persistent, append-only "notebook" for agents.

Agents forget everything between turns unless we persist what they need.
The Scratchpad is a tiny abstraction that lets an agent save notes,
recall them later, and accumulate working state across turns or sessions
without an external database.

Two layouts ship in the box:

    1. FileScratchpad  — one JSON file per scratchpad, atomic writes.
    2. DirectoryScratchpad — directory of JSONL fragments, useful when
        you want to keep notes per turn, per task, or per session.

Both support:
    - save(key, value)         — overwrite
    - append(key, value)       — add to a list under key
    - load(key, default)       — read
    - delete(key)              — remove
    - keys()                   — enumerate
    - snapshot() / restore()   — full dump / restore
    - save_text(...)           — write a multi-line note with a heading

The data model is deliberately simple — a dict of JSON-serializable
values plus an append-log of "turns" for history. No schema, no migrations.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Union


__all__ = [
    "Scratchpad",
    "FileScratchpad",
    "DirectoryScratchpad",
    "MemoryScratchpad",
]


class Scratchpad(Protocol):
    """Protocol every scratchpad backend satisfies."""

    def save(self, key: str, value: Any) -> None: ...
    def load(self, key: str, default: Any = None) -> Any: ...
    def append(self, key: str, value: Any) -> None: ...
    def delete(self, key: str) -> bool: ...
    def clear(self) -> None: ...
    def keys(self) -> List[str]: ...
    def snapshot(self) -> Dict[str, Any]: ...
    def restore(self, data: Dict[str, Any]) -> None: ...
    def save_text(self, heading: str, body: str, **meta: Any) -> None: ...
    def notes(self, since: Optional[float] = None) -> List[Dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON to `path` atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".scratch-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        # Clean up the partial file before bubbling up.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    os.replace(tmp_name, path)


def _atomic_write_line(path: Path, line: Dict[str, Any]) -> None:
    """Append one JSON line atomically (serialize + append).

    Writes a single temp file containing the previous contents followed by
    the new entry, then atomically renames it over the destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new_bytes = (json.dumps(line, default=str) + "\n").encode("utf-8")
    old_bytes = path.read_bytes() if path.exists() else b""
    fd, tmp_name = tempfile.mkstemp(prefix=".scratch-", dir=str(path.parent))
    try:
        os.write(fd, old_bytes)
        os.write(fd, new_bytes)
        os.fsync(fd)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# FileScratchpad — single JSON file per pad
# ---------------------------------------------------------------------------


class FileScratchpad:
    """Single-JSON-file scratchpad. Atomic, thread-safe.

    Args:
        path: Where to persist. Defaults to ~/.tiny-agent-toolkit/scratchpads/default.json
        auto_snapshot_every: If > 0, auto-snapshot to a timestamped file
            every N saves. Useful for "turn-N checkpoint" patterns.

    Layout::

        {
            "state": { "key": value, ... },
            "log": [ {"ts": ..., "heading": ..., "body": ..., "meta": {...}}, ... ]
        }
    """

    def __init__(
        self,
        path: Union[str, Path] = "~/.tiny-agent-toolkit/scratchpads/default.json",
        auto_snapshot_every: int = 0,
    ) -> None:
        self.path = Path(path).expanduser()
        self.auto_snapshot_every = auto_snapshot_every
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {}
        self._log: List[Dict[str, Any]] = []
        self._saves_since_snapshot = 0
        self._load()

    def _load(self) -> None:
        with self._lock:
            if self.path.exists():
                try:
                    with open(self.path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._state = dict(data.get("state", {}))
                    self._log = list(data.get("log", []))
                except (json.JSONDecodeError, OSError):
                    # Corrupt file — back it up and start fresh.
                    try:
                        self.path.rename(self.path.with_suffix(".corrupt.json"))
                    except OSError:
                        pass
                    self._state = {}
                    self._log = []

    def _persist(self) -> None:
        with self._lock:
            data = {"state": self._state, "log": self._log}
            _atomic_write_json(self.path, data)
            self._saves_since_snapshot += 1
            if (
                self.auto_snapshot_every > 0
                and self._saves_since_snapshot >= self.auto_snapshot_every
            ):
                self._snapshot_locked()
                self._saves_since_snapshot = 0

    def _snapshot_locked(self) -> None:
        snap_path = self.path.with_name(
            f"{self.path.stem}.snapshot.{int(time.time())}{self.path.suffix}"
        )
        data = {"state": self._state, "log": self._log}
        _atomic_write_json(snap_path, data)

    # ---- core API ----

    def save(self, key: str, value: Any) -> None:
        """Overwrite the value stored under `key`."""
        with self._lock:
            self._state[key] = value
            self._persist()

    def load(self, key: str, default: Any = None) -> Any:
        """Read the value stored under `key`, or `default` if absent."""
        with self._lock:
            return self._state.get(key, default)

    def append(self, key: str, value: Any) -> None:
        """Append `value` to the list under `key`. Creates the list if needed."""
        with self._lock:
            existing = self._state.get(key)
            if existing is None:
                self._state[key] = [value]
            elif isinstance(existing, list):
                existing.append(value)
            else:
                # Coerce: turn scalar into a 2-item list with the old value first
                self._state[key] = [existing, value]
            self._persist()

    def delete(self, key: str) -> bool:
        """Remove `key` entirely. Returns True if it existed."""
        with self._lock:
            if key in self._state:
                del self._state[key]
                self._persist()
                return True
            return False

    def clear(self) -> None:
        """Forget all stored state and notes."""
        with self._lock:
            self._state = {}
            self._log = []
            self._persist()

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._state.keys())

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of the entire scratchpad."""
        with self._lock:
            return {"state": json.loads(json.dumps(self._state)), "log": list(self._log)}

    def restore(self, data: Dict[str, Any]) -> None:
        """Overwrite the scratchpad from a snapshot dict."""
        with self._lock:
            self._state = dict(data.get("state", {}))
            self._log = list(data.get("log", []))
            self._persist()

    # ---- notes API ----

    def save_text(self, heading: str, body: str, **meta: Any) -> None:
        """Append a structured note (heading + body + metadata)."""
        with self._lock:
            entry = {
                "ts": time.time(),
                "heading": heading,
                "body": body,
                "meta": dict(meta),
            }
            self._log.append(entry)
            # Trim log to a sane bound (keep the most recent 1000 entries)
            if len(self._log) > 1000:
                self._log = self._log[-1000:]
            self._persist()

    def notes(self, since: Optional[float] = None) -> List[Dict[str, Any]]:
        """Return notes, optionally filtered by timestamp."""
        with self._lock:
            if since is None:
                return list(self._log)
            return [n for n in self._log if n.get("ts", 0) >= since]

    def __repr__(self) -> str:
        with self._lock:
            return f"FileScratchpad(path={str(self.path)!r}, keys={len(self._state)}, notes={len(self._log)})"


# ---------------------------------------------------------------------------
# DirectoryScratchpad — directory of JSONL fragments
# ---------------------------------------------------------------------------


class DirectoryScratchpad:
    """A scratchpad backed by a directory of small JSONL files.

    Useful when you want one file per turn, task, or session and the
    ability to ship them around or import them back later.

    Layout::

        <root>/
            state.json         # current state dict
            notes.jsonl        # one note per line, append-only
            snapshots/
                turn-001.json
                turn-002.json
    """

    def __init__(self, root: Union[str, Path] = "~/.tiny-agent-toolkit/scratchpads/default") -> None:
        self.root = Path(root).expanduser()
        self.state_path = self.root / "state.json"
        self.notes_path = self.root / "notes.jsonl"
        self.snapshots_dir = self.root / "snapshots"
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {}
        self._load_state()

    def _load_state(self) -> None:
        with self._lock:
            if self.state_path.exists():
                try:
                    with open(self.state_path, "r", encoding="utf-8") as f:
                        self._state = json.load(f)
                except (json.JSONDecodeError, OSError):
                    self._state = {}

    def _persist_state(self) -> None:
        with self._lock:
            _atomic_write_json(self.state_path, self._state)

    def _append_note(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            _atomic_write_line(self.notes_path, entry)

    # ---- core API ----

    def save(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = value
            self._persist_state()

    def load(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)

    def append(self, key: str, value: Any) -> None:
        with self._lock:
            existing = self._state.get(key)
            if existing is None:
                self._state[key] = [value]
            elif isinstance(existing, list):
                existing.append(value)
            else:
                self._state[key] = [existing, value]
            self._persist_state()

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._state:
                del self._state[key]
                self._persist_state()
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._state = {}
            self._persist_state()
            # Truncate notes.jsonl atomically
            tmp = self.notes_path.with_suffix(".tmp")
            tmp.touch()
            os.replace(tmp, self.notes_path)

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._state.keys())

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"state": json.loads(json.dumps(self._state))}

    def restore(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self._state = dict(data.get("state", {}))
            self._persist_state()

    def save_text(self, heading: str, body: str, **meta: Any) -> None:
        entry = {
            "ts": time.time(),
            "heading": heading,
            "body": body,
            "meta": dict(meta),
        }
        self._append_note(entry)

    def notes(self, since: Optional[float] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self.notes_path.exists():
            return out
        with open(self.notes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if since is None or obj.get("ts", 0) >= since:
                    out.append(obj)
        return out

    def checkpoint(self, label: str = "snapshot") -> Path:
        """Write a full snapshot file under snapshots/."""
        with self._lock:
            ts = int(time.time())
            safe_label = "".join(c for c in label if c.isalnum() or c in ("-", "_"))[:40]
            name = f"{ts}-{safe_label or 'snapshot'}.json"
            target = self.snapshots_dir / name
            _atomic_write_json(target, {"state": self._state, "notes": self.notes()})
            return target

    def __repr__(self) -> str:
        return f"DirectoryScratchpad(root={str(self.root)!r}, keys={len(self._state)})"


# ---------------------------------------------------------------------------
# MemoryScratchpad — in-process, no persistence (tests, ephemeral agents)
# ---------------------------------------------------------------------------


class MemoryScratchpad:
    """Pure in-memory scratchpad. Same API as FileScratchpad but no disk I/O."""

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}
        self._log: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def save(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = value

    def load(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)

    def append(self, key: str, value: Any) -> None:
        with self._lock:
            existing = self._state.get(key)
            if existing is None:
                self._state[key] = [value]
            elif isinstance(existing, list):
                existing.append(value)
            else:
                self._state[key] = [existing, value]

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._state.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._state.clear()
            self._log.clear()

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._state.keys())

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"state": json.loads(json.dumps(self._state)), "log": list(self._log)}

    def restore(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self._state = dict(data.get("state", {}))
            self._log = list(data.get("log", []))

    def save_text(self, heading: str, body: str, **meta: Any) -> None:
        with self._lock:
            self._log.append({
                "ts": time.time(),
                "heading": heading,
                "body": body,
                "meta": dict(meta),
            })

    def notes(self, since: Optional[float] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if since is None:
                return list(self._log)
            return [n for n in self._log if n.get("ts", 0) >= since]

    def __repr__(self) -> str:
        with self._lock:
            return f"MemoryScratchpad(keys={len(self._state)}, notes={len(self._log)})"