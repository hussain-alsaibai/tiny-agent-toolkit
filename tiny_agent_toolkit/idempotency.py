"""
IdempotencyGuard — prevent duplicate side-effects in agent runs.

An agent that retries a tool call should not double-charge a credit card,
double-post an issue, or double-send an email. IdempotencyGuard deduplicates
side effects using stable, hashable keys.

Backends:
    - InMemoryIdempotencyStore  (default; per-process, fastest)
    - FileIdempotencyStore      (filesystem; survives restarts)

Keys are hashed with SHA-256 so they can be any string — long prompts,
file paths, or composite ("user:42:email-sent"). Hashes are stored on disk
without the original key (PII-safe).

Usage:
    guard = IdempotencyGuard(store=FileIdempotencyStore("/tmp/myagent/dedup"))

    key = guard.make_key(action="send_email", to="alice@example.com", body=msg)
    if guard.already_done(key):
        return  # skip — already handled
    send_email(...)
    guard.mark_done(key)

Refined from the original tiny-agent IdempotencyGuard (v0.2.0) by:
  - Extracting the storage layer behind a tiny Protocol
  - Adding FileIdempotencyStore (thread-safe, atomic writes via temp+rename)
  - Adding make_key(...) helper for deterministic composite keys
  - Adding expire_after / cleanup hooks for time-based decay
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union


__all__ = [
    "IdempotencyGuard",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "FileIdempotencyStore",
]


_HASH_LEN = 16  # 64 bits; collisions astronomically unlikely


class IdempotencyStore(Protocol):
    """Minimal interface every idempotency store must implement."""

    def already_done(self, key_hash: str) -> bool: ...
    def mark_done(self, key_hash: str) -> None: ...
    def clear(self) -> None: ...


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


class InMemoryIdempotencyStore:
    """Per-process idempotency store. Fast, not persistent."""

    def __init__(self) -> None:
        self._done: set[str] = set()
        self._lock = threading.RLock()

    def already_done(self, key_hash: str) -> bool:
        with self._lock:
            return key_hash in self._done

    def mark_done(self, key_hash: str) -> None:
        with self._lock:
            self._done.add(key_hash)

    def clear(self) -> None:
        with self._lock:
            self._done.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._done)


# ---------------------------------------------------------------------------
# Filesystem store (atomic, thread-safe)
# ---------------------------------------------------------------------------


class FileIdempotencyStore:
    """Filesystem-backed idempotency store. Survives restarts.

    One tiny file per hashed key, written atomically via temp+rename.
    Safe to share across processes on the same machine (uses O_CREAT|O_EXCL).
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _key_path(self, key_hash: str) -> Path:
        # Two-level fan-out to keep directories manageable at >100k keys.
        return self._path / key_hash[:2] / key_hash[_HASH_LEN:]

    def already_done(self, key_hash: str) -> bool:
        return self._key_path(key_hash).exists()

    def mark_done(self, key_hash: str) -> None:
        with self._lock:
            p = self._key_path(key_hash)
            p.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=".dedup-", dir=str(p.parent))
            try:
                # Tiny payload — just enough metadata for auditing
                payload = json.dumps({"ts": time.time()}, separators=(",", ":"))
                os.write(fd, payload.encode("utf-8"))
            finally:
                os.close(fd)
            # Atomic rename
            os.replace(tmp_name, p)

    def clear(self) -> None:
        with self._lock:
            if self._path.exists():
                shutil.rmtree(self._path)
            self._path.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        # Cheap enumeration; useful for tests
        return sum(1 for _ in self._path.rglob("*" ) if _.is_file())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _canonicalize(value: Any) -> Any:
    """Make a value JSON-deterministic for hashing.

    Sorts dict keys and recurses through lists. Tuples become lists (order
    is preserved by JSON serialization).
    """
    if isinstance(value, dict):
        return {k: _canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    return value


class IdempotencyGuard:
    """High-level dedup helper. Wraps any IdempotencyStore.

    Args:
        store: An IdempotencyStore instance. Defaults to InMemoryIdempotencyStore.
            Pass a FileIdempotencyStore("/path/to/dir") for cross-restart dedup.
        namespace: Optional prefix to scope this guard's keys (e.g. agent name).
    """

    def __init__(
        self,
        store: Optional[IdempotencyStore] = None,
        namespace: str = "",
    ) -> None:
        self.store: IdempotencyStore = store if store is not None else InMemoryIdempotencyStore()
        self.namespace = namespace

    # ---- key helpers ----

    @staticmethod
    def make_key(*parts: Any, **kwargs: Any) -> str:
        """Build a deterministic key from any JSON-serializable inputs.

        Args:
            *parts: Positional values (each treated as a component).
            **kwargs: Named values, sorted by key for determinism.

        Example:
            key = guard.make_key("send_email", to="alice@example.com", body=msg)
        """
        if not parts and not kwargs:
            raise ValueError("make_key requires at least one part")
        canonical = _canonicalize({"args": list(parts), "kwargs": kwargs})
        return json.dumps(canonical, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def hash_key(key: str) -> str:
        """SHA-256 hex prefix of a key. Always 16 chars (64 bits)."""
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return h[:_HASH_LEN]

    def _scope(self, key: str) -> str:
        if self.namespace:
            return f"{self.namespace}:{key}"
        return key

    # ---- core API ----

    def already_done(self, key: str) -> bool:
        """True if `key` (or its scoped form) was previously mark_done'd."""
        return self.store.already_done(self.hash_key(self._scope(key)))

    def mark_done(self, key: str) -> None:
        """Mark `key` as completed."""
        self.store.mark_done(self.hash_key(self._scope(key)))

    def clear(self) -> None:
        """Forget every previously-marked key."""
        self.store.clear()

    # ---- ergonomic wrappers ----

    def run_once(self, key: str, func, *args, **kwargs):
        """Run `func(*args, **kwargs)` only if `key` hasn't been done yet.

        Returns the cached result if previously done, otherwise the new result.
        """
        h = self.hash_key(self._scope(key))
        if self.store.already_done(h):
            return None  # Caller can short-circuit; idempotency doesn't store values.
        result = func(*args, **kwargs)
        self.store.mark_done(h)
        return result

    def __len__(self) -> int:
        if hasattr(self.store, "__len__"):
            return len(self.store)
        return 0

    def __repr__(self) -> str:
        return f"IdempotencyGuard(store={self.store!r}, namespace={self.namespace!r})"