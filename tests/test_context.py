"""Tests for the Scratchpad implementations."""

import tempfile
from pathlib import Path

import pytest

from tiny_agent_toolkit import (
    DirectoryScratchpad,
    FileScratchpad,
    MemoryScratchpad,
)


class TestMemoryScratchpad:
    def test_basic(self):
        s = MemoryScratchpad()
        s.save("k", "v")
        assert s.load("k") == "v"
        assert "k" in s.keys()

    def test_default(self):
        s = MemoryScratchpad()
        assert s.load("missing", default="fallback") == "fallback"

    def test_append_creates_list(self):
        s = MemoryScratchpad()
        s.append("items", "a")
        s.append("items", "b")
        s.append("items", "c")
        assert s.load("items") == ["a", "b", "c"]

    def test_append_to_scalar(self):
        s = MemoryScratchpad()
        s.save("x", "first")
        s.append("x", "second")
        assert s.load("x") == ["first", "second"]

    def test_delete(self):
        s = MemoryScratchpad()
        s.save("k", "v")
        assert s.delete("k") is True
        assert s.delete("k") is False

    def test_clear(self):
        s = MemoryScratchpad()
        s.save("a", 1)
        s.save_text("h", "b")
        s.clear()
        assert s.load("a") is None
        assert s.notes() == []

    def test_snapshot_restore(self):
        s = MemoryScratchpad()
        s.save("a", 1)
        s.save("b", 2)
        snap = s.snapshot()
        s.clear()
        assert s.load("a") is None
        s.restore(snap)
        assert s.load("a") == 1
        assert s.load("b") == 2


class TestFileScratchpad:
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pad.json"
            s1 = FileScratchpad(path)
            s1.save("k", "v")
            s1.save_text("note", "hello")
            s2 = FileScratchpad(path)
            assert s2.load("k") == "v"
            assert len(s2.notes()) == 1

    def test_corrupt_file_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pad.json"
            path.write_text("not-json{")
            s = FileScratchpad(path)
            assert s.keys() == []

    def test_auto_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pad.json"
            s = FileScratchpad(path, auto_snapshot_every=2)
            s.save("a", 1)
            s.save("b", 2)
            # Two saves → should have written a snapshot
            snapshots = list(Path(td).glob("*.snapshot.*"))
            assert len(snapshots) >= 1


class TestDirectoryScratchpad:
    def test_basic(self):
        with tempfile.TemporaryDirectory() as td:
            s = DirectoryScratchpad(td)
            s.save("k", "v")
            assert s.load("k") == "v"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            DirectoryScratchpad(td).save("k", "v")
            s2 = DirectoryScratchpad(td)
            assert s2.load("k") == "v"

    def test_notes_appended(self):
        with tempfile.TemporaryDirectory() as td:
            s = DirectoryScratchpad(td)
            s.save_text("h1", "b1")
            s.save_text("h2", "b2")
            assert len(s.notes()) == 2

    def test_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            s = DirectoryScratchpad(td)
            s.save("a", 1)
            s.save_text("h", "b")
            path = s.checkpoint("test")
            assert path.exists()
            assert path.parent.name == "snapshots"

    def test_clear(self):
        with tempfile.TemporaryDirectory() as td:
            s = DirectoryScratchpad(td)
            s.save("a", 1)
            s.save_text("h", "b")
            s.clear()
            assert s.load("a") is None
            assert s.notes() == []