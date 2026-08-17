"""Tests for the IdempotencyGuard."""

import tempfile
from pathlib import Path

import pytest

from tiny_agent_toolkit import (
    FileIdempotencyStore,
    IdempotencyGuard,
    InMemoryIdempotencyStore,
)


class TestInMemoryIdempotencyStore:
    def test_basic(self):
        store = InMemoryIdempotencyStore()
        assert not store.already_done("abc123")
        store.mark_done("abc123")
        assert store.already_done("abc123")
        assert len(store) == 1

    def test_clear(self):
        store = InMemoryIdempotencyStore()
        store.mark_done("x")
        store.mark_done("y")
        store.clear()
        assert len(store) == 0


class TestFileIdempotencyStore:
    def test_basic(self):
        with tempfile.TemporaryDirectory() as td:
            store = FileIdempotencyStore(td)
            h = "deadbeef" * 4
            assert not store.already_done(h)
            store.mark_done(h)
            assert store.already_done(h)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            store1 = FileIdempotencyStore(td)
            h = IdempotencyGuard.hash_key("persistent-key")
            store1.mark_done(h)

            store2 = FileIdempotencyStore(td)
            assert store2.already_done(h)

    def test_clear(self):
        with tempfile.TemporaryDirectory() as td:
            store = FileIdempotencyStore(td)
            store.mark_done(IdempotencyGuard.hash_key("k1"))
            store.mark_done(IdempotencyGuard.hash_key("k2"))
            store.clear()
            assert len(store) == 0


class TestIdempotencyGuard:
    def test_make_key_deterministic(self):
        k1 = IdempotencyGuard.make_key("send_email", to="alice@example.com", body="hi")
        k2 = IdempotencyGuard.make_key("send_email", to="alice@example.com", body="hi")
        assert k1 == k2

    def test_make_key_kwargs_order_independent(self):
        k1 = IdempotencyGuard.make_key(a=1, b=2, c=3)
        k2 = IdempotencyGuard.make_key(c=3, b=2, a=1)
        assert k1 == k2

    def test_make_key_distinct_inputs(self):
        k1 = IdempotencyGuard.make_key(action="send", target="a")
        k2 = IdempotencyGuard.make_key(action="send", target="b")
        assert k1 != k2

    def test_hash_key_length(self):
        h = IdempotencyGuard.hash_key("any-string")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_basic_dedup_in_memory(self):
        guard = IdempotencyGuard(store=InMemoryIdempotencyStore())
        key = IdempotencyGuard.make_key("action", tool="square", x=7)
        assert not guard.already_done(key)
        guard.mark_done(key)
        assert guard.already_done(key)

    def test_namespace_isolation(self):
        store = InMemoryIdempotencyStore()
        g1 = IdempotencyGuard(store=store, namespace="agent-A")
        g2 = IdempotencyGuard(store=store, namespace="agent-B")
        key = IdempotencyGuard.make_key("shared")
        g1.mark_done(key)
        assert g1.already_done(key)
        assert not g2.already_done(key)

    def test_run_once_executes_once(self):
        guard = IdempotencyGuard(store=InMemoryIdempotencyStore())
        calls = []

        def side_effect():
            calls.append(1)

        key = IdempotencyGuard.make_key("side-effect")
        guard.run_once(key, side_effect)
        guard.run_once(key, side_effect)
        guard.run_once(key, side_effect)
        assert len(calls) == 1

    def test_empty_make_key_raises(self):
        with pytest.raises(ValueError):
            IdempotencyGuard.make_key()