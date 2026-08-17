"""Tests for the BudgetGovernor."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from tiny_agent_toolkit import (
    BudgetExceeded,
    BudgetGovernor,
    configure_global_budget,
    DEFAULT_PRICING,
    get_global_budget,
)


class TestBudgetGovernor:
    def test_basic_recording(self):
        gov = BudgetGovernor(limit_usd=10.0)
        e = gov.record("gpt-4o", input_tokens=1000, output_tokens=500)
        assert e.cost_usd > 0
        assert gov.total_cost == e.cost_usd
        assert gov.call_count == 1

    def test_pricing_separate_input_output(self):
        gov = BudgetGovernor(limit_usd=10.0)
        # gpt-4o: $2.50 input / $10.00 output per 1M
        cost = gov.estimate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(12.50)

    def test_per_call_limit(self):
        gov = BudgetGovernor(limit_per_call_usd=0.001)
        with pytest.raises(BudgetExceeded) as ei:
            gov.record("gpt-4o", input_tokens=10_000, output_tokens=10_000)
        assert ei.value.limit_type == "per_call"

    def test_total_limit(self):
        gov = BudgetGovernor(limit_usd=0.01)
        gov.record("gpt-4o-mini", input_tokens=10_000, output_tokens=10_000)
        # Second call would breach total
        with pytest.raises(BudgetExceeded) as ei:
            gov.record("gpt-4o-mini", input_tokens=10_000, output_tokens=10_000)
        assert ei.value.limit_type in ("total", "per_call")

    def test_daily_limit(self):
        # 50k in + 50k out at gpt-4o-mini = 50k/1e6*(0.15+0.60) = $0.0375 per call
        # Daily limit $0.05 → first call fits ($0.0375), second breaches ($0.075)
        gov = BudgetGovernor(limit_daily_usd=0.05)
        gov.record("gpt-4o-mini", input_tokens=50_000, output_tokens=50_000)
        with pytest.raises(BudgetExceeded) as ei:
            gov.record("gpt-4o-mini", input_tokens=50_000, output_tokens=50_000)
        assert ei.value.limit_type == "daily"

    def test_daily_limit_only(self):
        # Per-call and total unset, only daily cap.
        # Two calls of 100k each = $0.15 > $0.10 daily limit
        gov = BudgetGovernor(limit_daily_usd=0.10)
        gov.record("gpt-4o-mini", input_tokens=100_000, output_tokens=100_000)
        with pytest.raises(BudgetExceeded) as ei:
            gov.record("gpt-4o-mini", input_tokens=100_000, output_tokens=100_000)
        assert ei.value.limit_type == "daily"

    def test_can_afford(self):
        gov = BudgetGovernor(limit_usd=0.0001)  # very tight
        assert gov.can_afford("gpt-4o-mini", input_tokens=1, output_tokens=1) is True
        # 100k in + 100k out at gpt-4o-mini = $0.075 > $0.0001 → unaffordable
        assert gov.can_afford("gpt-4o-mini", input_tokens=100_000, output_tokens=100_000) is False

    def test_raise_disabled_records_metadata(self):
        gov = BudgetGovernor(limit_usd=0.0001, raise_on_exceeded=False)
        e = gov.record("gpt-4o-mini", input_tokens=1000, output_tokens=1000)
        assert e.metadata.get("budget_exceeded") is True

    def test_on_warn_fires_once(self):
        # 1M tokens gpt-4o-mini = $0.15 input. With limit=$1, warn_at=0.5:
        # first call ($0.15) < 0.5 → no warn
        # second call ($0.30) < 0.5 → no warn
        # third call ($0.45) < 0.5 → no warn
        # fourth call ($0.60) >= 0.5 → fires
        # fifth call → already fired
        fires = []
        gov = BudgetGovernor(limit_usd=1.0, warn_at=0.5, on_warn=lambda s, l: fires.append((s, l)))
        for _ in range(3):
            gov.record("gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
        assert len(fires) == 0
        gov.record("gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
        assert len(fires) == 1
        gov.record("gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
        assert len(fires) == 1  # still only once

    def test_on_exceeded_callback(self):
        # per_call limit of $0.0001, cost ~$0.0025 from 1k in + 1k out at gpt-4o pricing
        calls = []
        gov = BudgetGovernor(
            limit_usd=10.0,           # big total
            limit_per_call_usd=0.0001,  # tiny per-call
            on_exceeded=lambda t, l, c: calls.append(t),
        )
        with pytest.raises(BudgetExceeded):
            gov.record("gpt-4o", input_tokens=1_000, output_tokens=1_000)
        assert calls == ["per_call"]

    def test_fuzzy_model_match(self):
        gov = BudgetGovernor(limit_usd=100.0)
        # Fine-tuned model names should still match
        cost = gov.estimate_cost("ft:gpt-4o-mini-2024-07-18", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost > 0

    def test_set_price_override(self):
        gov = BudgetGovernor(limit_usd=100.0)
        gov.set_price("custom-model", input_per_m=5.0, output_per_m=15.0)
        cost = gov.estimate_cost("custom-model", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(20.0)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "budget.jsonl"
            gov1 = BudgetGovernor(limit_usd=100.0, persist_path=path)
            gov1.record("gpt-4o-mini", input_tokens=1000, output_tokens=500)
            assert path.exists()

            # Reload into a fresh governor
            gov2 = BudgetGovernor(limit_usd=100.0, persist_path=path)
            assert gov2.total_cost == gov1.total_cost
            assert gov2.call_count == gov1.call_count

    def test_reset(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "budget.jsonl"
            gov = BudgetGovernor(limit_usd=100.0, persist_path=path)
            gov.record("gpt-4o-mini", input_tokens=1000, output_tokens=500)
            gov.reset()
            assert gov.total_cost == 0
            assert gov.call_count == 0
            assert not path.exists()

    def test_stats(self):
        gov = BudgetGovernor(limit_usd=10.0)
        gov.record("gpt-4o-mini", input_tokens=100, output_tokens=50)
        gov.record("gpt-4o", input_tokens=200, output_tokens=100)
        s = gov.stats()
        assert s["calls"] == 2
        assert s["total_input_tokens"] == 300
        assert s["total_output_tokens"] == 150
        assert "gpt-4o-mini" in s["cost_by_model"]

    def test_remaining(self):
        gov = BudgetGovernor(limit_usd=1.0, limit_per_call_usd=0.5, limit_daily_usd=5.0)
        gov.record("gpt-4o-mini", input_tokens=1000, output_tokens=1000)
        r = gov.remaining
        assert "total" in r
        assert "per_call" in r
        assert "daily" in r
        assert r["total"] < 1.0

    def test_global_configure(self):
        b = configure_global_budget(limit_usd=1.0)
        assert get_global_budget() is b
        b.record("gpt-4o-mini", input_tokens=100, output_tokens=100)
        assert get_global_budget().total_cost > 0