"""
BudgetGovernor — runtime cost + token enforcement for AI agents.

Prevents runaway LLM spend with:
  - Per-model pricing (separate input/output rates)
  - Per-call, total, and daily limits
  - Persisted state across runs
  - Optional warn callback before a hard cap
  - Circuit breaking on repeated overspend

Usage:
    gov = BudgetGovernor(limit_usd=0.50, warn_at=0.7)
    gov.check("gpt-4o", input_tokens=200, output_tokens=80)
    # raises BudgetExceeded if limit crossed

Refined from the original tiny-agent BudgetGovernor (single-file PR) by
folding in the per-call/per-day/per-total split from tiny-cost.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union


__all__ = [
    "BudgetGovernor",
    "BudgetExceeded",
    "BudgetEntry",
    "DEFAULT_PRICING",
    "configure_global",
    "get_global_budget",
]


# Default pricing table — USD per 1M tokens
# (input_cost_per_million, output_cost_per_million)
DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o":           {"input":  2.50, "output": 10.00},
    "gpt-4o-mini":      {"input":  0.15, "output":  0.60},
    "gpt-4-turbo":      {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo":    {"input":  0.50, "output":  1.50},
    "o1-preview":       {"input": 15.00, "output": 60.00},
    "o1-mini":          {"input":  3.00, "output": 12.00},
    # Anthropic
    "claude-3-5-sonnet":   {"input":  3.00, "output": 15.00},
    "claude-3-5-haiku":    {"input":  0.80, "output":  4.00},
    "claude-3-opus":       {"input": 15.00, "output": 75.00},
    "claude-3-haiku":      {"input":  0.25, "output":  1.25},
    # Google
    "gemini-2.5-pro":      {"input":  1.25, "output":  5.00},
    "gemini-2.5-flash":    {"input":  0.075, "output": 0.30},
    "gemini-1.5-pro":      {"input":  1.25, "output":  5.00},
    "gemini-1.5-flash":    {"input":  0.075, "output": 0.30},
    # xAI
    "grok-2":              {"input":  2.00, "output": 10.00},
    # Local — free (set explicitly to opt-in)
    "llama3":              {"input":  0.00, "output":  0.00},
    "default":             {"input":  2.00, "output":  6.00},
}


class BudgetExceeded(Exception):
    """Raised when the agent's runtime budget is exceeded."""

    def __init__(
        self,
        message: str,
        limit_type: str = "total",
        limit_value: float = 0.0,
        current_value: float = 0.0,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.current_value = current_value
        self.model = model


@dataclass
class BudgetEntry:
    """A single recorded spend event."""
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BudgetGovernor:
    """Runtime cost + token enforcement with persistence.

    Args:
        pricing: Per-model pricing dict (input/output cost per 1M tokens).
            Defaults to DEFAULT_PRICING.
        limit_usd: Hard cap on total spend in USD (across the lifetime of
            this governor). None disables.
        limit_per_call_usd: Max USD allowed on a single LLM call.
        limit_daily_usd: Max USD allowed per UTC day.
        warn_at: Ratio (0.0–1.0) at which the on_warn callback fires.
        on_warn: Callable(spent, limit) invoked once when spend crosses warn_at.
        on_exceeded: Callable(limit_type, limit_val, current_val) invoked on
            every breach before raising.
        raise_on_exceeded: If True, raise BudgetExceeded. If False, record a
            metadata flag and continue (useful for soft caps in dev).
        persist_path: If set, write ledger to this JSONL file. Loaded on init.
    """

    def __init__(
        self,
        pricing: Optional[Dict[str, Dict[str, float]]] = None,
        limit_usd: Optional[float] = 1.0,
        limit_per_call_usd: Optional[float] = None,
        limit_daily_usd: Optional[float] = None,
        warn_at: float = 0.7,
        on_warn: Optional[Callable[[float, float], None]] = None,
        on_exceeded: Optional[Callable[[str, float, float], None]] = None,
        raise_on_exceeded: bool = True,
        persist_path: Optional[Union[str, Path]] = None,
    ) -> None:
        # Pricing: merge user overrides onto defaults
        self._pricing: Dict[str, Dict[str, float]] = {
            k: dict(v) for k, v in DEFAULT_PRICING.items()
        }
        if pricing:
            for k, v in pricing.items():
                if k in self._pricing:
                    self._pricing[k].update(v)
                else:
                    self._pricing[k] = dict(v)

        self.limit_usd = limit_usd
        self.limit_per_call_usd = limit_per_call_usd
        self.limit_daily_usd = limit_daily_usd
        self.warn_at = warn_at
        self.on_warn = on_warn
        self.on_exceeded = on_exceeded
        self.raise_on_exceeded = raise_on_exceeded
        self.persist_path = Path(persist_path) if persist_path else None

        self._lock = threading.RLock()
        self._entries: list[BudgetEntry] = []
        self._total = 0.0
        self._daily_total = 0.0
        self._daily_start = self._day_start()
        self._warned = False

        if self.persist_path:
            self._load_persisted()

    # ------------------------------------------------------------------
    # Pricing helpers
    # ------------------------------------------------------------------

    def set_price(self, model: str, input_per_m: float, output_per_m: float) -> None:
        """Override pricing for a specific model."""
        with self._lock:
            self._pricing[model] = {
                "input": float(input_per_m),
                "output": float(output_per_m),
            }

    def price_for(self, model: str) -> Dict[str, float]:
        """Return pricing dict for a model, with fuzzy match fallback."""
        if model in self._pricing:
            return self._pricing[model]
        model_lower = (model or "").lower()
        for k, v in self._pricing.items():
            if k != "default" and k in model_lower:
                return v
        return self._pricing["default"]

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost for a planned call."""
        p = self.price_for(model)
        return (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]

    def can_afford(self, model: str, input_tokens: int, output_tokens: int) -> bool:
        """Return True if this call would not breach any configured cap."""
        cost = self.estimate_cost(model, input_tokens, output_tokens)
        with self._lock:
            self._maybe_reset_daily()
            if self.limit_per_call_usd is not None and cost > self.limit_per_call_usd:
                return False
            if self.limit_usd is not None and (self._total + cost) > self.limit_usd:
                return False
            if self.limit_daily_usd is not None and (self._daily_total + cost) > self.limit_daily_usd:
                return False
            return True

    # ------------------------------------------------------------------
    # Recording + enforcement
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BudgetEntry:
        """Record a spend event. Raises BudgetExceeded if it breaches a cap."""
        cost = self.estimate_cost(model, input_tokens, output_tokens)
        metadata = dict(metadata or {})
        entry = BudgetEntry(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            request_id=request_id,
            metadata=metadata,
        )

        with self._lock:
            self._maybe_reset_daily()
            breach = self._find_breach(cost, model)

            if breach is not None:
                limit_type, limit_val, current = breach
                if self.on_exceeded:
                    self.on_exceeded(limit_type, limit_val, current)
                if self.raise_on_exceeded:
                    raise BudgetExceeded(
                        f"Budget exceeded: {limit_type} ${current:.4f} > ${limit_val:.4f}",
                        limit_type=limit_type,
                        limit_value=limit_val,
                        current_value=current,
                        model=model,
                    )
                metadata["budget_exceeded"] = True

            self._entries.append(entry)
            self._total += cost
            self._daily_total += cost

            # Warn (once) when crossing warn_at ratio on the total cap
            if (
                not self._warned
                and self.limit_usd
                and self.on_warn is not None
                and self._total / self.limit_usd >= self.warn_at
            ):
                self._warned = True
                self.on_warn(self._total, self.limit_usd)

            if self.persist_path:
                self._append_persisted(entry)

        return entry

    def check(self, model: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Convenience: record and silently raise on breach.

        Equivalent to record() but discards the return value.
        """
        self.record(model, input_tokens, output_tokens)

    def _find_breach(self, cost: float, model: str):
        """Return (limit_type, limit_val, current_val) for the first breach, else None.

        Order of checks: per-call → total → daily. The first one that fails
        wins, so callers can attribute the failure correctly.
        """
        if self.limit_per_call_usd is not None and cost > self.limit_per_call_usd:
            return ("per_call", self.limit_per_call_usd, cost)
        if self.limit_usd is not None and (self._total + cost) > self.limit_usd:
            return ("total", self.limit_usd, self._total + cost)
        if self.limit_daily_usd is not None and (self._daily_total + cost) > self.limit_daily_usd:
            return ("daily", self.limit_daily_usd, self._daily_total + cost)
        return None

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def spent(self) -> float:
        """Alias for total_cost for backwards compatibility."""
        return self.total_cost

    @property
    def total_cost(self) -> float:
        with self._lock:
            return self._total

    @property
    def daily_cost(self) -> float:
        with self._lock:
            self._maybe_reset_daily()
            return self._daily_total

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def remaining(self) -> Dict[str, float]:
        """Remaining budget per configured limit (USD)."""
        with self._lock:
            self._maybe_reset_daily()
            out: Dict[str, float] = {}
            if self.limit_usd is not None:
                out["total"] = round(max(0.0, self.limit_usd - self._total), 6)
            if self.limit_per_call_usd is not None:
                out["per_call"] = float(self.limit_per_call_usd)
            if self.limit_daily_usd is not None:
                out["daily"] = round(max(0.0, self.limit_daily_usd - self._daily_total), 6)
            return out

    def cost_by_model(self) -> Dict[str, float]:
        """Total USD cost aggregated by model name."""
        out: Dict[str, float] = {}
        with self._lock:
            for e in self._entries:
                out[e.model] = out.get(e.model, 0.0) + e.cost_usd
        return out

    def stats(self) -> Dict[str, Any]:
        """Return summary stats for the lifetime of this governor."""
        with self._lock:
            total_in = sum(e.input_tokens for e in self._entries)
            total_out = sum(e.output_tokens for e in self._entries)
            return {
                "calls": len(self._entries),
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "total_cost_usd": round(self._total, 6),
                "daily_cost_usd": round(self.daily_cost, 6),
                "remaining": self.remaining,
                "cost_by_model": {k: round(v, 6) for k, v in self.cost_by_model().items()},
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _append_persisted(self, entry: BudgetEntry) -> None:
        assert self.persist_path is not None
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.__dict__, default=str) + "\n")

    def _load_persisted(self) -> None:
        assert self.persist_path is not None
        if not self.persist_path.exists():
            return
        with open(self.persist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    entry = BudgetEntry(**obj)
                except Exception:
                    continue
                self._entries.append(entry)
                self._total += entry.cost_usd
                # Recompute daily bucket
                if entry.timestamp >= self._daily_start:
                    self._daily_total += entry.cost_usd
                else:
                    # Entry is older than current day; drop it from daily total.
                    pass

    def reset(self) -> None:
        """Clear all in-memory state and persisted ledger."""
        with self._lock:
            self._entries.clear()
            self._total = 0.0
            self._daily_total = 0.0
            self._daily_start = self._day_start()
            self._warned = False
        if self.persist_path and self.persist_path.exists():
            try:
                os.remove(self.persist_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Daily window helpers
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self) -> None:
        now = self._day_start()
        with self._lock:
            if now > self._daily_start:
                self._daily_start = now
                self._daily_total = 0.0

    @staticmethod
    def _day_start() -> float:
        import datetime
        today = datetime.datetime.now(datetime.timezone.utc).date()
        return datetime.datetime.combine(
            today,
            datetime.datetime.min.time(),
            tzinfo=datetime.timezone.utc,
        ).timestamp()

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"BudgetGovernor(spent=${self._total:.4f}, "
                f"limit=${self.limit_usd}, calls={len(self._entries)})"
            )


# ---------------------------------------------------------------------------
# Process-global convenience (matches tiny-cost's API)
# ---------------------------------------------------------------------------

_global: Optional[BudgetGovernor] = None
_global_lock = threading.Lock()


def configure_global(
    limit_usd: Optional[float] = None,
    limit_per_call_usd: Optional[float] = None,
    limit_daily_usd: Optional[float] = None,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
    raise_on_exceeded: bool = True,
    on_exceeded: Optional[Callable[[str, float, float], None]] = None,
    persist_path: Optional[Union[str, Path]] = None,
) -> BudgetGovernor:
    """Configure the process-global BudgetGovernor and return it."""
    global _global
    with _global_lock:
        _global = BudgetGovernor(
            pricing=pricing,
            limit_usd=limit_usd,
            limit_per_call_usd=limit_per_call_usd,
            limit_daily_usd=limit_daily_usd,
            raise_on_exceeded=raise_on_exceeded,
            on_exceeded=on_exceeded,
            persist_path=persist_path,
        )
        return _global


def get_global_budget() -> Optional[BudgetGovernor]:
    return _global