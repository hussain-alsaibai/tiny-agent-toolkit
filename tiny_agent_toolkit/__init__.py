"""
tiny_agent_toolkit — zero-dependency Python toolkit for reliable AI agents.

Modules:
    budget       — BudgetGovernor (per-model pricing, persistence, caps)
    idempotency  — IdempotencyGuard (file + memory dedup)
    context      — Scratchpad backends (file, directory, memory)
    tools        — Tool registry with OpenAI / Anthropic / MCP schema export
    sandbox      — Static AST sanitizer + subprocess runner with rlimits

All components are importable as `from tiny_agent_toolkit import X` and are
designed to compose. See examples/reliable_agent.py for a working agent
loop that uses every component.

MIT License — github.com/hussain-alsaibai/tiny-agent-toolkit
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "__version__",
    # budget
    "BudgetGovernor",
    "BudgetExceeded",
    "BudgetEntry",
    "DEFAULT_PRICING",
    "configure_global_budget",
    "get_global_budget",
    # idempotency
    "IdempotencyGuard",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "FileIdempotencyStore",
    # context
    "Scratchpad",
    "FileScratchpad",
    "DirectoryScratchpad",
    "MemoryScratchpad",
    # tools
    "Tool",
    "ToolRegistry",
    "ToolError",
    "tool",
    # sandbox
    "SandboxError",
    "SandboxTimeout",
    "SandboxViolation",
    "ResourceLimits",
    "SubprocessResult",
    "BANNED_NAMES",
    "BANNED_CALLS",
    "BANNED_IMPORTS",
    "BANNED_ATTRS",
    "inspect_source",
    "SanitizedExec",
    "run_in_subprocess",
    "is_safe",
]


# --- budget ---
from .budget import (
    BudgetGovernor,
    BudgetExceeded,
    BudgetEntry,
    DEFAULT_PRICING,
    configure_global as configure_global_budget,
    get_global_budget,
)

# --- idempotency ---
from .idempotency import (
    IdempotencyGuard,
    IdempotencyStore,
    InMemoryIdempotencyStore,
    FileIdempotencyStore,
)

# --- context ---
from .context import (
    Scratchpad,
    FileScratchpad,
    DirectoryScratchpad,
    MemoryScratchpad,
)

# --- tools ---
from .tools import (
    Tool,
    ToolRegistry,
    ToolError,
    tool,
)

# --- sandbox ---
from .sandbox import (
    SandboxError,
    SandboxTimeout,
    SandboxViolation,
    ResourceLimits,
    SubprocessResult,
    BANNED_NAMES,
    BANNED_CALLS,
    BANNED_IMPORTS,
    BANNED_ATTRS,
    inspect_source,
    SanitizedExec,
    run_in_subprocess,
    is_safe,
)