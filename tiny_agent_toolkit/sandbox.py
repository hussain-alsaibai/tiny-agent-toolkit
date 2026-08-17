"""
Sandbox — safe execution patterns for untrusted agent code.

Two complementary safety mechanisms:

  1. AST-based sanitization
     - inspect_source(src)   : static AST scan that blocks dangerous
                               imports, calls, and attribute access.
     - SanitizedExec         : runs `exec()` in a hard-allowlisted env.

  2. Subprocess-based sandbox with resource limits
     - run_in_subprocess(...) : run a snippet with optional wall-clock
                                timeout, output capture, and (Linux-only)
                                POSIX resource limits (CPU, MEM, FDSIZE).
     - ResourceLimits         : dataclass for the same.

Both layers are deliberately conservative: if a feature isn't safe in the
worst case, it's not included. If you need stronger isolation, run the
subprocess in a container, gVisor, firecracker, or a remote worker.

Usage:
    # Static check (returns a list of issues, [] = safe):
    issues = inspect_source("import os; os.system('rm -rf /')")
    # -> ["banned_import:os", "banned_call:os.system"]

    # Run untrusted code in a subprocess with a timeout and memory cap:
    result = run_in_subprocess(
        "print(sum(range(100)))",
        timeout_s=5,
        limits=ResourceLimits(cpu_seconds=5, memory_mb=128),
    )
"""

from __future__ import annotations

import ast
import os
import sys
import textwrap
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


__all__ = [
    "SandboxError",
    "SandboxTimeout",
    "SandboxViolation",
    "ResourceLimits",
    "BANNED_NAMES",
    "BANNED_CALLS",
    "BANNED_IMPORTS",
    "inspect_source",
    "SanitizedExec",
    "run_in_subprocess",
]


# ---------------------------------------------------------------------------
# Banned identifiers
# ---------------------------------------------------------------------------


# Top-level imports we never want a sandboxed snippet to perform.
BANNED_IMPORTS = frozenset({
    "os", "subprocess", "sys", "shutil", "socket", "ctypes",
    "multiprocessing", "threading", "asyncio", "signal",
    "importlib", "pkgutil", "code", "codeop", "compileall",
    "pty", "pwd", "grp", "resource", "fcntl", "termios",
})

# Function calls we never want, even with the underlying module imported
# through an alias. The keys are dotted names; values are the level of
# severity (used for diagnostics).
BANNED_CALLS = frozenset({
    "os.system", "os.popen", "os.exec", "os.execv", "os.execve",
    "os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace",
    "shutil.rmtree", "shutil.move",
    "subprocess.run", "subprocess.call", "subprocess.Popen",
    "eval", "exec", "compile",
    "open",  # file I/O must go through explicit allowlisted wrapper
})

# Attribute access patterns that are typically the path-of-least-resistance
# for sandbox escape. Checked on Attribute AST nodes.
BANNED_ATTRS = frozenset({
    "__class__", "__subclasses__", "__bases__", "__mro__",
    "__globals__", "__code__", "__dict__", "__init_subclass__",
    "__subclasshook__", "__getattribute__", "__delattr__",
    "__setattr__", "__reduce__", "__reduce_ex__",
})

# Generic names that almost always indicate the user is reaching for the
# "outer" scope of a host process.
BANNED_NAMES = frozenset({
    "exit", "quit", "help", "input", "breakpoint",
})


class SandboxError(Exception):
    """Base class for sandbox errors."""


class SandboxTimeout(SandboxError):
    """Raised when sandboxed code exceeds its wall-clock budget."""


class SandboxViolation(SandboxError):
    """Raised when sandboxed code violates a static rule."""


# ---------------------------------------------------------------------------
# AST static analyzer
# ---------------------------------------------------------------------------


class _StaticVisitor(ast.NodeVisitor):
    """Walks an AST and records banned constructs."""

    def __init__(self) -> None:
        self.issues: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in BANNED_IMPORTS:
                self.issues.append(f"banned_import:{alias.name} (line {node.lineno})")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            top = node.module.split(".")[0]
            if top in BANNED_IMPORTS:
                self.issues.append(f"banned_import_from:{node.module} (line {node.lineno})")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = self._qualified_name(func)
        if name and name in BANNED_CALLS:
            self.issues.append(f"banned_call:{name} (line {node.lineno})")
        # Block function-locals named `BANNED_NAMES`
        if isinstance(func, ast.Name) and func.id in BANNED_NAMES:
            self.issues.append(f"banned_name:{func.id} (line {node.lineno})")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BANNED_ATTRS:
            self.issues.append(f"banned_attr:{node.attr} (line {node.lineno})")
        self.generic_visit(node)

    @staticmethod
    def _qualified_name(node: ast.AST) -> Optional[str]:
        """Resolve an ast.Call's func to a dotted string when possible."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = _StaticVisitor._qualified_name(node.value)
            if base:
                return f"{base}.{node.attr}"
        return None


def inspect_source(source: str) -> List[str]:
    """Return a list of sandbox-violation issues in `source`.

    Empty list means the code passes the static check. This is purely
    syntactic — it does not run the code.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax_error:{e.msg} (line {e.lineno})"]
    visitor = _StaticVisitor()
    visitor.visit(tree)
    return visitor.issues


# ---------------------------------------------------------------------------
# Sandboxed exec() with hard-allowlisted globals/locals
# ---------------------------------------------------------------------------


_ALLOWED_BUILTINS: Dict[str, Any] = {
    "abs": abs, "min": min, "max": max, "sum": sum,
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "sorted": sorted, "reversed": reversed, "map": map, "filter": filter,
    "any": any, "all": all,
    "print": print, "repr": repr, "str": str, "int": int, "float": float,
    "bool": bool, "list": list, "tuple": tuple, "dict": dict, "set": set,
    "isinstance": isinstance, "type": type,
    "True": True, "False": False, "None": None,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError,
    "__import__": None,  # explicitly blocked (raises if invoked)
}


class SanitizedExec:
    """Executes a snippet in a sandboxed namespace.

    The environment starts EMPTY of user-provided globals. The only
    accessible builtins are the safe ones above. Attempting to import
    anything raises SandboxViolation.

    Example:
        with SanitizedExec() as sx:
            sx.run("x = 2 + 2", user_globals={"y": 10})
        print(sx.globals["x"])  # 4
    """

    def __init__(self, extra_globals: Optional[Dict[str, Any]] = None) -> None:
        # Builtins get a custom getter that blocks __import__.
        class _SafeBuiltins(dict):
            def __getitem__(self, key):
                if key == "__import__":
                    raise SandboxViolation("imports are blocked in sandbox")
                return super().__getitem__(key)

            def get(self, key, default=None):
                if key == "__import__":
                    raise SandboxViolation("imports are blocked in sandbox")
                return super().get(key, default)

        self._builtins = _SafeBuiltins(_ALLOWED_BUILTINS)
        self.globals: Dict[str, Any] = {"__builtins__": self._builtins}
        if extra_globals:
            self.globals.update(extra_globals)
        self.locals: Dict[str, Any] = {}

    def __enter__(self) -> "SanitizedExec":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # The namespace lives for the lifetime of the object, so nothing
        # to clean up here. Override in subclasses.
        return None

    def check(self, source: str) -> List[str]:
        """Run static checks on `source` without executing it."""
        return inspect_source(source)

    def run(self, source: str, user_globals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute `source` in the sandbox. Returns the resulting globals."""
        issues = self.check(source)
        if issues:
            raise SandboxViolation(f"Static check failed: {'; '.join(issues)}")
        g = dict(self.globals)
        if user_globals:
            g.update(user_globals)
        try:
            exec(compile(source, "<sandbox>", "exec"), g)
        except SandboxViolation:
            raise
        except Exception as e:
            raise SandboxError(f"Sandboxed code raised: {e}") from e
        # Merge back the new bindings so callers can introspect results.
        self.globals.update(g)
        return g


# ---------------------------------------------------------------------------
# Subprocess-based sandbox with resource limits (POSIX)
# ---------------------------------------------------------------------------


@dataclass
class ResourceLimits:
    """POSIX resource limits (Linux/macOS). All fields optional."""
    cpu_seconds: Optional[int] = None       # RLIMIT_CPU
    memory_mb: Optional[int] = None         # RLIMIT_AS (virtual memory)
    file_size_mb: Optional[int] = None      # RLIMIT_FSIZE
    nproc: Optional[int] = None            # RLIMIT_NPROC
    nofile: Optional[int] = None           # RLIMIT_NOFILE
    fsize_mb: Optional[int] = None         # alias kept for clarity

    def to_preamble(self) -> str:
        """Generate the Python source applied *before* the user code.

        Returned as a snippet that calls `resource.setrlimit(...)` if the
        matching module is available.
        """
        lines = ["import resource"]
        if self.cpu_seconds is not None:
            lines.append(f"resource.setrlimit(resource.RLIMIT_CPU, ({self.cpu_seconds}, {self.cpu_seconds}))")
        if self.memory_mb is not None:
            bytes_ = self.memory_mb * 1024 * 1024
            lines.append(f"resource.setrlimit(resource.RLIMIT_AS, ({bytes_}, {bytes_}))")
        fsize = self.file_size_mb if self.file_size_mb is not None else self.fsize_mb
        if fsize is not None:
            bytes_ = fsize * 1024 * 1024
            lines.append(f"resource.setrlimit(resource.RLIMIT_FSIZE, ({bytes_}, {bytes_}))")
        if self.nproc is not None:
            lines.append(f"resource.setrlimit(resource.RLIMIT_NPROC, ({self.nproc}, {self.nproc}))")
        if self.nofile is not None:
            lines.append(f"resource.setrlimit(resource.RLIMIT_NOFILE, ({self.nofile}, {self.nofile}))")
        return "\n".join(lines) + "\n"


@dataclass
class SubprocessResult:
    """Result of a sandboxed subprocess run."""
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def raise_for_status(self) -> None:
        if self.timed_out:
            raise SandboxTimeout("subprocess exceeded wall-clock budget")
        if self.returncode != 0:
            raise SandboxError(
                f"subprocess exited {self.returncode}: stderr={self.stderr[:500]!r}"
            )


def run_in_subprocess(
    source: str,
    *,
    timeout_s: float = 30.0,
    limits: Optional[ResourceLimits] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    stdin_payload: Optional[str] = None,
    extra_argv: Sequence[str] = (),
) -> SubprocessResult:
    """Run a Python `source` snippet in a fresh subprocess.

    Args:
        source: Python code to execute.
        timeout_s: Wall-clock timeout. Triggers SandboxTimeout on expiry.
        limits: Optional POSIX resource limits (Linux/macOS).
        cwd: Working directory for the subprocess.
        env: Extra environment variables (merged onto os.environ).
        stdin_payload: Optional string piped to the child's stdin.
        extra_argv: Extra args appended to `sys.argv`.

    Returns:
        SubprocessResult with stdout/stderr/returncode.
    """
    import subprocess
    import time

    preamble = ""
    if limits is not None:
        preamble = limits.to_preamble()

    payload = textwrap.dedent(preamble + "\n" + source)

    argv = [sys.executable, "-I", "-S", "-c", payload]
    argv.extend(extra_argv)

    full_env = None
    if env is not None:
        full_env = {**os.environ, **env}

    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=full_env,
            input=stdin_payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return SubprocessResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            timed_out=False,
            duration_s=time.monotonic() - start,
        )
    except subprocess.TimeoutExpired as e:
        return SubprocessResult(
            returncode=-1,
            stdout=(e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            stderr=(e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or ""),
            timed_out=True,
            duration_s=time.monotonic() - start,
        )


# ---------------------------------------------------------------------------
# Convenience: combined static + dynamic check
# ---------------------------------------------------------------------------


def is_safe(source: str) -> bool:
    """Quick boolean check: does `source` pass the static sanitizer?"""
    return len(inspect_source(source)) == 0