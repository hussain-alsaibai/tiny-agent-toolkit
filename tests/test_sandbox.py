"""Tests for the Sandbox static analyzer and subprocess runner."""

import pytest

from tiny_agent_toolkit import (
    ResourceLimits,
    SandboxError,
    SandboxTimeout,
    SandboxViolation,
    SanitizedExec,
    inspect_source,
    is_safe,
    run_in_subprocess,
)


class TestInspectSource:
    def test_banned_import(self):
        issues = inspect_source("import os")
        assert any("os" in i for i in issues)

    def test_banned_import_from(self):
        issues = inspect_source("from os import path")
        assert any("os" in i for i in issues)

    def test_banned_call(self):
        issues = inspect_source("os.system('rm -rf /')")
        # Even if `os` were importable, the call is banned
        assert any("system" in i.lower() for i in issues)

    def test_banned_attr(self):
        issues = inspect_source("().__class__.__mro__")
        assert any("__class__" in i or "__mro__" in i for i in issues)

    def test_safe_code(self):
        assert inspect_source("x = 2 + 2") == []
        assert inspect_source("def f(x):\n    return x * 2\n") == []
        assert inspect_source("[x*2 for x in range(10)]") == []

    def test_syntax_error_reported(self):
        issues = inspect_source("def f(:\n    pass")
        assert any("syntax" in i.lower() for i in issues)

    def test_is_safe_helper(self):
        assert is_safe("x = 1") is True
        assert is_safe("import os") is False


class TestSanitizedExec:
    def test_basic_execution(self):
        with SanitizedExec() as sx:
            g = sx.run("x = 2 + 2")
            assert g["x"] == 4

    def test_blocks_import(self):
        with SanitizedExec() as sx:
            with pytest.raises(SandboxViolation):
                sx.run("import os")

    def test_blocks_open(self):
        with SanitizedExec() as sx:
            with pytest.raises(SandboxViolation):
                sx.run("open('/etc/passwd')")

    def test_blocks_eval(self):
        with SanitizedExec() as sx:
            with pytest.raises(SandboxViolation):
                sx.run("eval('1+1')")

    def test_safe_builtins_work(self):
        with SanitizedExec() as sx:
            g = sx.run("x = sum(range(5))\ny = len('hello')")
            assert g["x"] == 10
            assert g["y"] == 5

    def test_extra_globals_visible(self):
        with SanitizedExec(extra_globals={"pi": 3.14159}) as sx:
            g = sx.run("x = pi * 2")
            assert g["x"] == pytest.approx(6.28318)

    def test_user_globals_override(self):
        with SanitizedExec() as sx:
            g = sx.run("y = my_value", user_globals={"my_value": 42})
            assert g["y"] == 42


class TestRunInSubprocess:
    def test_basic_print(self):
        r = run_in_subprocess("print('hello')\n", timeout_s=5)
        assert r.ok
        assert "hello" in r.stdout

    def test_exit_code(self):
        r = run_in_subprocess("import sys\nsys.exit(7)\n", timeout_s=5)
        assert r.returncode == 7
        assert not r.ok
        with pytest.raises(SandboxError):
            r.raise_for_status()

    def test_timeout(self):
        # Sleep longer than the timeout
        r = run_in_subprocess("import time\ntime.sleep(10)\n", timeout_s=0.5)
        assert r.timed_out
        with pytest.raises(SandboxTimeout):
            r.raise_for_status()

    def test_limits_applied(self):
        # ResourceLimits should be applied via the preamble. We don't assert
        # the OS-level enforcement here, but ensure no crash.
        r = run_in_subprocess("print('ok')\n", timeout_s=2, limits=ResourceLimits(memory_mb=64))
        assert r.ok

    def test_cwd_argument(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            r = run_in_subprocess(
                "import os\nprint(os.getcwd())\n",
                timeout_s=2,
                cwd=td,
            )
            assert r.ok
            assert r.stdout.strip() == os.path.realpath(td)

    def test_extra_env(self):
        r = run_in_subprocess(
            "import os\nprint(os.environ.get('TINY_AGENT_TEST', 'missing'))\n",
            timeout_s=2,
            env={"TINY_AGENT_TEST": "present"},
        )
        assert r.ok
        assert "present" in r.stdout