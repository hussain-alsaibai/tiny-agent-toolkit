"""Tests for the Tool registry and decorator."""

import json

import pytest

from tiny_agent_toolkit import Tool, ToolError, ToolRegistry, tool


class TestToolDecorator:
    def test_basic_decoration(self):
        @tool(description="Add two numbers")
        def add(a: float, b: float) -> float:
            return a + b

        assert hasattr(add, "__tool__")
        t = add.__tool__
        assert t.name == "add"
        assert t.description == "Add two numbers"
        assert "a" in t.parameters["properties"]
        assert "b" in t.parameters["properties"]
        assert set(t.parameters["required"]) == {"a", "b"}

    def test_docstring_used_when_no_description(self):
        @tool()
        def my_func(x: int) -> int:
            """First line of docstring.

            More docs.
            """
            return x

        assert my_func.__tool__.description == "First line of docstring."

    def test_no_docstring_or_description(self):
        @tool()
        def bare(x: int) -> int:
            return x

        assert bare.__tool__.description == ""

    def test_optional_parameter(self):
        @tool()
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}"

        params = greet.__tool__.parameters
        assert "name" in params["required"]
        # Optional params get a default
        assert "default" in params["properties"]["greeting"]

    def test_list_and_dict_annotations(self):
        @tool()
        def process(items: list, options: dict) -> dict:
            return {"count": len(items)}

        params = process.__tool__.parameters
        assert params["properties"]["items"]["type"] == "array"
        assert params["properties"]["options"]["type"] == "object"

    def test_typed_list(self):
        from typing import List
        @tool()
        def process(items: List[int]) -> int:
            return sum(items)

        params = process.__tool__.parameters
        assert params["properties"]["items"]["items"]["type"] == "integer"


class TestToolRegistry:
    def test_register_decorated(self):
        @tool(description="X")
        def my_fn(x: int) -> int:
            return x

        reg = ToolRegistry()
        reg.register(my_fn)
        assert reg.has("my_fn")
        assert len(reg) == 1

    def test_register_explicit(self):
        t = Tool(name="t", func=lambda: 1, description="t", parameters={"type": "object"})
        reg = ToolRegistry()
        reg.register(t)
        assert "t" in reg

    def test_register_function_helper(self):
        reg = ToolRegistry()

        def my_fn(a: int) -> int:
            """Doubles an integer."""
            return a * 2

        reg.register_function(my_fn)
        assert reg.has("my_fn")
        assert "Doubles" in reg.get("my_fn").description

    def test_duplicate_registration_raises(self):
        reg = ToolRegistry()
        reg.register(Tool(name="t", func=lambda: 1, description="", parameters={}))
        with pytest.raises(ToolError):
            reg.register(Tool(name="t", func=lambda: 2, description="", parameters={}))

    def test_call_with_json_string(self):
        reg = ToolRegistry()

        @tool()
        def add(a: float, b: float) -> float:
            return a + b

        reg.register(add)
        assert reg.call("add", '{"a": 1, "b": 2}') == 3.0

    def test_call_with_dict(self):
        reg = ToolRegistry()

        @tool()
        def add(a: float, b: float) -> float:
            return a + b

        reg.register(add)
        assert reg.call("add", {"a": 1, "b": 2}) == 3.0

    def test_call_no_args(self):
        reg = ToolRegistry()

        @tool()
        def hello() -> str:
            return "hi"

        reg.register(hello)
        assert reg.call("hello") == "hi"
        assert reg.call("hello", "") == "hi"
        assert reg.call("hello", None) == "hi"

    def test_call_invalid_json_raises(self):
        reg = ToolRegistry()

        @tool()
        def add(a: float, b: float) -> float:
            return a + b

        reg.register(add)
        with pytest.raises(ToolError):
            reg.call("add", "not-json")

    def test_call_unknown_tool_raises(self):
        reg = ToolRegistry()
        with pytest.raises(ToolError):
            reg.call("missing", {})

    def test_tool_error_wrapped(self):
        reg = ToolRegistry()

        @tool()
        def boom() -> int:
            raise ValueError("kaboom")

        reg.register(boom)
        with pytest.raises(ToolError) as ei:
            reg.call("boom")
        assert "kaboom" in str(ei.value)

    def test_to_openai(self):
        reg = ToolRegistry()

        @tool(description="Add")
        def add(a: float, b: float) -> float:
            return a + b

        reg.register(add)
        schemas = reg.to_openai()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "add"

    def test_to_anthropic(self):
        reg = ToolRegistry()

        @tool()
        def add(a: float, b: float) -> float:
            return a + b

        reg.register(add)
        schemas = reg.to_anthropic()
        assert "input_schema" in schemas[0]
        assert "name" in schemas[0]

    def test_to_mcp(self):
        reg = ToolRegistry()

        @tool(description="X")
        def my_fn() -> str:
            return "y"

        reg.register(my_fn)
        schemas = reg.to_mcp()
        assert "inputSchema" in schemas[0]
        assert schemas[0]["name"] == "my_fn"

    def test_to_mcp_list_payload(self):
        reg = ToolRegistry()

        @tool()
        def x() -> int:
            return 1

        reg.register(x)
        payload = reg.to_mcp_list_payload()
        assert "tools" in payload
        assert isinstance(payload["tools"], list)

    def test_unregister(self):
        reg = ToolRegistry()

        @tool()
        def x() -> int:
            return 1

        reg.register(x)
        assert reg.unregister("x") is True
        assert reg.unregister("x") is False
        assert not reg.has("x")