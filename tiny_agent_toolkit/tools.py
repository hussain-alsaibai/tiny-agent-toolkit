"""
Tool registry — single-file MCP-compatible tool dispatch.

A tool is a Python function with:
    - a JSON-schema describing its parameters
    - a one-line description (used as the schema description)

The registry stores tools and serializes them in three formats:
    - OpenAI / Anthropic function-calling JSON Schema
    - Anthropic-native tool schema (input_schema wrapper)
    - Model Context Protocol (MCP) tools/list JSON

Plus a tiny decorator `@tool(...)` that introspects the function signature
and produces a schema automatically (matches the original tiny-agent ergonomics).

Usage:
    from tiny_agent_toolkit import ToolRegistry, tool

    registry = ToolRegistry()

    @tool(description="Add two numbers")
    def add(a: float, b: float) -> float:
        return a + b

    registry.register(add)

    # Convert to formats any major LLM provider accepts:
    openai_schema = registry.to_openai()
    anthropic_schema = registry.to_anthropic()
    mcp_schema = registry.to_mcp()

    # Or call a tool by name (with JSON-string args, like an LLM would emit):
    result = registry.call('add', '{"a": 1, "b": 2}')
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin


__all__ = ["Tool", "ToolRegistry", "ToolError", "tool"]


# JSON-schema type mapping from Python annotations
_BASIC_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
    bytes: "string",
}


def _annotation_to_json_type(annotation: Any) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment.

    Handles Optional[T], List[T], Dict[str, T], and bare primitives.
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[T] is Union[T, None]
    if origin is Union and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_json_type(non_none[0])
        # Multi-type union — represent as anyOf
        return {"anyOf": [_annotation_to_json_type(a) for a in non_none]}

    if origin in (list, List):
        if args:
            return {"type": "array", "items": _annotation_to_json_type(args[0])}
        return {"type": "array"}

    if origin in (dict, Dict):
        if args and len(args) == 2:
            return {
                "type": "object",
                "additionalProperties": _annotation_to_json_type(args[1]),
            }
        return {"type": "object"}

    return {"type": _BASIC_TYPE_MAP.get(annotation, "string")}


def _build_schema_from_signature(func: Callable, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Auto-build a JSON Schema for a function's parameters."""
    if override is not None:
        return override

    sig = inspect.signature(func)
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            # *args / **kwargs — not supported in JSON schema; skip.
            continue

        annotation = param.annotation
        prop: Dict[str, Any] = _annotation_to_json_type(annotation)
        prop["description"] = (param.name or name).replace("_", " ")

        # If the param has a default, mention it
        if param.default is not inspect.Parameter.empty:
            try:
                prop["default"] = param.default
            except TypeError:
                pass  # some defaults aren't JSON-serializable; skip
        else:
            required.append(name)

        properties[name] = prop

    schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """Raised when a tool call fails."""


@dataclass
class Tool:
    """A registered tool.

    Attributes:
        name: Unique tool name (used by LLMs to call it).
        func: The underlying Python callable.
        description: One-line human description (becomes the schema description).
        parameters: JSON Schema dict describing the function arguments.
        metadata: Free-form dict for tags, version, owner, etc.
    """
    name: str
    func: Callable[..., Any]
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def call(self, **kwargs: Any) -> Any:
        """Invoke the tool with keyword arguments."""
        return self.func(**kwargs)

    # ---- schema serializers ----

    def to_openai(self) -> Dict[str, Any]:
        """OpenAI / Anthropic function-calling schema (the wire format
        every major provider except Anthropic uses)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic(self) -> Dict[str, Any]:
        """Anthropic-native tool schema. Anthropic wraps parameters in
        input_schema and does not use the function-calling envelope."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_mcp(self) -> Dict[str, Any]:
        """Model Context Protocol (MCP) tools/list entry.

        Format: https://modelcontextprotocol.io/specification/server/tools
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    **metadata: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that marks a function as a tool and attaches a Tool instance.

    The Tool is stored on the function's `__tool__` attribute. The registry
    picks it up from there.

    Example:
        @tool(description="Add two numbers")
        def add(a: float, b: float) -> float:
            return a + b

        # Manual schema (when auto-introspection isn't enough):
        @tool(schema={"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]})
        def square(x: float) -> float:
            return x * x
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        if description is not None:
            tool_desc = description
        elif func.__doc__:
            tool_desc = func.__doc__.strip().splitlines()[0]
        else:
            tool_desc = ""

        func.__tool__ = Tool(  # type: ignore[attr-defined]
            name=tool_name,
            func=func,
            description=tool_desc,
            parameters=_build_schema_from_signature(func, override=schema),
            metadata=dict(metadata),
        )
        return func
    return decorator


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """In-memory tool registry with schema export.

    Tools can be added by:
      1. Decorator + `register(func)` — picks up `func.__tool__`
      2. Direct `register(Tool(...))` — full control
      3. `register_function(callable, name=..., description=..., schema=...)`
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    # ---- registration ----

    def register(self, tool_or_func: Union[Tool, Callable[..., Any]]) -> Tool:
        """Register a Tool instance or a decorated function. Returns the Tool."""
        if isinstance(tool_or_func, Tool):
            if tool_or_func.name in self._tools:
                raise ToolError(f"Tool '{tool_or_func.name}' is already registered")
            self._tools[tool_or_func.name] = tool_or_func
            return tool_or_func

        func = tool_or_func
        tool_obj = getattr(func, "__tool__", None)
        if tool_obj is None:
            # Fall back to building one
            tool_obj = Tool(
                name=func.__name__,
                func=func,
                description=(func.__doc__ or "").strip().splitlines()[0] if func.__doc__ else "",
                parameters=_build_schema_from_signature(func),
            )
            func.__tool__ = tool_obj  # type: ignore[attr-defined]
        if tool_obj.name in self._tools:
            raise ToolError(f"Tool '{tool_obj.name}' is already registered")
        self._tools[tool_obj.name] = tool_obj
        return tool_obj

    def register_function(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        **metadata: Any,
    ) -> Tool:
        """Register a plain function (no decorator required)."""
        t = Tool(
            name=name or func.__name__,
            func=func,
            description=description or (func.__doc__ or "").strip().splitlines()[0] if func.__doc__ else "",
            parameters=_build_schema_from_signature(func, override=schema),
            metadata=dict(metadata),
        )
        return self.register(t)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if it existed."""
        return self._tools.pop(name, None) is not None

    # ---- lookup ----

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    # ---- invocation ----

    def call(self, name: str, arguments: Union[str, Dict[str, Any], None] = None) -> Any:
        """Invoke a tool by name. Accepts arguments as:
          - a JSON string  (typical LLM output)
          - a dict         (already-parsed)
          - None           (no arguments)
        """
        tool_obj = self.get(name)
        if arguments is None or arguments == "":
            arguments = {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                # Best-effort: try parsing 'a=1, b=2' style
                arguments = self._parse_loose(arguments)
                if arguments is None:
                    raise ToolError(f"Invalid JSON arguments for tool '{name}': {e}") from e
        if not isinstance(arguments, dict):
            raise ToolError(f"Tool '{name}' expected a JSON object, got {type(arguments).__name__}")
        try:
            return tool_obj.call(**arguments)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Tool '{name}' raised: {e}") from e

    @staticmethod
    def _parse_loose(s: str) -> Optional[Dict[str, Any]]:
        """Best-effort parse of 'key=value, key=value' style arguments."""
        out: Dict[str, Any] = {}
        # Try JSON-ish first
        try:
            return json.loads("{" + s + "}")
        except Exception:
            pass
        # Fall back to comma-split
        try:
            for part in s.split(","):
                if "=" not in part:
                    return None
                k, v = part.split("=", 1)
                k = k.strip().strip('"').strip("'")
                v = v.strip().strip('"').strip("'")
                # Type coercion
                if v.lower() in ("true", "false"):
                    out[k] = v.lower() == "true"
                else:
                    try:
                        out[k] = int(v)
                    except ValueError:
                        try:
                            out[k] = float(v)
                        except ValueError:
                            out[k] = v
            return out
        except Exception:
            return None

    # ---- schema export ----

    def to_openai(self) -> List[Dict[str, Any]]:
        """All tools in OpenAI function-calling format."""
        return [t.to_openai() for t in self._tools.values()]

    def to_anthropic(self) -> List[Dict[str, Any]]:
        """All tools in Anthropic-native format."""
        return [t.to_anthropic() for t in self._tools.values()]

    def to_mcp(self) -> List[Dict[str, Any]]:
        """All tools in MCP tools/list format.

        Wraps the result in the standard MCP envelope for parity with
        real MCP servers' responses.
        """
        return [t.to_mcp() for t in self._tools.values()]

    def to_mcp_list_payload(self) -> Dict[str, Any]:
        """Wrap `to_mcp()` in the JSON-RPC `result.tools` envelope."""
        return {"tools": self.to_mcp()}

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())!r})"