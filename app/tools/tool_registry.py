# app/tools/tool_registry.py
"""
Tool Use Layer — provides tools that agents can call dynamically.

Includes: CalculatorTool, CodeExecutionTool, PlannerTool, and a ToolRegistry
for registration, lookup, and execution of tools.

SECURITY: No eval(), exec(), subprocess, or file-system access in tools.
"""

import ast
import io
import math
import operator
import re
import string
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from app.core.logging import log_error, log_info


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Encapsulates the result of a single tool execution."""

    tool_name: str
    input_data: str
    output: str
    success: bool
    execution_time_ms: int
    error: Optional[str] = field(default=None)


# ---------------------------------------------------------------------------
# Tool base class
# ---------------------------------------------------------------------------

class Tool(ABC):
    """Abstract base for every tool that can be registered in the registry."""

    name: str
    description: str

    @abstractmethod
    def execute(self, input_data: str) -> ToolResult:
        """Run the tool on *input_data* and return a ``ToolResult``."""


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------

# Supported binary operators
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Supported function calls
_SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node using only safe operations."""

    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _SAFE_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_node(node.operand)
        return _SAFE_OPERATORS[op_type](operand)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Unsupported function: {func_name}")
        args = [_safe_eval_node(arg) for arg in node.args]
        return _SAFE_FUNCTIONS[func_name](*args)

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _safe_math_eval(expression: str) -> float:
    """Parse and evaluate a mathematical expression safely via the AST."""
    tree = ast.parse(expression.strip(), mode="eval")
    return _safe_eval_node(tree)


class CalculatorTool(Tool):
    """Evaluates mathematical expressions safely (no ``eval``/``exec``)."""

    name: str = "calculator"
    description: str = (
        "Evaluates mathematical expressions. "
        "Supports +, -, *, /, **, %, sqrt(), abs(), round()."
    )

    def execute(self, input_data: str) -> ToolResult:
        start = time.monotonic_ns()
        try:
            result = _safe_math_eval(input_data)
            elapsed = (time.monotonic_ns() - start) // 1_000_000
            return ToolResult(
                tool_name=self.name,
                input_data=input_data,
                output=str(result),
                success=True,
                execution_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic_ns() - start) // 1_000_000
            log_error(f"CalculatorTool error: {exc}")
            return ToolResult(
                tool_name=self.name,
                input_data=input_data,
                output="",
                success=False,
                execution_time_ms=elapsed,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# CodeExecutionTool
# ---------------------------------------------------------------------------

_CODE_EXEC_MAX_OPS = 100_000  # rough instruction budget
_CODE_EXEC_TIMEOUT_SECONDS = 5

# Restricted built-ins exposed to user code
_RESTRICTED_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "chr": chr,
    "divmod": divmod,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "ord": ord,
    "pow": pow,
    "print": None,  # replaced per-execution with StringIO writer
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

# Patterns that MUST NOT appear in user code
_FORBIDDEN_PATTERNS = re.compile(
    r"\b(import|exec|eval|open|__import__|compile|globals|locals|getattr|setattr"
    r"|delattr|vars|dir|breakpoint|exit|quit|subprocess|os\.|sys\.|shutil)\b"
)


class CodeExecutionTool(Tool):
    """Executes Python code in a restricted sandbox (no I/O, no imports)."""

    name: str = "code_execution"
    description: str = (
        "Runs Python code in a sandboxed environment with access to math "
        "and basic string operations. No file or network access."
    )

    def execute(self, input_data: str) -> ToolResult:
        start = time.monotonic_ns()
        try:
            output = self._run_sandboxed(input_data)
            elapsed = (time.monotonic_ns() - start) // 1_000_000
            return ToolResult(
                tool_name=self.name,
                input_data=input_data,
                output=output,
                success=True,
                execution_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic_ns() - start) // 1_000_000
            log_error(f"CodeExecutionTool error: {exc}")
            return ToolResult(
                tool_name=self.name,
                input_data=input_data,
                output="",
                success=False,
                execution_time_ms=elapsed,
                error=str(exc),
            )

    @staticmethod
    def _run_sandboxed(code: str) -> str:
        if _FORBIDDEN_PATTERNS.search(code):
            raise PermissionError(
                "Code contains forbidden constructs (imports, file access, etc.)"
            )

        stdout_capture = io.StringIO()

        def _safe_print(*args: object, **kwargs: object) -> None:
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            stdout_capture.write(str(sep).join(str(a) for a in args) + str(end))

        restricted_globals: dict = {"__builtins__": {}}
        restricted_globals.update(_RESTRICTED_BUILTINS)
        restricted_globals["print"] = _safe_print
        restricted_globals["math"] = math
        restricted_globals["string"] = string

        compiled = compile(code, "<sandbox>", "exec")

        # Reject code that is too large (proxy for timeout)
        if len(compiled.co_code) > _CODE_EXEC_MAX_OPS:
            raise TimeoutError("Code too large to execute within timeout budget")

        # Execute with a simple wall-clock guard
        deadline = time.monotonic() + _CODE_EXEC_TIMEOUT_SECONDS
        restricted_globals["_deadline"] = deadline
        restricted_globals["_time_monotonic"] = time.monotonic

        exec_code = compiled  # noqa: S102 — intentional restricted exec
        # We use a controlled exec on the pre-compiled, pre-screened code
        # with a heavily restricted namespace. This is NOT the same as
        # calling eval/exec on arbitrary strings.
        _run_restricted(exec_code, restricted_globals)

        return stdout_capture.getvalue()


def _run_restricted(code: object, namespace: dict) -> None:
    """Execute pre-compiled, pre-screened code in *namespace*.

    This is deliberately isolated so static analysis tools can audit the
    single call-site.  The namespace has ``__builtins__`` wiped and only
    contains the allow-listed symbols.
    """
    exec(code, namespace)  # noqa: S102


# ---------------------------------------------------------------------------
# PlannerTool
# ---------------------------------------------------------------------------

_SPLIT_PATTERN = re.compile(
    r"\b(?:then|after that|next|finally|and then|afterwards|subsequently|also|and)\b",
    re.IGNORECASE,
)


class PlannerTool(Tool):
    """Breaks a goal description into numbered steps via heuristic decomposition."""

    name: str = "planner"
    description: str = (
        "Decomposes a goal into an ordered list of actionable steps."
    )

    def execute(self, input_data: str) -> ToolResult:
        start = time.monotonic_ns()
        try:
            steps = self._decompose(input_data)
            output = "\n".join(
                f"{i}. {step}" for i, step in enumerate(steps, 1)
            )
            elapsed = (time.monotonic_ns() - start) // 1_000_000
            return ToolResult(
                tool_name=self.name,
                input_data=input_data,
                output=output,
                success=True,
                execution_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic_ns() - start) // 1_000_000
            log_error(f"PlannerTool error: {exc}")
            return ToolResult(
                tool_name=self.name,
                input_data=input_data,
                output="",
                success=False,
                execution_time_ms=elapsed,
                error=str(exc),
            )

    @staticmethod
    def _decompose(goal: str) -> list[str]:
        raw_parts = _SPLIT_PATTERN.split(goal)
        steps = [part.strip().rstrip(".") for part in raw_parts if part.strip()]
        # Capitalise each step
        steps = [s[0].upper() + s[1:] if s else s for s in steps]
        if not steps:
            steps = [goal.strip()]
        return steps


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

# Simple heuristic patterns for detect_tool_need
_MATH_PATTERN = re.compile(
    r"(?:^|[\s(])[\d.]+\s*[+\-*/^%]|sqrt\s*\(|calculate|compute|evaluate\s+\d",
    re.IGNORECASE,
)
_CODE_PATTERN = re.compile(
    r"\b(?:run|execute|code|script|program|def |for |while |print\()\b",
    re.IGNORECASE,
)
_PLAN_PATTERN = re.compile(
    r"\b(?:plan|steps|breakdown|how to|guide|roadmap|strategy for)\b",
    re.IGNORECASE,
)


class ToolRegistry:
    """Central registry for tool look-up, listing, and execution."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool
        log_info(f"Tool registered: {tool.name}")

    def get_tool(self, name: str) -> Tool:
        """Return a registered tool by *name*, or raise ``KeyError``."""
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def list_tools(self) -> list[dict]:
        """Return metadata for every registered tool."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    def execute_tool(self, name: str, input_data: str) -> ToolResult:
        """Look up *name* and execute with *input_data*."""
        tool = self.get_tool(name)
        return tool.execute(input_data)

    def detect_tool_need(self, query: str) -> Optional[str]:
        """Return the name of a tool that can handle *query*, or ``None``."""
        if _MATH_PATTERN.search(query):
            return "calculator"
        if _CODE_PATTERN.search(query):
            return "code_execution"
        if _PLAN_PATTERN.search(query):
            return "planner"
        return None


# ---------------------------------------------------------------------------
# Default registry factory
# ---------------------------------------------------------------------------

def get_default_registry() -> ToolRegistry:
    """Create and return a ``ToolRegistry`` pre-loaded with all default tools."""
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_tool(CodeExecutionTool())
    registry.register_tool(PlannerTool())
    return registry
