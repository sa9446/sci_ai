"""Insulated Execution Sandbox.

Threat model addressed:
  • Malicious/broken imports  → AST whitelist check before exec()
  • Dangerous built-ins       → restricted __builtins__ dict injected into namespace
  • Infinite loops / hangs    → daemon thread with join() timeout
  • Namespace leakage         → fresh isolated dict per execution; no shared state
  • stdout/stderr capture     → print() replaced in-namespace; warnings redirected

Known limitation: thread-based timeout cannot forcibly kill a CPU-bound tight
loop in pure C extensions (numpy, scipy).  For full isolation, replace the
threading backend with multiprocessing.Process + shared-memory result pipe.
"""
from __future__ import annotations

import ast
import builtins
import importlib
import io
import logging
import math
import threading
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)

# ─── Security constants ────────────────────────────────────────────────────────

ALLOWED_MODULE_ROOTS: frozenset[str] = frozenset({
    "numpy",
    "scipy",
    "sympy",
    "jax",
    "math",
    "cmath",
    "fractions",
    "decimal",
    "numbers",
    "functools",
    "itertools",
    "operator",
    "typing",
    "typing_extensions",
    "abc",
    "collections",
    "dataclasses",
    "enum",
})

# Full qualified names also allowed (e.g. "scipy.constants")
ALLOWED_MODULE_FULL: frozenset[str] = frozenset({
    "scipy.constants",
    "scipy.linalg",
    "scipy.optimize",
    "scipy.integrate",
    "scipy.special",
    "scipy.stats",
    "scipy.signal",
    "scipy.sparse",
    "jax.numpy",
    "jax.scipy",
    "jax.scipy.linalg",
    "sympy.physics",
    "sympy.physics.units",
    "sympy.parsing",
})

FORBIDDEN_CALL_NAMES: frozenset[str] = frozenset({
    "exec", "eval", "compile", "open", "input", "__import__",
    "globals", "locals", "vars", "dir",
    "breakpoint", "exit", "quit", "help",
    "memoryview", "bytearray",
})

FORBIDDEN_ATTRIBUTE_NAMES: frozenset[str] = frozenset({
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__code__", "__closure__", "__func__",
    "__builtins__", "__dict__", "__module__", "__qualname__",
    "__reduce__", "__reduce_ex__", "__init_subclass__",
    "__set_name__", "__slots__",
})


# ─── AST Security Checker ────────────────────────────────────────────────────

class ASTSecurityChecker(ast.NodeVisitor):
    """Walks generated AST and collects policy violations before any code runs."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def _flag(self, msg: str, lineno: int) -> None:
        self.violations.append(f"Line {lineno}: {msg}")

    # Import validation
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_MODULE_ROOTS and alias.name not in ALLOWED_MODULE_FULL:
                self._flag(f"Forbidden import '{alias.name}'", node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".")[0]
        if root not in ALLOWED_MODULE_ROOTS and module not in ALLOWED_MODULE_FULL:
            self._flag(f"Forbidden from-import '{module}'", node.lineno)
        self.generic_visit(node)

    # Call validation
    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
            self._flag(f"Forbidden call '{node.func.id}()'", node.lineno)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_CALL_NAMES:
                self._flag(f"Forbidden method call '.{node.func.attr}()'", node.lineno)
        self.generic_visit(node)

    # Attribute access validation (blocks dunder gadget chains)
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRIBUTE_NAMES:
            self._flag(f"Forbidden attribute access '.{node.attr}'", node.lineno)
        self.generic_visit(node)

    # Block subprocess-style shell invocation patterns
    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        # f-strings are fine; just recurse
        self.generic_visit(node)


# ─── Namespace Builder ───────────────────────────────────────────────────────

def _build_safe_import(allowed_roots: frozenset[str], allowed_full: frozenset[str]):
    """Returns a restricted __import__ that only admits whitelisted modules."""
    _real_import = builtins.__import__

    def _safe_import(
        name: str,
        glbls: dict | None = None,
        lcls: dict | None = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> Any:
        root = name.split(".")[0]
        if root not in allowed_roots and name not in allowed_full:
            raise ImportError(
                f"Module '{name}' is not permitted in the scientific sandbox. "
                f"Allowed roots: {sorted(allowed_roots)}"
            )
        return _real_import(name, glbls, lcls, fromlist, level)

    return _safe_import


def _build_execution_namespace(stdout_sink: io.StringIO) -> dict[str, Any]:
    """Construct a pre-populated, isolated namespace for one code execution."""
    ns: dict[str, Any] = {}

    # Pre-inject commonly needed scientific modules so the LLM code's import
    # statements resolve instantly (Python's import cache handles dedup).
    _preload = [
        ("numpy", "np"),
        ("scipy", "scipy"),
        ("scipy.constants", None),
        ("scipy.linalg", None),
        ("scipy.optimize", None),
        ("scipy.integrate", None),
        ("scipy.special", None),
        ("scipy.stats", None),
        ("sympy", "sp"),
        ("math", "math"),
        ("cmath", "cmath"),
    ]
    for mod_name, alias in _preload:
        try:
            mod = importlib.import_module(mod_name)
            short = mod_name.split(".")[-1]
            ns[short] = mod
            if alias:
                ns[alias] = mod
        except ImportError:
            logger.debug("Pre-import skipped (not installed): %s", mod_name)

    if settings.enable_jax:
        try:
            import jax
            import jax.numpy as jnp
            ns["jax"] = jax
            ns["jnp"] = jnp
        except ImportError:
            logger.debug("JAX not available")

    # Redirect print() so it writes to our captured buffer (thread-safe — each
    # execution gets its own StringIO, avoiding process-wide sys.stdout races).
    def _captured_print(*args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("file", stdout_sink)
        print(*args, **kwargs)

    ns["__builtins__"] = {
        # Numerics
        "abs": abs, "all": all, "any": any, "bin": bin,
        "bool": bool, "callable": callable, "chr": chr, "complex": complex,
        "divmod": divmod, "float": float, "format": format,
        "frozenset": frozenset, "hex": hex, "int": int, "isinstance": isinstance,
        "issubclass": issubclass, "iter": iter, "len": len, "list": list,
        "map": map, "max": max, "min": min, "next": next, "object": object,
        "oct": oct, "ord": ord, "pow": pow, "range": range, "repr": repr,
        "reversed": reversed, "round": round, "set": set, "slice": slice,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "type": type,
        "zip": zip, "enumerate": enumerate, "filter": filter,
        "dict": dict, "bytes": bytes,
        # I/O (captured)
        "print": _captured_print,
        # Exception types needed for try/except in generated code
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        "ArithmeticError": ArithmeticError, "ZeroDivisionError": ZeroDivisionError,
        "OverflowError": OverflowError, "RuntimeError": RuntimeError,
        "ImportError": ImportError, "IndexError": IndexError,
        "KeyError": KeyError, "AttributeError": AttributeError,
        "StopIteration": StopIteration, "NotImplementedError": NotImplementedError,
        # Constants
        "None": None, "True": True, "False": False,
        "NotImplemented": NotImplemented, "Ellipsis": Ellipsis,
        # Class machinery (needed for dataclasses, enums, etc.)
        "__build_class__": __build_class__,
        "__name__": "__sandbox__",
        "__doc__": None,
        # Safe import (whitelisted modules only)
        "__import__": _build_safe_import(ALLOWED_MODULE_ROOTS, ALLOWED_MODULE_FULL),
    }

    return ns


# ─── Execution Result ────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    success: bool
    result: dict[str, float] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0.0
    error_type: str = ""
    error_message: str = ""
    traceback_text: str = ""

    def summary(self) -> str:
        if self.success:
            return (
                f"OK in {self.execution_time_ms:.1f}ms — "
                f"{len(self.result)} result(s)"
            )
        return f"FAILED ({self.error_type}): {self.error_message}"


# ─── Sandbox ─────────────────────────────────────────────────────────────────

class InsulatedSandbox:
    """Executes LLM-generated scientific code in a hardened, isolated scope."""

    def __init__(self, timeout_seconds: int | None = None) -> None:
        self._timeout = timeout_seconds or settings.execution_timeout_seconds

    def execute(self, code: str) -> ExecutionResult:
        """Validate and run *code*, returning a structured :class:`ExecutionResult`."""

        # ── 1. Length guard ───────────────────────────────────────────────────
        if len(code) > settings.max_code_length:
            return ExecutionResult(
                success=False,
                error_type="CodeTooLongError",
                error_message=(
                    f"Code length {len(code)} chars exceeds maximum "
                    f"{settings.max_code_length}"
                ),
            )

        # ── 2. Syntax check ───────────────────────────────────────────────────
        try:
            tree = ast.parse(code, filename="<scientific_code>")
        except SyntaxError as exc:
            return ExecutionResult(
                success=False,
                error_type="SyntaxError",
                error_message=str(exc),
            )

        # ── 3. AST security scan ──────────────────────────────────────────────
        checker = ASTSecurityChecker()
        checker.visit(tree)
        if checker.violations:
            return ExecutionResult(
                success=False,
                error_type="SecurityViolationError",
                error_message=(
                    f"{len(checker.violations)} security violation(s) detected:\n"
                    + "\n".join(f"  • {v}" for v in checker.violations)
                ),
            )

        # ── 4. Compile once, reuse bytecode object ─────────────────────────────
        try:
            bytecode = compile(tree, "<scientific_code>", "exec")
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error_type="CompileError",
                error_message=str(exc),
            )

        # ── 5. Execute in isolated thread with timeout ────────────────────────
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        result_holder: dict[str, Any] = {"result": None, "error": None, "tb": ""}

        def _run() -> None:
            ns = _build_execution_namespace(stdout_buf)
            try:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    exec(bytecode, ns)  # noqa: S102 — sandboxed exec, intentional

                for w in caught_warnings:
                    stderr_buf.write(f"[WARNING] {w.category.__name__}: {w.message}\n")

                if "calculate_research" not in ns:
                    raise NameError(
                        "calculate_research() was not defined in the generated code"
                    )

                raw = ns["calculate_research"]()

                if not isinstance(raw, dict):
                    raise TypeError(
                        f"calculate_research() must return dict, got {type(raw).__name__}"
                    )

                # Coerce all values to float; raise early on non-numeric returns
                coerced: dict[str, float] = {}
                for k, v in raw.items():
                    if not isinstance(k, str):
                        raise TypeError(
                            f"Result key must be str, got {type(k).__name__}: {k!r}"
                        )
                    try:
                        coerced[k] = float(v)
                    except (TypeError, ValueError) as exc:
                        raise TypeError(
                            f"Result value for '{k}' is not numeric: {v!r}"
                        ) from exc

                result_holder["result"] = coerced

            except Exception as exc:  # noqa: BLE001
                result_holder["error"] = exc
                result_holder["tb"] = traceback.format_exc()

        start = time.perf_counter()
        thread = threading.Thread(target=_run, daemon=True, name="sci-sandbox")
        thread.start()
        thread.join(timeout=self._timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if thread.is_alive():
            logger.warning("Sandbox thread timed out after %ds", self._timeout)
            return ExecutionResult(
                success=False,
                error_type="TimeoutError",
                error_message=f"Execution exceeded {self._timeout}s timeout",
                execution_time_ms=elapsed_ms,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
            )

        if result_holder["error"] is not None:
            err = result_holder["error"]
            return ExecutionResult(
                success=False,
                error_type=type(err).__name__,
                error_message=str(err),
                traceback_text=result_holder["tb"],
                execution_time_ms=elapsed_ms,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
            )

        logger.info("Sandbox execution succeeded in %.1fms", elapsed_ms)
        return ExecutionResult(
            success=True,
            result=result_holder["result"],
            execution_time_ms=elapsed_ms,
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
        )
