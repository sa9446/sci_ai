"""Neuro-Symbolic Router — uses the LLM solely for structured parsing and code
generation.  All arithmetic is performed deterministically by the generated
code; the LLM never computes numerical results directly.
"""
from __future__ import annotations

import logging
from typing import Any

import anthropic
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Literal

from .config import settings

logger = logging.getLogger(__name__)

# ─── Structured-output contract ──────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a deterministic scientific computation planner for a research-grade engine.

Your role: parse a scientific query and emit a structured plan containing:
  1. Domain classification
  2. The primary equation in LaTeX
  3. Complete, self-contained Python code

━━━ MANDATORY RULES FOR executable_python_code ━━━

FUNCTION (exact signature):
  def calculate_research():
      ...
      return {"key": float_value, ...}

  • Zero parameters. No global state. Returns a plain Python dict.
  • Every value in the returned dict MUST be castable to Python float().

ALLOWED IMPORTS ONLY:
  import numpy as np
  import scipy / from scipy import ...
  from scipy import constants          # use this for physical constants
  import sympy as sp / from sympy import ...
  import jax.numpy as jnp             # optional
  import math, cmath

FORBIDDEN (will abort execution):
  os  sys  subprocess  socket  urllib  requests  pathlib  shutil
  open()  input()  exec()  eval()  compile()
  __import__  globals()  locals()  vars()
  Any file I/O or network access

NUMERICAL BEST PRACTICES:
  • Physical constants: always use scipy.constants (c, h, m_p, m_e, etc.)
  • Precision: np.float64 throughout; avoid Python int arithmetic on large values
  • Stability: guard division by zero; use np.clip() near singularities
  • Return keys: descriptive snake_case with units, e.g. "kinetic_energy_joules"
    or "lorentz_factor_dimensionless"
  • Return at least 4 distinct computed quantities
  • Cast all return values explicitly: {"key": float(numpy_scalar)}

LATEX:
  Write the symbolic_representation field as clean LaTeX, e.g.:
  $\\gamma = \\frac{1}{\\sqrt{1-\\beta^2}},\\quad K = (\\gamma-1)m_0c^2$
"""

_TOOL_NAME = "analyze_scientific_query"


# ─── Pydantic response model ──────────────────────────────────────────────────

class ScientificAnalysis(BaseModel):
    scientific_domain: str = Field(
        description=(
            "Canonical domain label, e.g. 'special_relativity', "
            "'quantum_mechanics', 'fluid_dynamics', 'thermodynamics', "
            "'electromagnetism', 'astrophysics', 'linear_algebra'"
        )
    )
    symbolic_representation: str = Field(
        description="Primary governing equation(s) in LaTeX notation."
    )
    variables: dict[str, str] = Field(
        description=(
            "Map of variable symbol → 'description [SI unit]', "
            "e.g. {\"m_0\": \"proton rest mass [kg]\", \"v\": \"particle velocity [m/s]\"}"
        )
    )
    expected_result_keys: list[str] = Field(
        description="Exact keys that calculate_research() will include in its return dict."
    )
    complexity_level: Literal["undergraduate", "graduate", "research"] = Field(
        description="Mathematical depth of the computation."
    )
    executable_python_code: str = Field(
        description=(
            "Complete Python source defining calculate_research() with no parameters. "
            "Must be importable as-is (no top-level side-effects outside the function). "
            "Returns dict[str, float]."
        )
    )

    @field_validator("executable_python_code")
    @classmethod
    def _must_have_function(cls, v: str) -> str:
        if "def calculate_research(" not in v:
            raise ValueError(
                "executable_python_code must define calculate_research() function"
            )
        return v

    @field_validator("expected_result_keys")
    @classmethod
    def _nonempty_keys(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("expected_result_keys must not be empty")
        return v


# ─── Router ───────────────────────────────────────────────────────────────────

class NeuralSymbolicRouter:
    """Routes a free-text scientific query to a deterministic computation plan.

    The LLM is invoked exactly once per call (or once per retry) using
    tool_choice="required" to guarantee structured output.  No arithmetic is
    performed by the model — that responsibility belongs to the generated code.
    """

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._tool_def = self._build_tool_definition()

    # ── Public API ────────────────────────────────────────────────────────────

    async def route(
        self,
        query: str,
        *,
        retry_context: str | None = None,
    ) -> ScientificAnalysis:
        """Parse *query* into a structured :class:`ScientificAnalysis`.

        Args:
            query: Natural-language scientific question.
            retry_context: On retries, the previous failure message is injected
                so the model can correct the generated code.
        """
        messages = self._build_messages(query, retry_context)

        logger.info("Calling LLM for structured routing (retry=%s)", retry_context is not None)

        response = await self._client.messages.create(
            model=settings.model_id,
            max_tokens=settings.max_tokens,
            system=_SYSTEM_PROMPT,
            tools=[self._tool_def],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=messages,
        )

        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_block is None:
            raise RuntimeError(
                "LLM response contained no tool_use block — unexpected finish_reason: "
                f"{response.stop_reason}"
            )

        logger.info(
            "Routing complete: domain=%s  complexity=%s  code_len=%d",
            tool_block.input.get("scientific_domain", "?"),
            tool_block.input.get("complexity_level", "?"),
            len(tool_block.input.get("executable_python_code", "")),
        )

        return ScientificAnalysis(**tool_block.input)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_tool_definition(self) -> dict[str, Any]:
        schema = ScientificAnalysis.model_json_schema()
        schema.pop("title", None)
        return {
            "name": _TOOL_NAME,
            "description": (
                "Analyze a scientific query and produce a structured computation plan "
                "with symbolic mathematics and executable Python code."
            ),
            "input_schema": schema,
        }

    @staticmethod
    def _build_messages(
        query: str, retry_context: str | None
    ) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [{"role": "user", "content": query.strip()}]
        if retry_context:
            msgs += [
                {
                    "role": "assistant",
                    "content": (
                        "I see the previous code had an issue. "
                        "I will regenerate it with the necessary corrections."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"The previous attempt failed with the following error:\n\n"
                        f"```\n{retry_context}\n```\n\n"
                        "Please regenerate calculate_research() fixing the reported issue."
                    ),
                },
            ]
        return msgs
