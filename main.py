"""End-to-end demonstration of the Scientific AI Model Engine.

Pipeline:
  query  →  NeuralSymbolicRouter (LLM structured parse)
         →  InsulatedSandbox     (deterministic code execution)
         →  ResultGroundingValidator (physics & dimensional checks)
         →  verified numerical output

Example: relativistic kinetic energy of a proton at 99.5% of light speed.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import textwrap
from pathlib import Path

# Make sure the package is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from sci_engine import (
    InsulatedSandbox,
    NeuralSymbolicRouter,
    ResultGroundingValidator,
    settings,
)
from sci_engine.executor import ExecutionResult
from sci_engine.agent import ScientificAnalysis
from sci_engine.validator import ValidationReport

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sci_engine.main")

# ─── Example query ───────────────────────────────────────────────────────────

QUERY = """\
A proton (rest mass m₀ = 1.67262192369e-27 kg) is accelerated in the Large
Hadron Collider to a velocity of v = 0.995c, where c is the speed of light.

Using special relativity, compute the following precisely:
  1. The Lorentz factor γ
  2. The relativistic kinetic energy K in Joules
  3. The kinetic energy K in MeV (mega-electronvolts) — use 1 eV = 1.602176634e-19 J
  4. The ratio of kinetic energy to rest-mass energy (K / E₀)
  5. The relativistic total energy E_total in Joules

Return all five quantities as a dictionary with descriptive keys that include
the units (e.g., "lorentz_factor_dimensionless", "kinetic_energy_joules").
"""

# ─── Formatting helpers ───────────────────────────────────────────────────────

_W = 72  # output width

def _bar(char: str = "─") -> str:
    return char * _W

def _header(title: str, char: str = "═") -> str:
    padding = (_W - len(title) - 2) // 2
    return f"{char * padding} {title} {char * (_W - padding - len(title) - 2)}"

def _fmt_value(val: float) -> str:
    if val == 0.0:
        return "0.000000"
    mag = abs(val)
    if 1e-3 <= mag < 1e7:
        return f"{val:>20.8f}"
    return f"{val:>20.6e}"


# ─── Pipeline ────────────────────────────────────────────────────────────────

async def run_pipeline(query: str) -> None:
    print()
    print(_header("SCIENTIFIC AI MODEL ENGINE", "═"))
    print(_header("Neuro-Symbolic Computation Pipeline"))
    print()

    if settings.model_backend == "local":
        from sci_engine.local_router import LocalTransformerRouter
        router = LocalTransformerRouter()
    else:
        if not settings.anthropic_api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
            print("       Create a .env file or export the variable before running.")
            sys.exit(1)
        router = NeuralSymbolicRouter()

    sandbox   = InsulatedSandbox()
    validator = ResultGroundingValidator()

    # ── Phase 1: Neuro-Symbolic Routing ──────────────────────────────────────
    print(_header("Phase 1 — Neuro-Symbolic Routing (LLM → Structured Plan)"))
    print()

    analysis: ScientificAnalysis | None = None
    last_error: str | None = None

    for attempt in range(1, settings.max_retries + 1):
        try:
            print(f"  Calling {settings.model_id} with structured tool_choice …")
            analysis = await router.route(query, retry_context=last_error)
            break
        except Exception as exc:
            last_error = str(exc)
            logger.error("Routing attempt %d/%d failed: %s", attempt, settings.max_retries, exc)
            if attempt == settings.max_retries:
                print(f"\n  FATAL: routing failed after {settings.max_retries} attempts.")
                raise

    assert analysis is not None

    print(f"\n  Domain       : {analysis.scientific_domain}")
    print(f"  Complexity   : {analysis.complexity_level}")
    print(f"\n  LaTeX equation:")
    for line in textwrap.wrap(analysis.symbolic_representation, width=66):
        print(f"    {line}")

    print(f"\n  Variables:")
    for sym, desc in analysis.variables.items():
        print(f"    {sym:12s}  {desc}")

    print(f"\n  Expected result keys:")
    for k in analysis.expected_result_keys:
        print(f"    • {k}")

    print(f"\n  Generated Python code ({len(analysis.executable_python_code)} chars):")
    print("  " + _bar())
    for line in analysis.executable_python_code.splitlines():
        print(f"  {line}")
    print("  " + _bar())

    # ── Phase 2: Insulated Execution ──────────────────────────────────────────
    print()
    print(_header("Phase 2 — Insulated Sandbox Execution"))
    print()

    exec_result: ExecutionResult | None = None

    for attempt in range(1, settings.max_retries + 1):
        print(f"  Executing in sandbox (attempt {attempt}/{settings.max_retries}) …")
        exec_result = sandbox.execute(analysis.executable_python_code)

        if exec_result.success:
            print(f"  Status       : SUCCESS  ({exec_result.execution_time_ms:.2f} ms)")
            break

        logger.error(
            "Sandbox attempt %d failed [%s]: %s",
            attempt, exec_result.error_type, exec_result.error_message,
        )
        print(f"  Status       : FAILED — {exec_result.error_type}")
        print(f"  Error        : {exec_result.error_message}")
        if exec_result.traceback_text:
            print(f"\n  Traceback:\n")
            for tb_line in exec_result.traceback_text.splitlines():
                print(f"    {tb_line}")

        if attempt < settings.max_retries:
            print(f"\n  Regenerating code with error context …")
            retry_ctx = (
                f"Execution failed with {exec_result.error_type}: "
                f"{exec_result.error_message}\n{exec_result.traceback_text}"
            )
            analysis = await router.route(query, retry_context=retry_ctx)
            print(f"  New code generated ({len(analysis.executable_python_code)} chars).")
        else:
            raise RuntimeError(
                f"Execution failed after {settings.max_retries} attempts. "
                f"Last error: {exec_result.error_message}"
            )

    assert exec_result is not None and exec_result.success

    if exec_result.stdout:
        print(f"\n  Captured stdout:\n    {exec_result.stdout.strip()}")

    print(f"\n  Raw numerical results ({len(exec_result.result)} values):")
    for key, val in exec_result.result.items():
        print(f"    {key:<45s} = {_fmt_value(val).strip()}")

    # ── Phase 3: Verification & Grounding ────────────────────────────────────
    print()
    print(_header("Phase 3 — Verification & Grounding Layer"))
    print()

    report: ValidationReport = validator.validate(
        results=exec_result.result,
        domain=analysis.scientific_domain,
        symbolic_repr=analysis.symbolic_representation,
    )

    overall = "PASSED ✓" if report.passed else "FAILED ✗"
    print(f"  Overall      : {overall}")
    print(f"  Dimensional  : {report.dimensional_check}")

    if report.errors:
        print(f"\n  Errors ({len(report.errors)}):")
        for issue in report.errors:
            print(f"    {issue}")

    if report.warnings:
        print(f"\n  Warnings ({len(report.warnings)}):")
        for issue in report.warnings:
            print(f"    {issue}")

    if report.notes:
        print(f"\n  Analysis notes:")
        for note in report.notes:
            print(f"    {note}")

    # ── Final Answer ──────────────────────────────────────────────────────────
    print()
    print(_header("FINAL VERIFIED RESULTS", "═"))
    print()

    if report.passed:
        print(f"  Query domain : {analysis.scientific_domain}")
        print(f"  Equation     : {analysis.symbolic_representation}")
        print()
        print(f"  {'Quantity':<45}  {'Value':>20}")
        print(f"  {_bar('·')}")
        for key, val in exec_result.result.items():
            print(f"  {key:<45}  {_fmt_value(val)}")
        print(f"  {_bar('·')}")
        print()
        print(f"  Computed in {exec_result.execution_time_ms:.2f} ms  |  "
              f"All physical bounds validated ✓")
    else:
        print("  WARNING: Results did not pass validation. See Phase 3 errors.")
        print()
        print("  Unvalidated results (use with caution):")
        for key, val in exec_result.result.items():
            print(f"    {key:<45}  {_fmt_value(val)}")

    print()
    print("═" * _W)
    print()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(run_pipeline(QUERY))
