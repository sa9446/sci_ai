"""Self-distillation dataset generator for Phase B.

Drives the EXISTING sci_engine pipeline (NeuralSymbolicRouter -> Claude API,
InsulatedSandbox, ResultGroundingValidator) — imported unmodified from the
parent project — over a large bank of synthetic science queries. Only
(query, ScientificAnalysis) pairs where the sandbox executed successfully
AND the validator passed are kept, so the Phase B training data meets the
exact same quality bar main.py already enforces for Claude's own output.

Needs ANTHROPIC_API_KEY set (same as the parent sci_ai_engine project).
Resumable: re-running appends to --out-path, skipping queries already logged.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

# sci_engine lives one level up, in the parent sci_ai_engine project.
sys.path.insert(0, str(Path(__file__).parent.parent))

from sci_engine import InsulatedSandbox, NeuralSymbolicRouter, ResultGroundingValidator, settings  # noqa: E402

# ─── Synthetic query bank ────────────────────────────────────────────────────
# Each template is filled with randomized parameters to generate diverse
# queries across domains, so Phase B doesn't overfit to one equation shape.

PARTICLES = [
    ("proton", "1.67262192369e-27"), ("electron", "9.1093837015e-31"),
    ("muon", "1.883531627e-28"), ("alpha particle", "6.6446573357e-27"),
]

_TEMPLATES: list[tuple[str, "callable[[], str]"]] = [
    ("special_relativity", lambda: (
        f"A {(p := random.choice(PARTICLES))[0]} (rest mass m0 = {p[1]} kg) is "
        f"accelerated to v = {random.choice([0.9, 0.95, 0.99, 0.995, 0.999])}c. "
        "Using special relativity, compute: the Lorentz factor, the relativistic "
        "kinetic energy in Joules, the kinetic energy in MeV, the ratio of kinetic "
        "to rest-mass energy, and the total relativistic energy in Joules. Return "
        "a dict with descriptive keys including units."
    )),
    ("thermodynamics", lambda: (
        f"An ideal gas has {random.choice([1, 2, 5, 10])} moles at a temperature of "
        f"{random.choice([250, 300, 373, 500])} K and pressure of "
        f"{random.choice([1, 2, 5])} atm. Compute the volume in cubic meters, the "
        "internal energy assuming a monatomic gas, the root-mean-square speed of "
        "the gas molecules (molar mass 0.032 kg/mol for O2), and the entropy change "
        "if the gas is isothermally expanded to double its volume. Return a dict "
        "with descriptive keys including units."
    )),
    ("quantum_mechanics", lambda: (
        f"A particle of mass {random.choice([9.109e-31, 1.673e-27])} kg is confined "
        f"to an infinite square well of width {random.choice([1e-9, 1e-10, 5e-10])} m. "
        "Compute the ground state energy in Joules and in eV, the energy of the "
        f"first excited state ({random.choice([2,3])} quantum number), the "
        "de Broglie wavelength at the ground state, and the probability of finding "
        "the particle in the central third of the well for the ground state. "
        "Return a dict with descriptive keys including units."
    )),
    ("electromagnetism", lambda: (
        f"Two point charges of {random.choice([1e-6, 2e-6, 5e-6])} C and "
        f"{random.choice([-1e-6, 3e-6])} C are separated by "
        f"{random.choice([0.01, 0.05, 0.1])} m in vacuum. Compute the Coulomb force "
        "magnitude in Newtons, the electric potential energy in Joules, the electric "
        "field magnitude at the midpoint between them, and the electric potential at "
        "that midpoint. Return a dict with descriptive keys including units."
    )),
    ("fluid_dynamics", lambda: (
        f"Water (density 1000 kg/m^3, viscosity 1.0e-3 Pa.s) flows through a pipe of "
        f"diameter {random.choice([0.02, 0.05, 0.1])} m at a velocity of "
        f"{random.choice([0.5, 1.0, 2.0])} m/s. Compute the Reynolds number, "
        "classify the flow regime via a numeric flag (0=laminar, 1=turbulent), the "
        "volumetric flow rate in cubic meters per second, and the dynamic pressure. "
        "Return a dict with descriptive keys including units."
    )),
    ("astrophysics", lambda: (
        f"A star has a surface temperature of {random.choice([5778, 9000, 3500])} K "
        f"and a radius of {random.choice([1, 2, 0.5])} times the Sun's radius "
        "(6.957e8 m). Using the Stefan-Boltzmann law, compute the luminosity in "
        "Watts, the peak emission wavelength via Wien's law, the luminosity ratio "
        "relative to the Sun (3.828e26 W), and the apparent brightness at a distance "
        "of 10 parsecs. Return a dict with descriptive keys including units."
    )),
]


def generate_queries(n: int, seed: int = 0) -> list[tuple[str, str]]:
    random.seed(seed)
    queries = []
    for _ in range(n):
        domain, query_fn = random.choice(_TEMPLATES)
        queries.append((domain, query_fn()))
    return queries


def _load_already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["query"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def distill(num_queries: int, out_path: Path, seed: int) -> None:
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set — same requirement as running main.py.")

    router = NeuralSymbolicRouter()
    sandbox = InsulatedSandbox()
    validator = ResultGroundingValidator()

    already_done = _load_already_done(out_path)
    queries = generate_queries(num_queries, seed=seed)

    kept, skipped, failed = 0, 0, 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as out_f:
        for i, (expected_domain, query) in enumerate(queries):
            if query in already_done:
                skipped += 1
                continue

            try:
                analysis = await router.route(query)
                exec_result = sandbox.execute(analysis.executable_python_code)
                if not exec_result.success:
                    failed += 1
                    continue

                report = validator.validate(
                    results=exec_result.result,
                    domain=analysis.scientific_domain,
                    symbolic_repr=analysis.symbolic_representation,
                )
                if not report.passed:
                    failed += 1
                    continue

                record = {
                    "query": query,
                    "expected_domain": expected_domain,
                    "scientific_domain": analysis.scientific_domain,
                    "symbolic_representation": analysis.symbolic_representation,
                    "variables": analysis.variables,
                    "expected_result_keys": analysis.expected_result_keys,
                    "complexity_level": analysis.complexity_level,
                    "executable_python_code": analysis.executable_python_code,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                kept += 1

            except Exception as exc:  # noqa: BLE001 — one bad query shouldn't kill the whole run
                print(f"  [{i}] error: {exc}")
                failed += 1

            if (i + 1) % 10 == 0:
                print(f"progress: {i+1}/{num_queries}  kept={kept}  failed={failed}  skipped={skipped}")

    print(f"Done. kept={kept}  failed={failed}  skipped={skipped}  -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the Phase B distillation dataset from the existing Claude-backed pipeline")
    p.add_argument("--out-path", type=Path, required=True)
    p.add_argument("--num-queries", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    asyncio.run(distill(args.num_queries, args.out_path, args.seed))


if __name__ == "__main__":
    main()
