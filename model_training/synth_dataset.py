"""Synthetic Phase B dataset generator — no Anthropic dependency.

Replaces the earlier Claude-distillation approach (which called the
sci_engine/NeuralSymbolicRouter -> Claude API to generate training examples).
Instead, each (query, ScientificAnalysis) pair is produced directly from a
hand-written physics formula per domain template: the randomized query
parameters are baked as literals into a generated calculate_research()
function, so the "ground truth" code is authored here, not asked of an LLM.

Because the code is ours (not LLM output), there's no need for the sandboxed
execution / AST security scan sci_engine uses for untrusted Claude output —
we just exec() it directly and sanity-check the result (finite floats only).

Resumable: re-running appends to --out-path, skipping queries already logged.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

# ─── Physical constants (avoids any dependency on scipy.constants at
# generation time; the generated code embeds its own constants so the
# training corpus is self-contained and matches what local_router.py's
# inference-time sandbox will execute later). ─────────────────────────────────
C_LIGHT = 299_792_458.0
ELEMENTARY_CHARGE = 1.602_176_634e-19
COULOMB_K = 8.987_551_792_3e9
GAS_CONSTANT = 8.314_462_618
PLANCK = 6.626_070_15e-34
STEFAN_BOLTZMANN = 5.670_374_419e-8
WIEN_B = 2.897_771_955e-3
SUN_LUMINOSITY = 3.828e26
SUN_RADIUS = 6.957e8
PARSEC = 3.085_677_581e16

PARTICLES = [
    ("proton", 1.672_621_923_69e-27), ("electron", 9.109_383_7015e-31),
    ("muon", 1.883_531_627e-28), ("alpha particle", 6.644_657_3357e-27),
]


def _finite_dict(d: dict[str, float]) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in d.values())


# ─── Domain generators ───────────────────────────────────────────────────────
# Each returns (query, record) where record has every ScientificAnalysis field
# except "executable_python_code" runs and its outputs are verified before
# the record is kept.

def gen_special_relativity(rng: random.Random) -> dict:
    name, m0 = rng.choice(PARTICLES)
    beta = rng.choice([0.9, 0.95, 0.99, 0.995, 0.999])
    query = (
        f"A {name} (rest mass m0 = {m0} kg) is accelerated to v = {beta}c. "
        "Using special relativity, compute: the Lorentz factor, the relativistic "
        "kinetic energy in Joules, the kinetic energy in MeV, the ratio of kinetic "
        "to rest-mass energy, and the total relativistic energy in Joules. Return "
        "a dict with descriptive keys including units."
    )
    code = f"""def calculate_research():
    m0 = {m0!r}
    beta = {beta!r}
    c = {C_LIGHT!r}
    e = {ELEMENTARY_CHARGE!r}
    gamma = 1.0 / (1.0 - beta ** 2) ** 0.5
    kinetic_energy_joules = (gamma - 1.0) * m0 * c ** 2
    kinetic_energy_mev = kinetic_energy_joules / e / 1e6
    ratio_ke_to_rest_energy_dimensionless = gamma - 1.0
    total_relativistic_energy_joules = gamma * m0 * c ** 2
    return {{
        "lorentz_factor_dimensionless": gamma,
        "relativistic_kinetic_energy_joules": kinetic_energy_joules,
        "kinetic_energy_mev": kinetic_energy_mev,
        "ratio_ke_to_rest_energy_dimensionless": ratio_ke_to_rest_energy_dimensionless,
        "total_relativistic_energy_joules": total_relativistic_energy_joules,
    }}
"""
    return {
        "query": query,
        "scientific_domain": "special_relativity",
        "symbolic_representation": r"$\gamma = \frac{1}{\sqrt{1-\beta^2}},\quad K = (\gamma-1)m_0c^2$",
        "variables": {"m_0": "rest mass [kg]", "beta": "velocity as fraction of c [dimensionless]", "c": "speed of light [m/s]"},
        "expected_result_keys": [
            "lorentz_factor_dimensionless", "relativistic_kinetic_energy_joules",
            "kinetic_energy_mev", "ratio_ke_to_rest_energy_dimensionless",
            "total_relativistic_energy_joules",
        ],
        "complexity_level": "undergraduate",
        "executable_python_code": code,
    }


def gen_thermodynamics(rng: random.Random) -> dict:
    n = rng.choice([1, 2, 5, 10])
    T = rng.choice([250, 300, 373, 500])
    P_atm = rng.choice([1, 2, 5])
    query = (
        f"An ideal gas has {n} moles at a temperature of {T} K and pressure of "
        f"{P_atm} atm. Compute the volume in cubic meters, the internal energy "
        "assuming a monatomic gas, the root-mean-square speed of the gas "
        "molecules (molar mass 0.032 kg/mol for O2), and the entropy change if "
        "the gas is isothermally expanded to double its volume. Return a dict "
        "with descriptive keys including units."
    )
    code = f"""def calculate_research():
    import math
    n = {n!r}
    T = {T!r}
    P = {P_atm!r} * 101325.0
    R = {GAS_CONSTANT!r}
    M = 0.032
    volume_cubic_meters = n * R * T / P
    internal_energy_joules = 1.5 * n * R * T
    rms_speed_meters_per_second = (3.0 * R * T / M) ** 0.5
    entropy_change_joules_per_kelvin = n * R * math.log(2.0)
    return {{
        "volume_cubic_meters": volume_cubic_meters,
        "internal_energy_joules": internal_energy_joules,
        "rms_speed_meters_per_second": rms_speed_meters_per_second,
        "entropy_change_joules_per_kelvin": entropy_change_joules_per_kelvin,
    }}
"""
    return {
        "query": query,
        "scientific_domain": "thermodynamics",
        "symbolic_representation": r"$PV = nRT,\quad U = \tfrac{3}{2}nRT,\quad \Delta S = nR\ln(V_2/V_1)$",
        "variables": {"n": "amount of substance [mol]", "T": "temperature [K]", "P": "pressure [Pa]", "R": "gas constant [J/(mol K)]"},
        "expected_result_keys": [
            "volume_cubic_meters", "internal_energy_joules",
            "rms_speed_meters_per_second", "entropy_change_joules_per_kelvin",
        ],
        "complexity_level": "undergraduate",
        "executable_python_code": code,
    }


def gen_quantum_mechanics(rng: random.Random) -> dict:
    m = rng.choice([9.109e-31, 1.673e-27])
    L = rng.choice([1e-9, 1e-10, 5e-10])
    n2 = rng.choice([2, 3])
    query = (
        f"A particle of mass {m} kg is confined to an infinite square well of "
        f"width {L} m. Compute the ground state energy in Joules and in eV, the "
        f"energy of the first excited state ({n2} quantum number), the de Broglie "
        "wavelength at the ground state, and the probability of finding the "
        "particle in the central third of the well for the ground state. Return "
        "a dict with descriptive keys including units."
    )
    code = f"""def calculate_research():
    import math
    m = {m!r}
    L = {L!r}
    n2 = {n2!r}
    h = {PLANCK!r}
    e = {ELEMENTARY_CHARGE!r}
    ground_state_energy_joules = (h ** 2 * 1 ** 2) / (8.0 * m * L ** 2)
    ground_state_energy_ev = ground_state_energy_joules / e
    excited_state_energy_joules = (h ** 2 * n2 ** 2) / (8.0 * m * L ** 2)
    momentum = (2.0 * m * ground_state_energy_joules) ** 0.5
    de_broglie_wavelength_meters = h / momentum
    probability_central_third_dimensionless = 1.0 / 3.0 + math.sqrt(3.0) / (2.0 * math.pi)
    return {{
        "ground_state_energy_joules": ground_state_energy_joules,
        "ground_state_energy_ev": ground_state_energy_ev,
        "excited_state_energy_joules": excited_state_energy_joules,
        "de_broglie_wavelength_meters": de_broglie_wavelength_meters,
        "probability_central_third_dimensionless": probability_central_third_dimensionless,
    }}
"""
    return {
        "query": query,
        "scientific_domain": "quantum_mechanics",
        "symbolic_representation": r"$E_n = \frac{n^2h^2}{8mL^2},\quad \lambda_{dB} = \frac{h}{p}$",
        "variables": {"m": "particle mass [kg]", "L": "well width [m]", "n": "quantum number [dimensionless]", "h": "Planck constant [J s]"},
        "expected_result_keys": [
            "ground_state_energy_joules", "ground_state_energy_ev",
            "excited_state_energy_joules", "de_broglie_wavelength_meters",
            "probability_central_third_dimensionless",
        ],
        "complexity_level": "undergraduate",
        "executable_python_code": code,
    }


def gen_electromagnetism(rng: random.Random) -> dict:
    q1 = rng.choice([1e-6, 2e-6, 5e-6])
    q2 = rng.choice([-1e-6, 3e-6])
    r = rng.choice([0.01, 0.05, 0.1])
    query = (
        f"Two point charges of {q1} C and {q2} C are separated by {r} m in "
        "vacuum. Compute the Coulomb force magnitude in Newtons, the electric "
        "potential energy in Joules, the electric field magnitude at the "
        "midpoint between them, and the electric potential at that midpoint. "
        "Return a dict with descriptive keys including units."
    )
    code = f"""def calculate_research():
    q1 = {q1!r}
    q2 = {q2!r}
    r = {r!r}
    k = {COULOMB_K!r}
    coulomb_force_newtons = abs(k * q1 * q2 / r ** 2)
    electric_potential_energy_joules = k * q1 * q2 / r
    d_mid = r / 2.0
    electric_field_magnitude_at_midpoint = abs(k * (q1 - q2) / d_mid ** 2)
    electric_potential_at_midpoint_volts = k * (q1 + q2) / d_mid
    return {{
        "coulomb_force_newtons": coulomb_force_newtons,
        "electric_potential_energy_joules": electric_potential_energy_joules,
        "electric_field_magnitude_at_midpoint": electric_field_magnitude_at_midpoint,
        "electric_potential_at_midpoint_volts": electric_potential_at_midpoint_volts,
    }}
"""
    return {
        "query": query,
        "scientific_domain": "electromagnetism",
        "symbolic_representation": r"$F = \frac{kq_1q_2}{r^2},\quad U = \frac{kq_1q_2}{r},\quad E = \frac{kq}{d^2},\quad V = \frac{kq}{d}$",
        "variables": {"q_1": "first point charge [C]", "q_2": "second point charge [C]", "r": "separation [m]", "k": "Coulomb constant [N m^2/C^2]"},
        "expected_result_keys": [
            "coulomb_force_newtons", "electric_potential_energy_joules",
            "electric_field_magnitude_at_midpoint", "electric_potential_at_midpoint_volts",
        ],
        "complexity_level": "undergraduate",
        "executable_python_code": code,
    }


def gen_fluid_dynamics(rng: random.Random) -> dict:
    D = rng.choice([0.02, 0.05, 0.1])
    v = rng.choice([0.5, 1.0, 2.0])
    query = (
        "Water (density 1000 kg/m^3, viscosity 1.0e-3 Pa.s) flows through a "
        f"pipe of diameter {D} m at a velocity of {v} m/s. Compute the Reynolds "
        "number, classify the flow regime via a numeric flag (0=laminar, "
        "1=turbulent), the volumetric flow rate in cubic meters per second, and "
        "the dynamic pressure. Return a dict with descriptive keys including "
        "units."
    )
    code = f"""def calculate_research():
    import math
    rho = 1000.0
    mu = 1.0e-3
    D = {D!r}
    v = {v!r}
    reynolds_number_dimensionless = rho * v * D / mu
    flow_regime_flag = 1.0 if reynolds_number_dimensionless > 2300.0 else 0.0
    volumetric_flow_rate_cubic_meters_per_second = v * math.pi * (D / 2.0) ** 2
    dynamic_pressure_pascals = 0.5 * rho * v ** 2
    return {{
        "reynolds_number_dimensionless": reynolds_number_dimensionless,
        "flow_regime_flag": flow_regime_flag,
        "volumetric_flow_rate_cubic_meters_per_second": volumetric_flow_rate_cubic_meters_per_second,
        "dynamic_pressure_pascals": dynamic_pressure_pascals,
    }}
"""
    return {
        "query": query,
        "scientific_domain": "fluid_dynamics",
        "symbolic_representation": r"$Re = \frac{\rho v D}{\mu},\quad Q = vA,\quad q = \tfrac{1}{2}\rho v^2$",
        "variables": {"rho": "fluid density [kg/m^3]", "v": "flow velocity [m/s]", "D": "pipe diameter [m]", "mu": "dynamic viscosity [Pa s]"},
        "expected_result_keys": [
            "reynolds_number_dimensionless", "flow_regime_flag",
            "volumetric_flow_rate_cubic_meters_per_second", "dynamic_pressure_pascals",
        ],
        "complexity_level": "undergraduate",
        "executable_python_code": code,
    }


def gen_astrophysics(rng: random.Random) -> dict:
    T = rng.choice([5778, 9000, 3500])
    R_mult = rng.choice([1, 2, 0.5])
    query = (
        f"A star has a surface temperature of {T} K and a radius of {R_mult} "
        "times the Sun's radius (6.957e8 m). Using the Stefan-Boltzmann law, "
        "compute the luminosity in Watts, the peak emission wavelength via "
        "Wien's law, the luminosity ratio relative to the Sun (3.828e26 W), and "
        "the apparent brightness at a distance of 10 parsecs. Return a dict "
        "with descriptive keys including units."
    )
    code = f"""def calculate_research():
    import math
    T = {T!r}
    R = {R_mult!r} * {SUN_RADIUS!r}
    sigma = {STEFAN_BOLTZMANN!r}
    wien_b = {WIEN_B!r}
    sun_luminosity = {SUN_LUMINOSITY!r}
    d = 10.0 * {PARSEC!r}
    luminosity_watts = 4.0 * math.pi * R ** 2 * sigma * T ** 4
    peak_wavelength_meters = wien_b / T
    luminosity_ratio_to_sun_dimensionless = luminosity_watts / sun_luminosity
    apparent_brightness_watts_per_square_meter = luminosity_watts / (4.0 * math.pi * d ** 2)
    return {{
        "luminosity_watts": luminosity_watts,
        "peak_wavelength_meters": peak_wavelength_meters,
        "luminosity_ratio_to_sun_dimensionless": luminosity_ratio_to_sun_dimensionless,
        "apparent_brightness_watts_per_square_meter": apparent_brightness_watts_per_square_meter,
    }}
"""
    return {
        "query": query,
        "scientific_domain": "astrophysics",
        "symbolic_representation": r"$L = 4\pi R^2\sigma T^4,\quad \lambda_{peak} = b/T,\quad F = \frac{L}{4\pi d^2}$",
        "variables": {"T": "surface temperature [K]", "R": "stellar radius [m]", "sigma": "Stefan-Boltzmann constant [W/(m^2 K^4)]", "d": "distance [m]"},
        "expected_result_keys": [
            "luminosity_watts", "peak_wavelength_meters",
            "luminosity_ratio_to_sun_dimensionless", "apparent_brightness_watts_per_square_meter",
        ],
        "complexity_level": "undergraduate",
        "executable_python_code": code,
    }


GENERATORS = [
    gen_special_relativity, gen_thermodynamics, gen_quantum_mechanics,
    gen_electromagnetism, gen_fluid_dynamics, gen_astrophysics,
]


def _run_and_check(code: str) -> dict[str, float] | None:
    """Exec our own generated code (trusted — we authored it, not an LLM) and
    sanity-check the output. Returns the result dict, or None if it failed."""
    ns: dict = {}
    try:
        exec(code, ns)  # noqa: S102 — trusted, locally authored code
        result = ns["calculate_research"]()
        if not isinstance(result, dict) or not _finite_dict(result):
            return None
        return {k: float(v) for k, v in result.items()}
    except Exception:
        return None


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


def generate(num_queries: int, out_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    already_done = _load_already_done(out_path)

    kept, skipped, failed = 0, 0, 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as out_f:
        for i in range(num_queries):
            record = rng.choice(GENERATORS)(rng)
            if record["query"] in already_done:
                skipped += 1
                continue

            result = _run_and_check(record["executable_python_code"])
            if result is None:
                failed += 1
                continue

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            kept += 1

            if (i + 1) % 200 == 0:
                print(f"progress: {i+1}/{num_queries}  kept={kept}  failed={failed}  skipped={skipped}")

    print(f"Done. kept={kept}  failed={failed}  skipped={skipped}  -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the Phase B fine-tune dataset with hand-written physics formulas (no Anthropic dependency)")
    p.add_argument("--out-path", type=Path, required=True)
    p.add_argument("--num-queries", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    generate(args.num_queries, args.out_path, args.seed)


if __name__ == "__main__":
    main()
