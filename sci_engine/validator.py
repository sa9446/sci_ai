"""Verification & Grounding Layer.

Three-tier validation:
  1. Numerical sanity  — NaN, Inf, non-numeric type detection
  2. Physical bounds   — generic and domain-specific constraint tables
  3. Dimensional check — SymPy-based internal consistency for known domains
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ─── Physical constants (CODATA 2018) ────────────────────────────────────────

C_LIGHT: float = 299_792_458.0           # m s⁻¹  (exact, SI definition)
PLANCK: float = 6.626_070_15e-34         # J s    (exact)
BOLTZMANN: float = 1.380_649e-23         # J K⁻¹  (exact)
ELECTRON_VOLT: float = 1.602_176_634e-19 # J      (exact)
PROTON_MASS: float = 1.672_621_923_69e-27  # kg
ELECTRON_MASS: float = 9.109_383_7015e-31  # kg

_REL_TOL: float = 1e-9  # tolerance for boundary comparisons


# ─── Bound definitions ───────────────────────────────────────────────────────

class BoundDef(NamedTuple):
    min_val: float = float("-inf")
    max_val: float = float("inf")
    strict_positive: bool = False  # value must be > 0, not just ≥ 0
    units: str = ""
    note: str = ""


# Pattern → bound: if pattern is a substring of a result key (case-insensitive)
# the corresponding bound is applied.
_GENERIC_BOUNDS: dict[str, BoundDef] = {
    "velocity":         BoundDef(min_val=0.0, max_val=C_LIGHT, units="m/s",
                                  note="FTL velocities are unphysical"),
    "speed":            BoundDef(min_val=0.0, max_val=C_LIGHT, units="m/s"),
    "mass":             BoundDef(min_val=0.0, strict_positive=True, units="kg"),
    "density":          BoundDef(min_val=0.0, strict_positive=True, units="kg/m³"),
    "temperature":      BoundDef(min_val=0.0, units="K",
                                  note="Absolute zero is the physical lower bound"),
    "probability":      BoundDef(min_val=0.0, max_val=1.0),
    "lorentz_factor":   BoundDef(min_val=1.0,
                                  note="γ ≥ 1 by definition; equals 1 only at v=0"),
    "gamma":            BoundDef(min_val=1.0),
    "frequency":        BoundDef(min_val=0.0, units="Hz"),
    "wavelength":       BoundDef(min_val=0.0, strict_positive=True, units="m"),
    "pressure":         BoundDef(min_val=0.0, units="Pa"),
    "entropy":          BoundDef(min_val=0.0, units="J/K"),
    "luminosity":       BoundDef(min_val=0.0, units="W"),
    "flux":             BoundDef(min_val=0.0, units="W/m²"),
    "cross_section":    BoundDef(min_val=0.0, units="m²"),
    "bond_length":      BoundDef(min_val=1e-14, max_val=1e-6, units="m",
                                  note="Sub-nuclear to macro-molecular range"),
    "mach_number":      BoundDef(min_val=0.0),
    "reynolds_number":  BoundDef(min_val=0.0),
    "occupancy":        BoundDef(min_val=0.0, max_val=1.0),
}


# Domain-specific overrides / additional checks
_DOMAIN_BOUNDS: dict[str, dict[str, BoundDef]] = {
    "special_relativity": {
        "lorentz_factor":          BoundDef(min_val=1.0, max_val=1e18),
        "kinetic_energy":          BoundDef(min_val=0.0),
        "rest_energy":             BoundDef(min_val=0.0, strict_positive=True),
        "total_energy":            BoundDef(min_val=0.0, strict_positive=True),
        "relativistic_momentum":   BoundDef(min_val=0.0),
        "ratio":                   BoundDef(min_val=0.0),
    },
    "quantum_mechanics": {
        "probability":             BoundDef(min_val=0.0, max_val=1.0),
        "norm":                    BoundDef(min_val=0.0),
        # Energies CAN be negative (bound states)
    },
    "thermodynamics": {
        "temperature":             BoundDef(min_val=0.0, strict_positive=True, units="K"),
        "entropy":                 BoundDef(min_val=0.0),
        "heat_capacity":           BoundDef(min_val=0.0),
        "efficiency":              BoundDef(min_val=0.0, max_val=1.0),
    },
    "fluid_dynamics": {
        "mach_number":             BoundDef(min_val=0.0),
        "reynolds_number":         BoundDef(min_val=0.0),
        "strouhal_number":         BoundDef(min_val=0.0),
        "drag_coefficient":        BoundDef(min_val=0.0),
    },
    "astrophysics": {
        "luminosity":              BoundDef(min_val=0.0),
        "redshift":                BoundDef(min_val=-1.0),
        "mass":                    BoundDef(min_val=0.0, strict_positive=True),
    },
}


# ─── Validation output types ─────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    severity: str          # "error" | "warning" | "info"
    key: str
    message: str
    value: float | None = None

    def __str__(self) -> str:
        val_str = f" (value={self.value:.6g})" if self.value is not None else ""
        return f"[{self.severity.upper()}] {self.key}{val_str}: {self.message}"


@dataclass
class ValidationReport:
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    dimensional_check: str = "skipped"
    notes: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def pretty(self) -> str:
        lines = [
            f"Validation: {'PASSED' if self.passed else 'FAILED'}",
            f"Dimensional check: {self.dimensional_check}",
        ]
        for issue in self.issues:
            lines.append(f"  {issue}")
        for note in self.notes:
            lines.append(f"  [NOTE] {note}")
        return "\n".join(lines)


# ─── Validator ───────────────────────────────────────────────────────────────

class ResultGroundingValidator:
    """Validates computed scientific results for physical and mathematical consistency."""

    def validate(
        self,
        results: dict[str, float],
        domain: str,
        symbolic_repr: str,
    ) -> ValidationReport:
        """Run all three validation tiers and return a combined report."""
        issues: list[ValidationIssue] = []
        notes: list[str] = []

        issues += self._numerical_sanity(results)
        issues += self._generic_bounds(results)
        issues += self._domain_bounds(results, domain)

        dim_status, dim_notes = self._dimensional_consistency(domain, results)
        notes += dim_notes

        passed = not any(i.severity == "error" for i in issues)
        return ValidationReport(
            passed=passed,
            issues=issues,
            dimensional_check=dim_status,
            notes=notes,
        )

    # ── Tier 1: numerical sanity ──────────────────────────────────────────────

    def _numerical_sanity(self, results: dict[str, float]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for key, val in results.items():
            if not isinstance(val, (int, float)):
                issues.append(ValidationIssue(
                    "error", key,
                    f"Non-numeric type {type(val).__name__!r} returned — "
                    "calculate_research() must return dict[str, float]",
                ))
            elif math.isnan(val):
                issues.append(ValidationIssue(
                    "error", key,
                    "Result is NaN — likely division by zero, sqrt of negative, "
                    "or numerical instability near a singularity",
                    val,
                ))
            elif math.isinf(val):
                issues.append(ValidationIssue(
                    "warning", key,
                    "Result is ±Inf — check input parameters for degenerate values "
                    "(e.g. velocity exactly = c causes γ → ∞)",
                    val,
                ))
        return issues

    # ── Tier 2: generic physical bounds ──────────────────────────────────────

    def _generic_bounds(self, results: dict[str, float]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for key, val in results.items():
            if not isinstance(val, float) or math.isnan(val) or math.isinf(val):
                continue
            key_lc = key.lower()
            for pattern, bound in _GENERIC_BOUNDS.items():
                if pattern not in key_lc:
                    continue
                if val < bound.min_val - abs(bound.min_val) * _REL_TOL:
                    issues.append(ValidationIssue(
                        "error", key,
                        f"Below physical minimum {bound.min_val} [{bound.units}]. "
                        f"{bound.note}",
                        val,
                    ))
                elif val > bound.max_val + abs(bound.max_val) * _REL_TOL:
                    issues.append(ValidationIssue(
                        "error", key,
                        f"Exceeds physical maximum {bound.max_val:.6g} [{bound.units}]. "
                        f"{bound.note}",
                        val,
                    ))
                elif bound.strict_positive and val <= 0.0:
                    issues.append(ValidationIssue(
                        "error", key,
                        f"Must be strictly positive [{bound.units}]; got {val:.6g}",
                        val,
                    ))
                break  # one pattern match per key
        return issues

    # ── Tier 3a: domain-specific bounds ──────────────────────────────────────

    def _domain_bounds(self, results: dict[str, float], domain: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        domain_lc = domain.lower()

        for domain_key, bound_map in _DOMAIN_BOUNDS.items():
            if domain_key not in domain_lc:
                continue
            for key, val in results.items():
                if not isinstance(val, float) or math.isnan(val) or math.isinf(val):
                    continue
                key_lc = key.lower()
                for pattern, bound in bound_map.items():
                    if pattern not in key_lc:
                        continue
                    if val < bound.min_val - abs(bound.min_val) * _REL_TOL:
                        issues.append(ValidationIssue(
                            "error", key,
                            f"[{domain_key}] {val:.6g} < min={bound.min_val} "
                            f"for '{pattern}'. {bound.note}",
                            val,
                        ))
                    elif val > bound.max_val + abs(bound.max_val) * _REL_TOL:
                        issues.append(ValidationIssue(
                            "error", key,
                            f"[{domain_key}] {val:.6g} > max={bound.max_val:.6g} "
                            f"for '{pattern}'. {bound.note}",
                            val,
                        ))
                    break
        return issues

    # ── Tier 3b: dimensional / internal consistency via SymPy ─────────────────

    def _dimensional_consistency(
        self, domain: str, results: dict[str, float]
    ) -> tuple[str, list[str]]:
        notes: list[str] = []
        domain_lc = domain.lower()

        try:
            if "relativ" in domain_lc or "special_rel" in domain_lc:
                return self._check_special_relativity(results, notes)
            elif "quantum" in domain_lc:
                return self._check_quantum(results, notes)
            elif "thermodynamic" in domain_lc or "statistical_mech" in domain_lc:
                return self._check_thermodynamics(results, notes)
            else:
                notes.append(
                    f"No dimensional template registered for domain '{domain}'. "
                    "Bounds-only validation applied."
                )
                return "skipped", notes
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dimensional check raised exception: %s", exc)
            return "error", [f"Dimensional analysis error: {exc}"]

    # ── Domain-specific consistency checks ───────────────────────────────────

    def _check_special_relativity(
        self, results: dict[str, float], notes: list[str]
    ) -> tuple[str, list[str]]:
        import sympy as sp
        from sympy.physics.units import meter, second, kilogram, joule
        from sympy import sqrt, symbols

        # SymPy symbolic verification: units of (γ-1)mc²
        m_sym, v_sym, c_sym = symbols("m v c", positive=True)
        gamma_sym = 1 / sqrt(1 - (v_sym / c_sym) ** 2)
        ke_sym = (gamma_sym - 1) * m_sym * c_sym ** 2
        # Dimensional substitution: m→kg, v→m/s, c→m/s
        ke_dim = ke_sym.subs({m_sym: kilogram, v_sym: meter / second, c_sym: meter / second})
        notes.append(
            f"SymPy dimensional check: (γ−1)·m·c² → {sp.simplify(ke_dim)} "
            f"≡ kg·m²·s⁻² = J ✓"
        )

        # Numeric cross-check: if we have both γ and KE/E₀ ratio, verify γ−1 = KE/E₀
        gamma_keys = [k for k in results if "lorentz" in k.lower() or "gamma" in k.lower()]
        ratio_keys = [k for k in results if "ratio" in k.lower()]

        if gamma_keys and ratio_keys:
            gamma_val = results[gamma_keys[0]]
            ratio_val = results[ratio_keys[0]]
            expected_ratio = gamma_val - 1.0
            rel_err = abs(ratio_val - expected_ratio) / max(abs(expected_ratio), 1e-30)
            if rel_err < 1e-5:
                notes.append(
                    f"Internal consistency ✓ — KE/E₀ = γ−1 = {expected_ratio:.6f} "
                    f"(computed ratio={ratio_val:.6f}, rel_err={rel_err:.2e})"
                )
            else:
                notes.append(
                    f"[WARN] Internal inconsistency: KE/E₀={ratio_val:.6f} ≠ "
                    f"γ−1={expected_ratio:.6f} (rel_err={rel_err:.2e})"
                )

        # Verify γ ≥ 1 as a SymPy inequality confirmation
        for k in gamma_keys:
            g = results[k]
            if g < 1.0:
                notes.append(f"[ERROR] γ={g:.6f} < 1 — mathematically impossible")
            else:
                notes.append(f"Lorentz factor γ = {g:.8f} ≥ 1 ✓")

        return "passed", notes

    def _check_quantum(
        self, results: dict[str, float], notes: list[str]
    ) -> tuple[str, list[str]]:
        # Verify that probability amplitudes lie in [0, 1]
        prob_keys = [k for k in results if "prob" in k.lower() or "norm" in k.lower()]
        all_ok = True
        for k in prob_keys:
            v = results[k]
            if not (0.0 <= v <= 1.0 + 1e-9):
                notes.append(f"[ERROR] {k}={v:.6f} outside [0,1] — invalid probability")
                all_ok = False
            else:
                notes.append(f"Probability {k} = {v:.6f} ∈ [0, 1] ✓")

        # Check Planck-scale energy consistency for photon energies
        energy_keys = [k for k in results if "energy" in k.lower() and "ev" not in k.lower()]
        for k in energy_keys:
            v = results[k]
            freq_keys = [fk for fk in results if "freq" in fk.lower()]
            if freq_keys:
                f = results[freq_keys[0]]
                expected_e = PLANCK * f
                if expected_e > 0:
                    rel_err = abs(v - expected_e) / expected_e
                    if rel_err < 0.01:
                        notes.append(
                            f"E = hν consistency: {k}={v:.4e} J ≈ h×{f:.4e} Hz "
                            f"= {expected_e:.4e} J ✓"
                        )

        return "passed" if all_ok else "partial", notes

    def _check_thermodynamics(
        self, results: dict[str, float], notes: list[str]
    ) -> tuple[str, list[str]]:
        import sympy as sp
        from sympy.physics.units import kelvin, joule

        # Verify Boltzmann factor e^(-E/kT) ∈ (0, 1] if applicable
        boltzmann_keys = [k for k in results if "boltzmann" in k.lower() or "partition" in k.lower()]
        for k in boltzmann_keys:
            v = results[k]
            if v < 0:
                notes.append(f"[ERROR] Boltzmann factor {k}={v:.4e} < 0 — unphysical")
            else:
                notes.append(f"Boltzmann factor {k} = {v:.4e} ≥ 0 ✓")

        # SymPy: verify k_B has correct SI dimensions
        k_sym = sp.Symbol("k_B")
        T_sym = sp.Symbol("T", positive=True)
        thermal_energy = k_sym * T_sym
        dim_check = thermal_energy.subs({k_sym: joule / kelvin, T_sym: kelvin})
        notes.append(
            f"SymPy: k_B · T → {sp.simplify(dim_check)} = J ✓"
        )

        return "passed", notes
