"""Lightweight internal benchmarks.

These checks are *smoke tests*, not a physics certification.

What this is for
----------------
- Catch obvious regressions: unit conversions, distribution normalization, and
  averaging integrator agreement.
- Optionally ("--slow"), compare the fast approximation regime selector
  (method="auto") against a partial-wave calculation on a few hand-picked
  parameter points where the approximations are expected to behave reasonably.

What this is *not* for
----------------------
- Proving correctness in the resonant/intermediate regime.
- Guaranteeing that every point in parameter space will be numerically stable.

If you need publication-grade accuracy in a tricky region, use
method="partial_wave" (spot checks) and/or independent cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .averaging import average_sigma_over_m
from .distributions import MaxwellRelativeDistribution
from .model import PotentialType, YukawaModel
from .rates import scattering_rate
from .cross_sections import sigma_over_m

@dataclass
class BenchmarkSummary:
    """Container for benchmark outputs."""

    results: Dict[str, Any]


def _safe_rel_err(a: float, b: float, floor: float = 1e-300) -> float:
    """Relative error |a-b|/max(|b|, floor)."""
    den = max(abs(b), floor)
    return float(abs(a - b) / den)


def run_benchmarks(include_slow: bool = False) -> BenchmarkSummary:
    """Run the built-in benchmark suite.

    Parameters
    ----------
    include_slow:
        If True, also run a partial-wave comparison on a small set of points.

    Returns
    -------
    JSON-serializable dict of benchmark results.
    """

    out: Dict[str, Any] = {}

    # --- (1) Rate conversion sanity check ---
    gamma = scattering_rate(
        rho=0.01,
        sigma_over_m_cm2_g=1.0,
        v_km_s=50.0,
        rho_unit="Msun/pc3",
        out_unit="1/Gyr",
    )
    out["rate_check_Gamma_Gyr_inv"] = float(gamma)
    out["rate_check_pass"] = bool(0.08 < gamma < 0.14)

    # --- (2) Relative-speed Maxwell distribution sanity ---
    dist = MaxwellRelativeDistribution(sigma_1d_km_s=100.0)
    out["maxwell_rel_norm"] = float(dist.normalization())
    out["maxwell_rel_mean"] = float(dist.mean_speed())
    out["maxwell_rel_mean_pass"] = bool(abs(out["maxwell_rel_mean"] - 225.6758334191025) < 1e-6)

    # --- (3) Averaging integrator agreement ---
    m = YukawaModel(20.0, 0.05, 0.01, potential=PotentialType.ATTRACTIVE)
    avg_lag = average_sigma_over_m(m, sigma_1d_km_s=50.0, moment_n=1.0, integrator="laguerre")
    avg_quad = average_sigma_over_m(m, sigma_1d_km_s=50.0, moment_n=1.0, integrator="quad")
    out["avg_laguerre_value"] = float(avg_lag.value)
    out["avg_quad_value"] = float(avg_quad.value)
    rel = _safe_rel_err(avg_lag.value, avg_quad.value)
    out["avg_integrator_rel_diff"] = float(rel)
    out["avg_integrator_pass"] = bool(rel < 1e-6)

    # --- (4) Optional: auto vs partial-wave sanity points ---
    out["auto_vs_pw_sanity_rel_errs"] = None
    out["auto_vs_pw_sanity_pass"] = None
    out["auto_vs_pw_stress_rel_errs"] = None
    out["auto_vs_pw_stress_errors"] = None

    if include_slow:
        sanity_points: List[Tuple[str, YukawaModel, float]] = [
            ("Born-like", YukawaModel(10.0, 0.04, 0.01, potential=PotentialType.ATTRACTIVE), 30.0),
            ("Classical-like", YukawaModel(20.0, 0.005, 0.2, potential=PotentialType.ATTRACTIVE), 100.0),
            ("Resonant-like (Hulth\u00e9n)", YukawaModel(15.0, 0.02, 0.18, potential=PotentialType.ATTRACTIVE), 30.0),
        ]

        stress_points: List[Tuple[str, YukawaModel, float]] = [
            ("Stress: intermediate", YukawaModel(50.0, 0.02, 0.02, potential=PotentialType.ATTRACTIVE), 200.0),
            ("Stress: resonance spike", YukawaModel(15.0, 0.02, 0.2, potential=PotentialType.ATTRACTIVE), 30.0),
            ("Stress: high-v (Kaplinghat-like)", YukawaModel(15.0, 0.017, 1.0 / 137.0, potential=PotentialType.ATTRACTIVE), 1500.0),
        ]

        sanity_errs: List[float] = []
        sanity_fail_reasons: List[str] = []
        for name, model, v in sanity_points:
            try:
                s_auto = float(sigma_over_m(v, model=model, method="auto", out_unit="cm2/g"))
                s_pw = float(sigma_over_m(v, model=model, method="partial_wave", out_unit="cm2/g"))
                sanity_errs.append(_safe_rel_err(s_auto, s_pw))
            except Exception as e:  # pragma: no cover (best-effort slow path)
                sanity_fail_reasons.append(f"{name}: {type(e).__name__}: {e}")
                sanity_errs.append(float("nan"))

        out["auto_vs_pw_sanity_rel_errs"] = sanity_errs
        # Pass only if finite and not too large.
        finite = [x for x in sanity_errs if np.isfinite(x)]
        out["auto_vs_pw_sanity_pass"] = bool(
            (len(finite) == len(sanity_points)) and (max(finite, default=float("inf")) < 0.5)
        )
        if sanity_fail_reasons:
            out["auto_vs_pw_sanity_errors"] = sanity_fail_reasons

        stress_errs: List[Optional[float]] = []
        stress_errors: List[str] = []
        for name, model, v in stress_points:
            try:
                s_auto = float(sigma_over_m(v, model=model, method="auto", out_unit="cm2/g"))
                s_pw = float(sigma_over_m(v, model=model, method="partial_wave", out_unit="cm2/g"))
                stress_errs.append(_safe_rel_err(s_auto, s_pw))
            except Exception as e:  # pragma: no cover
                stress_errs.append(None)
                stress_errors.append(f"{name}: {type(e).__name__}: {e}")

        out["auto_vs_pw_stress_rel_errs"] = stress_errs
        out["auto_vs_pw_stress_errors"] = stress_errors or None

    return BenchmarkSummary(results=out)
