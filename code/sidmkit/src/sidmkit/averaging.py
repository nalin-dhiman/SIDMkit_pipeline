"""Velocity-averaging utilities for SIDM observables.

Why this exists
---------------
Astrophysical systems do not probe σ(v) at a single sharp speed. Even when a
paper quotes an “effective velocity,” what enters many observables is an
average like ⟨σ(v_rel) v_rel⟩ over a system-dependent relative-velocity
distribution.

This module implements a pragmatic, explicit baseline:
- isotropic Maxwellian relative-speed distribution (optionally truncated),
- Gauss–Laguerre quadrature for efficient untruncated Maxwell averages,
- adaptive quadrature fallback for other cases.

This is *not* a replacement for full halo modeling. It is a step up from
single-velocity evaluation and it is written so you can swap the distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad

from .cross_sections import sigma_over_m
from .distributions import MaxwellRelativeDistribution
from .model import YukawaModel


Integrator = Literal["auto", "laguerre", "quad"]
Method = Literal["auto", "born", "classical", "hulthen", "partial_wave"]


@dataclass(frozen=True)
class VelocityAverageResult:
    """Container for common velocity-averaged quantities."""

    moment_n: float
    value: float
    unit: str
    distribution: str
    sigma_1d_km_s: float
    v_max_km_s: Optional[float]


def _laguerre_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    # Lazy import to keep import time low.
    from numpy.polynomial.laguerre import laggauss

    x, w = laggauss(n)
    return x.astype(float), w.astype(float)


def average_sigma_over_m(
    model: YukawaModel,
    sigma_1d_km_s: float,
    moment_n: float = 0.0,
    *,
    v_max_km_s: Optional[float] = None,
    distribution: Literal["maxwell"] = "maxwell",
    method: Method = "auto",
    integrator: Integrator = "auto",
    n_laguerre: int = 64,
) -> VelocityAverageResult:
    """Compute ⟨ (σ/m)(v_rel) v_rel^n ⟩ for a chosen distribution.

    Parameters
    ----------
    model:
        Yukawa model parameters.
    sigma_1d_km_s:
        One-dimensional dispersion (km/s) of the underlying single-particle Maxwellian.
    moment_n:
        Power n in v_rel^n.

        - n=0 gives ⟨σ/m⟩ in cm^2/g
        - n=1 gives ⟨σ v⟩/m in (cm^2/g)*(km/s)
        - n=2 gives ⟨σ v^2⟩/m in (cm^2/g)*(km/s)^2
        etc.

    v_max_km_s:
        Optional cutoff on v_rel for a truncated Maxwellian.
        If provided, we renormalize the PDF on [0, v_max_km_s].
    method:
        Cross section computation method (see :func:`sidmkit.cross_sections.sigma_over_m`).
    integrator:
        - "laguerre" is fast but only valid for *untruncated* Maxwell.
        - "quad" is slower but robust for truncated cases.
        - "auto" chooses laguerre if v_max_km_s is None, else quad.
    n_laguerre:
        Number of Gauss–Laguerre nodes used when integrator="laguerre".
        64 is a good baseline for smooth integrands; increase for sharp resonances.

    Returns
    -------
    VelocityAverageResult
        value is in units of (cm^2/g)*(km/s)^n.

    Critical notes (be honest)
    --------------------------
    1) The distribution model matters. Maxwellian is a convenience, not a law of nature.
    2) If σ(v) has narrow resonances, any quadrature can miss them. In that case:
       - increase n_laguerre,
       - or use integrator="quad" with tightened tolerances,
       - or do brute-force integration on a custom grid.
    """
    if sigma_1d_km_s <= 0:
        raise ValueError("sigma_1d_km_s must be > 0")
    if n_laguerre < 16:
        raise ValueError("n_laguerre must be >= 16 for stability")

    dist = MaxwellRelativeDistribution(sigma_1d_km_s=float(sigma_1d_km_s), v_max_km_s=v_max_km_s)

    if integrator == "auto":
        integrator = "laguerre" if v_max_km_s is None else "quad"

    if integrator == "laguerre":
        if v_max_km_s is not None:
            raise ValueError("laguerre integrator is only valid for untruncated Maxwell (v_max_km_s=None)")
        # Using x = (v/(2σ))^2 mapping:
        #  f(v)dv  -> (2/√π) x^{1/2} e^{-x} dx
        x, w = _laguerre_nodes_weights(int(n_laguerre))
        v = 2.0 * dist.sigma * np.sqrt(x)  # km/s
        som = sigma_over_m(v, model=model, method=method, out_unit="cm2/g")  # cm^2/g
        integrand = (2.0 / np.sqrt(np.pi)) * np.sqrt(x) * som * (v**moment_n)
        val = float(np.sum(w * integrand))
    elif integrator == "quad":
        # Integrate in v space with PDF, with renormalization handled by dist.pdf.
        v_max = dist.v_max_km_s

        def integrand(v: float) -> float:
            som = float(sigma_over_m(v, model=model, method=method, out_unit="cm2/g"))
            return som * (v**moment_n) * float(dist.pdf(v))

        upper = float(v_max) if v_max is not None else np.inf
        # Tight-ish tolerances; users can copy and adjust for heavy inference.
        val, _ = quad(integrand, 0.0, upper, epsabs=0.0, epsrel=5e-4, limit=200)
        val = float(val)
    else:
        raise ValueError(f"Unknown integrator: {integrator}")

    unit = "cm^2/g" if moment_n == 0 else f"(cm^2/g)*(km/s)^{moment_n:g}"
    return VelocityAverageResult(
        moment_n=float(moment_n),
        value=val,
        unit=unit,
        distribution="maxwell",
        sigma_1d_km_s=float(sigma_1d_km_s),
        v_max_km_s=v_max_km_s,
    )


def effective_sigma_over_m_from_avg(
    avg_sigma_v_over_m: float,
    mean_v_rel_km_s: float,
) -> float:
    """Convert ⟨σ v⟩/m into an effective σ/m by dividing by ⟨v⟩.

    This is occasionally useful for comparison to papers that quote σ/m at a
    single “characteristic” velocity, but it is *not* a universal mapping.
    """
    if mean_v_rel_km_s <= 0:
        raise ValueError("mean_v_rel_km_s must be > 0")
    return float(avg_sigma_v_over_m / mean_v_rel_km_s)

