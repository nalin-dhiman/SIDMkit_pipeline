"""Velocity distributions used for velocity-averaging SIDM observables.

This module is intentionally small and explicit.

In many SIDM applications the *relative speed* distribution matters, because the
per-particle scattering rate involves the average ⟨σ(v_rel) v_rel⟩.

For an isotropic Maxwellian single-particle velocity distribution with 1D
dispersion σ_1D, the relative speed v_rel = |v1 - v2| follows a Maxwell speed
distribution with component dispersion √2 σ_1D.

We implement the relative-speed PDF and a few analytic moments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erf, gammainc, gamma


@dataclass(frozen=True)
class MaxwellRelativeDistribution:
    """Relative-speed Maxwellian distribution.

    Parameters
    ----------
    sigma_1d_km_s:
        One-dimensional velocity dispersion (km/s) of the *single-particle* distribution.
        The relative-speed distribution is derived assuming both particles are drawn from
        this same Maxwellian.
    v_max_km_s:
        Optional truncation of the relative-speed distribution at v_max.
        This is a crude but sometimes useful approximation to a truncated halo distribution.
        If provided, the PDF is renormalized on [0, v_max].

    Notes
    -----
    Untruncated relative-speed PDF:

        f(v) = (1 / (2 √π σ^3)) v^2 exp[- v^2 / (4 σ^2)] ,  v ≥ 0

    where σ = sigma_1d_km_s.

    This is normalized: ∫_0^∞ f(v) dv = 1.

    The n-th moment (untruncated) is

        ⟨v^n⟩ = (2/√π) (2σ)^n Γ((n+3)/2).

    The mean is ⟨v⟩ = 4σ/√π.
    """

    sigma_1d_km_s: float
    v_max_km_s: Optional[float] = None

    def __post_init__(self) -> None:
        if not (self.sigma_1d_km_s > 0):
            raise ValueError("sigma_1d_km_s must be > 0")
        if self.v_max_km_s is not None and not (self.v_max_km_s > 0):
            raise ValueError("v_max_km_s must be > 0 when provided")

    @property
    def sigma(self) -> float:
        return float(self.sigma_1d_km_s)

    def _xmax(self) -> Optional[float]:
        if self.v_max_km_s is None:
            return None
        return (self.v_max_km_s / (2.0 * self.sigma)) ** 2

    def normalization(self) -> float:
        """Return normalization Z = ∫ f_untr(v) dv over support.

        For the untruncated distribution this is 1. For the truncated distribution it is
        erf(a) - (2a/√π) exp(-a^2) with a=v_max/(2σ).
        """
        if self.v_max_km_s is None:
            return 1.0
        a = self.v_max_km_s / (2.0 * self.sigma)
        return float(erf(a) - (2.0 * a / np.sqrt(np.pi)) * np.exp(-a * a))

    def pdf(self, v_km_s: ArrayLike) -> np.ndarray:
        """Relative-speed PDF f(v) in (km/s)^-1.

        Returns an array with the same shape as v_km_s.
        """
        v = np.asarray(v_km_s, dtype=float)
        sigma = self.sigma
        f = (1.0 / (2.0 * np.sqrt(np.pi) * sigma**3)) * v**2 * np.exp(-(v**2) / (4.0 * sigma**2))
        if self.v_max_km_s is not None:
            f = np.where(v <= self.v_max_km_s, f, 0.0)
            Z = self.normalization()
            f = f / Z
        return f

    def moment(self, n: float) -> float:
        """Return ⟨v^n⟩ for the (possibly truncated) distribution.

        For non-integer n this is still well-defined as long as n > -3.
        """
        if n <= -3:
            raise ValueError("Moment requires n > -3 for MaxwellRelativeDistribution.")
        sigma = self.sigma
        pref = (2.0 / np.sqrt(np.pi)) * (2.0 * sigma) ** n
        s = 0.5 * (n + 3.0)
        if self.v_max_km_s is None:
            return float(pref * gamma(s))
        # truncated: use lower incomplete gamma
        xmax = float(self._xmax())
        lower_inc = gammainc(s, xmax) * gamma(s)
        Z = self.normalization()
        return float(pref * lower_inc / Z)

    def mean_speed(self) -> float:
        return self.moment(1.0)

    def rms_speed(self) -> float:
        return float(np.sqrt(self.moment(2.0)))

