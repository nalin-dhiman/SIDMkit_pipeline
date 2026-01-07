"""Toy rotation-curve likelihood with an SIDM-inspired cored halo.

This is *not* a faithful reproduction of any single paper's full analysis.
It is meant to be a transparent, auditable demonstration of how to connect:

    particle model → σ(v) → (Γ t_age = 1) radius r1 → a cored halo profile → V_circ(r).

The core model
--------------
We use a simple pseudo-isothermal sphere in the inner region:

    ρ_iso(r) = ρ0 / (1 + (r/r_c)^2)

and match it to an NFW halo at r1 by requiring continuity of density and
enclosed mass at r1.

Outside r1 we keep the NFW mass profile, shifted to ensure mass continuity.

This is a *toy* semi-analytic model. It is useful for examples and unit tests,
but publication-grade rotation-curve fits generally need:
- baryonic potentials and contraction/expansion,
- a more careful SIDM gravothermal model,
- marginalization over M200, c200, and stellar M/L, etc.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from ..halo import G_KPC_KMS2_PER_MSUN, NFWHalo, r1_interaction_radius_kpc
from ..model import YukawaModel


@dataclass(frozen=True)
class RotationCurveData:
    r_kpc: np.ndarray
    v_obs_km_s: np.ndarray
    v_err_km_s: np.ndarray
    v_bar_km_s: Optional[np.ndarray] = None

    @staticmethod
    def from_csv(path: str) -> "RotationCurveData":
        r, v, e, vb = [], [], [], []
        has_vb = False
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r.append(float(row["r_kpc"]))
                v.append(float(row["v_obs_km_s"]))
                e.append(float(row["v_err_km_s"]))
                if "v_bar_km_s" in row and row["v_bar_km_s"] not in (None, ""):
                    has_vb = True
                    vb.append(float(row["v_bar_km_s"]))
        r = np.asarray(r, dtype=float)
        v = np.asarray(v, dtype=float)
        e = np.asarray(e, dtype=float)
        vbar = np.asarray(vb, dtype=float) if has_vb else None
        return RotationCurveData(r_kpc=r, v_obs_km_s=v, v_err_km_s=e, v_bar_km_s=vbar)


def _m_iso(r: ArrayLike, rho0: float, r_c: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    x = r / r_c
    return 4.0 * np.pi * rho0 * r_c**3 * (x - np.arctan(x))


def _solve_isothermal_core_params(halo: NFWHalo, r1_kpc: float) -> tuple[float, float]:
    """Match pseudo-isothermal core to NFW at r1.

    Returns
    -------
    rho0_msun_kpc3, r_c_kpc
    """
    r1 = float(r1_kpc)
    rho1 = float(halo.rho_msun_kpc3(r1))
    m1 = float(halo.m_enclosed_msun(r1))

    if rho1 <= 0 or m1 <= 0:
        raise ValueError("Invalid NFW values at r1")

    # Solve for x = r1/r_c
    def F(x: float) -> float:
        if x <= 0:
            return 1e99
        r_c = r1 / x
        rho0 = rho1 * (1.0 + x * x)
        m_iso_1 = float(_m_iso(r1, rho0=rho0, r_c=r_c))
        return m_iso_1 - m1

    # Bracket x. x small -> huge core (low mass); x large -> small core (high density)
    # We find a bracket by scanning.
    xs = np.logspace(-3, 3, 200)
    vals = np.array([F(x) for x in xs])
    # Find a sign change
    sign = np.sign(vals)
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    if idx.size == 0:
        # F might be always positive/negative due to the toy nature; fall back to x=1
        x_star = 1.0
    else:
        i0 = int(idx[0])
        x_star = brentq(F, float(xs[i0]), float(xs[i0 + 1]), maxiter=200)

    r_c = r1 / x_star
    rho0 = rho1 * (1.0 + x_star * x_star)
    return float(rho0), float(r_c)


def sidm_cored_mass_profile_msun(
    halo: NFWHalo,
    r_kpc: ArrayLike,
    *,
    r1_kpc: float,
) -> np.ndarray:
    """Hybrid mass profile: isothermal core inside r1, NFW outside (mass-matched)."""
    r = np.asarray(r_kpc, dtype=float)
    rho0, r_c = _solve_isothermal_core_params(halo, r1_kpc=r1_kpc)
    m1_nfw = float(halo.m_enclosed_msun(r1_kpc))
    m1_core = float(_m_iso(r1_kpc, rho0=rho0, r_c=r_c))
    m = np.zeros_like(r, dtype=float)
    inside = r <= r1_kpc
    m[inside] = _m_iso(r[inside], rho0=rho0, r_c=r_c)
    # outside: shift NFW mass to preserve continuity
    m[~inside] = halo.m_enclosed_msun(r[~inside]) - m1_nfw + m1_core
    return m


def sidm_cored_vcirc_km_s(
    halo: NFWHalo,
    r_kpc: ArrayLike,
    *,
    r1_kpc: float,
) -> np.ndarray:
    r = np.asarray(r_kpc, dtype=float)
    m = sidm_cored_mass_profile_msun(halo, r, r1_kpc=r1_kpc)
    return np.sqrt(G_KPC_KMS2_PER_MSUN * m / r)


@dataclass(frozen=True)
class RotationCurveLikelihood:
    data: RotationCurveData
    m200_msun: float
    c200: float
    t_age_gyr: float = 10.0

    def log_likelihood(
        self,
        model: YukawaModel,
        *,
        method: str = "auto",
        sidm: bool = True,
    ) -> float:
        halo = NFWHalo(m200_msun=self.m200_msun, c200=self.c200)
        r = self.data.r_kpc

        if sidm:
            r1 = r1_interaction_radius_kpc(model, halo, t_age_gyr=self.t_age_gyr, method=method, average=True, v_mode="jeans")
            v_dm = sidm_cored_vcirc_km_s(halo, r, r1_kpc=r1)
        else:
            v_dm = halo.v_circ_km_s(r)

        if self.data.v_bar_km_s is not None:
            v_tot = np.sqrt(v_dm**2 + self.data.v_bar_km_s**2)
        else:
            v_tot = v_dm

        chi2 = np.sum(((v_tot - self.data.v_obs_km_s) / self.data.v_err_km_s) ** 2)
        return float(-0.5 * chi2)

