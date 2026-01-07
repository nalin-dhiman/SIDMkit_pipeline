"""Halo-level mapping utilities (micro → macro).

This module is intentionally conservative: it implements *one* widely used
semi-analytic mapping between a particle-physics self-interaction model and a
halo-scale “interaction radius” r1, defined by the condition that a typical DM
particle has scattered O(1) times over the halo age.

A common definition in the SIDM literature is:

    Γ(r1) t_age = 1,

where Γ(r) is the per-particle scattering rate. For an isotropic distribution:

    Γ(r) = (ρ(r)/m) ⟨σ(v_rel) v_rel⟩.

See e.g. the discussion around r1 in reviews (Tulin & Yu; Adhikari et al.).
We implement this mapping for an NFW halo with an isotropic Jeans velocity
dispersion to provide a reasonable baseline.

Be critical / limitations
-------------------------
- This is *not* a replacement for a full gravothermal evolution or
  baryon-inclusive SIDM simulation.
- The result depends on: halo age t_age, outer truncation, velocity model,
  and choice of observable (σ_T vs σ̃_T vs σ_V).
- For serious work, treat r1 as a diagnostic, not a “guaranteed core radius”.

Still, r1 is useful for:
- quick consistency checks,
- connecting σ/m to the scale where self-interactions become important,
- making the package capable of halo-profile likelihood examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.optimize import brentq

from .averaging import average_sigma_over_m
from .cross_sections import sigma_over_m
from .model import YukawaModel

# Gravitational constant in kpc (km/s)^2 / Msun
G_KPC_KMS2_PER_MSUN = 4.300917270036279e-6


@dataclass(frozen=True)
class NFWHalo:
    """An NFW halo specified by (M200, c200) at z≈0.

    Parameters
    ----------
    m200_msun:
        Mass within r200 where mean enclosed density is 200 ρ_c.
    c200:
        Concentration parameter c200 = r200 / r_s.
    h0_km_s_mpc:
        Hubble constant used to compute ρ_c. Defaults to 70 km/s/Mpc.

    Notes
    -----
    This is a minimal cosmology dependence just to turn (M200, c200) into
    (r_s, ρ_s). For z≠0 analyses, you should supply an appropriate ρ_c(z).
    """

    m200_msun: float
    c200: float
    h0_km_s_mpc: float = 70.0

    def __post_init__(self) -> None:
        if not (self.m200_msun > 0):
            raise ValueError("m200_msun must be > 0")
        if not (self.c200 > 0):
            raise ValueError("c200 must be > 0")
        if not (self.h0_km_s_mpc > 0):
            raise ValueError("h0_km_s_mpc must be > 0")

    @property
    def rho_crit_msun_kpc3(self) -> float:
        # H0 in km/s/kpc
        h0 = self.h0_km_s_mpc / 1000.0
        return float(3.0 * h0 * h0 / (8.0 * np.pi * G_KPC_KMS2_PER_MSUN))

    @property
    def r200_kpc(self) -> float:
        rho_c = self.rho_crit_msun_kpc3
        return float((3.0 * self.m200_msun / (4.0 * np.pi * 200.0 * rho_c)) ** (1.0 / 3.0))

    @property
    def r_s_kpc(self) -> float:
        return float(self.r200_kpc / self.c200)

    @property
    def rho_s_msun_kpc3(self) -> float:
        c = self.c200
        f = np.log(1.0 + c) - c / (1.0 + c)
        return float(self.m200_msun / (4.0 * np.pi * self.r_s_kpc**3 * f))

    def rho_msun_kpc3(self, r_kpc: ArrayLike) -> np.ndarray:
        r = np.asarray(r_kpc, dtype=float)
        x = r / self.r_s_kpc
        return self.rho_s_msun_kpc3 / (x * (1.0 + x) ** 2)

    def m_enclosed_msun(self, r_kpc: ArrayLike) -> np.ndarray:
        r = np.asarray(r_kpc, dtype=float)
        x = r / self.r_s_kpc
        f = np.log(1.0 + x) - x / (1.0 + x)
        return 4.0 * np.pi * self.rho_s_msun_kpc3 * self.r_s_kpc**3 * f

    def v_circ_km_s(self, r_kpc: ArrayLike) -> np.ndarray:
        r = np.asarray(r_kpc, dtype=float)
        m = self.m_enclosed_msun(r)
        return np.sqrt(G_KPC_KMS2_PER_MSUN * m / r)

    def vmax_rmax(self) -> tuple[float, float]:
        # For NFW, Vmax at r≈2.16258 r_s
        rmax = 2.16258 * self.r_s_kpc
        vmax = float(self.v_circ_km_s(rmax))
        return vmax, rmax

    def sigma_1d_jeans_km_s(self, r_kpc: float) -> float:
        """Isotropic Jeans 1D velocity dispersion for NFW (numerical).

        Uses boundary condition σ_r(r200)=0 and integrates to r200.
        For isotropic β=0, the radial and tangential dispersions are equal,
        and we use σ_1D ≈ σ_r.

        This is a reasonable baseline for “characteristic” dispersion inside r200.
        """
        r = float(r_kpc)
        if r <= 0:
            raise ValueError("r_kpc must be > 0")
        r200 = self.r200_kpc
        if r >= r200:
            return 0.0

        def integrand(s: float) -> float:
            rho = float(self.rho_msun_kpc3(s))
            m = float(self.m_enclosed_msun(s))
            return rho * G_KPC_KMS2_PER_MSUN * m / (s * s)

        val, _ = quad(integrand, r, r200, epsabs=0.0, epsrel=2e-4, limit=200)
        rho_r = float(self.rho_msun_kpc3(r))
        if rho_r <= 0:
            return 0.0
        sigma2 = val / rho_r
        return float(np.sqrt(max(sigma2, 0.0)))


def scattering_rate_profile(
    model: YukawaModel,
    halo: NFWHalo,
    r_kpc: ArrayLike,
    *,
    t_age_gyr: float = 10.0,
    method: str = "auto",
    average: bool = True,
    v_mode: Literal["jeans", "vmax"] = "jeans",
) -> np.ndarray:
    """Compute Γ(r) in 1/Gyr for an array of radii.

    We compute:
        Γ(r) = (ρ(r)/m) ⟨σ v⟩

    If average=False, we approximate ⟨σ v⟩ by σ(v_rep) v_rep where v_rep is either:
        - v_rep = 4 σ_1D / √π  (if v_mode="jeans")
        - v_rep = Vmax        (if v_mode="vmax")

    If average=True, we compute a Maxwell relative-speed average using the local
    σ_1D (jeans) or using a single σ_1D derived from Vmax (vmax mode).

    Returns
    -------
    np.ndarray
        Γ(r) in 1/Gyr with same shape as r_kpc.
    """
    if t_age_gyr <= 0:
        raise ValueError("t_age_gyr must be > 0")
    r = np.asarray(r_kpc, dtype=float)

    rho_msun_kpc3 = halo.rho_msun_kpc3(r)  # Msun/kpc^3
    # Convert Msun/kpc^3 to g/cm^3 using constants from sidmkit.constants if available.
    # We keep the conversion local to avoid a hard dependency on astropy.
    # 1 Msun = 1.98847e33 g, 1 kpc = 3.085677581e21 cm
    MSUN_G = 1.98847e33
    KPC_CM = 3.085677581e21
    rho_g_cm3 = rho_msun_kpc3 * MSUN_G / (KPC_CM**3)

    # number density n = rho/m, with m in g. m_chi in GeV; convert GeV to g.
    GEV_G = 1.78266192e-24
    m_g = model.m_chi_gev * GEV_G
    n_cm3 = rho_g_cm3 / m_g

    Gamma = np.zeros_like(r, dtype=float)  # 1/s in intermediate
    for i, ri in np.ndenumerate(r):
        if ri <= 0 or n_cm3[i] <= 0:
            Gamma[i] = 0.0
            continue
        if v_mode == "jeans":
            sigma1d = halo.sigma_1d_jeans_km_s(float(ri))
            if sigma1d <= 0:
                Gamma[i] = 0.0
                continue
            if average:
                avg = average_sigma_over_m(model, sigma_1d_km_s=sigma1d, moment_n=1.0, method=method)
                sigma_v_over_m = avg.value  # (cm^2/g)*(km/s)
                # Convert km/s to cm/s for rate
                sigma_v_over_m_cms = sigma_v_over_m * 1.0e5
                Gamma[i] = n_cm3[i] * (sigma_v_over_m_cms * m_g)  # since (σ/m)*m = σ
            else:
                v_rep = 4.0 * sigma1d / np.sqrt(np.pi)
                som = float(sigma_over_m(v_rep, model=model, method=method, out_unit="cm2/g"))
                Gamma[i] = n_cm3[i] * (som * m_g) * (v_rep * 1.0e5)
        elif v_mode == "vmax":
            vmax, _ = halo.vmax_rmax()
            # crude mapping from Vmax to a Maxwell sigma1d: for isothermal, v_c = √2 σ1D
            sigma1d = vmax / np.sqrt(2.0)
            if average:
                avg = average_sigma_over_m(model, sigma_1d_km_s=sigma1d, moment_n=1.0, method=method)
                sigma_v_over_m_cms = avg.value * 1.0e5
                Gamma[i] = n_cm3[i] * (sigma_v_over_m_cms * m_g)
            else:
                v_rep = vmax
                som = float(sigma_over_m(v_rep, model=model, method=method, out_unit="cm2/g"))
                Gamma[i] = n_cm3[i] * (som * m_g) * (v_rep * 1.0e5)
        else:
            raise ValueError(f"Unknown v_mode: {v_mode}")

    # Convert 1/s to 1/Gyr
    SEC_PER_GYR = 1.0e9 * 365.25 * 24.0 * 3600.0
    return Gamma * SEC_PER_GYR


def r1_interaction_radius_kpc(
    model: YukawaModel,
    halo: NFWHalo,
    *,
    t_age_gyr: float = 10.0,
    method: str = "auto",
    average: bool = True,
    v_mode: Literal["jeans", "vmax"] = "jeans",
    bracket_kpc: Optional[tuple[float, float]] = None,
) -> float:
    """Solve for r1 where Γ(r1) t_age = 1.

    Returns r1 in kpc.

    If no root exists within the bracket, raises ValueError.
    """
    if t_age_gyr <= 0:
        raise ValueError("t_age_gyr must be > 0")
    r200 = halo.r200_kpc
    if bracket_kpc is None:
        # bracket inside (very small, r200)
        bracket_kpc = (1e-4 * halo.r_s_kpc, 0.999 * r200)

    a, b = bracket_kpc
    if not (0 < a < b <= r200):
        raise ValueError("Invalid bracket_kpc; require 0 < a < b <= r200")

    def f(r: float) -> float:
        Gamma = float(
            scattering_rate_profile(
                model,
                halo,
                np.array([r]),
                t_age_gyr=t_age_gyr,
                method=method,
                average=average,
                v_mode=v_mode,
            )[0]
        )
        return Gamma * t_age_gyr - 1.0

    fa = f(a)
    fb = f(b)
    # If interactions are extremely weak, Γ t_age < 1 everywhere -> no r1.
    if fa < 0 and fb < 0:
        raise ValueError("No r1 root: Γ(r) t_age < 1 over the bracket (interactions too weak).")

    # If interactions are extremely strong, Γ t_age > 1 everywhere -> r1 above r200.
    if fa > 0 and fb > 0:
        # In this case r1 would be > b; return b as a conservative cap.
        return float(b)

    r1 = brentq(f, a, b, maxiter=200)
    return float(r1)
