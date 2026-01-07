"""Micro-to-macro translators: scattering rates, mean free paths, etc.

A common SIDM diagnostic is the per-particle scattering rate:

    Γ = (ρ/m) σ_T v  =  ρ (σ_T/m) v

This module provides simple, explicit conversions so you can sanity-check
"does this cross section matter in this halo?"
"""

from __future__ import annotations

from typing import Literal

from .constants import (
    C_CM_S,
    MSUN_PER_PC3_TO_G_PER_CM3,
    SEC_PER_GYR,
)


def scattering_rate(
    *,
    rho: float,
    sigma_over_m_cm2_g: float,
    v_km_s: float,
    rho_unit: Literal["g/cm3", "Msun/pc3"] = "Msun/pc3",
    out_unit: Literal["1/s", "1/Gyr"] = "1/Gyr",
) -> float:
    """Compute the SIDM scattering rate Γ.

    Parameters
    ----------
    rho:
        Mass density.
    sigma_over_m_cm2_g:
        Cross section per unit mass σ/m in cm^2/g.
    v_km_s:
        Relative velocity in km/s.
    rho_unit:
        Unit for `rho`.
    out_unit:
        Output rate in 1/s or 1/Gyr.

    Returns
    -------
    Scattering rate Γ.
    """
    if rho_unit == "Msun/pc3":
        rho_g_cm3 = rho * MSUN_PER_PC3_TO_G_PER_CM3
    elif rho_unit == "g/cm3":
        rho_g_cm3 = rho
    else:
        raise ValueError("rho_unit must be 'g/cm3' or 'Msun/pc3'")

    v_cm_s = v_km_s * 1.0e5
    rate_s = rho_g_cm3 * sigma_over_m_cm2_g * v_cm_s  # 1/s
    if out_unit == "1/s":
        return rate_s
    if out_unit == "1/Gyr":
        return rate_s * SEC_PER_GYR
    raise ValueError("out_unit must be '1/s' or '1/Gyr'")


def mean_free_path(
    *,
    rho: float,
    sigma_over_m_cm2_g: float,
    rho_unit: Literal["g/cm3", "Msun/pc3"] = "Msun/pc3",
    out_unit: Literal["cm", "kpc"] = "kpc",
) -> float:
    """Compute the mean free path λ = 1/(ρ σ/m)."""
    if rho_unit == "Msun/pc3":
        rho_g_cm3 = rho * MSUN_PER_PC3_TO_G_PER_CM3
    elif rho_unit == "g/cm3":
        rho_g_cm3 = rho
    else:
        raise ValueError("rho_unit must be 'g/cm3' or 'Msun/pc3'")

    lam_cm = 1.0 / (rho_g_cm3 * sigma_over_m_cm2_g)
    if out_unit == "cm":
        return lam_cm
    if out_unit == "kpc":
        cm_per_kpc = 3.085_677_581e21
        return lam_cm / cm_per_kpc
    raise ValueError("out_unit must be 'cm' or 'kpc'")
