"""Physical constants and unit conversions.

This module keeps constants explicit and localized. The goal is to avoid hidden
unit assumptions in the rest of the code.

Conventions:
- Natural units with \hbar = c = 1 are used internally for particle-physics
  formulae. Masses are in GeV, velocities are dimensionless (v/c).
- Conversions are provided to cm, g, km/s, etc.

Numerical constants are hard-coded (not fetched from the web) for full
reproducibility.
"""

from __future__ import annotations

import math

# --- Fundamental numerical constants (CODATA values, commonly used) ---

# Speed of light in vacuum.
C_KM_S: float = 299_792.458  # km / s
C_CM_S: float = C_KM_S * 1.0e5  # cm / s

# Planck's constant times c in GeV·cm.
# ħc = 0.1973269804 GeV·fm and 1 fm = 1e-13 cm.
HBAR_C_GEV_CM: float = 0.197_326_980_4e-13  # GeV·cm

# Conversion: 1 GeV^{-2} to cm^2.
# In natural units, [length] = 1/GeV, so (ħc)^2 gives cm^2 per GeV^{-2}.
GEV_INV2_TO_CM2: float = HBAR_C_GEV_CM**2  # cm^2 / GeV^{-2}

# Mass conversion: 1 GeV/c^2 to grams.
GEV_TO_G: float = 1.782_661_92e-24  # g / GeV

# Time conversions
SEC_PER_GYR: float = 1.0e9 * 365.25 * 24 * 3600  # s in a Gyr (Julian year)

# Astronomical density conversions
MSUN_G: float = 1.988_47e33  # g
PC_CM: float = 3.085_677_581e18  # cm
MSUN_PER_PC3_TO_G_PER_CM3: float = MSUN_G / (PC_CM**3)

# Convenience: cm^2/g expressed as cm^2/GeV
# (useful because older SIDM literature often quotes σ/m in cm^2/GeV).
CM2_PER_G_TO_CM2_PER_GEV: float = GEV_TO_G  # since 1 GeV corresponds to GEV_TO_G grams

# And inverse:
CM2_PER_GEV_TO_CM2_PER_G: float = 1.0 / GEV_TO_G

# Derived: convert (σ/m) in GeV^{-3} to cm^2/g.
# (σ in GeV^{-2}) / (m in GeV) = GeV^{-3}
GEV_INV3_TO_CM2_PER_G: float = GEV_INV2_TO_CM2 / GEV_TO_G

def km_s_to_v_over_c(v_km_s: float) -> float:
    """Convert km/s to dimensionless v/c."""
    return v_km_s / C_KM_S

def v_over_c_to_km_s(v_over_c: float) -> float:
    """Convert dimensionless v/c to km/s."""
    return v_over_c * C_KM_S

def sigma_over_m_gev3_to_cm2_g(sigma_over_m_gev3: float) -> float:
    """Convert σ/m from GeV^{-3} to cm^2/g."""
    return sigma_over_m_gev3 * GEV_INV3_TO_CM2_PER_G

def sigma_gev2_to_cm2(sigma_gev_inv2: float) -> float:
    """Convert a cross section from GeV^{-2} to cm^2."""
    return sigma_gev_inv2 * GEV_INV2_TO_CM2

def mass_gev_to_g(mass_gev: float) -> float:
    """Convert mass from GeV to grams."""
    return mass_gev * GEV_TO_G
