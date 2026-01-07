import math

import numpy as np
import pytest

from sidmkit.cross_sections import sigma_transfer_born, sigma_transfer_classical, sigma_transfer_hulthen, sigma_over_m
from sidmkit.model import PotentialType, YukawaModel
from sidmkit.rates import scattering_rate


def test_rate_scaling_canonical():
    rate = scattering_rate(rho=0.01, rho_unit="Msun/pc3", sigma_over_m_cm2_g=1.0, v_km_s=50.0, out_unit="1/Gyr")
    assert abs(rate - 0.1) < 0.05


def test_born_formula_matches_manual():
    v = 1e-3
    mchi = 10.0
    mmed = 0.1
    alpha = 1e-4
    xi2 = (mchi * v / mmed) ** 2
    manual = (8 * math.pi * alpha**2 / (mchi**2 * v**4)) * (math.log(1 + xi2) - xi2 / (1 + xi2))
    got = float(sigma_transfer_born(v, m_chi_gev=mchi, m_med_gev=mmed, alpha=alpha))
    assert math.isfinite(got)
    assert got > 0
    assert abs(got - manual) / manual < 1e-12


def test_classical_positive():
    v = np.array([1e-4, 1e-3, 1e-2])
    sig = sigma_transfer_classical(v, m_chi_gev=10.0, m_med_gev=0.01, alpha=0.1, potential=PotentialType.ATTRACTIVE)
    assert np.all(sig > 0)
    sig2 = sigma_transfer_classical(v, m_chi_gev=10.0, m_med_gev=0.01, alpha=0.1, potential=PotentialType.REPULSIVE)
    assert np.all(sig2 > 0)


def test_hulthen_positive():
    v = np.array([1e-5, 1e-4, 1e-3])
    sig = sigma_transfer_hulthen(v, m_chi_gev=10.0, m_med_gev=0.01, alpha=1e-3, potential=PotentialType.ATTRACTIVE, kappa=1.6)
    assert np.all(sig >= 0)


def test_sigma_over_m_units_cm2g():
    model = YukawaModel(10.0, 0.01, 1e-3, potential=PotentialType.ATTRACTIVE)
    val = float(sigma_over_m(30.0, model, v_unit="km/s", method="auto", out_unit="cm2/g"))
    assert math.isfinite(val)
    assert val >= 0


def test_invalid_velocity_raises():
    model = YukawaModel(10.0, 0.01, 1e-3)
    with pytest.raises(ValueError):
        sigma_over_m(0.0, model, v_unit="km/s")
