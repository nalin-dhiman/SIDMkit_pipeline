"""Cross sections for SIDM.

This module implements commonly-used semi-analytic expressions for the *transfer*
(momentum-transfer) cross section for a Yukawa potential, plus an optional
partial-wave solver for numerical reference.

The focus here is on transparency and correct units, not on squeezing every last
percent of speed.

References (see the papers you provided):
- Mini-review on self-interacting dark matter: classical + Hulthén formulae and
  regime conditions.
- "Self-interacting dark matter with observable ΔNeff": Born formula and
  definitions (β, etc.).
"""

from __future__ import annotations

import math
import warnings
from typing import Literal, Union

import numpy as np
from scipy import integrate
from scipy.special import loggamma, spherical_jn, spherical_yn

from .constants import km_s_to_v_over_c, sigma_over_m_gev3_to_cm2_g
from .model import PotentialType, YukawaModel

ArrayLike = Union[float, np.ndarray]


def _asarray(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _validate_velocity(v: np.ndarray) -> None:
    if np.any(v <= 0):
        raise ValueError("Relative velocity must be > 0.")


def sigma_transfer_born(
    v_rel: ArrayLike,
    *,
    m_chi_gev: float,
    m_med_gev: float,
    alpha: float,
) -> np.ndarray:
    """Born-regime transfer cross section σ_T for a Yukawa potential.

    Implements the common expression (equivalent forms appear in multiple SIDM reviews):

        σ_T^Born = (8π α^2)/(m_χ^2 v^4) [ ln(1 + ξ^2) - ξ^2/(1 + ξ^2) ]

    where ξ = m_χ v / m_med.

    Parameters
    ----------
    v_rel:
        Relative velocity, **dimensionless** (v/c). May be float or array.
    m_chi_gev, m_med_gev, alpha:
        Model parameters.

    Returns
    -------
    σ_T in natural units (GeV^{-2}).
    """
    v = _asarray(v_rel)
    _validate_velocity(v)

    if m_chi_gev <= 0 or m_med_gev <= 0 or alpha <= 0:
        raise ValueError("m_chi_gev, m_med_gev, alpha must be > 0")

    xi2 = (m_chi_gev * v / m_med_gev) ** 2
    term = np.log1p(xi2) - xi2 / (1.0 + xi2)
    sigma = (8.0 * math.pi * alpha**2 / (m_chi_gev**2 * v**4)) * term
    return sigma


def sigma_transfer_classical(
    v_rel: ArrayLike,
    *,
    m_chi_gev: float,
    m_med_gev: float,
    alpha: float,
    potential: PotentialType,
) -> np.ndarray:
    """Classical-regime transfer cross section σ_T for Yukawa potential.

    Uses the piecewise plasma-fit formula commonly used in SIDM literature.

    Define:

        β = 2 α m_med / (m_χ v^2)

    Attractive (Mini-review Eq. (10)):

        β < 1e-1:  σ_T = 4π/m_med^2 * β^2 ln(1 + β^{-1})
        1e-1 ≤ β ≤ 1e3: σ_T = 8π/m_med^2 * β^2/(1 + 1.5 β^{1.65})
        β > 1e3: σ_T = π/m_med^2 * (ln β + 1 - 1/(2 ln β))^2

    Repulsive (Mini-review Eq. (11)):

        β < 1: σ_T = 2π/m_med^2 * β^2 ln(1 + β^{-2})
        β ≥ 1: σ_T = π/m_med^2 * (ln(2β) - ln ln(2β))^2

    Parameters
    ----------
    v_rel:
        Relative velocity, dimensionless (v/c).
    potential:
        Attractive or repulsive channel.

    Returns
    -------
    σ_T in GeV^{-2}.
    """
    v = _asarray(v_rel)
    _validate_velocity(v)

    beta = 2.0 * alpha * m_med_gev / (m_chi_gev * v**2)

    if potential is PotentialType.ATTRACTIVE:
        sigma = np.empty_like(beta)

        m1 = beta < 1.0e-1
        m2 = (beta >= 1.0e-1) & (beta <= 1.0e3)
        m3 = beta > 1.0e3

        if np.any(m1):
            b = beta[m1]
            sigma[m1] = (4.0 * math.pi / m_med_gev**2) * b**2 * np.log1p(1.0 / b)

        if np.any(m2):
            b = beta[m2]
            sigma[m2] = (8.0 * math.pi / m_med_gev**2) * (b**2 / (1.0 + 1.5 * b**1.65))

        if np.any(m3):
            b = beta[m3]
            ln_b = np.log(b)
            sigma[m3] = (math.pi / m_med_gev**2) * (ln_b + 1.0 - 0.5 / ln_b) ** 2

        return sigma

    # Repulsive
    sigma = np.empty_like(beta)
    m1 = beta < 1.0
    m2 = beta >= 1.0

    if np.any(m1):
        b = beta[m1]
        sigma[m1] = (2.0 * math.pi / m_med_gev**2) * b**2 * np.log1p(1.0 / b**2)

    if np.any(m2):
        b = beta[m2]
        ln2b = np.log(2.0 * b)
        sigma[m2] = (math.pi / m_med_gev**2) * (ln2b - np.log(ln2b)) ** 2

    return sigma


def sigma_transfer_hulthen(
    v_rel: ArrayLike,
    *,
    m_chi_gev: float,
    m_med_gev: float,
    alpha: float,
    potential: PotentialType,
    kappa: float = 1.6,
) -> np.ndarray:
    """Hulthén approximation for transfer cross section in the resonant regime.

    The Hulthén potential approximates the Yukawa potential but allows an analytic
    expression for the s-wave phase shift δ0, and hence σ_T.

        σ_T^Hulthén = 16π sin^2(δ0) / (m_χ^2 v^2)

    with

        δ0 = Arg[ i Γ(i η) / ( Γ(λ+) Γ(λ-) ) ]
        η = m_χ v / (κ m_med)

    For the attractive channel:

        λ± = 1 + i m_χ v/(2 κ m_med) ± sqrt( α m_χ/(κ m_med) - (m_χ^2 v^2)/(4 κ^2 m_med^2) )

    For the repulsive channel:

        λ± = 1 + i m_χ v/(2 κ m_med) ± i sqrt( α m_χ/(κ m_med) + (m_χ^2 v^2)/(4 κ^2 m_med^2) )

    Notes
    -----
    - We compute δ0 using `loggamma` to avoid overflow/underflow in Γ(z).
    - This is a non-perturbative approximation, not a guarantee of percent-level accuracy.

    Returns σ_T in GeV^{-2}.
    """
    v = _asarray(v_rel)
    _validate_velocity(v)

    if kappa <= 0:
        raise ValueError("kappa must be > 0")

    eta = m_chi_gev * v / (kappa * m_med_gev)

    base = 1.0 + 1j * m_chi_gev * v / (2.0 * kappa * m_med_gev)

    if potential is PotentialType.ATTRACTIVE:
        under_sqrt = alpha * m_chi_gev / (kappa * m_med_gev) - (m_chi_gev * v) ** 2 / (
            4.0 * kappa**2 * m_med_gev**2
        )
        sqrt_term = np.sqrt(under_sqrt + 0j)
        lam_plus = base + sqrt_term
        lam_minus = base - sqrt_term
    else:
        under_sqrt = alpha * m_chi_gev / (kappa * m_med_gev) + (m_chi_gev * v) ** 2 / (
            4.0 * kappa**2 * m_med_gev**2
        )
        sqrt_term = 1j * np.sqrt(under_sqrt)
        lam_plus = base + sqrt_term
        lam_minus = base - sqrt_term

    log_i = 1j * (math.pi / 2.0)  # log(i)
    log_num = loggamma(1j * eta)
    log_den = loggamma(lam_plus) + loggamma(lam_minus)
    log_z = log_i + log_num - log_den

    delta0 = np.mod(np.imag(log_z) + math.pi, 2.0 * math.pi) - math.pi

    sigma = 16.0 * math.pi * (np.sin(delta0) ** 2) / (m_chi_gev**2 * v**2)
    return np.real_if_close(sigma, tol=1e-10).astype(float)


def _auto_method(model: YukawaModel, v_over_c: np.ndarray) -> Literal["born", "classical", "hulthen"]:
    born_param = model.alpha * model.m_chi_gev / model.m_med_gev
    ratio = model.m_chi_gev * v_over_c / model.m_med_gev

    if born_param < 0.1:
        return "born"
    if np.all(ratio >= 1.0):
        return "classical"
    return "hulthen"


def sigma_transfer(
    v_rel: ArrayLike,
    model: YukawaModel,
    *,
    v_unit: Literal["c", "km/s"] = "km/s",
    method: Literal["auto", "born", "classical", "hulthen", "partial_wave"] = "auto",
    warn: bool = True,
    **partial_wave_kwargs,
) -> np.ndarray:
    """Compute the transfer cross section σ_T for a Yukawa SIDM model.

    Parameters
    ----------
    v_rel:
        Relative velocity. Interpretation depends on `v_unit`.
    v_unit:
        - "km/s": `v_rel` is in km/s (default)
        - "c": `v_rel` is dimensionless (v/c)
    method:
        - "auto": Born/classical/Hulthén based on heuristics
        - "born", "classical", "hulthen": force a specific approximation
        - "partial_wave": numerical partial-wave solver (slow; validation)

    Returns
    -------
    σ_T in GeV^{-2}.
    """
    v = _asarray(v_rel)
    if v_unit == "km/s":
        v_over_c = km_s_to_v_over_c(v)
    elif v_unit == "c":
        v_over_c = v
    else:
        raise ValueError("v_unit must be 'km/s' or 'c'")
    _validate_velocity(v_over_c)

    chosen: str = method
    if method == "auto":
        chosen = _auto_method(model, v_over_c)

        born_param = model.alpha * model.m_chi_gev / model.m_med_gev
        if warn and (0.05 < born_param < 2.0):
            warnings.warn(
                "Auto-regime selection is heuristic and may be inaccurate in the intermediate regime "
                f"(α mχ/m_med ≈ {born_param:.2g}). Consider method='partial_wave' for spot checks.",
                RuntimeWarning,
            )

    if chosen == "born":
        return sigma_transfer_born(
            v_over_c, m_chi_gev=model.m_chi_gev, m_med_gev=model.m_med_gev, alpha=model.alpha
        )
    if chosen == "classical":
        return sigma_transfer_classical(
            v_over_c,
            m_chi_gev=model.m_chi_gev,
            m_med_gev=model.m_med_gev,
            alpha=model.alpha,
            potential=model.potential,
        )
    if chosen == "hulthen":
        return sigma_transfer_hulthen(
            v_over_c,
            m_chi_gev=model.m_chi_gev,
            m_med_gev=model.m_med_gev,
            alpha=model.alpha,
            potential=model.potential,
            kappa=model.kappa_hulthen,
        )
    if chosen == "partial_wave":
        return sigma_transfer_partial_wave(
            v_over_c,
            m_chi_gev=model.m_chi_gev,
            m_med_gev=model.m_med_gev,
            alpha=model.alpha,
            potential=model.potential,
            **partial_wave_kwargs,
        )
    raise ValueError(f"Unknown method: {method}")


def sigma_over_m(
    v_rel: ArrayLike,
    model: YukawaModel,
    *,
    v_unit: Literal["c", "km/s"] = "km/s",
    method: Literal["auto", "born", "classical", "hulthen", "partial_wave"] = "auto",
    out_unit: Literal["cm2/g", "GeV^-3"] = "cm2/g",
    **kwargs,
) -> np.ndarray:
    """Compute σ_T/m_χ.

    Returns:
    - out_unit="GeV^-3": natural-units (GeV^{-3})
    - out_unit="cm2/g": observational convention
    """
    sigma = sigma_transfer(v_rel, model, v_unit=v_unit, method=method, **kwargs)
    sigma_over_m_gev3 = sigma / model.m_chi_gev
    if out_unit == "GeV^-3":
        return sigma_over_m_gev3
    if out_unit == "cm2/g":
        return sigma_over_m_gev3_to_cm2_g(sigma_over_m_gev3)
    raise ValueError("out_unit must be 'cm2/g' or 'GeV^-3'")


# -----------------------------------------------------------------------------
# Partial-wave solver (numerical reference)
# -----------------------------------------------------------------------------

def _yukawa_potential(r: float, *, alpha: float, m_med_gev: float, potential: PotentialType) -> float:
    sign = potential.sign
    return float(sign * alpha * math.exp(-m_med_gev * r) / r)


def _phase_shift_single_l(
    l: int,
    *,
    m_chi_gev: float,
    m_med_gev: float,
    alpha: float,
    potential: PotentialType,
    v_over_c: float,
    r_max: float,
    r0: float,
    rtol: float,
    atol: float,
    max_step: float | None,
) -> float:
    mu = m_chi_gev / 2.0
    k = mu * v_over_c  # = m v /2

    def rhs(r: float, y: np.ndarray) -> np.ndarray:
        u, up = y
        V = _yukawa_potential(r, alpha=alpha, m_med_gev=m_med_gev, potential=potential)
        upp = (l * (l + 1) / (r * r) + 2.0 * mu * V - k * k) * u
        return np.array([up, upp], dtype=float)

    # Regular solution near the origin: u(r) ~ r^{l+1}. Overall normalization is arbitrary,
    # but a physically-motivated scaling avoids catastrophic growth for large l (which can
    # otherwise overflow the integrator at moderate r).
    #
    # The regular solution satisfies (u'/u)(r0) approx (l+1)/r0 for sufficiently small r0.
    # We choose a normalization u0 ∝ r0^{l+1} when representable, and otherwise rescale to
    # a safe floor; the phase shift is insensitive to this overall scale.
    log_u0 = (l + 1) * math.log(r0)
    log_floor = -700.0  # exp(-700) ~ 5e-305, safely above underflow in float64
    u0 = math.exp(max(log_u0, log_floor))
    up0 = u0 * (l + 1) / r0
    y0 = np.array([u0, up0], dtype=float)

    sol = integrate.solve_ivp(
        rhs,
        t_span=(r0, r_max),
        y0=y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=np.inf if max_step is None else max_step,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for l={l}: {sol.message}")

    u = float(sol.y[0, -1])
    up = float(sol.y[1, -1])

    x = k * r_max
    j = float(spherical_jn(l, x))
    y = float(spherical_yn(l, x))
    jp = float(spherical_jn(l, x, derivative=True))
    yp = float(spherical_yn(l, x, derivative=True))

    # We are solving for u(r) = r R_l(r). The free solutions are therefore
    # J(r) = r j_l(kr) and Y(r) = r y_l(kr), not j_l and y_l by themselves.
    J = r_max * j
    Y = r_max * y

    # Derivatives with respect to r:
    # d/dr [r j_l(kr)] = j_l(kr) + (kr) j_l'(kr)
    Jp = j + x * jp
    Yp = y + x * yp

    # tan δ = (u' J - u J') / (u' Y - u Y')
    num = up * J - u * Jp
    den = up * Y - u * Yp
    delta = math.atan2(num, den)
    return delta


def sigma_transfer_partial_wave(
    v_rel: ArrayLike,
    *,
    m_chi_gev: float,
    m_med_gev: float,
    alpha: float,
    potential: PotentialType,
    l_max: int | None = None,
    r_max: float | None = None,
    r0: float | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float | None = None,
) -> np.ndarray:
    """Numerical partial-wave computation of σ_T.

    Intended mainly for validation/benchmarking.

    Uses the identity:

        σ_T = (4π/k^2) Σ_{l=0}^{∞} (l+1) sin^2(δ_{l+1} - δ_l)

    where k = μ v, μ = m_χ/2 for equal-mass scattering.

    Parameters
    ----------
    v_rel:
        Relative velocity in dimensionless units v/c.

    Returns
    -------
    σ_T in GeV^{-2}.
    """
    v = _asarray(v_rel)
    _validate_velocity(v)

    mu = m_chi_gev / 2.0

    out = np.empty_like(v)

    for idx, vi in np.ndenumerate(v):
        vi_f = float(vi)
        ki = mu * vi_f
        if ki <= 0:
            raise ValueError("k must be positive")

        # Default r_max: go far enough that the Yukawa potential is exponentially negligible.
        # Matching uses exact spherical Bessel functions, so we do NOT need kr_max >> 1.
        r_max_i = r_max if r_max is not None else (50.0 / m_med_gev)
        r0_i = r0 if r0 is not None else 1e-6 * min(1.0 / m_med_gev, 1.0 / ki)

        if l_max is None:
            # Heuristic: partial waves contribute significantly up to impact parameters
            # b ~ few / m_med. With l ~ k b, a rough upper bound is l_max ~ O(k/m_med).
            # The prefactor (20) is a safety margin; increase for very high precision.
            l_max_i = int(math.ceil(20.0 * ki / m_med_gev + 10.0))
            l_max_i = max(l_max_i, 10)
        else:
            l_max_i = int(l_max)

        deltas = np.empty(l_max_i + 2, dtype=float)
        for l in range(l_max_i + 2):
            deltas[l] = _phase_shift_single_l(
                l,
                m_chi_gev=m_chi_gev,
                m_med_gev=m_med_gev,
                alpha=alpha,
                potential=potential,
                v_over_c=vi_f,
                r_max=float(r_max_i),
                r0=float(r0_i),
                rtol=rtol,
                atol=atol,
                max_step=np.inf if max_step is None else max_step,
            )

        diff = deltas[1:] - deltas[:-1]
        # sum l=0..l_max of (l+1) sin^2(δ_{l+1}-δ_l)
        ell = np.arange(0, l_max_i + 1, dtype=float)
        sigma = (4.0 * math.pi / (ki**2)) * float(np.sum((ell + 1.0) * (np.sin(diff[: l_max_i + 1]) ** 2)))
        out[idx] = sigma

    return out
