"""Validation helpers.

Validation philosophy
---------------------
To be reviewer-proof, we want at least three layers:

1) Unit tests for formulas and unit conversions.
2) Internal numerical cross-checks (e.g., auto approximation vs partial-wave).
3) External checks against published benchmark points/curves, when possible.

This module provides small utilities for (2) and a basic scaffold for (3).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .averaging import average_sigma_over_m
from .cross_sections import sigma_over_m
from .model import YukawaModel


@dataclass(frozen=True)
class CurveComparison:
    max_abs_rel_err: float
    rms_abs_rel_err: float
    n_points: int


def generate_sigma_curve(
    model: YukawaModel,
    *,
    v_min_km_s: float = 1.0,
    v_max_km_s: float = 5000.0,
    n: int = 200,
    method: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    v = np.geomspace(v_min_km_s, v_max_km_s, n)
    y = sigma_over_m(v, model=model, method=method, out_unit="cm2/g")
    return v, np.asarray(y, dtype=float)


def save_curve_csv(path: str, v_km_s: np.ndarray, y: np.ndarray, *, y_name: str = "sigma_over_m_cm2_g") -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["v_km_s", y_name])
        for vi, yi in zip(v_km_s, y):
            w.writerow([float(vi), float(yi)])


def load_curve_csv(path: str) -> tuple[np.ndarray, np.ndarray, str]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 2:
            raise ValueError("CSV must have at least two columns: v_km_s, y")
        y_name = header[1]
        rows = [(float(r[0]), float(r[1])) for r in reader if r]
    v = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=float)
    return v, y, y_name


def compare_curves(
    v_ref: np.ndarray,
    y_ref: np.ndarray,
    v_pred: np.ndarray,
    y_pred: np.ndarray,
) -> CurveComparison:
    # Interpolate prediction onto reference v-grid in log space
    logv_pred = np.log10(v_pred)
    logy_pred = np.log10(np.clip(y_pred, 1e-300, np.inf))
    logy_interp = np.interp(np.log10(v_ref), logv_pred, logy_pred)
    y_interp = 10 ** logy_interp
    rel = np.abs((y_interp - y_ref) / np.clip(y_ref, 1e-300, np.inf))
    return CurveComparison(
        max_abs_rel_err=float(np.max(rel)),
        rms_abs_rel_err=float(np.sqrt(np.mean(rel**2))),
        n_points=int(rel.size),
    )


def literature_fig13_left_points() -> list[dict]:
    """Return a small, auto-digitized set of points from Tulin & Yu Fig. 13 (left).

    This exists as a *demonstration* and regression test scaffold.
    The points are approximate and should not be used as authoritative data.

    Each entry is:
        {"v_km_s": ..., "sigma_v_over_m": ...}

    Units: (cm^2/g)*(km/s).
    """
    # These were extracted from the PDF at 400dpi using a crude color-based
    # marker detector and should be treated as approximate.
    return [
        {"v_km_s": 23.2, "sigma_v_over_m": 26.8},
        {"v_km_s": 46.2, "sigma_v_over_m": 256.},
        {"v_km_s": 105.6, "sigma_v_over_m": 178.},
        {"v_km_s": 134.3, "sigma_v_over_m": 6.7},
        {"v_km_s": 140.4, "sigma_v_over_m": 28.3},
        {"v_km_s": 156.5, "sigma_v_over_m": 864.},
        {"v_km_s": 164.8, "sigma_v_over_m": 18.0},
        {"v_km_s": 250.7, "sigma_v_over_m": 9.2},
        {"v_km_s": 1190.0, "sigma_v_over_m": 118.},
        {"v_km_s": 1205.0, "sigma_v_over_m": 333.},
        {"v_km_s": 1561.0, "sigma_v_over_m": 695.},
    ]


def chi2_against_fig13_left(
    model: YukawaModel,
    *,
    method: str = "auto",
    fractional_sigma: float = 0.35,
) -> float:
    """Compute a rough χ² against the (approximate) Fig.13 left points.

    This is *not* a publication-quality reproduction. It is intended as:
    - a sanity check (are we in the right ballpark?),
    - a regression test (did the curve drastically change?).

    We assume independent log-normal-like uncertainties using a fractional sigma.
    """
    pts = literature_fig13_left_points()
    v = np.array([p["v_km_s"] for p in pts], dtype=float)
    y_obs = np.array([p["sigma_v_over_m"] for p in pts], dtype=float)

    # Predict y = σ(v)*v (no additional averaging; the figure is already velocity-weighted).
    y_pred = sigma_over_m(v, model=model, method=method, out_unit="cm2/g") * v
    sig = fractional_sigma * y_obs
    chi2 = np.sum(((y_pred - y_obs) / sig) ** 2)
    return float(chi2)

