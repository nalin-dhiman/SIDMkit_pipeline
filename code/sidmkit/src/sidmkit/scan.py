"""Parameter-space scanning utilities.

This is a brute-force grid scan intended for:
- quick exploration,
- producing a table of model predictions vs summary constraints,
- regression testing.

It is not optimized (no parallelism) and it is not a replacement for proper
inference. For that, see :mod:`sidmkit.inference` and dataset likelihoods.
"""

from __future__ import annotations

import csv
import re
from typing import Iterable, Sequence

from .constraints import Constraint, DEFAULT_CONSTRAINTS, evaluate_constraints
from .model import PotentialType, YukawaModel


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:60] if len(s) > 60 else s


def grid_scan(
    *,
    m_chi_grid: Sequence[float],
    m_med_grid: Sequence[float],
    alpha_grid: Sequence[float],
    potential: PotentialType = PotentialType.ATTRACTIVE,
    constraints: Iterable[Constraint] = DEFAULT_CONSTRAINTS,
    method: str = "auto",
) -> list[dict]:
    """Run a brute-force grid scan and return JSON/CSV-friendly dict rows."""
    out: list[dict] = []
    constraints = list(constraints)
    for m_chi in m_chi_grid:
        for m_med in m_med_grid:
            for alpha in alpha_grid:
                model = YukawaModel(m_chi_gev=m_chi, m_med_gev=m_med, alpha=alpha, potential=potential)
                evals = evaluate_constraints(model, constraints, method=method)
                row: dict = {
                    "m_chi_gev": float(m_chi),
                    "m_med_gev": float(m_med),
                    "alpha": float(alpha),
                    "potential": potential.value,
                    "all_pass": bool(all(e.passed for e in evals)),
                }
                for e in evals:
                    key = _slug(e.constraint.name)
                    row[f"pred_{key}"] = float(e.predicted)
                    row[f"pass_{key}"] = bool(e.passed)
                out.append(row)
    return out


def write_csv(rows: list[dict], path: str) -> None:
    """Write scan results to CSV."""
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

