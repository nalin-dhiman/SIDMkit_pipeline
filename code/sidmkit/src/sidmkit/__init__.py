"""sidmkit: a transparent toolkit for SIDM micro→macro work.

The public API is intentionally small. Most functionality is accessible via the
CLI (`sidmkit --help`), but common building blocks are also importable.

Notes
-----
We keep the top-level namespace deliberately minimal to reduce accidental
backwards-compatibility burdens. More specialized functionality lives in
submodules (e.g. :mod:`sidmkit.likelihoods`).
"""

from __future__ import annotations

from .model import YukawaModel, PotentialType
from .cross_sections import sigma_over_m, sigma_transfer
from .averaging import average_sigma_over_m
from .constraints import (
    Constraint,
    VelocitySpec,
    list_constraint_sets,
    get_constraint_set,
    evaluate_constraints,
)
from .halo import NFWHalo, r1_interaction_radius_kpc

__all__ = [
    "YukawaModel",
    "PotentialType",
    "sigma_over_m",
    "sigma_transfer",
    "average_sigma_over_m",
    "Constraint",
    "VelocitySpec",
    "list_constraint_sets",
    "get_constraint_set",
    "evaluate_constraints",
    "NFWHalo",
    "r1_interaction_radius_kpc",
]

# Keep a simple version string for downstream tooling.
__version__ = "0.3.0"
