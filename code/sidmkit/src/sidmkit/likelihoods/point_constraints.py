"""Point-constraint likelihoods (summary-level but explicit).

This module is a bridge between:
- simple band checks (“is σ/m between 0.1 and 10 at 30 km/s?”), and
- full dataset forward modelling (rotation curves, lensing maps, mergers, ...).

A *point constraint* is a statement about an observable at a given velocity scale
(or a velocity-averaged scale). We provide a smooth log-likelihood so these can
be used in inference pipelines.

This is not a replacement for a real dataset likelihood, but it *is*:
- reproducible,
- easy to audit,
- easy to extend.

JSON schema
-----------
`PointConstraintDataset.from_json(path)` expects a JSON file with:

{
  "name": "...",
  "constraints": [
    {
      "name": "...",
      "kind": "upper|lower|band|measurement",
      "observable": "sigma_over_m|sigma_v_over_m",
      "v_km_s": 1000.0,              # for point
      "v_min_km_s": 500.0,           # for range (optional)
      "v_max_km_s": 2000.0,          # for range (optional)
      "lower": 0.1,
      "upper": 1.0,
      "mean": 0.5,
      "sigma": 0.1,
      "average": false,              # if true, use Maxwellian average
      "sigma_1d_km_s": 150.0,        # required if average=true
      "reference": "...",
      "notes": "..."
    }
  ]
}

Only the fields relevant to the chosen `kind` are required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

from ..constraints import Constraint, ConstraintEvaluation, VelocitySpec, evaluate_constraints, constraints_loglike_gaussian
from ..model import YukawaModel


@dataclass(frozen=True)
class PointConstraint:
    """A thin wrapper around :class:`sidmkit.constraints.Constraint` with averaging info."""

    constraint: Constraint
    average: bool = False
    sigma_1d_km_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.average and (self.sigma_1d_km_s is None or self.sigma_1d_km_s <= 0):
            raise ValueError("average=True requires sigma_1d_km_s > 0")


@dataclass(frozen=True)
class PointConstraintDataset:
    name: str
    points: Sequence[PointConstraint]

    def log_likelihood(self, model: YukawaModel, *, method: str = "auto") -> float:
        evals: list[ConstraintEvaluation] = []
        for p in self.points:
            ev = evaluate_constraints(
                model,
                [p.constraint],
                method=method,
                average=p.average,
                sigma_1d_km_s=p.sigma_1d_km_s,
            )[0]
            evals.append(ev)
        return constraints_loglike_gaussian(evals)

    @staticmethod
    def from_constraints(
        name: str,
        constraints: Sequence[Constraint],
        *,
        average: bool = False,
        sigma_1d_km_s: Optional[float] = None,
    ) -> "PointConstraintDataset":
        pts = [PointConstraint(c, average=average, sigma_1d_km_s=sigma_1d_km_s) for c in constraints]
        return PointConstraintDataset(name=name, points=pts)

    @staticmethod
    def from_json(path: str) -> "PointConstraintDataset":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        name = str(obj.get("name", "point-constraints"))
        constraints_in = obj.get("constraints", [])
        pts: list[PointConstraint] = []
        for cobj in constraints_in:
            kind = cobj["kind"]
            observable = cobj["observable"]
            cname = cobj.get("name", "constraint")

            # Velocity spec
            if "v_km_s" in cobj:
                vel = VelocitySpec.point(float(cobj["v_km_s"]))
            elif "v_min_km_s" in cobj and "v_max_km_s" in cobj:
                vel = VelocitySpec.range(float(cobj["v_min_km_s"]), float(cobj["v_max_km_s"]))
            else:
                raise ValueError(f"Constraint '{cname}' missing velocity fields")

            kwargs = dict(
                name=cname,
                kind=kind,
                observable=observable,
                velocity=vel,
                reference=str(cobj.get("reference", "")),
                notes=str(cobj.get("notes", "")),
            )

            if kind in ("upper", "band"):
                kwargs["upper"] = float(cobj["upper"])
            if kind in ("lower", "band"):
                kwargs["lower"] = float(cobj["lower"])
            if kind == "measurement":
                kwargs["mean"] = float(cobj["mean"])
                kwargs["sigma"] = float(cobj["sigma"])

            c = Constraint(**kwargs)

            avg = bool(cobj.get("average", False))
            sig = cobj.get("sigma_1d_km_s", None)
            sig_val = float(sig) if sig is not None else None
            pts.append(PointConstraint(constraint=c, average=avg, sigma_1d_km_s=sig_val))

        return PointConstraintDataset(name=name, points=pts)

