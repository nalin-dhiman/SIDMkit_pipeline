"""Constraint helpers and curated constraint sets.

This module serves two purposes:

1) A small data model for representing common SIDM constraints in a way that is
   easy to evaluate consistently.
2) A curated library of *summary* constraints from the literature.

Be critical / reviewer-proof notes
----------------------------------
- Most published “constraints” are *model-dependent* summaries.
- Many papers quote σ/m at a single “characteristic” velocity, but what enters
  a real observable may be an average ⟨σ v⟩/m or even a higher velocity moment.
- Treat the curated sets as *starting points*. For a serious analysis you
  should implement a dataset-specific likelihood (rotation curves, lensing,
  cluster mergers, ...).

We still include curated sets because they are extremely useful for:
- fast sanity checks,
- regression tests (“did my code change the curve?”),
- exploratory parameter scans.

Some recent reviews define alternative angular weights for identical particles,
e.g. a modified transfer cross section ∫(1-|cosθ|) dσ/dΩ. We expose a field for
that, but note that most astrophysical constraints are still usually phrased in
terms of the momentum-transfer cross section σ_T.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import numpy as np

from .averaging import average_sigma_over_m
from .cross_sections import sigma_over_m
from .model import YukawaModel


Observable = Literal["sigma_over_m", "sigma_v_over_m"]
ConstraintKind = Literal["upper", "lower", "band", "measurement"]
VelocityKind = Literal["point", "range"]


@dataclass(frozen=True)
class VelocitySpec:
    """How a constraint specifies the relevant velocity scale."""

    kind: VelocityKind
    v_km_s: Optional[float] = None
    v_min_km_s: Optional[float] = None
    v_max_km_s: Optional[float] = None

    @staticmethod
    def point(v_km_s: float) -> "VelocitySpec":
        return VelocitySpec(kind="point", v_km_s=float(v_km_s))

    @staticmethod
    def range(v_min_km_s: float, v_max_km_s: float) -> "VelocitySpec":
        if v_min_km_s <= 0 or v_max_km_s <= 0 or v_max_km_s <= v_min_km_s:
            raise ValueError("Require 0 < v_min < v_max for VelocitySpec.range")
        return VelocitySpec(kind="range", v_min_km_s=float(v_min_km_s), v_max_km_s=float(v_max_km_s))

    def representative(self) -> float:
        """Return a single representative speed in km/s.

        For ranges we use the geometric mean by default (appropriate on log axes).
        """
        if self.kind == "point":
            if self.v_km_s is None:
                raise ValueError("VelocitySpec.point requires v_km_s")
            return float(self.v_km_s)
        if self.v_min_km_s is None or self.v_max_km_s is None:
            raise ValueError("VelocitySpec.range requires v_min_km_s and v_max_km_s")
        return float(np.sqrt(self.v_min_km_s * self.v_max_km_s))


@dataclass(frozen=True)
class Constraint:
    """A single constraint or preferred band."""

    name: str
    kind: ConstraintKind
    observable: Observable
    velocity: VelocitySpec

    # For upper/lower/band constraints (in the unit implied by observable)
    lower: Optional[float] = None
    upper: Optional[float] = None

    # For measurement constraints (Gaussian; asymmetric optional)
    mean: Optional[float] = None
    sigma: Optional[float] = None

    # Meta
    reference: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind in ("upper", "lower", "band"):
            if self.kind == "upper" and self.upper is None:
                raise ValueError("upper constraint requires 'upper'")
            if self.kind == "lower" and self.lower is None:
                raise ValueError("lower constraint requires 'lower'")
            if self.kind == "band" and (self.lower is None or self.upper is None):
                raise ValueError("band constraint requires both 'lower' and 'upper'")
        if self.kind == "measurement":
            if self.mean is None or self.sigma is None:
                raise ValueError("measurement requires mean and sigma")


@dataclass(frozen=True)
class ConstraintEvaluation:
    constraint: Constraint
    predicted: float
    passed: bool


def _predict_observable(
    model: YukawaModel,
    c: Constraint,
    *,
    method: str = "auto",
    average: bool = False,
    sigma_1d_km_s: Optional[float] = None,
) -> float:
    """Predict the observable for a constraint.

    If average=False we evaluate at a representative speed:
        sigma_over_m(v_rep) or sigma_over_m(v_rep)*v_rep

    If average=True we compute a Maxwellian velocity average with given sigma_1d_km_s:
        ⟨σ/m⟩ or ⟨σ v⟩/m
    """
    v_rep = c.velocity.representative()
    if not average:
        som = float(sigma_over_m(v_rep, model=model, method=method, out_unit="cm2/g"))
        if c.observable == "sigma_over_m":
            return som
        if c.observable == "sigma_v_over_m":
            return som * v_rep
        raise ValueError(f"Unknown observable: {c.observable}")

    if sigma_1d_km_s is None:
        raise ValueError("sigma_1d_km_s must be provided when average=True")
    moment_n = 0.0 if c.observable == "sigma_over_m" else 1.0
    res = average_sigma_over_m(model, sigma_1d_km_s=sigma_1d_km_s, moment_n=moment_n, method=method)
    return float(res.value)


def evaluate_constraints(
    model: YukawaModel,
    constraints: Sequence[Constraint],
    *,
    method: str = "auto",
    average: bool = False,
    sigma_1d_km_s: Optional[float] = None,
) -> list[ConstraintEvaluation]:
    """Evaluate a list of constraints.

    Parameters
    ----------
    average:
        If True, replace point evaluation with Maxwellian velocity-averaging using sigma_1d_km_s.
        This is a simple but often more realistic comparison for systems with broad velocity
        distributions.
    """
    out: list[ConstraintEvaluation] = []
    for c in constraints:
        pred = _predict_observable(model, c, method=method, average=average, sigma_1d_km_s=sigma_1d_km_s)
        passed = True
        if c.kind == "upper":
            passed = pred <= float(c.upper)
        elif c.kind == "lower":
            passed = pred >= float(c.lower)
        elif c.kind == "band":
            passed = (pred >= float(c.lower)) and (pred <= float(c.upper))
        elif c.kind == "measurement":
            # A measurement is not a pass/fail constraint; treat "passed" as within 2σ.
            passed = abs(pred - float(c.mean)) <= 2.0 * float(c.sigma)
        else:
            raise ValueError(f"Unknown constraint kind: {c.kind}")
        out.append(ConstraintEvaluation(constraint=c, predicted=float(pred), passed=bool(passed)))
    return out


def constraints_loglike_gaussian(
    evals: Sequence[ConstraintEvaluation],
    *,
    sigma_softening: float = 0.0,
) -> float:
    """Turn constraint evaluations into a *toy* log-likelihood.

    This is intentionally conservative and smooth:

    - measurement: Gaussian log-likelihood
    - upper/lower: one-sided Gaussian penalty beyond the bound
    - band: Gaussian penalty outside band (distance to nearest edge)

    Parameters
    ----------
    sigma_softening:
        Additive softening term to avoid infinite slopes when uncertainties are tiny.
        Units are the same as the observable for each constraint.

    Notes
    -----
    This is NOT a substitute for a dataset-specific likelihood. It is meant for
    demonstrations and quick inference pipelines.
    """
    ll = 0.0
    for e in evals:
        c = e.constraint
        pred = e.predicted
        if c.kind == "measurement":
            sig = float(c.sigma) + float(sigma_softening)
            ll += -0.5 * ((pred - float(c.mean)) / sig) ** 2
        elif c.kind == "upper":
            if pred <= float(c.upper):
                continue
            # soft penalty: assume 20% relative uncertainty unless told otherwise
            sig = 0.2 * float(c.upper) + float(sigma_softening)
            ll += -0.5 * ((pred - float(c.upper)) / sig) ** 2
        elif c.kind == "lower":
            if pred >= float(c.lower):
                continue
            sig = 0.2 * float(c.lower) + float(sigma_softening)
            ll += -0.5 * ((pred - float(c.lower)) / sig) ** 2
        elif c.kind == "band":
            lo = float(c.lower)
            hi = float(c.upper)
            if lo <= pred <= hi:
                continue
            # distance to nearest edge
            d = min(abs(pred - lo), abs(pred - hi))
            sig = 0.2 * (hi - lo) + float(sigma_softening)
            ll += -0.5 * (d / sig) ** 2
        else:
            raise ValueError(f"Unknown constraint kind: {c.kind}")
    return float(ll)


# ---------------------------------------------------------------------------
# Curated constraint sets (summary-level)
# ---------------------------------------------------------------------------

def _set_illustrative() -> list[Constraint]:
    # This is the original toy set shipped with sidmkit v0.1.x.
    return [
        Constraint(
            name="Dwarf galaxies (core formation)",
            kind="band",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(30.0),
            lower=0.1,
            upper=10.0,
            reference="Illustrative summary (common SIDM heuristic)",
        ),
        Constraint(
            name="LSB / spiral galaxies (rotation curves)",
            kind="band",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(100.0),
            lower=0.1,
            upper=10.0,
            reference="Illustrative summary (common SIDM heuristic)",
        ),
        Constraint(
            name="Clusters (merging / lensing)",
            kind="upper",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(1000.0),
            upper=1.0,
            reference="Illustrative summary (common SIDM heuristic)",
        ),
        Constraint(
            name="Bullet Cluster",
            kind="upper",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(4000.0),
            upper=0.7,
            reference="Illustrative summary; see e.g. Tulin & Yu review Table I",
        ),
    ]


def _set_tulin_yu_table1() -> list[Constraint]:
    # Values are taken from Table I of Tulin & Yu (2017/2018) review:
    # "Dark Matter Self-interactions and Small Scale Structure".
    # See parsed Table I lines for v_rel and σ/m entries.
    return [
        # Positive observations / preferred ranges
        Constraint(
            name="Cores in spiral & dwarf/LSB galaxies (rotation curves)",
            kind="lower",
            observable="sigma_over_m",
            velocity=VelocitySpec.range(30.0, 200.0),
            lower=1.0,
            reference="Literature review Table I: rotation curves; v_rel 30–200 km/s; σ/m ≳ 1 cm^2/g",
        ),
        Constraint(
            name="Too-big-to-fail (Milky Way)",
            kind="lower",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(50.0),
            lower=0.6,
            reference="Literature review Table I: TBTF MW; v_rel 50 km/s; σ/m ≳ 0.6 cm^2/g",
        ),
        Constraint(
            name="Too-big-to-fail (Local Group)",
            kind="lower",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(50.0),
            lower=0.5,
            reference="Literature review Table I: TBTF Local Group; v_rel 50 km/s; σ/m ≳ 0.5 cm^2/g",
        ),
        Constraint(
            name="Cores in clusters (stellar dispersion, lensing)",
            kind="measurement",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(1500.0),
            mean=0.1,
            sigma=0.05,
            reference="Literature review Table I: clusters; v_rel 1500 km/s; σ/m ~ 0.1 cm^2/g",
            notes="The quoted value is approximate; we encode a loose ±0.05 for demonstration.",
        ),
        Constraint(
            name="Abell 3827 subhalo merger (DM–galaxy offset)",
            kind="measurement",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(1500.0),
            mean=1.5,
            sigma=0.5,
            reference="Literature review Table I: Abell 3827; v_rel 1500 km/s; σ/m ~ 1.5 cm^2/g",
            notes="Treated as a noisy 'measurement' for demo; interpretations vary.",
        ),
        Constraint(
            name="Abell 520 cluster merger (DM–galaxy offset)",
            kind="measurement",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(2500.0),
            mean=1.0,
            sigma=0.5,
            reference="Literature review Table I: Abell 520; v_rel 2000–3000 km/s; σ/m ~ 1 cm^2/g",
            notes="We use v_rep=2500 km/s and a loose uncertainty.",
        ),
        # Constraints
        Constraint(
            name="Halo shapes / ellipticity (cluster lensing surveys)",
            kind="upper",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(1300.0),
            upper=1.0,
            reference="Literature review Table I: halo shapes; v_rel 1300 km/s; σ/m ≲ 1 cm^2/g",
        ),
        Constraint(
            name="Substructure mergers (DM–galaxy offsets)",
            kind="upper",
            observable="sigma_over_m",
            velocity=VelocitySpec.range(500.0, 4000.0),
            upper=2.0,
            reference="Literature review Table I: substructure mergers; v_rel ~500–4000 km/s; σ/m ≲ 2 cm^2/g",
        ),
        Constraint(
            name="Merging clusters (post-merger halo survival, optical depth τ<1)",
            kind="upper",
            observable="sigma_over_m",
            velocity=VelocitySpec.range(2000.0, 4000.0),
            upper=3.0,
            reference="Literature review Table I: merging clusters; v_rel 2000–4000 km/s; σ/m ≲ few cm^2/g",
            notes="We encode 'few' as 3 cm^2/g.",
        ),
        Constraint(
            name="Bullet Cluster (mass-to-light ratio)",
            kind="upper",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(4000.0),
            upper=0.7,
            reference="Literature review Table I: Bullet Cluster; v_rel 4000 km/s; σ/m ≲ 0.7 cm^2/g",
        ),
    ]


def _set_adhikari_table1() -> list[Constraint]:
    # Table I from Adhikari et al. "Astrophysical Tests of Dark Matter Self-Interactions" (2025)
    # contains a modern summary of limits.
    # We encode them as upper/lower constraints.
    return [
        Constraint(
            name="Clusters v>2000 (strong lensing)",
            kind="upper",
            observable="sigma_over_m",
            velocity=VelocitySpec.range(2000.0, 4000.0),
            upper=0.28,
            reference="Literature review Table I: clusters v>2000 strong lensing σ/m < 0.28",
        ),
        Constraint(
            name="Clusters v>2000 (BCG offsets)",
            kind="upper",
            observable="sigma_over_m",
            velocity=VelocitySpec.range(2000.0, 4000.0),
            upper=0.39,
            reference="Literature review Table I: clusters v>2000 BCG offsets σ/m < 0.39",
        ),
        Constraint(
            name="Clusters v>2000 (merging clusters)",
            kind="upper",
            observable="sigma_over_m",
            velocity=VelocitySpec.range(2000.0, 4000.0),
            upper=1.25,
            reference="Literature review Table I: clusters v>2000 merging clusters σ/m < 1.25",
        ),
        Constraint(
            name="Groups 500<v<2000 (strong lensing)",
            kind="upper",
            observable="sigma_over_m",
            velocity=VelocitySpec.range(500.0, 2000.0),
            upper=0.9,
            reference="Literature review Table I: groups 500<v<2000 strong lensing σ/m < 0.9",
        ),
        Constraint(
            name="Groups 500<v<2000 (strong lensing, combined analysis)",
            kind="upper",
            observable="sigma_over_m",
            velocity=VelocitySpec.range(500.0, 2000.0),
            upper=0.13,
            reference="Literature review Table I: groups 500<v<2000 strong lensing σ/m < 0.13",
            notes="This is a stringent combined limit; treat with care about systematics.",
        ),
        Constraint(
            name="Groups 500<v<2000 (weak lensing)",
            kind="upper",
            observable="sigma_over_m",
            velocity=VelocitySpec.range(500.0, 2000.0),
            upper=1.0,
            reference="Literature review Table I: groups 500<v<2000 weak lensing σ/m < 1",
        ),
        Constraint(
            name="Galaxy halos 100<v<500 (rotation curves)",
            kind="lower",
            observable="sigma_over_m",
            velocity=VelocitySpec.range(100.0, 500.0),
            lower=3.0,
            reference="Literature review Table I: galaxy halos rotation curves σ/m > 3",
            notes="This is a 'preferred' value from specific fits; not a universal requirement.",
        ),
        Constraint(
            name="Galaxy halos (satellite counts / evaporation)",
            kind="upper",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(200.0),
            upper=10.0,
            reference="Literature review Table I: satellite counts σ/m < 5–10 at ~200 km/s",
            notes="We encode the upper end (10) as a conservative bound.",
        ),
        Constraint(
            name="Dwarf halos v<100 (density profiles)",
            kind="lower",
            observable="sigma_over_m",
            velocity=VelocitySpec.range(10.0, 100.0),
            lower=3.0,
            reference="Literature review Table I: dwarf halos density profiles σ/m ≥ 3",
            notes="Simulation-motivated; treat as a broad preference, not a hard bound.",
        ),
    ]


def _set_kaplinghat2016_summary() -> list[Constraint]:
    # From "Current status of self SIDM" summary: constant cross section fits
    # clusters: 0.10^{+0.03}_{-0.02} cm^2/g
    # galaxies: 1.9^{+0.6}_{-0.5} cm^2/g
    return [
        Constraint(
            name="Clusters (constant σ/m fit)",
            kind="measurement",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(1500.0),
            mean=0.10,
            sigma=0.03,
            reference="Literature constant-σ/m summary: clusters best-fit σ/m=0.10^{+0.03}_{-0.02}",
        ),
        Constraint(
            name="Galaxies (constant σ/m fit)",
            kind="measurement",
            observable="sigma_over_m",
            velocity=VelocitySpec.point(100.0),
            mean=1.9,
            sigma=0.6,
            reference="Literature constant-σ/m summary: galaxies best-fit σ/m=1.9^{+0.6}_{-0.5}",
        ),
    ]


_PRIMARY_CONSTRAINT_SETS = {
    # Minimal / demo set used by docs and tests
    "illustrative": _set_illustrative,

    # Curated literature compilations (names avoid author strings in the CLI)
    "literature_table1_compilation": _set_tulin_yu_table1,
    "literature_table1_summary": _set_adhikari_table1,
    "constant_sigma_summary": _set_kaplinghat2016_summary,
}

# Backwards-compatible aliases (kept for older scripts; hidden from --list-sets by default)
_CONSTRAINT_SET_ALIASES = {
    "tulin_yu_table1": "literature_table1_compilation",
    "adhikari_table1": "literature_table1_summary",
    "kaplinghat2016_summary": "constant_sigma_summary",
}


def list_constraint_sets(include_aliases: bool = False) -> list[str]:
    keys = sorted(_PRIMARY_CONSTRAINT_SETS.keys())
    if include_aliases:
        keys = sorted(set(keys) | set(_CONSTRAINT_SET_ALIASES.keys()))
    return keys


def resolve_constraint_set_name(name: str) -> str:
    """Return the canonical constraint-set key for *name* (aliases allowed)."""
    key = name.strip()
    return _CONSTRAINT_SET_ALIASES.get(key, key)

def get_constraint_set(name: str) -> list[Constraint]:
    key = name.strip()
    key = _CONSTRAINT_SET_ALIASES.get(key, key)
    if key not in _PRIMARY_CONSTRAINT_SETS:
        raise KeyError(
            f"Unknown constraint set '{name}'. Available: {', '.join(list_constraint_sets(include_aliases=True))}"
        )
    return list(_PRIMARY_CONSTRAINT_SETS[key]())


# Backwards compatibility: old name used by CLI and docs.
DEFAULT_CONSTRAINTS: list[Constraint] = _set_illustrative()

