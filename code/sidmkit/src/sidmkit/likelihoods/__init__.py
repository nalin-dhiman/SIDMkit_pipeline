"""Dataset-specific likelihoods.

The curated constraint sets in :mod:`sidmkit.constraints` are useful, but they
are not a substitute for dataset-level inference. This package provides small,
extensible likelihood implementations.

Currently included:
- point constraints (fast, summary-level, but explicit and reproducible)
- rotation curve likelihood (toy semi-analytic halo mapping; demo-quality)

For publication-grade work you should:
- implement the forward model appropriate to your dataset,
- verify velocity averaging and observable definitions,
- validate against the original analysis pipeline of the dataset.
"""

from .point_constraints import PointConstraint, PointConstraintDataset

from .rotation_curve import RotationCurveData, RotationCurveLikelihood
