"""Quickstart example for sidmkit (v0.2+).

Run from repo root (after `pip install -e .`):
    python examples/quickstart.py
"""

from sidmkit import (
    YukawaModel,
    PotentialType,
    sigma_over_m,
    average_sigma_over_m,
    get_constraint_set,
    evaluate_constraints,
)

model = YukawaModel(m_chi_gev=10.0, m_med_gev=0.05, alpha=1e-2, potential=PotentialType.ATTRACTIVE)

print("Sigma/m at representative velocities:")
for v in [30, 100, 1000, 3000]:
    print(f"v={v:4.0f} km/s  sigma/m = {float(sigma_over_m(v, model=model)):.3g} cm^2/g")

print("\nVelocity-averaged <σ v>/m for σ1D=50 km/s:")
avg = average_sigma_over_m(model, sigma_1d_km_s=50.0, moment_n=1.0)
print(f"<σ v>/m = {avg.value:.3g} {avg.unit}")

print("\nEvaluate a curated constraint set:")
constraints = get_constraint_set("tulin_yu_table1")
evals = evaluate_constraints(model, constraints)
for e in evals:
    print(f"{e.constraint.name:45s}  pred={e.predicted:.3g}  pass={e.passed}")

