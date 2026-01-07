"""Small grid scan demo.

This is NOT an inference workflow; it's a quick check that the pipeline runs end-to-end.
"""

from sidmkit import PotentialType
from sidmkit.scan import grid_scan

rows = grid_scan(
    m_chi_grid=[1.0, 10.0, 100.0],
    m_med_grid=[0.01, 0.04, 0.1],
    alpha_grid=[1e-3, 1e-2, 1e-1],
    potential=PotentialType.ATTRACTIVE,
)

# Print passing points.
for r in rows:
    if r["pass_all"]:
        print(r)
