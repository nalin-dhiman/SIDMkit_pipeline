# SPARC batch outputs (already computed)

This directory contains the outputs produced by running the batch fitter on the
SPARC rotmod sample.

## Key files

- `outputs/sparc_chunks/`  
  Chunk directories. Each contains:
  - `fits/<GALAXY>.json` (per-galaxy fit results for each model)
  - `plots/<GALAXY>_fit.png` (per-galaxy plot)
  - `summary.json`, `summary.csv` (row-per-fit summary)
  - `chunk_stats.json` (meta info)

- `outputs/sparc_all_summary.json`  
  Merged summary across all chunks (two rows per galaxy: NFW and Burkert).

- `outputs/sparc_report/`  
  Original report produced in the run that generated this submission bundle.

- `outputs/sparc_report_v2/`  
  Report regenerated with the updated `sidmkit.sparc_batch report` command.

## Quick population numbers (from `sparc_report_v2/population_stats.json`)

- N galaxies with both models: 191
- Median ΔBIC: 1.812
- Mean ΔBIC: 12.915
- Fraction Burkert-preferred (ΔBIC>0): 0.654
- Strong Burkert (ΔBIC>6): 0.325
- Strong NFW (ΔBIC<-6): 0.147

**Caution:** this is a phenomenological baseline comparison; do not interpret it
as a microphysical SIDM inference without additional modeling.
