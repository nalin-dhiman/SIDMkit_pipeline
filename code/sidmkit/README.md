# sidmkit

A transparent toolkit for **self-interacting dark matter (SIDM)** micro→macro work:

- Yukawa / dark-photon style self-interaction cross sections (Born / classical / Hulthén / partial-wave)
- **Velocity averaging** (Maxwellian relative-speed baseline) for ⟨σ/m⟩ and ⟨σ v⟩/m
- Curated **summary constraint sets** (from reviews) for fast sanity checks
- A simple halo-level mapping via the **interaction radius** `r1` (Γ(r1) t_age = 1)
- Lightweight **likelihood scaffolds** + a tiny Metropolis MCMC for runnable end-to-end demos
- Benchmarks & regression numbers to catch numerical/unit regressions

> sidmkit is designed to be easy to audit and extend.
> The included constraints and likelihoods are *starting points*, not publication-grade analyses.

---

## Install

```bash
git clone
cd sidmkit
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

---

## CLI overview

```bash
sidmkit --help
sidmkit sigma --help
sidmkit avg --help
sidmkit constraints --help
sidmkit halo --help
sidmkit likelihood --help
sidmkit infer --help
sidmkit benchmark --help
sidmkit validate --help
```

---

## End-to-end pipeline

This is the **recommended** reproducible workflow.

### 0) Run internal benchmarks (sanity + regression)

```bash
sidmkit benchmark
# optional slow partial-wave checks:
sidmkit benchmark --slow
```

### 1) Compute σ/m(v) and save a curve

Example model:
- mχ = 10 GeV
- m_med = 0.05 GeV (50 MeV)
- α = 0.01
- attractive Yukawa

```bash
sidmkit sigma --mchi 10 --mmed 0.05 --alpha 0.01 --potential attractive \
  --vmin 1 --vmax 5000 --n 200 --csv sigma_curve.csv
```

### 2) Do the velocity average (this is upgrade #1)

Compute ⟨σ v⟩/m for a Maxwellian relative-speed distribution with σ1D = 50 km/s:

```bash
sidmkit avg --mchi 10 --mmed 0.05 --alpha 0.01 --potential attractive \
  --sigma1d 50 --moment 1
```

- `moment=0` → ⟨σ/m⟩
- `moment=1` → ⟨σ v⟩/m

### 3) Check curated constraint sets (upgrade #2)

List available sets:

```bash
sidmkit constraints --list-sets --mchi 10 --mmed 0.05 --alpha 0.01
```

Evaluate a modern summary set (example: Adhikari+ review Table I):

```bash
sidmkit constraints --set literature_table1_summary --mchi 10 --mmed 0.05 --alpha 0.01
```

Or evaluate the classic Tulin & Yu Table I set:

```bash
sidmkit constraints --set literature_table1_compilation --mchi 10 --mmed 0.05 --alpha 0.01
```

### 4) Halo-level mapping: compute r1 for an NFW halo (upgrade #3)

Example: MW-like halo (M200=1e12 Msun, c200=10), age 10 Gyr:

```bash
sidmkit halo --mchi 10 --mmed 0.05 --alpha 0.01 \
  --m200 1e12 --c200 10 --tage 10 --vmode jeans
```

This solves for **r1** where Γ(r1) t_age = 1 using:
- NFW density profile,
- isotropic Jeans velocity dispersion,
- Maxwellian relative-speed averaging for ⟨σ v⟩/m.

### 5) Dataset-style likelihood + inference (upgrades #4 and #5)

We ship two “starter” likelihoods:

1) **Point constraints dataset** (JSON): reproducible, auditable, summary-level.
2) **Toy rotation-curve likelihood** (CSV): demonstrates micro→macro → V_circ(r).

#### 5a) Point-constraints likelihood

Example dataset file included:

- `examples/datasets/kaplinghat2016_summary_plus_bullet.json`

Evaluate log-likelihood at a model point:

```bash
sidmkit likelihood --kind points --data examples/datasets/kaplinghat2016_summary_plus_bullet.json \
  --mchi 15 --mmed 0.017 --alpha 0.00729927 --potential attractive
```

Run a small Metropolis MCMC:

```bash
sidmkit infer --data examples/datasets/kaplinghat2016_summary_plus_bullet.json \
  --potential attractive --method auto --n-steps 4000 --seed 0
```

#### 5b) Toy rotation-curve likelihood (demo)

Example dataset file included:

- `examples/datasets/toy_rotation_curve.csv`

Evaluate the rotation-curve log-likelihood (you must specify a halo M200,c200):

```bash
sidmkit likelihood --kind rotation_curve --data examples/datasets/toy_rotation_curve.csv \
  --m200 1e11 --c200 12 --tage 10 \
  --mchi 10 --mmed 0.05 --alpha 0.01 --potential attractive
```

> Be critical: the rotation-curve model is deliberately simple (isothermal core matched to NFW at r1).
> It exists to show the mechanics of a forward model, not to replace full analyses.

---

## Validation hooks (upgrade #4)

### Validate against an external curve (your digitized data)

If you digitize a paper’s σ/m(v) curve into a CSV with columns:
`v_km_s, sigma_over_m_cm2_g`, you can compare:

```bash
sidmkit validate --target curve --reference your_curve.csv \
  --mchi 10 --mmed 0.05 --alpha 0.01 --potential attractive
```



This is a non-authoritative regression test using a small set of auto-digitized points:

```bash
sidmkit validate --target fig13 --mchi 15 --mmed 0.017 --alpha 0.00729927 --potential attractive
```


---

## SPARC rotation-curve batch fits (phenomenological baseline)

sidmkit also ships a self-contained batch fitter for the SPARC `*_rotmod.dat` files.
It fits simple halo profiles (NFW, Burkert) plus stellar mass-to-light parameters.

Install plotting extra (recommended):

```bash
pip install -e ".[plot]"
```

### 1) Batch fit in chunks (recommended)

```bash
python -m sidmkit.sparc_batch batch   --inputs path/to/Rotmod_LTG path/to/Rotmod_ETG   --outdir outputs/sparc_chunks/chunk_0   --skip 0 --limit 25   --plots --plot-format png
```

Run additional chunks by increasing `--skip` (e.g. 25, 50, ...). Use `--resume`
to make reruns idempotent.

### 2) Merge chunk summaries

```bash
python -m sidmkit.sparc_batch merge   --inputs outputs/sparc_chunks/chunk_*/summary.json   --out outputs/sparc_all_summary.json
```

### 3) Population report

```bash
python -m sidmkit.sparc_batch report   --summary-json outputs/sparc_all_summary.json   --outdir outputs/sparc_report
```

> Be critical: these are halo-profile fits (baseline model comparison), not a
> full SIDM microphysics → core-size inference of SPARC.


---

## Python quickstart

```bash
python examples/quickstart.py
```

---

## License

MIT
