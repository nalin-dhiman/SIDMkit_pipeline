# SIDMKit + SPARC submission runbook

This folder is a *complete* reproducibility bundle:
- `code/` contains the PyPI-ready `sidmkit` source tree (editable-install friendly).
- `results/` contains the already-generated SPARC batch outputs (plots + summaries).
- `paper/` contains the LaTeX manuscript + figures + a compiled PDF.

This runbook tells you how to re-run everything from scratch.

---

## 0) Environment (recommended)

Create a fresh conda env (or use venv):

```bash
conda create -n sidmkit python=3.11 -y
conda activate sidmkit
```

Install the package in editable mode (with plotting extras):

```bash
cd code/sidmkit
pip install -e ".[plot,dev]"
```

Sanity check:

```bash
sidmkit benchmark
sidmkit-sparc --help
```

---

## 1) SPARC data layout

You need the SPARC rotmod files (LTG + ETG). There are two supported options:

### Option A (recommended): unzip the datasets

```bash
mkdir -p sparc_data
unzip -q Rotmod_LTG.zip -d sparc_data/Rotmod_LTG
unzip -q Rotmod_ETG.zip -d sparc_data/Rotmod_ETG
```

### Option B: point directly at the zip files

This is supported but slower than unzipping.

---

## 2) Run the batch fits in chunks

From the repository root (or anywhere):

```bash
python -m sidmkit.sparc_batch batch \
  --inputs sparc_data/Rotmod_LTG sparc_data/Rotmod_ETG \
  --outdir outputs/sparc_chunks/chunk_0 \
  --skip 0 --limit 25 \
  --plots --plot-format png
```

Repeat for additional chunks by incrementing `--skip`:

- chunk_1: `--skip 25 --limit 25`
- chunk_2: `--skip 50 --limit 25`
- ...

Useful flags:
- `--resume` skips galaxies that already have `outputs/.../fits/<galaxy>.json`
- `--no-priors` disables the default weak M/L priors (pure max-likelihood fits)
- `--loss soft_l1` can reduce sensitivity to outliers

---

## 3) Merge chunk summaries into one file

```bash
python -m sidmkit.sparc_batch merge \
  --inputs outputs/sparc_chunks/chunk_*/summary.json \
  --out outputs/sparc_all_summary.json
```

---

## 4) Make population-level plots

```bash
python -m sidmkit.sparc_batch report \
  --summary-json outputs/sparc_all_summary.json \
  --outdir outputs/sparc_report \
  --plot-format png
```

This writes:
- `delta_bic_hist.png`
- `chi2red_hist.png`
- `chi2red_scatter.png`
- `nfw_logrs_hist.png`
- `burkert_logr0_hist.png`
- `population_stats.json`

---

## 5) Build the paper PDF

```bash
cd paper
make
```

Outputs: `paper/sidmkit_sparc_submission.pdf`

---

## 6) PyPI packaging checklist

From `code/sidmkit/`:

```bash
python -m build
twine check dist/*
```

This produces `dist/sidmkit-*.tar.gz` and `dist/sidmkit-*.whl`.

(Uploading requires credentials and is intentionally not automated here.)
