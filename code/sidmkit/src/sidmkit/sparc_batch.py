"""SPARC/rotmod batch fitting utilities.

This module provides a *phenomenological* rotation-curve fitting pipeline for
the SPARC "rotmod" files (gas + stellar templates + a parametric DM halo).

Design goals
------------
- Reproducible, robust batch processing across the full SPARC sample.
- Explicit, inspectable assumptions (priors, bounds, objective function).
- "Submission-grade" ergonomics for large runs via chunking:
    --skip N
    --limit N
  so you can generate outputs in manageable pieces (or parallelize).

Non-goals (important!)
----------------------
- This is **not** a full microphysical SIDM inference of SPARC (yet).
  We fit NFW/Burkert halos as a baseline. Mapping Yukawa SIDM parameters to
  galaxy-scale core sizes requires additional modeling (e.g. Jeans/heat
  conduction calibrations, baryonic systematics), which we intentionally
  separate from this batch-fitting layer.

Example (chunked batch run)
---------------------------
Unzip the SPARC rotmod zips (recommended; faster than reading inside zip):

  unzip -q Rotmod_LTG.zip -d sparc_data/Rotmod_LTG
  unzip -q Rotmod_ETG.zip -d sparc_data/Rotmod_ETG

Run 25 galaxies at a time:

  python -m sidmkit.sparc_batch batch \
    --inputs sparc_data/Rotmod_LTG sparc_data/Rotmod_ETG \
    --outdir outputs/chunk_0 \
    --skip 0 --limit 25 \
    --plots --plot-format png

Merge chunk summaries:

  python -m sidmkit.sparc_batch merge \
    --inputs outputs/chunk_*/summary.json \
    --out outputs/sparc_all_summary.json

Population report:

  python -m sidmkit.sparc_batch report \
    --summary-json outputs/sparc_all_summary.json \
    --outdir outputs/sparc_report
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import least_squares

# Matplotlib is an optional dependency. Import lazily to avoid slow import-time
# side effects (e.g. font cache builds) when users only want numeric outputs.
_HAVE_MPL = False
plt = None  # type: ignore


def _lazy_import_mpl() -> None:
    global _HAVE_MPL, plt
    if _HAVE_MPL:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as _plt

        plt = _plt
        _HAVE_MPL = True
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Plotting requested but matplotlib is not available. "
            "Install it with: pip install matplotlib"
        ) from e


# Gravitational constant in units: (km/s)^2 * kpc / Msun
G_KPC_KMS2_PER_MSUN = 4.30091e-6


_DIST_RE = re.compile(r"Distance\s*=\s*([0-9.]+)\s*Mpc", re.IGNORECASE)


@dataclass(frozen=True)
class RotmodData:
    """Minimal SPARC rotmod data needed for baryon+DM rotation curve fits."""

    galaxy: str
    source: str
    distance_mpc: float | None
    r_kpc: np.ndarray
    v_obs_km_s: np.ndarray
    v_err_km_s: np.ndarray
    v_gas_km_s: np.ndarray
    v_disk_km_s: np.ndarray
    v_bul_km_s: np.ndarray
    sb_disk_lsun_pc2: np.ndarray | None = None
    sb_bul_lsun_pc2: np.ndarray | None = None

    def has_bulge(self) -> bool:
        # Some files include a bulge column but it may be all zeros.
        return bool(np.any(np.abs(self.v_bul_km_s) > 1e-8))

    @property
    def n(self) -> int:
        return int(self.r_kpc.size)


@dataclass
class FitResult:
    """Fit summary for one galaxy and one halo model."""

    galaxy: str
    source: str
    distance_mpc: float | None
    model: str
    n_points: int
    k_params: int
    chi2: float
    dof: int
    chi2_red: float | None
    aic: float | None
    bic: float | None
    ups_disk: float
    ups_bul: float | None
    halo_params: dict
    success: bool
    message: str
    runtime_s: float


def _read_text_from_path(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_text_from_zip(zip_path: Path, inner: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(inner, "r") as f:
            return f.read().decode("utf-8", errors="replace")


def iter_rotmod_files(inputs: Sequence[str]) -> list[tuple[str, str, str]]:
    """Return a sorted list of rotmod sources.

    Each item is (galaxy_name, source_id, file_text).

    For the SPARC-sized sample, holding all file text in memory is acceptable.
    """

    items: list[tuple[str, str, str]] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.rglob("*_rotmod.dat")):
                galaxy = f.name.replace("_rotmod.dat", "")
                src = str(f)
                items.append((galaxy, src, _read_text_from_path(f)))
        elif p.is_file() and p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p, "r") as z:
                names = [n for n in z.namelist() if n.lower().endswith("_rotmod.dat")]
            for n in sorted(names):
                galaxy = Path(n).name.replace("_rotmod.dat", "")
                src = f"{p}!{n}"
                items.append((galaxy, src, _read_text_from_zip(p, n)))
        elif p.is_file() and p.name.lower().endswith("_rotmod.dat"):
            galaxy = p.name.replace("_rotmod.dat", "")
            src = str(p)
            items.append((galaxy, src, _read_text_from_path(p)))
        else:
            raise FileNotFoundError(f"Input not found or unsupported: {raw}")
    items.sort(key=lambda t: (t[0].lower(), t[1]))
    return items


def parse_rotmod_text(galaxy: str, source: str, text: str) -> RotmodData:
    """Parse a SPARC rotmod file (LTG or ETG style)."""

    distance_mpc: float | None = None
    for line in text.splitlines()[:30]:
        m = _DIST_RE.search(line)
        if m:
            try:
                distance_mpc = float(m.group(1))
            except Exception:
                distance_mpc = None
            break

    arr = np.genfromtxt(io.StringIO(text), comments="#")
    arr = np.atleast_2d(arr)
    if arr.size == 0 or arr.shape[1] < 6:
        raise ValueError(f"{source}: expected >=6 numeric columns, got shape={arr.shape}")

    r = arr[:, 0].astype(float)
    vobs = arr[:, 1].astype(float)
    verr = arr[:, 2].astype(float)
    vgas = arr[:, 3].astype(float)
    vdisk = arr[:, 4].astype(float)
    vbul = arr[:, 5].astype(float)

    sbdisk = arr[:, 6].astype(float) if arr.shape[1] >= 7 else None
    sbbul = arr[:, 7].astype(float) if arr.shape[1] >= 8 else None

    # Remove non-positive radii, non-finite, and invalid uncertainties.
    mask = (
        np.isfinite(r)
        & np.isfinite(vobs)
        & np.isfinite(verr)
        & np.isfinite(vgas)
        & np.isfinite(vdisk)
        & np.isfinite(vbul)
        & (r > 0)
        & (verr > 0)
    )
    if mask.sum() < 2:
        raise ValueError(f"{source}: too few valid data points after cleaning")

    r = r[mask]
    vobs = vobs[mask]
    verr = verr[mask]
    vgas = vgas[mask]
    vdisk = vdisk[mask]
    vbul = vbul[mask]
    if sbdisk is not None:
        sbdisk = sbdisk[mask]
    if sbbul is not None:
        sbbul = sbbul[mask]

    return RotmodData(
        galaxy=galaxy,
        source=source,
        distance_mpc=distance_mpc,
        r_kpc=r,
        v_obs_km_s=vobs,
        v_err_km_s=verr,
        v_gas_km_s=vgas,
        v_disk_km_s=vdisk,
        v_bul_km_s=vbul,
        sb_disk_lsun_pc2=sbdisk,
        sb_bul_lsun_pc2=sbbul,
    )


def _nfw_v2_km2_s2(r_kpc: np.ndarray, log10_rhos: float, log10_rs_kpc: float) -> np.ndarray:
    """NFW circular speed squared from (rho_s, r_s).

    Parameters
    ----------
    log10_rhos:
        log10 of rho_s in Msun/kpc^3
    log10_rs_kpc:
        log10 of r_s in kpc
    """

    rhos = 10.0 ** log10_rhos
    rs = 10.0 ** log10_rs_kpc
    x = r_kpc / rs
    x = np.clip(x, 1e-10, None)
    m = 4.0 * math.pi * rhos * rs**3 * (np.log(1.0 + x) - x / (1.0 + x))
    v2 = G_KPC_KMS2_PER_MSUN * m / r_kpc
    return v2


def _burkert_v2_km2_s2(r_kpc: np.ndarray, log10_rho0: float, log10_r0_kpc: float) -> np.ndarray:
    """Burkert circular speed squared from (rho0, r0).

    Burkert profile:
        rho(r) = rho0 r0^3 / ((r+r0)(r^2+r0^2))

    Enclosed mass (analytic):
        M(r) = pi rho0 r0^3 [ ln((1+x)^2(1+x^2)) - 2 arctan(x) ],  x=r/r0
    """

    rho0 = 10.0 ** log10_rho0
    r0 = 10.0 ** log10_r0_kpc
    x = r_kpc / r0
    x = np.clip(x, 1e-10, None)
    m = math.pi * rho0 * r0**3 * (np.log((1.0 + x) ** 2 * (1.0 + x**2)) - 2.0 * np.arctan(x))
    v2 = G_KPC_KMS2_PER_MSUN * m / r_kpc
    return v2


def _combine_baryons_v2(data: RotmodData, ups_disk: float, ups_bul: float | None) -> np.ndarray:
    """Combine SPARC baryonic templates in quadrature.

    SPARC rotmod files provide *template velocities* for gas/disk/bulge. For a
    given stellar mass-to-light ratio (\\Upsilon_*), the corresponding template
    contributes \\sqrt{\\Upsilon_*} times the template velocity, so the squared
    contribution scales linearly with \\Upsilon_*.
    """

    v2 = (data.v_gas_km_s**2) + ups_disk * (data.v_disk_km_s**2)
    if ups_bul is not None:
        v2 = v2 + ups_bul * (data.v_bul_km_s**2)
    return v2


def fit_rotmod(
    data: RotmodData,
    model: Literal["nfw", "burkert"],
    *,
    use_priors: bool = True,
    ups_disk_prior: tuple[float, float] = (0.5, 0.1),
    ups_bul_prior: tuple[float, float] = (0.7, 0.15),
    loss: Literal["linear", "soft_l1"] = "linear",
    max_nfev: int = 8000,
) -> FitResult:
    """Fit a halo model to one SPARC rotmod galaxy.

    Notes
    -----
    - If `use_priors=True`, the fit is a MAP estimate with Gaussian priors on
      \\Upsilon_*. We still compute *data-only* chi^2 for AIC/BIC reporting.
      This is a pragmatic compromise for batch robustness; for strict
      likelihood-based information criteria comparisons, run with `--no-priors`.
    """

    t0 = time.time()
    has_bul = data.has_bulge()

    # Parameter vector: [log10_rho, log10_rscale, ups_disk, (ups_bul)]
    # Bounds are wide but finite to avoid pathological optimizer wandering.
    if model == "nfw":
        halo_fun = lambda r, p0, p1: _nfw_v2_km2_s2(r, p0, p1)
        halo_param_names = ("log10_rho_s_msun_kpc3", "log10_r_s_kpc")
        x0_halo = np.array([7.5, 1.0])  # rho_s ~ 3e7, r_s ~ 10 kpc
        lb_halo = np.array([3.0, -2.0])
        ub_halo = np.array([11.0, 2.5])
    elif model == "burkert":
        halo_fun = lambda r, p0, p1: _burkert_v2_km2_s2(r, p0, p1)
        halo_param_names = ("log10_rho0_msun_kpc3", "log10_r0_kpc")
        x0_halo = np.array([7.5, 0.5])  # rho0 ~ 3e7, r0 ~ 3 kpc
        lb_halo = np.array([3.0, -2.0])
        ub_halo = np.array([11.0, 2.5])
    else:
        raise ValueError(f"Unknown model: {model}")

    ups_d0 = float(max(0.01, ups_disk_prior[0]))
    ups_b0 = float(max(0.01, ups_bul_prior[0]))

    if has_bul:
        x0 = np.concatenate([x0_halo, [ups_d0, ups_b0]])
        lb = np.concatenate([lb_halo, [0.0, 0.0]])
        ub = np.concatenate([ub_halo, [2.5, 3.0]])
    else:
        x0 = np.concatenate([x0_halo, [ups_d0]])
        lb = np.concatenate([lb_halo, [0.0]])
        ub = np.concatenate([ub_halo, [2.5]])

    r = data.r_kpc
    vobs = data.v_obs_km_s
    verr = data.v_err_km_s

    def residuals(x: np.ndarray) -> np.ndarray:
        p0, p1 = float(x[0]), float(x[1])
        ups_d = float(x[2])
        ups_b = float(x[3]) if has_bul else None

        v2_bar = _combine_baryons_v2(data, ups_d, ups_b)
        v2_dm = halo_fun(r, p0, p1)
        v2_tot = np.clip(v2_bar + v2_dm, 0.0, None)
        vmod = np.sqrt(v2_tot)
        res = (vmod - vobs) / verr

        if use_priors:
            mu_d, sig_d = ups_disk_prior
            if sig_d > 0:
                res = np.concatenate([res, [(ups_d - mu_d) / sig_d]])
            if has_bul:
                mu_b, sig_b = ups_bul_prior
                if sig_b > 0:
                    res = np.concatenate([res, [(ups_b - mu_b) / sig_b]])
        return res

    try:
        sol = least_squares(residuals, x0, bounds=(lb, ub), loss=loss, max_nfev=max_nfev)
        success = bool(sol.success)
        msg = str(sol.message)
        xbest = sol.x
    except Exception as e:  # pragma: no cover
        success = False
        msg = f"least_squares failed: {type(e).__name__}: {e}"
        xbest = x0

    # Data-only chi^2 for reporting (exclude priors).
    p0, p1 = float(xbest[0]), float(xbest[1])
    ups_d = float(xbest[2])
    ups_b = float(xbest[3]) if has_bul else None
    v2_bar = _combine_baryons_v2(data, ups_d, ups_b)
    v2_dm = halo_fun(r, p0, p1)
    v2_tot = np.clip(v2_bar + v2_dm, 0.0, None)
    vmod = np.sqrt(v2_tot)
    chi2 = float(np.sum(((vmod - vobs) / verr) ** 2))

    k = 3 + (1 if has_bul else 0)
    n = int(r.size)
    dof = n - k
    chi2_red = float(chi2 / dof) if dof > 0 else None
    aic = float(chi2 + 2 * k) if n > 0 else None
    bic = float(chi2 + k * math.log(n)) if n > 1 else None

    halo_params = {halo_param_names[0]: p0, halo_param_names[1]: p1}
    rt = time.time() - t0
    return FitResult(
        galaxy=data.galaxy,
        source=data.source,
        distance_mpc=data.distance_mpc,
        model=model,
        n_points=n,
        k_params=k,
        chi2=chi2,
        dof=dof,
        chi2_red=chi2_red,
        aic=aic,
        bic=bic,
        ups_disk=ups_d,
        ups_bul=ups_b,
        halo_params=halo_params,
        success=success,
        message=msg,
        runtime_s=float(rt),
    )


def _ensure_mpl(need_plots: bool) -> None:
    if need_plots:
        _lazy_import_mpl()


def plot_galaxy_fit(data: RotmodData, results: Sequence[FitResult], outpath: Path, *, title: str | None = None) -> None:
    """Create a paper-style fit+residual plot for one galaxy.

    We plot:
    - observed V_rot with uncertainties,
    - gas/disk/bulge components (scaled by best-fit \\Upsilon_*),
    - total baryons,
    - total model curves for each halo profile,
    - residuals (data-model)/sigma.
    """

    _ensure_mpl(True)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    r = data.r_kpc
    vobs = data.v_obs_km_s
    verr = data.v_err_km_s

    # Use a representative stellar M/L for component display.
    # (The model curves below always use each model's own \Upsilon_* values.)
    ups_d = float(np.median([r.ups_disk for r in results]))
    ups_b_vals = [r.ups_bul for r in results if r.ups_bul is not None]
    ups_b = float(np.median(ups_b_vals)) if ups_b_vals else None

    v_gas = np.abs(data.v_gas_km_s)
    v_disk = math.sqrt(max(ups_d, 0.0)) * np.abs(data.v_disk_km_s)
    v_bul = (math.sqrt(max(ups_b, 0.0)) * np.abs(data.v_bul_km_s)) if ups_b is not None else None
    v_bar = np.sqrt(np.clip(_combine_baryons_v2(data, ups_d, ups_b), 0.0, None))

    fig = plt.figure(figsize=(6.4, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    ax.errorbar(r, vobs, yerr=verr, fmt="o", ms=4, lw=1, label="Observed")

    # Components
    ax.plot(r, v_gas, ls="--", lw=1.5, label="Gas")
    ax.plot(r, v_disk, ls="--", lw=1.5, label=fr"Disk ($\Upsilon_{{\star,d}}$={ups_d:.2g})")
    if v_bul is not None and np.any(v_bul > 0):
        ax.plot(r, v_bul, ls="--", lw=1.5, label=fr"Bulge ($\Upsilon_{{\star,b}}$={ups_b:.2g})")
    ax.plot(r, v_bar, ls=":", lw=2.0, label="Total baryons")

    colors = {"nfw": "C0", "burkert": "C2"}

    for res in results:
        if res.model == "nfw":
            v2_dm = _nfw_v2_km2_s2(r, res.halo_params["log10_rho_s_msun_kpc3"], res.halo_params["log10_r_s_kpc"])
        elif res.model == "burkert":
            v2_dm = _burkert_v2_km2_s2(r, res.halo_params["log10_rho0_msun_kpc3"], res.halo_params["log10_r0_kpc"])
        else:  # pragma: no cover
            continue

        v2_tot = np.clip(_combine_baryons_v2(data, res.ups_disk, res.ups_bul) + v2_dm, 0.0, None)
        vmod = np.sqrt(v2_tot)
        col = colors.get(res.model, None)
        lab = f"{res.model.upper()}  $\\chi^2_\\nu$={res.chi2_red:.2g}" if res.chi2_red is not None else res.model.upper()
        ax.plot(r, vmod, lw=2.2, color=col, label=lab)

        resid = (vobs - vmod) / verr
        axr.plot(r, resid, marker="o", ms=3, lw=1, color=col, label=res.model.upper())

    ax.set_ylabel(r"$V_{\rm rot}$ [km/s]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.5, loc="best", ncol=2)

    axr.axhline(0.0, color="k", lw=1)
    axr.set_xlabel(r"$r$ [kpc]")
    axr.set_ylabel(r"resid $/\sigma$")
    axr.grid(True, alpha=0.25)
    axr.set_ylim(-5, 5)

    if title is None:
        title = data.galaxy
        if data.distance_mpc is not None:
            title += f"  (D={data.distance_mpc:g} Mpc)"
    ax.set_title(title, fontsize=10)

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(path: Path, rows: Sequence[FitResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten halo_params keys for CSV.
    keys: set[str] = set()
    for r in rows:
        keys.update(r.halo_params.keys())
    halo_keys = sorted(keys)

    fieldnames = [
        "galaxy",
        "source",
        "distance_mpc",
        "model",
        "n_points",
        "k_params",
        "chi2",
        "dof",
        "chi2_red",
        "aic",
        "bic",
        "ups_disk",
        "ups_bul",
        "success",
        "message",
        "runtime_s",
    ] + halo_keys

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            d = {
                "galaxy": r.galaxy,
                "source": r.source,
                "distance_mpc": r.distance_mpc,
                "model": r.model,
                "n_points": r.n_points,
                "k_params": r.k_params,
                "chi2": r.chi2,
                "dof": r.dof,
                "chi2_red": r.chi2_red,
                "aic": r.aic,
                "bic": r.bic,
                "ups_disk": r.ups_disk,
                "ups_bul": r.ups_bul,
                "success": r.success,
                "message": r.message,
                "runtime_s": r.runtime_s,
            }
            for k in halo_keys:
                d[k] = r.halo_params.get(k)
            w.writerow(d)


def write_summary_json(path: Path, rows: Sequence[FitResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [asdict(r) for r in rows]
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def _group_by_galaxy(rows: Sequence[FitResult]) -> dict[str, list[FitResult]]:
    d: dict[str, list[FitResult]] = {}
    for r in rows:
        d.setdefault(r.galaxy, []).append(r)
    return d


def report_from_summary(summary_json: Path, outdir: Path, *, plot_format: str = "png") -> None:
    """Generate population plots from a merged summary.json.

    The intent is lightweight, reviewer-friendly sanity: show which model fits
    better across the sample and whether there are obvious pathologies such as
    many fits saturating parameter bounds.
    """

    _ensure_mpl(True)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [FitResult(**r) for r in json.loads(summary_json.read_text(encoding="utf-8"))]
    by_g = _group_by_galaxy(rows)

    # Collect per-galaxy paired stats
    delta_bic: list[float] = []
    chi2red_nfw: list[float] = []
    chi2red_bur: list[float] = []
    ups_disk: list[float] = []
    nfw_rs: list[float] = []
    bur_r0: list[float] = []
    nfw_rs_hit_upper = 0
    nfw_rs_hit_lower = 0

    # These bounds must match fit_rotmod() for interpretability.
    NFW_LOGRS_LB, NFW_LOGRS_UB = -2.0, 2.5

    for g, rr in by_g.items():
        r_nfw = next((x for x in rr if x.model == "nfw"), None)
        r_bur = next((x for x in rr if x.model == "burkert"), None)
        if r_nfw is None or r_bur is None:
            continue

        if (r_nfw.bic is not None) and (r_bur.bic is not None):
            delta_bic.append(float(r_nfw.bic - r_bur.bic))

        if r_nfw.chi2_red is not None:
            chi2red_nfw.append(float(r_nfw.chi2_red))
        if r_bur.chi2_red is not None:
            chi2red_bur.append(float(r_bur.chi2_red))

        ups_disk.append(float(r_bur.ups_disk))  # representative

        try:
            log_rs = float(r_nfw.halo_params.get("log10_r_s_kpc"))
            nfw_rs.append(log_rs)
            if abs(log_rs - NFW_LOGRS_UB) < 1e-9:
                nfw_rs_hit_upper += 1
            if abs(log_rs - NFW_LOGRS_LB) < 1e-9:
                nfw_rs_hit_lower += 1
        except Exception:
            pass

        try:
            bur_r0.append(float(r_bur.halo_params.get("log10_r0_kpc")))
        except Exception:
            pass

    delta_bic_arr = np.array(delta_bic, dtype=float)
    chi2red_nfw_arr = np.array(chi2red_nfw, dtype=float)
    chi2red_bur_arr = np.array(chi2red_bur, dtype=float)

    # --- Plot: ΔBIC histogram (positive => Burkert preferred)
    fig = plt.figure(figsize=(6.4, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(delta_bic_arr, bins=30)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel(r"$\Delta \mathrm{BIC} = \mathrm{BIC}_{\rm NFW}-\mathrm{BIC}_{\rm Burkert}$")
    ax.set_ylabel("Number of galaxies")
    ax.set_title("Model preference (positive = Burkert preferred)")
    ax.grid(True, alpha=0.25)
    fig.savefig(outdir / f"delta_bic_hist.{plot_format}", bbox_inches="tight")
    plt.close(fig)

    # --- Plot: reduced chi2 distributions
    fig = plt.figure(figsize=(6.4, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(chi2red_nfw_arr, bins=30, alpha=0.6, label="NFW")
    ax.hist(chi2red_bur_arr, bins=30, alpha=0.6, label="Burkert")
    ax.set_xlabel(r"$\chi^2_\nu$")
    ax.set_ylabel("Number of galaxies")
    ax.set_title("Reduced chi-square distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.savefig(outdir / f"chi2red_hist.{plot_format}", bbox_inches="tight")
    plt.close(fig)

    # --- Plot: chi2red scatter
    m = min(len(chi2red_nfw_arr), len(chi2red_bur_arr))
    if m > 0:
        fig = plt.figure(figsize=(5.0, 5.0))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(chi2red_nfw_arr[:m], chi2red_bur_arr[:m], s=10)
        mx = float(max(np.max(chi2red_nfw_arr[:m]), np.max(chi2red_bur_arr[:m]), 1.0))
        ax.plot([0, mx], [0, mx], lw=1, color="k")
        ax.set_xlabel(r"$\chi^2_{\nu,{\rm NFW}}$")
        ax.set_ylabel(r"$\chi^2_{\nu,{\rm Burkert}}$")
        ax.set_title("Per-galaxy fit quality")
        ax.grid(True, alpha=0.25)
        fig.savefig(outdir / f"chi2red_scatter.{plot_format}", bbox_inches="tight")
        plt.close(fig)

    # --- Plot: NFW r_s (log10) distribution + boundary markers
    if len(nfw_rs) > 0:
        rs = np.array(nfw_rs, dtype=float)
        fig = plt.figure(figsize=(6.4, 4.0))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(rs, bins=30)
        ax.axvline(NFW_LOGRS_LB, color="k", lw=1, ls=":")
        ax.axvline(NFW_LOGRS_UB, color="k", lw=1, ls=":")
        ax.set_xlabel(r"$\log_{10}(r_s/{\rm kpc})$  (NFW)")
        ax.set_ylabel("Number of galaxies")
        ax.set_title("NFW scale-radius fit distribution")
        ax.grid(True, alpha=0.25)
        fig.savefig(outdir / f"nfw_logrs_hist.{plot_format}", bbox_inches="tight")
        plt.close(fig)

    # --- Plot: Burkert r0 (log10) distribution
    if len(bur_r0) > 0:
        r0 = np.array(bur_r0, dtype=float)
        fig = plt.figure(figsize=(6.4, 4.0))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(r0, bins=30)
        ax.set_xlabel(r"$\log_{10}(r_0/{\rm kpc})$  (Burkert)")
        ax.set_ylabel("Number of galaxies")
        ax.set_title("Burkert core-radius fit distribution")
        ax.grid(True, alpha=0.25)
        fig.savefig(outdir / f"burkert_logr0_hist.{plot_format}", bbox_inches="tight")
        plt.close(fig)

    # --- Stats JSON (for paper text)
    def _safe_mean(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else float("nan")

    def _safe_median(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(np.median(x)) if x.size else float("nan")

    stats = {
        "n_galaxies_with_both_models": int(delta_bic_arr.size),
        "delta_bic_median": _safe_median(delta_bic_arr),
        "delta_bic_mean": _safe_mean(delta_bic_arr),
        "frac_burkert_preferred": float(np.mean(delta_bic_arr > 0.0)) if delta_bic_arr.size else float("nan"),
        "frac_strong_burkert_delta_bic_gt_6": float(np.mean(delta_bic_arr > 6.0)) if delta_bic_arr.size else float("nan"),
        "frac_strong_nfw_delta_bic_lt_-6": float(np.mean(delta_bic_arr < -6.0)) if delta_bic_arr.size else float("nan"),
        "chi2red_median_nfw": _safe_median(chi2red_nfw_arr),
        "chi2red_median_burkert": _safe_median(chi2red_bur_arr),
        "chi2red_mean_nfw": _safe_mean(chi2red_nfw_arr),
        "chi2red_mean_burkert": _safe_mean(chi2red_bur_arr),
        "nfw_logrs_hit_upper_frac": float(nfw_rs_hit_upper / max(int(delta_bic_arr.size), 1)),
        "nfw_logrs_hit_lower_frac": float(nfw_rs_hit_lower / max(int(delta_bic_arr.size), 1)),
    }
    (outdir / "population_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def cmd_batch(args: argparse.Namespace) -> int:
    _ensure_mpl(args.plots)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fits_dir = outdir / "fits"
    plots_dir = outdir / "plots"
    fits_dir.mkdir(exist_ok=True)
    if args.plots:
        plots_dir.mkdir(exist_ok=True)

    sources = iter_rotmod_files(args.inputs)
    total = len(sources)

    start = int(args.skip)
    end = total if args.limit is None else min(total, start + int(args.limit))
    subset = sources[start:end]

    if not args.quiet:
        print(f"Found {total} rotmod files; processing [{start}:{end}) -> {len(subset)} files")

    all_rows: list[FitResult] = []
    n_fail = 0

    for galaxy, src, text in subset:
        try:
            data = parse_rotmod_text(galaxy, src, text)
        except Exception as e:
            n_fail += 1
            if not args.quiet:
                print(f"[FAIL parse] {galaxy}  ({src})  {type(e).__name__}: {e}")
            continue

        # Optional resume: if a per-galaxy JSON exists, skip
        if args.resume:
            gjson = fits_dir / f"{data.galaxy}.json"
            if gjson.exists():
                if not args.quiet:
                    print(f"[skip resume] {data.galaxy}")
                try:
                    rr = json.loads(gjson.read_text(encoding="utf-8"))
                    for r in rr:
                        all_rows.append(FitResult(**r))
                except Exception:
                    pass
                continue

        results: list[FitResult] = []
        for m in args.models:
            fr = fit_rotmod(
                data,
                m,
                use_priors=not args.no_priors,
                ups_disk_prior=(args.ups_disk_mu, args.ups_disk_sigma),
                ups_bul_prior=(args.ups_bul_mu, args.ups_bul_sigma),
                loss=args.loss,
                max_nfev=args.max_nfev,
            )
            results.append(fr)
            all_rows.append(fr)

        # Write per-galaxy JSON
        gjson = fits_dir / f"{data.galaxy}.json"
        gjson.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")

        # Plot
        if args.plots:
            plot_path = plots_dir / f"{data.galaxy}_fit.{args.plot_format}"
            plot_galaxy_fit(data, results, plot_path)

        if not args.quiet:
            short = ", ".join(
                [
                    f"{r.model}: chi2_red={r.chi2_red:.2g}" if r.chi2_red is not None else f"{r.model}: chi2={r.chi2:.2g}"
                    for r in results
                ]
            )
            print(f"[OK] {data.galaxy}  n={data.n}  {short}")

    # Write chunk summary
    write_summary_csv(outdir / "summary.csv", all_rows)
    write_summary_json(outdir / "summary.json", all_rows)

    stats = {
        "n_total": total,
        "n_selected": len(subset),
        "skip": start,
        "limit": args.limit,
        "n_fit_rows": len(all_rows),
        "n_fail_parse": n_fail,
        "models": list(args.models),
        "plots": bool(args.plots),
        "use_priors": bool(not args.no_priors),
    }
    (outdir / "chunk_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(stats, indent=2))
    return 0 if n_fail == 0 else 2


def cmd_merge(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for p in args.inputs:
        jp = Path(p)
        rr = json.loads(jp.read_text(encoding="utf-8"))
        for r in rr:
            # De-duplication key (galaxy, model, source) to make merges idempotent.
            key = (str(r.get("galaxy")), str(r.get("model")), str(r.get("source")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Merged {len(args.inputs)} files -> {out}  (rows={len(rows)})")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report_from_summary(Path(args.summary_json), Path(args.outdir), plot_format=args.plot_format)
    print(f"Wrote report to {args.outdir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sidmkit.sparc_batch", description="SPARC rotmod batch fitting")
    sub = p.add_subparsers(dest="cmd", required=True)

    # batch
    pb = sub.add_parser("batch", help="Fit NFW/Burkert to many rotmod files (with chunking).")
    pb.add_argument("--inputs", nargs="+", required=True, help="Directories, .zip files, or individual *_rotmod.dat files")
    pb.add_argument("--outdir", required=True, help="Output directory (will be created)")
    pb.add_argument("--models", nargs="+", default=["nfw", "burkert"], choices=["nfw", "burkert"], help="Models to fit")
    pb.add_argument("--skip", type=int, default=0, help="Skip the first N rotmod files (for chunking)")
    pb.add_argument("--limit", type=int, default=None, help="Process at most N files after --skip (for chunking)")
    pb.add_argument("--plots", action="store_true", help="Generate per-galaxy fit plots")
    pb.add_argument("--plot-format", choices=["png", "pdf"], default="png")
    pb.add_argument("--resume", action="store_true", help="Skip galaxies that already have outputs/fits/<gal>.json")
    pb.add_argument("--no-priors", action="store_true", help="Disable mass-to-light priors (pure max-likelihood fit)")
    pb.add_argument("--ups-disk-mu", type=float, default=0.5)
    pb.add_argument("--ups-disk-sigma", type=float, default=0.1)
    pb.add_argument("--ups-bul-mu", type=float, default=0.7)
    pb.add_argument("--ups-bul-sigma", type=float, default=0.15)
    pb.add_argument("--loss", choices=["linear", "soft_l1"], default="linear")
    pb.add_argument("--max-nfev", type=int, default=8000, help="Max function evals per model fit")
    pb.add_argument("--quiet", action="store_true")
    pb.set_defaults(func=cmd_batch)

    # merge
    pm = sub.add_parser("merge", help="Merge multiple chunk summary.json files into one JSON.")
    pm.add_argument("--inputs", nargs="+", required=True, help="Paths to summary.json files")
    pm.add_argument("--out", required=True, help="Output merged JSON path")
    pm.set_defaults(func=cmd_merge)

    # report
    pr = sub.add_parser("report", help="Generate population plots from a merged summary JSON.")
    pr.add_argument("--summary-json", required=True, help="Merged summary.json path")
    pr.add_argument("--outdir", required=True, help="Output directory")
    pr.add_argument("--plot-format", choices=["png", "pdf"], default="png")
    pr.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
