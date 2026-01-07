"""SPARC/rotmod batch fitting utilities.

This module is intentionally self-contained so you can run it immediately in an
editable install.

Example (chunked batch run):
---------------------------
1) Unzip the SPARC rotmod zips (recommended; faster than reading inside zip):

   unzip -q Rotmod_LTG.zip -d sparc_data/Rotmod_LTG
   unzip -q Rotmod_ETG.zip -d sparc_data/Rotmod_ETG

2) Run 25 galaxies at a time:

   python -m sidmkit.sparc_batch batch \
     --inputs sparc_data/Rotmod_LTG sparc_data/Rotmod_ETG \
     --outdir outputs/chunk_0 \
     --skip 0 --limit 25 \
     --plots --plot-format png

Design notes
------------
* We fit phenomenological halo models (NFW, Burkert) to SPARC "rotmod" files.
  This is *not* yet a microphysical SIDM inference pipeline; it is a robust
  rotation-curve fitting and model-comparison module.
* The goal is reproducibility and robustness across many galaxies, not squeezing
  every last bit of statistical efficiency out of each object.
* Chunking (--skip/--limit) is the submission-grade solution for large batches.
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

try:
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False

from scipy.optimize import least_squares


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
        return np.any(np.abs(self.v_bul_km_s) > 1e-8)

    @property
    def n(self) -> int:
        return int(self.r_kpc.size)


@dataclass
class FitResult:
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

    *If* you point this at a large directory/zip, it will load all file text
    into memory; for SPARC-sized samples this is fine.
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
    for line in text.splitlines()[:20]:
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

    # Basic sanity: remove non-positive radii and non-finite values.
    mask = np.isfinite(r) & np.isfinite(vobs) & np.isfinite(verr) & (r > 0) & (verr > 0)
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
    log10_rhos : log10 of rho_s in Msun/kpc^3
    log10_rs_kpc : log10 of r_s in kpc
    """

    rhos = 10.0 ** log10_rhos
    rs = 10.0 ** log10_rs_kpc
    x = r_kpc / rs
    # Avoid log singularities for tiny x.
    x = np.clip(x, 1e-10, None)
    m = 4.0 * math.pi * rhos * rs**3 * (np.log(1.0 + x) - x / (1.0 + x))
    v2 = G_KPC_KMS2_PER_MSUN * m / r_kpc
    return v2


def _burkert_v2_km2_s2(r_kpc: np.ndarray, log10_rho0: float, log10_r0_kpc: float) -> np.ndarray:
    """Burkert circular speed squared from (rho0, r0).

    Burkert density profile:
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


def _combine_baryons_v2(
    data: RotmodData,
    ups_disk: float,
    ups_bul: float | None,
) -> np.ndarray:
    # SPARC templates are velocities; total baryonic contribution adds in quadrature.
    # Note: some files contain negative template velocities (rare); squaring fixes this.
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
    """Fit a halo model to one SPARC rotmod galaxy."""

    t0 = time.time()
    has_bul = data.has_bulge()

    # Parameter vector: [log10_rho, log10_rscale, ups_disk, (ups_bul)]
    # Mentioned bounds are wide but finite to avoid pathological optimizer wandering.
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

    # Mass-to-light starting points from priors.
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
        sol = least_squares(
            residuals,
            x0,
            bounds=(lb, ub),
            loss=loss,
            max_nfev=max_nfev,
        )
        success = bool(sol.success)
        msg = str(sol.message)
        xbest = sol.x
    except Exception as e:  # pragma: no cover
        success = False
        msg = f"least_squares failed: {e}"
        xbest = x0

    # Compute data-only chi2 (exclude priors) for information criteria.
    p0, p1 = float(xbest[0]), float(xbest[1])
    ups_d = float(xbest[2])
    ups_b = float(xbest[3]) if has_bul else None
    v2_bar = _combine_baryons_v2(data, ups_d, ups_b)
    v2_dm = halo_fun(r, p0, p1)
    v2_tot = np.clip(v2_bar + v2_dm, 0.0, None)
    vmod = np.sqrt(v2_tot)
    chi2 = float(np.sum(((vmod - vobs) / verr) ** 2))

    # Parameter count for ICs: halo(2) + ups_disk(1) + ups_bul(optional)
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
    if need_plots and not _HAVE_MPL:
        raise RuntimeError(
            "Plotting requested but matplotlib is not available. "
            "Install it with: pip install matplotlib"
        )


def plot_galaxy_fit(
    data: RotmodData,
    results: Sequence[FitResult],
    outpath: Path,
    *,
    title: str | None = None,
) -> None:
    """Create a paper-style fit+residual plot for one galaxy."""

    _ensure_mpl(True)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    r = data.r_kpc
    vobs = data.v_obs_km_s
    verr = data.v_err_km_s

    # Baryonic curve with best-fit ups from first result (they usually agree).
    ups_d = results[0].ups_disk
    ups_b = results[0].ups_bul
    vbar = np.sqrt(np.clip(_combine_baryons_v2(data, ups_d, ups_b), 0.0, None))

    fig = plt.figure(figsize=(6.4, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    ax.errorbar(r, vobs, yerr=verr, fmt="o", ms=4, lw=1, label="Observed")
    ax.plot(r, vbar, ls=":", lw=2, label=r"Baryons ($\Upsilon_*$ fit)")

    colors = {
        "nfw": "C0",
        "burkert": "C2",
    }

    for res in results:
        # reconstruct model curve
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
        ax.plot(r, vmod, lw=2, color=col, label=lab)

        # residuals
        resid = (vobs - vmod) / verr
        axr.plot(r, resid, marker="o", ms=3, lw=1, color=col, label=res.model.upper())

    ax.set_ylabel(r"$V_{\rm rot}$ [km/s]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")

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
    keys = set()
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


def report_from_summary(summary_json: Path, outdir: Path) -> None:
    """Generate simple population plots from a merged summary.json."""

    _ensure_mpl(True)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [FitResult(**r) for r in json.loads(summary_json.read_text(encoding="utf-8"))]

    # compute delta BIC between NFW and Burkert
    by_g = _group_by_galaxy(rows)
    delta = []
    for _, rr in by_g.items():
        r_nfw = next((x for x in rr if x.model == "nfw"), None)
        r_bur = next((x for x in rr if x.model == "burkert"), None)
        if r_nfw and r_bur and r_nfw.bic is not None and r_bur.bic is not None:
            delta.append(r_nfw.bic - r_bur.bic)

    if len(delta) == 0:
        raise RuntimeError("No galaxies with both NFW and Burkert BIC values found.")

    delta = np.array(delta, dtype=float)
    fig = plt.figure(figsize=(6.4, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(delta, bins=30)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel(r"$\Delta \mathrm{BIC} = \mathrm{BIC}_{\rm NFW}-\mathrm{BIC}_{\rm Burkert}$")
    ax.set_ylabel("Number of galaxies")
    ax.set_title("Model preference (positive = Burkert preferred)")
    ax.grid(True, alpha=0.25)
    fig.savefig(outdir / "delta_bic_hist.png", bbox_inches="tight")
    plt.close(fig)

    stats = {
        "n_galaxies_with_delta_bic": int(delta.size),
        "delta_bic_median": float(np.median(delta)),
        "delta_bic_mean": float(np.mean(delta)),
        "frac_burkert_preferred": float(np.mean(delta > 0.0)),
        "frac_strong_burkert": float(np.mean(delta > 6.0)),
        "frac_strong_nfw": float(np.mean(delta < -6.0)),
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

    # chunk selection
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
                print(f"[FAIL parse] {galaxy}  ({src})  {e}")
            continue

        # optional resume: if a per-galaxy JSON exists, skip
        if args.resume:
            gjson = fits_dir / f"{data.galaxy}.json"
            if gjson.exists():
                if not args.quiet:
                    print(f"[skip resume] {data.galaxy}")
                # include in summary for this chunk by reading it back
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

        # write per-galaxy JSON
        gjson = fits_dir / f"{data.galaxy}.json"
        gjson.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")

        # plot
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

    # write chunk summary
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
    }
    (outdir / "chunk_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(stats, indent=2))
    return 0 if n_fail == 0 else 2


def cmd_merge(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for p in args.inputs:
        jp = Path(p)
        rr = json.loads(jp.read_text(encoding="utf-8"))
        rows.extend(rr)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Merged {len(args.inputs)} files -> {out}  (rows={len(rows)})")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report_from_summary(Path(args.summary_json), Path(args.outdir))
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
    pb.add_argument("--no-priors", action="store_true", help="Disable mass-to-light priors")
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
    pr.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
