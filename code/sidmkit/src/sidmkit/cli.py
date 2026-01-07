"""Command-line interface for sidmkit.

Design goals
------------
- Small number of commands, each doing one thing.
- Explicit units (km/s, GeV, cm^2/g).
- Outputs are reproducible and easy to redirect to CSV.

Run `sidmkit --help` or `sidmkit <cmd> --help` for details.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .averaging import average_sigma_over_m
from .cross_sections import sigma_over_m
from .bench import run_benchmarks
from .constraints import get_constraint_set, list_constraint_sets, resolve_constraint_set_name, evaluate_constraints
from .inference import YukawaLogPrior, infer_yukawa_from_dataset_mcmc
from .likelihoods import PointConstraintDataset, RotationCurveData, RotationCurveLikelihood
from .model import PotentialType, YukawaModel
from .scan import grid_scan, write_csv
from .validate import (
    chi2_against_fig13_left,
    generate_sigma_curve,
    load_curve_csv,
    compare_curves,
    save_curve_csv,
)
from .halo import NFWHalo, r1_interaction_radius_kpc


def _add_model_args(p: argparse.ArgumentParser, required: bool = True) -> None:
    p.add_argument("--mchi", type=float, required=required, help="DM mass mχ in GeV")
    p.add_argument("--mmed", type=float, required=required, help="Mediator mass m_med in GeV (e.g. 0.05 = 50 MeV)")
    p.add_argument("--alpha", type=float, required=required, help="Dark fine-structure constant α")
    p.add_argument(
        "--potential",
        choices=["attractive", "repulsive"],
        default="attractive",
        help="Yukawa potential sign",
    )
    p.add_argument(
        "--method",
        choices=["auto", "born", "classical", "hulthen", "partial_wave"],
        default="auto",
        help="Cross-section computation method",
    )


def _parse_model(args: argparse.Namespace) -> YukawaModel:
    pot = PotentialType.ATTRACTIVE if args.potential == "attractive" else PotentialType.REPULSIVE
    return YukawaModel(m_chi_gev=args.mchi, m_med_gev=args.mmed, alpha=args.alpha, potential=pot)


def cmd_sigma(args: argparse.Namespace) -> int:
    model = _parse_model(args)
    if args.v is not None:
        v = np.array(args.v, dtype=float)
    else:
        v = np.geomspace(args.vmin, args.vmax, args.n)
    y = sigma_over_m(v, model=model, method=args.method, out_unit="cm2/g")
    if args.csv:
        save_curve_csv(args.csv, v, y, y_name="sigma_over_m_cm2_g")
    else:
        for vi, yi in zip(v, np.atleast_1d(y)):
            print(f"{vi:10.4g} km/s  sigma/m = {yi:12.4g} cm^2/g")
    return 0


def cmd_avg(args: argparse.Namespace) -> int:
    model = _parse_model(args)
    res = average_sigma_over_m(
        model,
        sigma_1d_km_s=args.sigma1d,
        moment_n=args.moment,
        v_max_km_s=args.vmax if args.vmax is not None else None,
        method=args.method,
        integrator=args.integrator,
        n_laguerre=args.n_laguerre,
    )
    print(json.dumps(res.__dict__, indent=2))
    return 0


def cmd_constraints(args: argparse.Namespace) -> int:
    """Evaluate a named constraint set.

    Notes
    -----
    - `--list-sets` does not require model parameters.
    - For machine-readable reporting, use `--json-out` and/or `--csv-out`.
    """
    if args.list_sets:
        for name in list_constraint_sets():
            print(name)
        return 0

    # Model args are optional at parse-time (to allow --list-sets), so enforce here.
    missing = []
    for flag, attr in [("--mchi", "mchi"), ("--mmed", "mmed"), ("--alpha", "alpha")]:
        if getattr(args, attr, None) is None:
            missing.append(flag)
    if missing:
        raise SystemExit(
            "sidmkit constraints: error: missing required arguments %s (unless --list-sets)" % ", ".join(missing)
        )

    model = _parse_model(args)
    canonical_set = resolve_constraint_set_name(args.set)
    constraints = get_constraint_set(args.set)

    evals = evaluate_constraints(
        model,
        constraints,
        method=args.method,
        average=args.average,
        sigma_1d_km_s=args.sigma1d,
    )

    all_pass = all(e.passed for e in evals)

    # Build a structured report for file output.
    report = {
        "constraint_set": canonical_set,
        "constraint_set_requested": args.set,
        "model": {
            "mchi_gev": model.m_chi_gev,
            "mmed_gev": model.m_med_gev,
            "alpha": model.alpha,
            "potential": str(model.potential),
        },
        "options": {
            "method": args.method,
            "average": bool(args.average),
            "sigma_1d_km_s": (float(args.sigma1d) if args.sigma1d is not None else None),
        },
        "all_pass": bool(all_pass),
        "n_constraints": len(evals),
        "constraints": [],
    }

    csv_rows = []
    for e in evals:
        c = e.constraint
        unit = "cm^2/g" if c.observable == "sigma_over_m" else "(cm^2/g)*(km/s)"
        v_rep = float(c.velocity.representative())
        v_min = float(c.velocity.v_min_km_s) if c.velocity.kind == "range" and c.velocity.v_min_km_s is not None else None
        v_max = float(c.velocity.v_max_km_s) if c.velocity.kind == "range" and c.velocity.v_max_km_s is not None else None
        row = {
            "name": c.name,
            "kind": c.kind,
            "observable": c.observable,
            "v_rep_km_s": v_rep,
            "v_min_km_s": v_min,
            "v_max_km_s": v_max,
            "predicted": float(e.predicted),
            "predicted_unit": unit,
            "passed": bool(e.passed),
            "reference": c.reference,
            # Bound/measurement fields (filled below as relevant).
            "lower": None,
            "upper": None,
            "central": None,
            "sigma": None,
        }
        if c.kind == "upper":
            row["upper"] = c.upper
        elif c.kind == "lower":
            row["lower"] = c.lower
        elif c.kind == "band":
            row["lower"] = c.lower
            row["upper"] = c.upper
        elif c.kind == "measurement":
            row["central"] = c.mean
            row["sigma"] = c.sigma
        report["constraints"].append(row)
        csv_rows.append(row)

    # Optional file outputs.
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(csv_rows, out_path)

    # Human-readable report.
    if not args.quiet:
        # If the user requested an alias, show the canonical key too.
        if args.set != canonical_set:
            print(f"Constraint set: {canonical_set}")
        else:
            print(f"Constraint set: {canonical_set}")

        for e in evals:
            c = e.constraint
            unit = "cm^2/g" if c.observable == "sigma_over_m" else "(cm^2/g)*(km/s)"
            status = "PASS" if e.passed else "FAIL"
            print(f"[{status:7s}] {c.name}")
            if c.velocity.kind == "point":
                print(f"  v: {c.velocity.v_km_s} km/s")
            else:
                v_rep = c.velocity.representative()
                print(f"  v: [{c.velocity.v_min_km_s}, {c.velocity.v_max_km_s}] km/s (rep {v_rep} km/s)")
            print(f"  predicted: {e.predicted:.6g} {unit}")
            if c.kind == "upper":
                print(f"  upper: {c.upper:g} {unit}")
            elif c.kind == "lower":
                print(f"  lower: {c.lower:g} {unit}")
            elif c.kind == "band":
                print(f"  band: [{c.lower:g}, {c.upper:g}] {unit}")
            elif c.kind == "measurement":
                print(f"  central ± sigma: {c.mean:g} ± {c.sigma:g} {unit}")
            print(f"  ref: {c.reference}")
            if c.notes:
                print(f"  notes: {c.notes}")
    return 0 if all_pass else 2


def cmd_scan(args: argparse.Namespace) -> int:
    mchi = np.geomspace(args.mchi_min, args.mchi_max, args.mchi_n)
    mmed = np.geomspace(args.mmed_min, args.mmed_max, args.mmed_n)
    alpha = np.geomspace(args.alpha_min, args.alpha_max, args.alpha_n)

    pot = PotentialType.ATTRACTIVE if args.potential == "attractive" else PotentialType.REPULSIVE
    constraints = get_constraint_set(args.set)

    rows = grid_scan(
        m_chi_grid=mchi,
        m_med_grid=mmed,
        alpha_grid=alpha,
        potential=pot,
        constraints=constraints,
        method=args.method,
    )
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


def cmd_halo(args: argparse.Namespace) -> int:
    model = _parse_model(args)
    halo = NFWHalo(m200_msun=args.m200, c200=args.c200, h0_km_s_mpc=args.h0)
    r1 = r1_interaction_radius_kpc(
        model,
        halo,
        t_age_gyr=args.tage,
        method=args.method,
        average=not args.no_average,
        v_mode=args.vmode,
    )
    vmax, rmax = halo.vmax_rmax()
    out = {
        "r1_kpc": r1,
        "r200_kpc": halo.r200_kpc,
        "r_s_kpc": halo.r_s_kpc,
        "vmax_km_s": vmax,
        "rmax_kpc": rmax,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_likelihood(args: argparse.Namespace) -> int:
    model = _parse_model(args)
    if args.kind == "points":
        ds = PointConstraintDataset.from_json(args.data)
        ll = ds.log_likelihood(model, method=args.method)
        print(json.dumps({"dataset": ds.name, "log_likelihood": ll}, indent=2))
        return 0
    if args.kind == "rotation_curve":
        data = RotationCurveData.from_csv(args.data)
        lk = RotationCurveLikelihood(data=data, m200_msun=args.m200, c200=args.c200, t_age_gyr=args.tage)
        ll = lk.log_likelihood(model, method=args.method, sidm=not args.no_sidm)
        print(json.dumps({"dataset": "rotation_curve", "log_likelihood": ll}, indent=2))
        return 0
    raise ValueError(f"Unknown likelihood kind: {args.kind}")


def cmd_infer(args: argparse.Namespace) -> int:
    ds = PointConstraintDataset.from_json(args.data)

    prior = YukawaLogPrior(
        log10_mchi_gev=(args.logmchi_min, args.logmchi_max),
        log10_mmed_gev=(args.logmmed_min, args.logmmed_max),
        log10_alpha=(args.logalpha_min, args.logalpha_max),
        potential="attractive" if args.potential == "attractive" else "repulsive",
    )

    # starting point: mid of box
    x0 = np.array(
        [
            0.5 * (args.logmchi_min + args.logmchi_max),
            0.5 * (args.logmmed_min + args.logmmed_max),
            0.5 * (args.logalpha_min + args.logalpha_max),
        ],
        dtype=float,
    )
    step = np.array([args.step_logmchi, args.step_logmmed, args.step_logalpha], dtype=float)

    res = infer_yukawa_from_dataset_mcmc(
        lambda m: ds.log_likelihood(m, method=args.method),
        prior=prior,
        x0=x0,
        step_scales=step,
        n_steps=args.n_steps,
        method=args.method,
        seed=args.seed,
    )

    # summary
    burn = int(args.burn_frac * args.n_steps)
    chain = res.chain[burn:]
    mean = chain.mean(axis=0)
    std = chain.std(axis=0)
    out = {
        "dataset": ds.name,
        "n_steps": args.n_steps,
        "burn": burn,
        "accept_rate": res.accept_rate,
        "posterior_mean_log10": mean.tolist(),
        "posterior_std_log10": std.tolist(),
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    summ = run_benchmarks(include_slow=args.slow)
    print(json.dumps(summ.results, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    model = _parse_model(args)
    if args.target == "fig13":
        chi2 = chi2_against_fig13_left(model, method=args.method)
        print(json.dumps({"target": "fig13_left", "chi2": chi2}, indent=2))
        return 0
    if args.target == "curve":
        v_pred, y_pred = generate_sigma_curve(model, v_min_km_s=args.vmin, v_max_km_s=args.vmax, n=args.n, method=args.method)
        v_ref, y_ref, y_name = load_curve_csv(args.reference)
        comp = compare_curves(v_ref, y_ref, v_pred, y_pred)
        print(json.dumps({"target": "curve", "reference": args.reference, "y_name": y_name, **comp.__dict__}, indent=2))
        return 0
    raise ValueError(f"Unknown validate target: {args.target}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sidmkit", description="SIDM micro→macro helper toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    # sigma
    ps = sub.add_parser("sigma", help="Compute sigma/m at one or more velocities, or output a curve.")
    _add_model_args(ps)
    g = ps.add_mutually_exclusive_group(required=False)
    g.add_argument("--v", type=float, nargs="+", help="One or more velocities in km/s")
    g.add_argument("--curve", action="store_true", help="Generate a log-spaced curve")
    ps.add_argument("--vmin", type=float, default=1.0)
    ps.add_argument("--vmax", type=float, default=5000.0)
    ps.add_argument("--n", type=int, default=100)
    ps.add_argument("--csv", type=str, help="Write curve/points to CSV")
    ps.set_defaults(func=cmd_sigma)

    # avg
    pa = sub.add_parser("avg", help="Velocity-average: compute ⟨(σ/m) v^n⟩ for a Maxwellian relative-speed distribution.")
    _add_model_args(pa)
    pa.add_argument("--sigma1d", type=float, required=True, help="1D dispersion σ_1D of halo particle speeds (km/s)")
    pa.add_argument("--moment", type=float, default=1.0, help="Velocity moment n (0 -> ⟨σ/m⟩, 1 -> ⟨σ v⟩/m)")
    pa.add_argument("--vmax", type=float, default=None, help="Optional truncation v_max (km/s) of relative speed")
    pa.add_argument("--integrator", choices=["auto", "laguerre", "quad"], default="auto")
    pa.add_argument("--n-laguerre", type=int, default=64, help="Laguerre nodes when integrator=laguerre")
    pa.set_defaults(func=cmd_avg)

    # constraints
    pc = sub.add_parser("constraints", help="Evaluate a curated constraint set.")
    _add_model_args(pc, required=False)
    pc.add_argument("--list-sets", action="store_true", help="List available constraint sets and exit")
    pc.add_argument("--set", type=str, default="literature_table1_summary", help="Constraint set name")
    pc.add_argument("--average", action="store_true", help="Use Maxwellian velocity averaging instead of point evaluation")
    pc.add_argument("--sigma1d", type=float, default=None, help="Required if --average: σ_1D (km/s)")
    pc.add_argument("--json-out", type=str, default=None, help="Write a machine-readable JSON report to this path")
    pc.add_argument("--csv-out", type=str, default=None, help="Write a per-constraint CSV table to this path")
    pc.add_argument("--quiet", action="store_true", help="Suppress the human-readable report (useful with --json-out/--csv-out)")
    pc.set_defaults(func=cmd_constraints)

    # scan
    pscan = sub.add_parser("scan", help="Grid-scan Yukawa model parameters against a constraint set and write CSV.")
    pscan.add_argument("--mchi-min", type=float, required=True)
    pscan.add_argument("--mchi-max", type=float, required=True)
    pscan.add_argument("--mchi-n", type=int, default=8)
    pscan.add_argument("--mmed-min", type=float, required=True)
    pscan.add_argument("--mmed-max", type=float, required=True)
    pscan.add_argument("--mmed-n", type=int, default=8)
    pscan.add_argument("--alpha-min", type=float, required=True)
    pscan.add_argument("--alpha-max", type=float, required=True)
    pscan.add_argument("--alpha-n", type=int, default=8)
    pscan.add_argument("--potential", choices=["attractive", "repulsive"], default="attractive")
    pscan.add_argument("--method", choices=["auto", "born", "classical", "hulthen", "partial_wave"], default="auto")
    pscan.add_argument("--set", type=str, default="illustrative")
    pscan.add_argument("--out", type=str, required=True, help="Output CSV path")
    pscan.set_defaults(func=cmd_scan)

    # halo
    ph = sub.add_parser("halo", help="Compute the interaction radius r1 for an NFW halo.")
    _add_model_args(ph)
    ph.add_argument("--m200", type=float, required=True, help="Halo M200 in Msun")
    ph.add_argument("--c200", type=float, required=True, help="Halo concentration c200")
    ph.add_argument("--h0", type=float, default=70.0, help="H0 in km/s/Mpc used for rho_crit")
    ph.add_argument("--tage", type=float, default=10.0, help="Halo age in Gyr")
    ph.add_argument("--vmode", choices=["jeans", "vmax"], default="jeans")
    ph.add_argument("--no-average", action="store_true", help="Disable velocity averaging inside Γ(r)")
    ph.set_defaults(func=cmd_halo)

    # likelihood
    pl = sub.add_parser("likelihood", help="Evaluate a dataset log-likelihood for a model.")
    _add_model_args(pl)
    pl.add_argument("--kind", choices=["points", "rotation_curve"], required=True)
    pl.add_argument("--data", type=str, required=True, help="Path to dataset file (JSON or CSV depending on kind)")
    pl.add_argument("--m200", type=float, help="(rotation_curve) Halo M200 in Msun")
    pl.add_argument("--c200", type=float, help="(rotation_curve) Halo concentration c200")
    pl.add_argument("--tage", type=float, default=10.0)
    pl.add_argument("--no-sidm", action="store_true", help="(rotation_curve) Use plain NFW (no core)")
    pl.set_defaults(func=cmd_likelihood)

    # infer
    pi = sub.add_parser("infer", help="Run a simple Metropolis MCMC against a point-constraint dataset (JSON).")
    pi.add_argument("--data", type=str, required=True)
    pi.add_argument("--method", choices=["auto", "born", "classical", "hulthen", "partial_wave"], default="auto")
    pi.add_argument("--potential", choices=["attractive", "repulsive"], default="attractive")
    pi.add_argument("--logmchi-min", type=float, default=0.0)
    pi.add_argument("--logmchi-max", type=float, default=2.5)
    pi.add_argument("--logmmed-min", type=float, default=-5.0)
    pi.add_argument("--logmmed-max", type=float, default=-1.0)
    pi.add_argument("--logalpha-min", type=float, default=-6.0)
    pi.add_argument("--logalpha-max", type=float, default=-0.5)
    pi.add_argument("--step-logmchi", type=float, default=0.05)
    pi.add_argument("--step-logmmed", type=float, default=0.05)
    pi.add_argument("--step-logalpha", type=float, default=0.05)
    pi.add_argument("--n-steps", type=int, default=3000)
    pi.add_argument("--burn-frac", type=float, default=0.3)
    pi.add_argument("--seed", type=int, default=None)
    pi.set_defaults(func=cmd_infer)

    # benchmark
    pb = sub.add_parser("benchmark", help="Run internal benchmarks / smoke tests.")
    pb.add_argument("--slow", action="store_true", help="Include slower checks")
    pb.set_defaults(func=cmd_benchmark)

    # validate
    pv = sub.add_parser("validate", help="External-ish validation scaffolds.")
    _add_model_args(pv)
    pv.add_argument("--target", choices=["fig13", "curve"], required=True)
    pv.add_argument("--reference", type=str, help="(curve) Reference CSV with v_km_s,y columns")
    pv.add_argument("--vmin", type=float, default=1.0)
    pv.add_argument("--vmax", type=float, default=5000.0)
    pv.add_argument("--n", type=int, default=200)
    pv.set_defaults(func=cmd_validate)

    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

