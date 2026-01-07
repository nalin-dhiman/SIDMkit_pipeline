import numpy as np

from sidmkit.sparc_batch import parse_rotmod_text, fit_rotmod


SYNTH_ROTMOD = """# Example synthetic SPARC rotmod-like file
# Distance = 10.0 Mpc
# r  vobs  err  vgas  vdisk  vbul
1.0  50.0  5.0  10.0  20.0  0.0
2.0  70.0  5.0  12.0  25.0  0.0
3.0  85.0  5.0  13.0  28.0  0.0
4.0  95.0  6.0  14.0  30.0  0.0
5.0  100.0 6.0  14.0  31.0  0.0
"""


def test_parse_rotmod_text_basic():
    data = parse_rotmod_text("SYNTH", "memory", SYNTH_ROTMOD)
    assert data.galaxy == "SYNTH"
    assert data.distance_mpc == 10.0
    assert data.n == 5
    assert np.all(data.r_kpc > 0)
    assert np.all(data.v_err_km_s > 0)


def test_fit_rotmod_runs_nfw_and_burkert():
    data = parse_rotmod_text("SYNTH", "memory", SYNTH_ROTMOD)

    r1 = fit_rotmod(data, "nfw", use_priors=False, max_nfev=2000)
    r2 = fit_rotmod(data, "burkert", use_priors=False, max_nfev=2000)

    assert r1.model == "nfw"
    assert r2.model == "burkert"
    assert r1.n_points == data.n
    assert r2.n_points == data.n
    assert np.isfinite(r1.chi2)
    assert np.isfinite(r2.chi2)
    # should return a sensible non-negative chi2
    assert r1.chi2 >= 0
    assert r2.chi2 >= 0
