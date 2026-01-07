"""Simple inference utilities (grid + Metropolis MCMC).

This module exists to make sidmkit *end-to-end runnable* without pulling in a
large inference stack.

Be critical
-----------
For serious parameter inference you will likely want:
- `emcee` (ensemble MCMC),
- `dynesty` / `ultranest` (nested sampling),
- or a gradient-based method if you have differentiability.

We ship a small Metropolis–Hastings sampler because:
- it is dependency-light,
- it is easy to audit,
- it is enough for examples, regression tests, and quick exploratory fits.

If you want publication-grade posteriors, treat this as scaffolding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

from .model import YukawaModel


@dataclass(frozen=True)
class YukawaLogPrior:
    """Independent uniform priors in log10 space for (mχ, m_med, α)."""

    log10_mchi_gev: tuple[float, float] = (0.0, 3.0)  # 1 to 1e3 GeV
    log10_mmed_gev: tuple[float, float] = (-6.0, -1.0)  # 1e-6 to 0.1 GeV
    log10_alpha: tuple[float, float] = (-6.0, -0.5)  # very weak to ~0.3
    potential: Literal["attractive", "repulsive"] = "attractive"

    def logpdf(self, theta: np.ndarray) -> float:
        lmchi, lmmed, lalpha = map(float, theta)
        a, b = self.log10_mchi_gev
        if not (a <= lmchi <= b):
            return -np.inf
        a, b = self.log10_mmed_gev
        if not (a <= lmmed <= b):
            return -np.inf
        a, b = self.log10_alpha
        if not (a <= lalpha <= b):
            return -np.inf
        return 0.0

    def to_model(self, theta: np.ndarray) -> YukawaModel:
        lmchi, lmmed, lalpha = map(float, theta)
        return YukawaModel(
            m_chi_gev=10 ** lmchi,
            m_med_gev=10 ** lmmed,
            alpha=10 ** lalpha,
            potential=self.potential,
        )


@dataclass
class MCMCResult:
    chain: np.ndarray  # (n_steps, n_dim)
    logp: np.ndarray  # (n_steps,)
    accept_rate: float


def metropolis_hastings(
    log_prob: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    step_cov: np.ndarray,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
) -> MCMCResult:
    """Basic random-walk Metropolis–Hastings sampler."""
    if rng is None:
        rng = np.random.default_rng()

    x0 = np.asarray(x0, dtype=float)
    n_dim = x0.size
    if step_cov.shape != (n_dim, n_dim):
        raise ValueError("step_cov must be (n_dim, n_dim)")
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")

    L = np.linalg.cholesky(step_cov)
    chain = np.zeros((n_steps, n_dim), dtype=float)
    logp = np.zeros(n_steps, dtype=float)

    x = x0.copy()
    lp = float(log_prob(x))
    acc = 0
    for i in range(n_steps):
        prop = x + L @ rng.normal(size=n_dim)
        lp_prop = float(log_prob(prop))
        if np.isfinite(lp_prop) and (np.log(rng.random()) < (lp_prop - lp)):
            x = prop
            lp = lp_prop
            acc += 1
        chain[i] = x
        logp[i] = lp

    return MCMCResult(chain=chain, logp=logp, accept_rate=acc / n_steps)


def infer_yukawa_from_dataset_mcmc(
    dataset_loglike: Callable[[YukawaModel], float],
    *,
    prior: YukawaLogPrior,
    x0: np.ndarray,
    step_scales: np.ndarray,
    n_steps: int = 5000,
    method: str = "auto",
    seed: Optional[int] = None,
) -> MCMCResult:
    """Convenience wrapper: infer Yukawa parameters against a dataset loglike."""
    rng = np.random.default_rng(seed)

    step_scales = np.asarray(step_scales, dtype=float)
    if step_scales.shape != (3,):
        raise ValueError("step_scales must be shape (3,) for (log mchi, log mmed, log alpha)")
    step_cov = np.diag(step_scales**2)

    def log_post(theta: np.ndarray) -> float:
        lp = prior.logpdf(theta)
        if not np.isfinite(lp):
            return -np.inf
        model = prior.to_model(theta)
        ll = float(dataset_loglike(model))
        return lp + ll

    return metropolis_hastings(log_post, x0=np.asarray(x0, dtype=float), step_cov=step_cov, n_steps=n_steps, rng=rng)

