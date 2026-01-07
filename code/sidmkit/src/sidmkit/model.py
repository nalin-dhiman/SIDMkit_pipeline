"""Model containers.

Right now we implement the most commonly-used toy model for SIDM phenomenology:
a Yukawa potential from a light mediator.

V(r) = ± α \exp(-m_med r)/r

The sign choice depends on the mediator and the scattering channel:
- scalar mediator: purely attractive
- vector mediator: attractive for particle–antiparticle and repulsive for like-charge

We keep this explicit so users don't accidentally mix conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PotentialType(str, Enum):
    """Sign convention for the Yukawa potential."""

    ATTRACTIVE = "attractive"
    REPULSIVE = "repulsive"

    @property
    def sign(self) -> float:
        # In V(r) = sign * α exp(-m r)/r: attractive means V < 0.
        return -1.0 if self is PotentialType.ATTRACTIVE else +1.0


@dataclass(frozen=True)
class YukawaModel:
    """Yukawa SIDM model parameters.

    Parameters
    ----------
    m_chi_gev:
        Dark matter mass m_χ in GeV.
    m_med_gev:
        Mediator mass m_φ (or M_X) in GeV.
    alpha:
        Dark fine-structure constant α = g^2/(4π).
    potential:
        Attractive or repulsive channel.
    kappa_hulthen:
        Matching parameter κ≈1.6 used in the Hulthén approximation.
    """

    m_chi_gev: float
    m_med_gev: float
    alpha: float
    potential: PotentialType = PotentialType.ATTRACTIVE
    kappa_hulthen: float = 1.6

    def __post_init__(self) -> None:
        if self.m_chi_gev <= 0:
            raise ValueError("m_chi_gev must be > 0")
        if self.m_med_gev <= 0:
            raise ValueError("m_med_gev must be > 0")
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        if self.kappa_hulthen <= 0:
            raise ValueError("kappa_hulthen must be > 0")

    @property
    def g_squared(self) -> float:
        """Return g^2 = 4π α."""
        import math

        return 4.0 * math.pi * self.alpha
