"""Reproducible prescribed wrinkle fields, independent of mesh and optics."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WrinkleField:
    amplitude_m: float
    seed: int = 7

    def heights(self, radial_fraction, theta):
        """Temporary height field on an annulus, periodic in azimuth [radians].

        Peak absolute nodal height equals amplitude_m on the supplied grid.
        There are no folds or overhangs, so normal illumination has no mutual
        facet shadowing. Inner/outer rings are flat. This remains a placeholder
        for a future wrinkle model; replace this method without changing optics.
        """
        rng = np.random.default_rng(self.seed)
        height = np.zeros_like(radial_fraction, dtype=float)
        for _ in range(7):
            kr = rng.uniform(1.5, 5.0)
            azimuthal_mode = rng.integers(2, 13)
            phase = rng.uniform(0, 2*np.pi)
            height += rng.uniform(.3, 1.0)*np.sin(
                2*np.pi*kr*radial_fraction+azimuthal_mode*theta+phase)
        height *= np.sin(np.pi*radial_fraction)
        height = np.where((radial_fraction == 0) | (radial_fraction == 1), 0, height)
        return height * self.amplitude_m/max(np.max(np.abs(height)), 1e-15)


@dataclass(frozen=True)
class StochasticWrinkleField:
    """Bounded Gaussian spectral ensemble, not an elastic equilibrium model.

    Random coefficients, radial frequencies, azimuthal modes, and phases give
    reproducible smooth asymmetric surfaces. Mode limits avoid wavelengths below
    four mesh intervals. Peak *nodal* absolute height is amplitude_m. Low modes
    include m=0 and m=1 so ensemble symmetry does not artificially cancel torque.
    """
    amplitude_m: float
    seed: int = 7
    radial_modes: int = 6
    angular_modes: int = 12
    terms: int = 32

    def heights(self, radial_fraction, theta):
        rng = np.random.default_rng(self.seed)
        height = np.zeros_like(radial_fraction, dtype=float)
        for _ in range(self.terms):
            n = int(rng.integers(1, self.radial_modes+1))
            m = int(rng.integers(0, self.angular_modes+1))
            coefficient = rng.normal() / np.sqrt(n*n+(m/3)**2)
            phase = rng.uniform(0, 2*np.pi)
            height += coefficient*np.sin(n*np.pi*radial_fraction)*np.cos(m*theta+phase)
        height = np.where((radial_fraction == 0) | (radial_fraction == 1), 0, height)
        return height*self.amplitude_m/max(float(np.max(np.abs(height))), 1e-15)
