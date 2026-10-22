"""Earth-centered orbital-plane frame, independent of Earth's rotation."""
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..constants import EARTH_RADIUS


@dataclass(frozen=True)
class SatelliteOrbitalFrame:
    """Origin: Earth. X: ascending node; Z: orbit normal; Y completes RHS.

    Inclination and RAAN fix the plane in geocentric equatorial axes. At zero
    inclination, RAAN simply sets a reference direction. This circular orbit
    is prescribed, without SRP feedback, J2 precession, or aerodynamic drag.
    """
    config: Config

    @property
    def rotation_to_geocentric(self):
        inc, node = np.deg2rad([self.config.inclination_deg, self.config.raan_deg])
        ci, si, cn, sn = np.cos(inc), np.sin(inc), np.cos(node), np.sin(node)
        return np.array([[cn, -sn*ci, sn*si],
                         [sn, cn*ci, -cn*si],
                         [0, si, ci]])

    def vectors_to_geocentric(self, vectors):
        return np.einsum("ij,...j->...i", self.rotation_to_geocentric, vectors)

    def vectors_from_geocentric(self, vectors):
        return np.einsum("ji,...j->...i", self.rotation_to_geocentric, vectors)

    def positions_orbital(self, seconds):
        phase = (np.asarray(seconds)*2*np.pi/self.config.period_s
                 + np.deg2rad(self.config.phase_deg))
        return (EARTH_RADIUS+self.config.altitude_m)*np.stack(
            (np.cos(phase), np.sin(phase), np.zeros_like(phase)), axis=-1)

    def positions_geocentric(self, seconds):
        return self.vectors_to_geocentric(self.positions_orbital(seconds))
