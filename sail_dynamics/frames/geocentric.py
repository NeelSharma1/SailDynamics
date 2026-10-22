"""Earth-centered nonrotating J2000 equatorial reference frame (ECI)."""
from dataclasses import dataclass

import numpy as np

from ..config import Config
from .heliocentric import HeliocentricEclipticFrame


@dataclass(frozen=True)
class GeocentricEquatorialFrame:
    """Origin: Earth. X: J2000 equinox. Z: celestial north; Y completes RHS.

    Obliquity is a fixed orientation between ecliptic/equatorial axes, not a
    model of Earth's spin. No Earth-fixed rotating frame is needed.
    """
    obliquity_deg: float = 23.43928

    @property
    def ecliptic_rotation(self):
        angle = np.deg2rad(self.obliquity_deg)
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def vectors_from_ecliptic(self, vectors):
        return np.einsum("ij,...j->...i", self.ecliptic_rotation, vectors)

    def positions_from_heliocentric(self, positions, earth_position):
        return self.vectors_from_ecliptic(np.asarray(positions)-earth_position)

    def positions_to_heliocentric(self, positions, earth_position):
        return np.einsum("ji,...j->...i", self.ecliptic_rotation, positions) + earth_position

    def sun_position(self, seconds, config: Config):
        earth = HeliocentricEclipticFrame().earth_position(seconds, config)
        return self.positions_from_heliocentric(np.zeros_like(earth), earth)
