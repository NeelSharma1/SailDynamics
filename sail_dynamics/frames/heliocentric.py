"""Sun-centered J2000 ecliptic frame and Earth's approximate revolution."""
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..constants import AU, J2000


@dataclass(frozen=True)
class HeliocentricEclipticFrame:
    """Origin: Sun. X: J2000 vernal equinox. Z: north ecliptic pole.

    Uses JPL's approximate 1800–2050 Earth/EM-barycenter elements and linear
    century rates. UTC replaces TDB. Lunar offsets and perturbations are omitted.
    https://ssd.jpl.nasa.gov/planets/approx_pos.html
    """

    def earth_position(self, seconds, config: Config):
        t = ((config.start - J2000).total_seconds() + np.asarray(seconds)) / (36525*86400)
        a = (1.00000261 + .00000562*t)*AU
        e = .01671123 - .00004392*t
        inc = np.deg2rad(-.00001531 - .01294668*t)
        longitude = np.deg2rad(100.46457166 + 35999.37244981*t)
        perihelion = np.deg2rad(102.93768193 + .32327364*t)
        mean_anomaly = (longitude-perihelion+np.pi) % (2*np.pi)-np.pi
        eccentric_anomaly = mean_anomaly.copy()
        for _ in range(6):
            eccentric_anomaly -= ((eccentric_anomaly-e*np.sin(eccentric_anomaly)-mean_anomaly)
                                   / (1-e*np.cos(eccentric_anomaly)))
        xp = a*(np.cos(eccentric_anomaly)-e)
        yp = a*np.sqrt(1-e**2)*np.sin(eccentric_anomaly)
        x = np.cos(perihelion)*xp-np.sin(perihelion)*yp
        y_rotated = np.sin(perihelion)*xp+np.cos(perihelion)*yp
        return np.stack((x, y_rotated*np.cos(inc), y_rotated*np.sin(inc)), axis=-1)
