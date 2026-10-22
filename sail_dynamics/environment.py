"""Sun/orbit geometry assembled through explicit reference-frame objects."""
from dataclasses import dataclass
import numpy as np
from .config import Config
from .constants import AU, SOLAR_CONSTANT
from .frames.geocentric import GeocentricEquatorialFrame
from .frames.orbital import SatelliteOrbitalFrame
from .shadows import visible_sun_fraction


@dataclass
class Environment:
    seconds: np.ndarray
    satellite_m: np.ndarray
    sun_unit: np.ndarray
    distance_m: np.ndarray
    sunlight: np.ndarray
    unshadowed_flux: np.ndarray
    flux: np.ndarray


def environment(seconds, config: Config):
    seconds = np.asarray(seconds, dtype=float)
    satellite = SatelliteOrbitalFrame(config).positions_geocentric(seconds)
    sun = GeocentricEquatorialFrame().sun_position(seconds, config)
    delta = sun - satellite
    distance = np.linalg.norm(delta, axis=-1)
    sunlight = visible_sun_fraction(satellite, sun)
    unshadowed = SOLAR_CONSTANT * (AU / distance)**2
    return Environment(seconds, satellite, delta / distance[..., None], distance,
                       sunlight, unshadowed, sunlight * unshadowed)
