"""Each frame object declares its origin, axes, and coordinate transformations."""
from .heliocentric import HeliocentricEclipticFrame
from .geocentric import GeocentricEquatorialFrame
from .orbital import SatelliteOrbitalFrame
from .sail_body import SailBodyFrame

__all__ = ["HeliocentricEclipticFrame", "GeocentricEquatorialFrame",
           "SatelliteOrbitalFrame", "SailBodyFrame"]
