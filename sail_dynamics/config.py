"""Shared immutable simulation settings; angles are in degrees, lengths in meters."""
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
from .constants import EARTH_RADIUS, EARTH_MU


@dataclass(frozen=True)
class Config:
    year: int = 2026
    altitude_m: float = 400_000.0
    inclination_deg: float = 51.6
    raan_deg: float = 0.0
    phase_deg: float = 0.0
    diameter_m: float = 10.0
    sail_shape: str = "circular"
    hub_diameter_m: float = 1.0
    radial_cells: int = 12
    angular_segments: int = 96
    wrinkle_m: float = 0.18
    seed: int = 7
    wrinkle_model: str = "modal"
    specular: float = 0.85
    diffuse: float = 0.10
    hub_specular: float = 0.0
    hub_diffuse: float = 0.0
    mass_kg: float = 20.0
    attachment_m: tuple[float, float, float] = (0.0, 0.0, -0.8)

    def __post_init__(self):
        if len(self.attachment_m) != 3:
            raise ValueError("Attachment position must have three coordinates.")
        numeric = (self.altitude_m, self.inclination_deg, self.raan_deg,
                   self.phase_deg, self.diameter_m, self.hub_diameter_m, self.wrinkle_m,
                   self.specular, self.diffuse, self.hub_specular, self.hub_diffuse,
                   self.mass_kg, *self.attachment_m)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("All configuration values must be finite.")
        if self.altitude_m <= 0 or self.diameter_m <= 0 or self.mass_kg <= 0:
            raise ValueError("Altitude, sail diameter, and total mass must be positive.")
        if not 0 < self.hub_diameter_m < self.diameter_m:
            raise ValueError("Hub diameter must be positive and smaller than sail diameter.")
        if not isinstance(self.radial_cells, int) or self.radial_cells < 2:
            raise ValueError("Use at least two radial mesh cells.")
        if not isinstance(self.angular_segments, int) or self.angular_segments < 8:
            raise ValueError("Use at least eight angular segments.")
        if self.sail_shape not in ("circular", "octagonal"):
            raise ValueError("Sail shape must be circular or octagonal.")
        if self.sail_shape == "octagonal":
            if self.angular_segments % 8:
                raise ValueError("An octagonal sail requires --segments to be a multiple of eight.")
            # A regular octagon of area A has apothem sqrt(A/(8*tan(pi/8))).
            area = self.angular_segments/8*np.sin(2*np.pi/self.angular_segments)*self.diameter_m**2
            across_sides = 2*np.sqrt(area/(8*np.tan(np.pi/8)))
            if self.hub_diameter_m >= across_sides:
                raise ValueError(f"Hub must fit inside the octagon: diameter must be less than {across_sides:.6g} m.")
        if self.wrinkle_m < 0:
            raise ValueError("Wrinkle amplitude cannot be negative.")
        if self.wrinkle_model not in ("modal", "stochastic"):
            raise ValueError("Wrinkle model must be modal or stochastic.")
        for label, specular, diffuse in (("Sail", self.specular, self.diffuse),
                                         ("Hub", self.hub_specular, self.hub_diffuse)):
            if not 0 <= specular <= 1 or not 0 <= diffuse <= 1:
                raise ValueError(f"{label} optical fractions must be between zero and one.")
            if specular + diffuse > 1:
                raise ValueError(f"{label} specular + diffuse reflectivity cannot exceed one.")
        if not 1800 <= self.year <= 2049:
            raise ValueError("The approximate ephemeris supports years 1800–2049.")
        if not 0 <= self.inclination_deg <= 180:
            raise ValueError("Inclination must be between 0 and 180 degrees.")

    @property
    def start(self):
        return datetime(self.year, 1, 1, tzinfo=timezone.utc)

    @property
    def duration_s(self):
        return (datetime(self.year + 1, 1, 1, tzinfo=timezone.utc) - self.start).total_seconds()

    @property
    def period_s(self):
        return 2 * np.pi * np.sqrt((EARTH_RADIUS + self.altitude_m) ** 3 / EARTH_MU)
