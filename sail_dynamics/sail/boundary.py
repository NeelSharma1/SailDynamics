"""Outer sail outlines with a shared projected-area convention."""
import numpy as np

from ..config import Config


class SailBoundary:
    """Keep the existing circular mesh area when selecting a regular octagon.

    diameter_m is the reference circle diameter for either shape. The octagon
    has corners at 0, 45, ... degrees in sail axes. Angular mesh segments must
    include those corners; added boundary nodes lie on the eight straight sides.
    Hub geometry is independent of this outline.
    """
    def __init__(self, config: Config):
        self.config = config

    @property
    def projected_area_m2(self):
        n = self.config.angular_segments
        return n/8*np.sin(2*np.pi/n)*self.config.diameter_m**2

    @property
    def max_radius_m(self):
        if self.config.sail_shape == 'octagonal':
            return np.sqrt(self.projected_area_m2/(2*np.sqrt(2)))
        return self.config.diameter_m/2

    @property
    def min_radius_m(self):
        if self.config.sail_shape == 'octagonal':
            return self.max_radius_m*np.cos(np.pi/8)
        return self.max_radius_m

    def radii(self, theta):
        theta = np.asarray(theta, dtype=float)
        if self.config.sail_shape == 'circular':
            return np.full_like(theta, self.max_radius_m)
        # Each side is a line at the apothem from the origin. Its intersection
        # with the spoke at theta gives the radius of the true straight edge.
        local = np.remainder(theta, np.pi/4)-np.pi/8
        return self.min_radius_m/np.cos(local)

    def metadata(self):
        return {
            'shape': self.config.sail_shape,
            'reference_circle_diameter_m': self.config.diameter_m,
            'total_projected_area_m2': float(self.projected_area_m2),
            'outer_corner_to_corner_m': float(2*self.max_radius_m),
            'outer_side_to_side_m': float(2*self.min_radius_m),
        }
