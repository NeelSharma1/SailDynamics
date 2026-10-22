"""Rigid central hub, separate from the flexible membrane and its wrinkles."""
from dataclasses import replace

import numpy as np

from ..config import Config
from .facets import Mesh
from .optics import facet_loads


class Hub:
    """Flat opaque disk at sail Z=0; normally illuminated front face only.

    The hub shares the sail attitude. It is absorbing by default, with its own
    configurable reflection fractions. Its polygonal rim matches the membrane
    exactly so there is neither a gap nor overlapping area.
    """
    def __init__(self, config: Config):
        self.config = config
        theta = np.arange(config.angular_segments)*2*np.pi/config.angular_segments
        radius = config.hub_diameter_m/2
        rim = np.column_stack((radius*np.cos(theta), radius*np.sin(theta), np.zeros_like(theta)))
        vertices = np.vstack(([0, 0, 0], rim))
        i = np.arange(config.angular_segments)
        triangles = np.column_stack((np.zeros_like(i), i+1, (i+1) % len(i)+1))
        self.mesh = Mesh.from_vertices(vertices, triangles)
        self.optical_config = replace(config, specular=config.hub_specular, diffuse=config.hub_diffuse)
        self.unit_facets, _ = self.facet_loads(1.0)
        self.unit_force = self.unit_facets.sum(axis=0)
        lever = self.mesh.centers-np.asarray(config.attachment_m)
        self.unit_torque = np.cross(lever, self.unit_facets).sum(axis=0)
        self.projected_area = self.mesh.areas.sum()

    def facet_loads(self, flux):
        return facet_loads(self.mesh, np.asarray(flux), self.optical_config)
