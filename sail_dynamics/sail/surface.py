"""Flexible sail triangulated along radial spokes from hub to chosen outline."""
import numpy as np
from ..config import Config
from .facets import Mesh
from .wrinkles import WrinkleField, StochasticWrinkleField
from .optics import facet_loads
from .boundary import SailBoundary


def make_mesh(config: Config) -> Mesh:
    """Single-valued height field: no folds/overhangs or self-shadowing.

    Periodic connectivity closes the angular seam without duplicated vertices.
    The inner ring meets the rigid hub; the outer ring defines the sail rim.
    Both rings are flat. Alternating diagonals reduce triangulation bias.
    """
    n, sectors = config.radial_cells, config.angular_segments
    radial_fraction, theta = np.meshgrid(np.linspace(0, 1, n + 1),
                                         np.arange(sectors)*2*np.pi/sectors, indexing="ij")
    inner = config.hub_diameter_m/2
    outer = SailBoundary(config).radii(theta)
    radius = inner + (outer-inner)*radial_fraction
    field = (StochasticWrinkleField(config.wrinkle_m, config.seed,
                                    max(1, n//2), max(1, sectors//4))
             if config.wrinkle_model == "stochastic" else WrinkleField(config.wrinkle_m, config.seed))
    height = field.heights(radial_fraction, theta)
    vertices = np.column_stack(((radius*np.cos(theta)).ravel(),
                                (radius*np.sin(theta)).ravel(), height.ravel()))
    triangles = []
    for j in range(n):
        for i in range(sectors):
            a, b = j*sectors+i, (j+1)*sectors+i
            c, d = j*sectors+(i+1) % sectors, (j+1)*sectors+(i+1) % sectors
            triangles.extend(((a, b, d), (a, d, c)) if (i+j) % 2 == 0
                             else ((a, b, c), (b, d, c)))
    triangles = np.asarray(triangles, dtype=int)
    return Mesh.from_vertices(vertices, triangles)


class Sail:
    """Geometry and all forces live in the Sun-tracking sail-body frame."""
    def __init__(self, config: Config):
        self.config = config
        self.boundary = SailBoundary(config)
        self.mesh = make_mesh(config)
        # With fixed normals and Sun-tracking attitude all loads scale with flux.
        self.unit_facets, _ = facet_loads(self.mesh, 1.0, config)
        self.unit_force = self.unit_facets.sum(axis=0)
        lever = self.mesh.centers - np.asarray(config.attachment_m)
        self.unit_torque = np.cross(lever, self.unit_facets).sum(axis=0)
        self.projected_area = np.sum(self.mesh.areas * self.mesh.normals[:, 2])

    def facet_loads(self, flux):
        return facet_loads(self.mesh, np.asarray(flux), self.config)
