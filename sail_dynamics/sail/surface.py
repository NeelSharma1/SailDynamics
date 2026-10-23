"""Flexible sail triangulated along radial spokes from hub to chosen outline."""
import numpy as np
from ..config import Config
from .facets import Mesh
from .wrinkles import WrinkleField, StochasticWrinkleField
from .optics import facet_loads
from .boundary import SailBoundary
from .contraction import area_preserving_vertices


def make_mesh(config: Config) -> Mesh:
    """Single-valued height field: no folds/overhangs or self-shadowing.

    Periodic connectivity closes the angular seam without duplicated vertices.
    The inner ring meets the rigid hub; the outer ring defines the sail rim.
    Both rings are flat. The outer rim contracts to conserve total membrane
    material area, while the hub ring stays fixed. Alternating diagonals reduce
    triangulation bias.
    """
    n, sectors = config.radial_cells, config.angular_segments
    radial_fraction, theta = np.meshgrid(np.linspace(0, 1, n + 1),
                                         np.arange(sectors)*2*np.pi/sectors, indexing="ij")
    inner = config.hub_diameter_m/2
    boundary = SailBoundary(config)
    outer = boundary.radii(theta)
    field = (StochasticWrinkleField(config.wrinkle_m, config.seed,
                                    max(1, n//2), max(1, sectors//4))
             if config.wrinkle_model == "stochastic" else WrinkleField(config.wrinkle_m, config.seed))
    height = field.heights(radial_fraction, theta)
    triangles = []
    for j in range(n):
        for i in range(sectors):
            a, b = j*sectors+i, (j+1)*sectors+i
            c, d = j*sectors+(i+1) % sectors, (j+1)*sectors+(i+1) % sectors
            triangles.extend(((a, b, d), (a, d, c)) if (i+j) % 2 == 0
                             else ((a, b, c), (b, d, c)))
    triangles = np.asarray(triangles, dtype=int)
    hub_area = sectors/2*np.sin(2*np.pi/sectors)*inner**2
    material_area = boundary.reference_area_m2-hub_area
    vertices = area_preserving_vertices(radial_fraction, theta, outer, inner,
                                        height, triangles, material_area, boundary.min_radius_m)
    return Mesh.from_vertices(vertices, triangles)


class Sail:
    """Geometry and all forces live in the Sun-tracking sail-body frame."""
    def __init__(self, config: Config):
        self.config = config
        self.mesh = make_mesh(config)
        nominal = SailBoundary(config)
        scale = np.linalg.norm(self.mesh.vertices[-config.angular_segments:, :2], axis=-1).max()/nominal.max_radius_m
        self.boundary = SailBoundary(config, scale)
        self.material_area = float(self.mesh.areas.sum())
        # With fixed normals and Sun-tracking attitude all loads scale with flux.
        self.unit_facets, _ = facet_loads(self.mesh, 1.0, config)
        self.unit_force = self.unit_facets.sum(axis=0)
        lever = self.mesh.centers - np.asarray(config.attachment_m)
        self.unit_torque = np.cross(lever, self.unit_facets).sum(axis=0)
        self.projected_area = np.sum(self.mesh.areas * self.mesh.normals[:, 2])

    def metadata(self):
        return dict(self.boundary.metadata(), membrane_material_area_m2=self.material_area,
                    membrane_projected_area_m2=float(self.projected_area),
                    contraction_model='Uniform outer-outline scale, fixed hub; total membrane area conserved. '
                                      'Local triangle areas/edge lengths are not constrained.')

    def facet_loads(self, flux):
        return facet_loads(self.mesh, np.asarray(flux), self.config)
