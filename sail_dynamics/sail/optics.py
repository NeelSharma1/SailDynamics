"""Momentum exchange for absorbing, specular, and Lambertian diffuse facets."""
import numpy as np
from ..config import Config
from ..constants import LIGHT_SPEED
from .facets import Mesh


def facet_loads(mesh: Mesh, flux: float, config: Config):
    """Per-facet force [N] and incident irradiance [W/m² of actual area].

    F = I A mu/c * [(1-r_s) k - (2 r_s mu + 2 r_d/3) n], k=(0,0,-1).
    Unreflected light is absorbed; diffuse reflection is Lambertian. No thermal
    reradiation or transmission. Flux may be a scalar or one value per facet.
    """
    mu = np.maximum(mesh.normals[:, 2], 0.0)
    irradiance = flux * mu
    momentum = irradiance * mesh.areas / LIGHT_SPEED
    direction = -(2 * config.specular * mu + 2 * config.diffuse / 3)[:, None] * mesh.normals
    direction[:, 2] -= 1 - config.specular
    return momentum[:, None] * direction, irradiance
