"""Spacecraft-centered Sun-tracking sail reference frame."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SailBodyFrame:
    """Origin: sail center. +Z: toward Sun; X: north cross Z; Y: Z cross X.

    The attached assembly shares this attitude. Near the celestial pole an
    alternate reference axis avoids degeneracy. No separate spin is modeled.
    Positions require translation; force vectors require rotation only.
    """
    sun_unit_geocentric: np.ndarray

    @property
    def rotation_to_geocentric(self):
        z = np.asarray(self.sun_unit_geocentric, dtype=float)
        z = z / np.linalg.norm(z, axis=-1, keepdims=True)
        ref = np.broadcast_to([0.0, 0.0, 1.0], z.shape).copy()
        ref[np.abs(z[..., 2]) > .99] = [0.0, 1.0, 0.0]
        x = np.cross(ref, z)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return np.stack((x, np.cross(z, x), z), axis=-1)

    def vectors_to_geocentric(self, vectors):
        return np.einsum("...ij,...j->...i", self.rotation_to_geocentric, vectors)

    def vectors_from_geocentric(self, vectors):
        return np.einsum("...ji,...j->...i", self.rotation_to_geocentric, vectors)

    def positions_to_geocentric(self, positions, sail_center_geocentric):
        return self.vectors_to_geocentric(positions) + sail_center_geocentric

    def positions_from_geocentric(self, positions, sail_center_geocentric):
        return self.vectors_from_geocentric(np.asarray(positions)-sail_center_geocentric)
