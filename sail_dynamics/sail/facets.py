"""Triangular facets: vertices, connectivity, centers, normals, and actual areas."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray
    triangles: np.ndarray
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray

    @classmethod
    def from_vertices(cls, vertices, triangles):
        vertices, triangles = np.asarray(vertices, float), np.asarray(triangles, int)
        points = vertices[triangles]
        cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        twice_area = np.linalg.norm(cross, axis=1)
        if np.any(twice_area <= 0):
            raise ValueError("Degenerate triangular facet.")
        return cls(vertices, triangles, points.mean(axis=1),
                   cross / twice_area[:, None], twice_area / 2)
