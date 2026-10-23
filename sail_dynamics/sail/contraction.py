"""Conserve total membrane area by contracting a prescribed height-field mesh.

This is a geometric closure, not a local inextensibility or stress solver.
The rigid hub stays fixed and the outer outline retains its shape. A single
outline scale is chosen so summed 3-D triangle areas equal the flat material
area. Individual triangle areas and edge lengths need not be conserved.
"""
import numpy as np


def area_preserving_vertices(fraction, theta, outer, inner, height, triangles,
                             material_area, minimum_outer_radius):
    def vertices(scale):
        radius = inner+(scale*outer-inner)*fraction
        return np.column_stack(((radius*np.cos(theta)).ravel(),
                                (radius*np.sin(theta)).ravel(), height.ravel()))

    nominal = vertices(1.0)
    if not np.any(height):
        return nominal

    # Vertices are affine in scale, so facet cross products are quadratic.
    # Reuse these coefficients during bisection instead of rebuilding meshes.
    base = vertices(0.0)
    delta = nominal-base
    a = base[triangles[:, 1:]]-base[triangles[:, :1]]
    b = delta[triangles[:, 1:]]-delta[triangles[:, :1]]
    c0 = np.cross(a[:, 0], a[:, 1])
    c1 = np.cross(a[:, 0], b[:, 1])+np.cross(b[:, 0], a[:, 1])
    c2 = np.cross(b[:, 0], b[:, 1])

    def area(scale):
        return .5*np.linalg.norm(c0+scale*c1+scale*scale*c2, axis=-1).sum()

    # The contracted outer outline must still contain the entire circular hub.
    low = inner/minimum_outer_radius*(1+1e-12)
    high = 1.0
    if area(low) >= material_area:
        raise ValueError("Cannot conserve membrane area for this wrinkle pattern while keeping "
                         "the outer rim outside the fixed hub. Reduce wrinkle height or hub diameter.")
    for _ in range(44):
        mid = (low+high)/2
        if area(mid) > material_area:
            high = mid
        else:
            low = mid
    return vertices((low+high)/2)
