"""Earth occultation in the geocentric equatorial frame."""
import numpy as np
from .constants import EARTH_RADIUS, SUN_RADIUS


def shadow_angles(satellite, sun_geocentric):
    """Apparent Sun radius, Earth radius, and center separation [radians]."""
    satellite, sun_geocentric = np.broadcast_arrays(
        np.asarray(satellite, dtype=float), np.asarray(sun_geocentric, dtype=float))
    to_sun = sun_geocentric - satellite
    earth_distance = np.linalg.norm(satellite, axis=-1)
    sun_distance = np.linalg.norm(to_sun, axis=-1)
    if np.any(earth_distance <= EARTH_RADIUS):
        raise ValueError("Shadow observer must be above Earth's surface.")
    alpha = np.arcsin(np.clip(SUN_RADIUS / sun_distance, 0, 1))
    beta = np.arcsin(EARTH_RADIUS / earth_distance)
    separation = np.arccos(np.clip(np.sum(-satellite * to_sun, axis=-1)
                                   / (earth_distance * sun_distance), -1, 1))
    return alpha, beta, separation


def disc_visible_fraction(alpha, beta, separation):
    """Visible fraction of a uniform Sun using planar angular-disc overlap."""
    alpha, beta, separation = np.broadcast_arrays(alpha, beta, separation)
    shape = alpha.shape
    a, b, d = (v.reshape(-1) for v in (alpha, beta, separation))
    visible = np.ones_like(a)
    contained = d <= np.abs(b - a)
    visible[contained] = 1 - np.minimum(1, (b[contained] / a[contained]) ** 2)
    partial = (~contained) & (d < a + b)
    ap, bp, dp = a[partial], b[partial], d[partial]
    theta_a = np.arccos(np.clip((dp**2 + ap**2 - bp**2) / (2 * dp * ap), -1, 1))
    theta_b = np.arccos(np.clip((dp**2 + bp**2 - ap**2) / (2 * dp * bp), -1, 1))
    radicand = (-dp + ap + bp) * (dp + ap - bp) * (dp - ap + bp) * (dp + ap + bp)
    overlap = ap**2 * theta_a + bp**2 * theta_b - 0.5 * np.sqrt(np.maximum(radicand, 0))
    visible[partial] = 1 - overlap / (np.pi * ap**2)
    return np.clip(visible.reshape(shape), 0, 1)


def visible_sun_fraction(satellite, sun_geocentric):
    """Spherical opaque Earth; no atmosphere/refraction/solar limb darkening."""
    return disc_visible_fraction(*shadow_angles(satellite, sun_geocentric))
