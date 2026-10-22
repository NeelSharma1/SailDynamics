"""Minute grid plus fine sampling of every finite-Sun eclipse transition."""
import numpy as np

from .constants import EARTH_RADIUS, AU
from .environment import environment
from .illumination import angular_position_bounds
from .shadows import shadow_angles
from .sail.boundary import SailBoundary


def time_grid(duration_s, step_s=60.0):
    times = np.arange(0, duration_s, step_s, dtype=float)
    return np.append(times, float(duration_s))


def eclipse_refinement_times(config, duration_s, step_s=.25, extent_m=None):
    """Locate orbit minima, then bisect outer and inner solar-disc contacts.

    Locating each orbit's minimum also detects grazing penumbra events that a
    minute grid could miss completely. Windows are expanded by a bounding sphere
    so the first/last *facet* can enter/leave shadow outside the center contacts.
    Both ingress and egress are included. A clipped year endpoint is retained.
    """
    if not np.isfinite(step_s) or step_s <= 0:
        raise ValueError("Refinement timestep must be positive and finite.")
    if extent_m is None:
        extent_m = np.hypot(SailBoundary(config).max_radius_m, config.wrinkle_m)
    db, dd, da = angular_position_bounds(EARTH_RADIUS+config.altitude_m, extent_m, .9*AU)
    margin = db+dd+da

    def contact(seconds, inner=False):
        env = environment(seconds, config)
        sun = env.satellite_m+env.sun_unit*env.distance_m[..., None]
        alpha, beta, separation = shadow_angles(env.satellite_m, sun)
        return separation-beta + (alpha+margin if inner else -alpha-margin)

    coarse = time_grid(duration_s)
    g = contact(coarse)
    minima = np.flatnonzero((g[1:-1] <= g[:-2]) & (g[1:-1] < g[2:]))+1
    if g[0] < 0:
        minima = np.r_[0, minima]
    if g[-1] < 0:
        minima = np.r_[minima, len(coarse)-1]
    if not len(minima):
        return np.empty(0), 0
    left = coarse[np.maximum(minima-1, 0)].copy()
    right = coarse[np.minimum(minima+1, len(coarse)-1)].copy()
    for _ in range(32):
        p, q = left+(right-left)/3, right-(right-left)/3
        choose_left = contact(p) < contact(q)
        left, right = np.where(choose_left, left, p), np.where(choose_left, q, right)
    centers = (left+right)/2
    centers = centers[contact(centers) < 0]
    if not len(centers):
        return np.empty(0), 0

    def root(side, inner=False, selected=None):
        center = centers if selected is None else centers[selected]
        outside = np.clip(center+side*config.period_s/2, 0, duration_s)
        inside = center.copy()
        clipped = contact(outside, inner) <= 0
        for _ in range(32):
            mid = (outside+inside)/2
            is_inside = contact(mid, inner) <= 0
            inside = np.where(is_inside, mid, inside)
            outside = np.where(is_inside, outside, mid)
        return np.where(clipped, np.clip(center+side*config.period_s/2, 0, duration_s),
                        (outside+inside)/2)

    ingress, egress = root(-1), root(1)
    deep = contact(centers, inner=True) < 0
    inner_in, inner_out = ingress.copy(), egress.copy()
    if np.any(deep):
        inner_in[deep], inner_out[deep] = root(-1, True, deep), root(1, True, deep)
    windows = []
    for i in range(len(centers)):
        spans = ((ingress[i], inner_in[i]), (inner_out[i], egress[i])) if deep[i] else (
            (ingress[i], egress[i]),)
        for start, end in spans:
            if end >= start:
                windows.append(np.linspace(start, end, max(1, int(np.ceil((end-start)/step_s)))+1))
    return np.unique(np.concatenate(windows)) if windows else np.empty(0), len(centers)
