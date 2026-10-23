"""Earth occultation evaluated separately at every membrane/hub facet center.

Fixed, parallel central Sun rays set the optical response. Finite solar-disc
visibility varies spatially across the sail. This does not integrate optical
force directions over the partially occulted solar disc. Inverse-square flux
is evaluated at the assembly center (negligible variation on meter scales).
"""
from dataclasses import dataclass

import numpy as np

from .constants import AU, EARTH_RADIUS, SUN_RADIUS
from .environment import environment
from .frames.sail_body import SailBodyFrame
from .loads import Loads
from .shadows import shadow_angles, disc_visible_fraction


def angular_position_bounds(observer_radius, extent_m, sun_distance):
    """Conservative changes of apparent directions/radii over a bounding sphere.

    The observer is on the circular orbit. These bounds allow *exact* full-light
    and full-umbra shortcuts and conservative force/torque pruning, rather than
    substituting the center illumination for a partially lit surface.
    """
    r, length = observer_radius, extent_m
    if length >= r-EARTH_RADIUS:
        raise ValueError("Sail bounding sphere must stay above Earth's surface.")
    beta = np.arcsin(EARTH_RADIUS/r)
    dbeta = max(np.arcsin(EARTH_RADIUS/(r-length))-beta,
                beta-np.arcsin(EARTH_RADIUS/(r+length)))
    ddirection = np.arcsin(length/r)+np.arcsin(length/np.asarray(sun_distance))
    alpha = np.arcsin(SUN_RADIUS/np.asarray(sun_distance))
    dalpha = np.arcsin(SUN_RADIUS/(np.asarray(sun_distance)-length))-alpha
    return dbeta, ddirection, dalpha


@dataclass
class FacetSnapshot:
    environment: object
    loads: Loads
    beam_flux: np.ndarray
    sunlight: np.ndarray
    sail_count: int


class SpatialLoadEvaluator:
    def __init__(self, config, sail, hub):
        self.config, self.sail, self.hub = config, sail, hub
        self.sail_count = len(sail.mesh.areas)
        self.centers = np.vstack((sail.mesh.centers, hub.mesh.centers))
        self.center_radius_squared = np.sum(self.centers**2, axis=-1)
        self.unit_forces = np.vstack((sail.unit_facets, hub.unit_facets))
        lever = self.centers-np.asarray(config.attachment_m)
        self.unit_moments = np.cross(lever, self.unit_forces)
        self.projected_areas = np.concatenate((sail.mesh.areas*sail.mesh.normals[:, 2],
                                               hub.mesh.areas))
        # Vertex extent bounds every point of each triangle, not just centroids.
        self.extent = max(np.linalg.norm(sail.mesh.vertices, axis=-1).max(),
                          np.linalg.norm(hub.mesh.vertices, axis=-1).max())
        angular_position_bounds(EARTH_RADIUS+config.altitude_m, self.extent, .9*AU)

    def illumination_bounds(self, env):
        sun = env.satellite_m+env.sun_unit*env.distance_m[..., None]
        alpha, beta, separation = shadow_angles(env.satellite_m, sun)
        db, dd, da = angular_position_bounds(EARTH_RADIUS+self.config.altitude_m,
                                             self.extent, env.distance_m)
        # Fraction is monotone in occultor radius and angular separation. The
        # solar-radius change is bounded separately by swept-disc area / area.
        radius_error = 4*da/np.maximum(alpha-da, 1e-15)
        low = disc_visible_fraction(alpha, beta+db, np.maximum(separation-dd, 0))-radius_error
        high = disc_visible_fraction(alpha, np.maximum(beta-db, 0), separation+dd)+radius_error
        low, high = np.clip(low, 0, 1), np.clip(high, 0, 1)
        # Exact sphere-contact predicates avoid tiny radius-error false positives.
        full = separation-dd >= beta+db+alpha+da
        dark = separation+dd <= beta-db-alpha-da
        low, high = np.where(full, 1, low), np.where(full, 1, high)
        low, high = np.where(dark, 0, low), np.where(dark, 0, high)
        return low, high

    def facet_flux(self, env):
        """Return [..., facet] beam flux and visible fraction, including the hub."""
        # Work in the body frame: the Sun vector is exactly (0,0,d). Expanding
        # the dot products avoids enormous [time,facet,3] inertial temporaries.
        # This is the same geometry as translating every centroid into ECI.
        position = SailBodyFrame(env.sun_unit).vectors_from_geocentric(env.satellite_m)
        dot = np.matmul(position, self.centers.T)
        radius2 = np.sum(env.satellite_m**2, axis=-1)[..., None]
        earth_distance = np.sqrt(radius2+2*dot+self.center_radius_squared)
        d = env.distance_m[..., None]
        qz = self.centers[:, 2]
        sun_distance = np.sqrt(d*d-2*d*qz+self.center_radius_squared)
        earth_sun_dot = -d*(position[..., 2, None]+qz)+dot+self.center_radius_squared
        separation = np.arccos(np.clip(earth_sun_dot/(earth_distance*sun_distance), -1, 1))
        alpha = np.arcsin(SUN_RADIUS/sun_distance)
        beta = np.arcsin(EARTH_RADIUS/earth_distance)
        visibility = disc_visible_fraction(alpha, beta, separation)
        return env.unshadowed_flux[..., None]*visibility, visibility

    def from_flux(self, env, flux):
        n = self.sail_count
        sf = np.matmul(flux[..., :n], self.unit_forces[:n])
        hf = np.matmul(flux[..., n:], self.unit_forces[n:])
        st = np.matmul(flux[..., :n], self.unit_moments[:n])
        ht = np.matmul(flux[..., n:], self.unit_moments[n:])
        body = sf+hf
        return Loads(body, SailBodyFrame(env.sun_unit).vectors_to_geocentric(body), st+ht,
                     np.einsum('...f,f->...', flux, self.projected_areas), sf, st, hf, ht)

    def loads(self, env, chunk_size=128):
        """All requested epochs, bounded memory; exact centroid shadows near limb."""
        shape = env.seconds.shape
        times = env.seconds.reshape(-1)
        flux = env.flux.reshape(-1)
        sf = flux[:, None]*self.sail.unit_force
        hf = flux[:, None]*self.hub.unit_force
        st = flux[:, None]*self.sail.unit_torque
        ht = flux[:, None]*self.hub.unit_torque
        power = flux*(self.sail.projected_area+self.hub.projected_area)
        low, high = self.illumination_bounds(env)
        partial = np.flatnonzero((low.reshape(-1) < 1) & (high.reshape(-1) > 0))
        for start in range(0, len(partial), chunk_size):
            indices = partial[start:start+chunk_size]
            local_env = environment(times[indices], self.config)
            beam_flux, _ = self.facet_flux(local_env)
            local = self.from_flux(local_env, beam_flux)
            sf[indices], hf[indices] = local.sail_forces_body, local.hub_forces_body
            st[indices], ht[indices] = local.sail_torques_body, local.hub_torques_body
            power[indices] = local.power_w
        sf, hf, st, ht = (v.reshape(shape+(3,)) for v in (sf, hf, st, ht))
        body = sf+hf
        return Loads(body, SailBodyFrame(env.sun_unit).vectors_to_geocentric(body), st+ht,
                     power.reshape(shape), sf, st, hf, ht)

    def snapshot(self, seconds):
        env = environment(float(seconds), self.config)
        flux, sunlight = self.facet_flux(env)
        return FacetSnapshot(env, self.from_flux(env, flux), flux, sunlight, self.sail_count)

    def magnitude_upper_bounds(self, env):
        """Triangle-inequality bounds in the same order as the study's metrics."""
        low, high = self.illumination_bounds(env)
        delta = np.maximum(env.sunlight-low, high-env.sunlight)
        vectors = (self.unit_forces, self.unit_moments)
        return np.stack([env.unshadowed_flux*(env.sunlight*np.linalg.norm(v.sum(axis=0))
                         + delta*np.linalg.norm(v, axis=-1).sum()) for v in vectors], axis=-1)
