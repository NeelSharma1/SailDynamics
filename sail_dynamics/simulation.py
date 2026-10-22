"""Orchestration and hourly exports; all geometry and optics belong to their objects."""
import csv
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import numpy as np
from .config import Config
from .constants import AU
from .environment import Environment, environment
from .loads import Loads
from .illumination import SpatialLoadEvaluator
from .sail.surface import Sail
from .sail.hub import Hub


class Simulation:
    def __init__(self, config: Config):
        self.config = config
        self.sail = Sail(config)
        self.hub = Hub(config)
        self.spatial = SpatialLoadEvaluator(config, self.sail, self.hub)
        self.hours = np.arange(0, config.duration_s + 1, 3600.0)
        self.year = environment(self.hours, config)
        self.annual_loads = self.loads(self.year)

    def loads(self, env: Environment):
        return self.spatial.loads(env)

    @property
    def unit_force(self):
        return self.sail.unit_force + self.hub.unit_force

    @property
    def unit_torque(self):
        return self.sail.unit_torque + self.hub.unit_torque

    @property
    def projected_area(self):
        return self.sail.projected_area + self.hub.projected_area

    def change_wrinkles(self, amplitude: float):
        self._change_geometry(wrinkle_m=float(amplitude))

    def change_hub_diameter(self, diameter: float):
        self._change_geometry(hub_diameter_m=float(diameter))

    def change_shape(self, shape: str):
        self._change_geometry(sail_shape=shape)

    def _change_geometry(self, **changes):
        config = replace(self.config, **changes)
        sail, hub = Sail(config), Hub(config)
        spatial = SpatialLoadEvaluator(config, sail, hub)
        self.config, self.sail, self.hub = config, sail, hub
        self.spatial = spatial
        self.annual_loads = self.loads(self.year)

    def export_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        loads, env = self.annual_loads, self.year
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["utc", "elapsed_hours", "sunlight_fraction", "sun_distance_au",
                             "beam_irradiance_W_m2", "intercepted_power_W",
                             "force_body_x_N", "force_body_y_N", "force_body_z_N",
                             "force_eci_x_N", "force_eci_y_N", "force_eci_z_N", "force_norm_N",
                             "acceleration_norm_m_s2", "torque_body_x_Nm", "torque_body_y_Nm",
                             "torque_body_z_Nm", "satellite_eci_x_m", "satellite_eci_y_m",
                             "satellite_eci_z_m", "sail_force_norm_N", "hub_force_norm_N",
                             "reference_circle_diameter_m", "hub_diameter_m", "membrane_projected_area_m2",
                             "hub_projected_area_m2", "sail_shape", "outer_corner_to_corner_m",
                             "outer_side_to_side_m"])
            for i, seconds in enumerate(self.hours):
                force_norm = np.linalg.norm(loads.forces_body[i])
                writer.writerow([(self.config.start + timedelta(seconds=float(seconds))).isoformat(),
                                 seconds / 3600, env.sunlight[i], env.distance_m[i] / AU,
                                 env.flux[i], loads.power_w[i], *loads.forces_body[i],
                                 *loads.forces_inertial[i], force_norm,
                                 force_norm / self.config.mass_kg, *loads.torques_body[i],
                                 *env.satellite_m[i],
                                 np.linalg.norm(loads.sail_forces_body[i]),
                                 np.linalg.norm(loads.hub_forces_body[i]),
                                 self.config.diameter_m, self.config.hub_diameter_m,
                                 self.sail.projected_area, self.hub.projected_area,
                                 self.config.sail_shape, 2*self.sail.boundary.max_radius_m,
                                 2*self.sail.boundary.min_radius_m])
        return path
