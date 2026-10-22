"""Load vectors in SI; torque is about Config.attachment_m in sail axes."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Loads:
    forces_body: np.ndarray
    forces_inertial: np.ndarray
    torques_body: np.ndarray
    power_w: np.ndarray
    sail_forces_body: np.ndarray
    sail_torques_body: np.ndarray
    hub_forces_body: np.ndarray
    hub_torques_body: np.ndarray
