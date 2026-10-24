"""flightgym vectorized env for ACMPC.

VisionEnv_v1 already takes collective thrust + body rate (CTBR) actions in
[-1, 1], normalized the same way as ACMPC's MPC output:
  thrust = a0 * T + T,  T = 4 * thrust_max / mass / 2
  omega  = a[1:] * omega_max
flightlib derives thrust_max from motor_omega_max (2800 rad/s gives 12.25 N per
rotor), so the config is adjusted to drone.py's limits.
The observation gets the raw drone state [p, q(wxyz), v, w] (13) appended, which
the MPC needs unnormalized; acmpc_policy.py splits it off again.
"""
import io

import numpy as np
from gymnasium import spaces
from ruamel.yaml import YAML

from flightgym import VisionEnv_v1
from rpg_baselines.torch.envs.vec_env_wrapper import FlightEnvVec

from acmpc_policy import STATE_DIM

# Must match diff_mpc_drones/drone.py (DroneDx.omega_max, thrust_max), which the
# MPC uses for its input bounds and to normalize its output.
ACMPC_OMEGA_MAX = [10.0, 10.0, 4.0]
ACMPC_THRUST_MAX = 8.5  # N per rotor


def motor_omega_for_thrust(thrust_map, thrust):
    # Solve thrust_map[0] w^2 + thrust_map[1] w + thrust_map[2] = thrust for w > 0
    a, b, c = thrust_map
    return (-b + np.sqrt(b * b - 4 * a * (c - thrust))) / (2 * a)


def acmpc_config(cfg):
    """VisionEnv config with the quadrotor limits of ACMPC's MPC model (these
    also match the ROS sim's kingfisher.yaml)."""
    cfg = dict(cfg)
    dyn = cfg["quadrotor_dynamics"] = dict(cfg["quadrotor_dynamics"])
    dyn["omega_max"] = ACMPC_OMEGA_MAX
    dyn["motor_omega_max"] = float(motor_omega_for_thrust(dyn["thrust_map"], ACMPC_THRUST_MAX))
    return cfg


def make_vision_env(cfg, num_envs=None):
    cfg = dict(cfg)
    if num_envs is not None:
        cfg["simulation"] = dict(cfg["simulation"])
        cfg["simulation"]["num_envs"] = num_envs
    buf = io.StringIO()
    YAML().dump(cfg, buf)
    return ACMPCFlightEnvVec(VisionEnv_v1(buf.getvalue(), False))


class ACMPCFlightEnvVec(FlightEnvVec):
    def __init__(self, impl):
        super().__init__(impl)
        n = self.obs_dim + STATE_DIM
        self._acmpc_observation_space = spaces.Box(
            -np.inf * np.ones(n), np.inf * np.ones(n), dtype=np.float64
        )

    @property
    def observation_space(self):
        return self._acmpc_observation_space

    def drone_state(self):
        # getQuadState: [t, p(3), q(4), v(3), w(3), a(3), motor omega(4), ...]
        return self.getQuadState()[:, 1 : 1 + STATE_DIM].copy()

    def normalize_obs(self, obs):
        # FlightEnvVec calls this on every step/reset/getObs: normalize the task
        # observation with the running stats, then append the raw drone state.
        return np.concatenate([super().normalize_obs(obs), self.drone_state()], axis=1)
