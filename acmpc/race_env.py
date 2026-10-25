"""Drone racing through gates: the ACMPC paper's task (Sec. IV-A).

A vectorized stable-baselines3 VecEnv in plain numpy (no flightgym / ROS), so it
trains anywhere, e.g. on Colab.

Quadrotor: collective thrust + body rate (CTBR) commands. The default dynamics
("dodgeros") are a port of the ROS sim the policies are flown in
(dodgelib's QuadrotorSimulator with LowLevelControllerSimple, kingfisher.yaml):
a P body-rate controller computes torques, the allocation turns them into
per-rotor thrusts clamped to [0, thrust_max], and the motors follow with a
first-order lag. At full collective thrust there is no torque left, so body
rates respond slowly. Commands reach the drone after a delay (ROS round trip
plus the sim loop; 20 ms fits a recorded ROS flight), randomized per episode.
"simple" is the old idealized model: first-order thrust and body-rate
responses, no delay. Neither matches ACMPC's MPC model (drone.py) exactly.

Observation (paper Sec. IV-A1), 36 values, normalized with running stats:
  v (3), R (9, row-major), corners of the next gate relative to the drone (12),
  corners of the gate after it relative to the next gate's corners (12)
followed by the raw drone state [p, q(wxyz), v, w] (13) that ACMPC's MPC needs
(acmpc_policy.py splits it off again).

Action: [c, wx, wy, wz] in [-1, 1], normalized as in acmpc_env.py:
  c = (a0 + 1) * 4 * thrust_max / mass / 2,  w = a[1:] * omega_max

Reward (paper eq. 10): -10 crash, +10 gate passed, +10 race finished, otherwise
progress towards the next gate center - 0.01 * |w|.
"""
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from race_track import Track

OBS_DIM = 36
STATE_DIM = 13

# drone.py / kingfisher.yaml
MASS = 0.752
THRUST_MAX = 8.5  # N per rotor
OMEGA_MAX = np.array([10.0, 10.0, 4.0])
GRAVITY = np.array([0.0, 0.0, -9.8066])
C_HALF = 4 * THRUST_MAX / MASS / 2

# dodgeros sim: envsim/parameters/quads/kingfisher.yaml, dodgelib defaults
SIM_MASS = 0.750
INERTIA = np.array([0.0025, 0.0021, 0.0043])
K_RATE = np.array([20.0, 20.0, 40.0])  # LowLevelControllerSimple P gain (1/s)
TAU_MOTOR = 0.033
K_THRUST = 1.562522e-06  # rotor thrust = K_THRUST * omega^2
_ROTOR_X = np.array([0.0775, -0.0775, -0.0775, 0.0775])  # fr, bl, br, fl
_ROTOR_Y = np.array([-0.1, 0.1, -0.1, 0.1])
# [collective thrust, torque x, y, z] = ALLOC @ rotor thrusts
ALLOC = np.stack([np.ones(4), _ROTOR_Y, -_ROTOR_X, 0.022 * np.array([-1.0, -1.0, 1.0, 1.0])])
ALLOC_INV = np.linalg.inv(ALLOC)

DEFAULTS = dict(
    track="splits",
    laps=2,
    max_steps=1000,         # 20 s at 50 Hz
    ctrl_dt=0.02,           # = the MPC's dt
    sim_substeps=20,        # 1 ms, as dodgeros
    dynamics="dodgeros",    # or "simple"
    cmd_delay=(0.01, 0.03), # command delay (s), uniform per episode, at most 2 * ctrl_dt
    tau_thrust=0.033,       # "simple": thrust time constant (s)
    tau_rate=0.03,          # "simple": body-rate tracking time constant (s)
    rate_penalty=0.01,
    start_noise=0.25,       # start position: cube of 0.5 m side, as in the paper
    random_start_prob=0.5,  # fraction of episodes that start before a random gate
    # Disturbances (all off for nominal training; eval_race.py sweeps them)
    mass_scale=1.0,         # true mass / model mass
    omega_max_scale=1.0,    # body-rate limit of the "real" drone
    drag=0.0,               # linear drag (1/s)
    wind=(0.0, 0.0, 0.0),   # constant external force (N)
)


def quat_mul(a, b):
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], 1)


def quat_rot(q):
    """(B, 4) wxyz -> (B, 3, 3)."""
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
    ], 1)


def yaw_quat(yaw):
    return np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1)


class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        m, v, n = x.mean(0), x.var(0), x.shape[0]
        delta = m - self.mean
        tot = self.count + n
        self.mean = self.mean + delta * n / tot
        self.var = (self.var * self.count + v * n + delta ** 2 * self.count * n / tot) / tot
        self.count = tot


class RaceEnv(VecEnv):
    def __init__(self, num_envs=100, seed=0, **cfg):
        self.cfg = dict(DEFAULTS, **cfg)
        self.track = cfg.get("track_obj") or Track(self.cfg["track"])
        self.cfg.pop("track_obj", None)
        self.render_mode = None
        n_obs = OBS_DIM + STATE_DIM
        super().__init__(
            num_envs,
            spaces.Box(-np.inf, np.inf, (n_obs,), np.float32),
            spaces.Box(-1.0, 1.0, (4,), np.float32),
        )
        self.rng = np.random.default_rng(seed)
        self.rms = RunningMeanStd(OBS_DIM)
        self.training = True
        self.norm_obs = True

        n = num_envs
        self.p = np.zeros((n, 3))
        self.q = np.zeros((n, 4))
        self.v = np.zeros((n, 3))
        self.w = np.zeros((n, 3))
        self.c = np.zeros(n)            # "simple": actual mass-normalized collective thrust
        self.motors = np.zeros((n, 4))  # "dodgeros": rotor speeds (rad/s)
        # Last three commands [c, wx, wy, wz] (oldest first) and each env's delay
        self.cmds = np.zeros((n, 3, 4))
        self.delay = np.zeros(n)
        self.target = np.zeros(n, dtype=int)
        self.passed = np.zeros(n, dtype=int)
        self.steps = np.zeros(n, dtype=int)
        self.ep_ret = np.zeros(n)
        self.ep_dist = np.zeros(n)
        self._actions = None

    # ---- VecEnv API -------------------------------------------------------
    def reset(self):
        if self._seeds[0] is not None:
            self.rng = np.random.default_rng(self._seeds[0])
            self._reset_seeds()
        self._reset_idx(np.arange(self.num_envs))
        return self._obs(update=False)

    def step_async(self, actions):
        self._actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)

    def step_wait(self):
        cfg, tr = self.cfg, self.track
        a = self._actions
        w_lim = OMEGA_MAX * cfg["omega_max_scale"]
        cmd = np.concatenate([(a[:, :1] + 1.0) * C_HALF, np.clip(a[:, 1:] * OMEGA_MAX, -w_lim, w_lim)], 1)
        self.cmds = np.concatenate([self.cmds[:, 1:], cmd[:, None]], 1)

        p_prev = self.p.copy()
        rows = np.arange(self.num_envs)
        dt = cfg["ctrl_dt"] / cfg["sim_substeps"]
        k_c = 1 - np.exp(-dt / cfg["tau_thrust"])
        k_w = 1 - np.exp(-dt / cfg["tau_rate"])
        k_m = 1 - np.exp(-dt / TAU_MOTOR)
        mass = SIM_MASS * cfg["mass_scale"]
        ext = np.asarray(cfg["wind"]) / mass + GRAVITY
        for k in range(cfg["sim_substeps"]):
            # Each command takes effect self.delay after it was sent
            late = np.clip(np.ceil((self.delay - k * dt) / cfg["ctrl_dt"] - 1e-9), 0, 2).astype(int)
            u = self.cmds[rows, 2 - late]
            z_b = quat_rot(self.q)[:, :, 2]
            if cfg["dynamics"] == "simple":
                self.c += (u[:, 0] - self.c) * k_c
                self.w += (u[:, 1:] - self.w) * k_w
                acc = z_b * (self.c / cfg["mass_scale"])[:, None]
            else:
                # LowLevelControllerSimple: it assumes the nominal mass
                jw = INERTIA * self.w
                torque = INERTIA * K_RATE * (u[:, 1:] - self.w) + np.cross(self.w, jw)
                f_des = np.clip(np.concatenate([SIM_MASS * u[:, :1], torque], 1) @ ALLOC_INV.T, 0.0, THRUST_MAX)
                self.motors += (np.sqrt(f_des / K_THRUST) - self.motors) * k_m
                f = (K_THRUST * self.motors ** 2) @ ALLOC.T
                acc = z_b * (f[:, :1] / mass)
                self.w += (f[:, 1:] - np.cross(self.w, jw)) / INERTIA * dt
            self.v += (acc + ext - cfg["drag"] * self.v) * dt
            self.p += self.v * dt
            self.q = quat_mul(self.q, self._exp_map(self.w * dt))
        self.q /= np.linalg.norm(self.q, axis=1, keepdims=True)
        self.steps += 1
        self.ep_dist += np.linalg.norm(self.p - p_prev, axis=1)

        idx = np.arange(self.num_envs)
        g = tr.centers[self.target]
        rew = (np.linalg.norm(g - p_prev, axis=1) - np.linalg.norm(g - self.p, axis=1)
               - cfg["rate_penalty"] * np.linalg.norm(self.w, axis=1))

        passed, hit = tr.crossing(p_prev, self.p)
        gate_passed = passed[idx, self.target]
        self.passed += gate_passed
        self.target = np.where(gate_passed, (self.target + 1) % tr.num_gates, self.target)
        finished = self.passed >= cfg["laps"] * tr.num_gates
        out = np.any((self.p < tr.bounds[0]) | (self.p > tr.bounds[1]), axis=1)
        crashed = hit.any(1) | out
        rew = np.where(gate_passed, 10.0, rew)
        rew = np.where(finished, 10.0, rew)
        rew = np.where(crashed, -10.0, rew)
        self.ep_ret += rew

        terminated = crashed | finished
        truncated = ~terminated & (self.steps >= cfg["max_steps"])
        dones = terminated | truncated

        obs = self._obs(update=self.training)
        infos = [{} for _ in range(self.num_envs)]
        done_idx = np.flatnonzero(dones)
        for i in done_idx:
            t = self.steps[i] * cfg["ctrl_dt"]
            infos[i] = {
                "terminal_observation": obs[i].copy(),
                "TimeLimit.truncated": bool(truncated[i]),
                "episode": {"r": float(self.ep_ret[i]), "l": int(self.steps[i]), "t": t},
                "gates": int(self.passed[i]),
                "success": bool(finished[i] and not crashed[i]),
                "crash": bool(crashed[i]),
                "speed": float(self.ep_dist[i] / t),
            }
        if len(done_idx):
            self._reset_idx(done_idx)
            obs[done_idx] = self._obs(update=False)[done_idx]
        return obs, rew.astype(np.float32), dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs)] * len(self._indices(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._indices(indices))

    def _indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        return [indices] if isinstance(indices, int) else indices

    # ---- observation normalization ---------------------------------------
    def save_rms(self, path):
        np.savez(path, mean=self.rms.mean, var=self.rms.var, count=self.rms.count)

    def load_rms(self, path):
        d = np.load(path)
        self.rms.mean, self.rms.var, self.rms.count = d["mean"], d["var"], float(d["count"])

    # ---- internals ----------------------------------------------------------
    @staticmethod
    def _exp_map(rv):
        angle = np.linalg.norm(rv, axis=1, keepdims=True)
        axis = rv / np.maximum(angle, 1e-9)
        return np.concatenate([np.cos(angle / 2), axis * np.sin(angle / 2)], 1)

    def _reset_idx(self, idx):
        tr, cfg, n = self.track, self.cfg, len(idx)
        noise = self.rng.uniform(-cfg["start_noise"], cfg["start_noise"], (n, 3))
        target = np.zeros(n, dtype=int)
        pos = tr.start_pos + noise
        vel = np.zeros((n, 3))
        # Some episodes start between two gates, so later gates get trained early
        rand = self.rng.random(n) < cfg["random_start_prob"]
        if rand.any():
            k = rand.sum()
            gate = self.rng.integers(0, tr.num_gates, k)
            prev = np.where((gate == 0)[:, None], tr.start_pos, tr.centers[gate - 1])
            frac = self.rng.uniform(0.2, 0.8, (k, 1))
            pos[rand] = prev + frac * (tr.centers[gate] - prev) + 2 * noise[rand]
            heading = tr.centers[gate] - pos[rand]
            heading /= np.linalg.norm(heading, axis=1, keepdims=True)
            vel[rand] = heading * self.rng.uniform(0.0, 5.0, (k, 1))
            target[rand] = gate
        pos[:, 2] = np.maximum(pos[:, 2], 0.5)
        d = tr.centers[target] - pos
        yaw = np.arctan2(d[:, 1], d[:, 0])

        self.p[idx] = pos
        self.v[idx] = vel
        self.q[idx] = yaw_quat(yaw)
        self.w[idx] = 0.0
        hover = -GRAVITY[2] * cfg["mass_scale"]
        self.c[idx] = hover
        self.motors[idx] = np.sqrt(SIM_MASS * hover / 4 / K_THRUST)
        self.cmds[idx] = [hover, 0.0, 0.0, 0.0]
        lo, hi = cfg["cmd_delay"]
        assert hi <= 2 * cfg["ctrl_dt"], "cmd_delay is limited to two control steps"
        self.delay[idx] = self.rng.uniform(lo, hi, n)
        self.target[idx] = target
        self.passed[idx] = 0
        self.steps[idx] = 0
        self.ep_ret[idx] = 0.0
        self.ep_dist[idx] = 0.0

    def task_obs(self):
        tr = self.track
        nxt = tr.corners[self.target]                              # (B, 4, 3)
        after = tr.corners[(self.target + 1) % tr.num_gates]
        return np.concatenate([
            self.v,
            quat_rot(self.q).reshape(-1, 9),
            (nxt - self.p[:, None, :]).reshape(-1, 12),
            (after - nxt).reshape(-1, 12),
        ], 1)

    def drone_state(self):
        return np.concatenate([self.p, self.q, self.v, self.w], 1)

    def normalize(self, obs):
        return np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8), -10.0, 10.0)

    def _obs(self, update):
        obs = self.task_obs()
        if update:
            self.rms.update(obs)
        if self.norm_obs:
            obs = self.normalize(obs)
        return np.concatenate([obs, self.drone_state()], 1).astype(np.float32)
