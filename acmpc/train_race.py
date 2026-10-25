#!/usr/bin/env python3
"""Train AC-MPC or AC-MLP on gate racing (race_env.py) with PPO.

Hyperparameters from the ACMPC paper, Table II. Runs need no flightgym or ROS,
so they also train on Colab (see colab_train_race.ipynb); a CUDA GPU is used
when available.

  python acmpc/train_race.py --policy mpc --track splits     # AC-MPC
  python acmpc/train_race.py --policy mlp --track splits     # AC-MLP baseline

Each run goes to acmpc/saved/race/<MPC|MLP>_<track>_<n>/ with
  config.json, Policy/iter_<k>.pth, RMS/iter_<k>.npz and TensorBoard logs.
Evaluate with eval_race.py, fly in the ROS sim with run_race_ros.py.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Upstream ACMPC networks (training_modules) and the differentiable MPC
_acmpc = os.environ.get("ACMPC_PATH", os.path.join(HERE, "..", "src", "acmpc_public"))
for sub in ("training_modules", "diff_mpc_drones", "mpc.pytorch"):
    sys.path.insert(0, os.path.join(_acmpc, sub))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from acmpc_policy import MlpMpcPolicy, MlpOnlyPolicy  # noqa: E402
from race_env import DEFAULTS, RaceEnv  # noqa: E402

POLICIES = {"mpc": MlpMpcPolicy, "mlp": MlpOnlyPolicy}
# Paper Table II: AC-MPC needs less exploration noise than AC-MLP
LOG_STD_INIT = {"mpc": -1.2, "mlp": -0.5}


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=POLICIES, default="mpc")
    p.add_argument("--track", default="splits", help="name in acmpc/tracks/ or a path to a track yaml")
    p.add_argument("--T", type=int, default=5, help="MPC horizon (ACMPC_T)")
    p.add_argument("--iterations", type=int, default=2000, help="PPO iterations (num_envs * n_steps samples each)")
    p.add_argument("--num_envs", type=int, default=100)
    p.add_argument("--n_steps", type=int, default=250)
    p.add_argument("--laps", type=int, default=2)
    p.add_argument("--log_std_init", type=float, default=None, help="default: -1.2 (mpc) / -0.5 (mlp)")
    p.add_argument("--save_every", type=int, default=50, help="iterations between checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--log_dir", default=os.path.join(HERE, "saved", "race"))
    p.add_argument("--resume", help="continue RUN_DIR from its latest checkpoint, or RUN_DIR:ITER from that one")
    return p


def linear_schedule(start, end):
    return lambda progress_remaining: end + (start - end) * progress_remaining


def next_run_dir(log_dir, prefix):
    n = 1 + max([int(d.rsplit("_", 1)[1]) for d in glob.glob(os.path.join(log_dir, prefix + "_*"))
                 if d.rsplit("_", 1)[1].isdigit()] + [0])
    return os.path.join(log_dir, "{0}_{1}".format(prefix, n))


class RaceCallback(BaseCallback):
    """Logs racing metrics of finished training episodes and saves checkpoints."""

    def __init__(self, run_dir, save_every, iteration=0):
        super().__init__()
        self.run_dir = run_dir
        self.save_every = save_every
        self.iteration = iteration
        self.start = iteration
        self.episodes = []
        self.t0 = time.time()

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episodes.append((info["success"], info["crash"], info["gates"], info["speed"]))
        return True

    def _on_rollout_end(self):
        self.iteration += 1
        if self.episodes:
            s, c, g, v = np.array(self.episodes, dtype=np.float64).T
            self.logger.record("race/success_rate", s.mean())
            self.logger.record("race/crash_rate", c.mean())
            self.logger.record("race/gates_passed", g.mean())
            self.logger.record("race/avg_speed", v.mean())
            self.episodes = []
        self.logger.record("race/iteration", self.iteration)
        self.logger.record("time/iteration_s", (time.time() - self.t0) / (self.iteration - self.start))
        if self.iteration % self.save_every == 0:
            self.save(self.iteration)

    def save(self, it):
        if not all(torch.isfinite(p).all() for p in self.model.policy.parameters()):
            print("Policy has NaN/inf parameters, not saving iteration", it)
            return
        os.makedirs(os.path.join(self.run_dir, "Policy"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "RMS"), exist_ok=True)
        # SB3's policy.save() stores the state dict and the constructor args
        self.model.policy.save(os.path.join(self.run_dir, "Policy", "iter_{0:05d}.pth".format(it)))
        self.training_env.save_rms(os.path.join(self.run_dir, "RMS", "iter_{0:05d}.npz".format(it)))


def main():
    args = parser().parse_args()
    start_it = 0
    if args.resume:
        run_dir, _, resume_it = args.resume.partition(":")
        run_dir = run_dir.rstrip("/")
        with open(os.path.join(run_dir, "config.json")) as f:
            saved_args = json.load(f)
        for k in ("policy", "track", "T", "num_envs", "n_steps", "laps", "seed"):
            setattr(args, k, saved_args[k])
        args.log_std_init = saved_args["log_std_init"]
        # Runs from before the dodgeros model have no "dynamics" entry
        dynamics = saved_args["env"].get("dynamics", "simple")
        if dynamics != DEFAULTS["dynamics"]:
            sys.exit("{0} was trained with the {1!r} dynamics, race_env.py now simulates {2!r}: "
                     "start a new run".format(run_dir, dynamics, DEFAULTS["dynamics"]))
        start_it = int(resume_it) if resume_it else max(
            int(os.path.basename(p)[5:10]) for p in glob.glob(os.path.join(run_dir, "Policy", "iter_*.pth")))
    os.environ["ACMPC_T"] = str(args.T)
    # On a GPU, solve the MPC for the whole PPO batch at once: the solver is
    # launch-bound, and the default 1024-sample chunks run one after another
    if args.device != "cpu" and torch.cuda.is_available():
        os.environ.setdefault("ACMPC_CHUNK", str(args.n_steps * args.num_envs))
    torch.manual_seed(args.seed + start_it)

    env = RaceEnv(num_envs=args.num_envs, seed=args.seed + start_it, track=args.track, laps=args.laps)
    log_std = args.log_std_init if args.log_std_init is not None else LOG_STD_INIT[args.policy]
    if not args.resume:
        run_dir = next_run_dir(args.log_dir, "{0}_{1}".format(args.policy.upper(), env.track.name))
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(dict(vars(args), log_std_init=log_std, track_path=env.track.path, env=env.cfg), f, indent=2)
    print("Run directory:", run_dir, "from iteration", start_it)

    model = PPO(
        POLICIES[args.policy],
        env,
        learning_rate=linear_schedule(3e-4, 1e-5),
        n_steps=args.n_steps,
        batch_size=args.n_steps * args.num_envs,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(log_std_init=log_std),
        tensorboard_log=run_dir,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    if start_it:
        saved = torch.load(os.path.join(run_dir, "Policy", "iter_{0:05d}.pth".format(start_it)),
                           map_location=model.device, weights_only=False)
        model.policy.load_state_dict(saved["state_dict"])
        env.load_rms(os.path.join(run_dir, "RMS", "iter_{0:05d}.npz".format(start_it)))
        # Continue the step count, so the learning-rate schedule picks up where
        # it stopped and TensorBoard continues the last tb_<n> curve
        model.num_timesteps = start_it * args.n_steps * args.num_envs
    callback = RaceCallback(run_dir, args.save_every, start_it)
    try:
        model.learn(total_timesteps=max(args.iterations - start_it, 1) * args.n_steps * args.num_envs,
                    callback=callback, tb_log_name="tb", reset_num_timesteps=not start_it)
    finally:
        callback.save(callback.iteration)
        print("Saved", run_dir, "iteration", callback.iteration)


if __name__ == "__main__":
    main()
