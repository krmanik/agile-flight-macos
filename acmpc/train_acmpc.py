#!/usr/bin/env python3
"""Train ACMPC (MLP -> differentiable MPC actor, MLP critic) with PPO on
flightgym's VisionEnv_v1 (state-based obstacle avoidance, CTBR actions).

The paper's racing environment is not public; this uses the DodgeDrone task.

  python acmpc/train_acmpc.py                        # ACMPC, T=5
  python acmpc/train_acmpc.py --policy mlp           # model-free baseline
  python acmpc/train_acmpc.py --train 0 --trial 1 --iter 50
"""
import argparse
import os
import sys

import torch
from ruamel.yaml import YAML

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acmpc_env import acmpc_config, make_vision_env  # noqa: E402
from acmpc_policy import MlpMpcPolicy, MlpOnlyPolicy  # noqa: E402
from rpg_baselines.torch.common.ppo import PPO  # noqa: E402
from rpg_baselines.torch.common.util import test_policy  # noqa: E402

POLICIES = {"mpc": MlpMpcPolicy, "mlp": MlpOnlyPolicy}


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=POLICIES, default="mpc")
    p.add_argument("--T", type=int, default=5, help="MPC horizon (ACMPC_T)")
    p.add_argument("--train", type=int, default=1)
    p.add_argument("--num_envs", type=int, default=100)
    p.add_argument("--n_steps", type=int, default=250)
    p.add_argument("--timesteps", type=float, default=5e7)
    p.add_argument("--log_interval", type=int, nargs=2, default=(10, 50),
                   help="iterations between log lines / checkpoints + eval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trial", type=int, default=1, help="run to evaluate: saved/<POLICY>_<trial>")
    p.add_argument("--iter", type=int, default=50, help="checkpoint to evaluate")
    p.add_argument("--render", type=int, default=0, help="evaluate with the Unity renderer")
    p.add_argument("--log_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved"))
    return p


def main():
    args = parser().parse_args()
    os.environ["ACMPC_T"] = str(args.T)
    torch.manual_seed(args.seed)

    cfg = acmpc_config(YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml")))
    if args.render:
        cfg["unity"]["render"] = "yes"
    eval_env = make_vision_env(cfg, num_envs=1)

    if args.train:
        env = make_vision_env(cfg, num_envs=args.num_envs)
        env.seed(args.seed)
        os.makedirs(args.log_dir, exist_ok=True)
        model = PPO(
            policy=POLICIES[args.policy],
            policy_kwargs=dict(log_std_init=-0.5),
            env=env,
            eval_env=eval_env,
            # The MPC (or the baseline's Linear head) already outputs normalized actions
            use_tanh_act=False,
            n_steps=args.n_steps,
            batch_size=args.n_steps * args.num_envs,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            clip_range=0.2,
            env_cfg=cfg,
            tensorboard_log=args.log_dir,
            seed=args.seed,
            verbose=1,
        )
        model.learn(
            total_timesteps=int(args.timesteps),
            log_interval=tuple(args.log_interval),
            tb_log_name=args.policy.upper(),
        )
        return

    run = os.path.join(args.log_dir, "{0}_{1}".format(args.policy.upper(), args.trial))
    # Own checkpoint: SB3's policy.save() pickles the constructor args
    saved = torch.load(os.path.join(run, "Policy/iter_{0:05d}.pth".format(args.iter)),
                       map_location="cpu", weights_only=False)
    policy = POLICIES[args.policy](**saved["data"])
    policy.load_state_dict(saved["state_dict"])
    eval_env.load_rms(os.path.join(run, "RMS/iter_{0:05d}.npz".format(args.iter)))
    proc = None
    if args.render:
        import subprocess
        proc = subprocess.Popen(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64")
    try:
        test_policy(eval_env, policy, render=args.render)
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
