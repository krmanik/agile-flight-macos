#!/usr/bin/env python3
"""FlashSAC (Kim et al., arXiv 2604.04539) on the gate-racing task.

The same MDP as the PPO runs of acmpc/train_race.py (acmpc/race_env.py: observation, CTBR
action, reward eq. 10, 50 Hz, 1000 steps), trained off-policy with FlashSAC
(github.com/Holiday-Robot/FlashSAC, pixi run setup-flashsac):

  python flashsac/flashsac_race.py                                   # dodgeros dynamics, splits
  python flashsac/flashsac_race.py --seed 1 --run_name SAC_splits_s1
  python flashsac/flashsac_race.py --resume acmpc/saved/race/SAC_splits_dodgeros_1

Differences to the PPO setup, all needed for off-policy learning:
- Observations use a fixed normalization (OBS_SCALE) instead of running
  statistics, so transitions in the replay buffer keep their meaning. The
  networks' input batch norm does the rest.
- Asymmetric actor-critic (unless --no_privileged): the actor sees the paper's
  36 observations, the critic also the full drone state, rotor speeds, command
  delay and the last commands.
- Crashes and finishes are terminal, timeouts are truncations (bootstrapped).

Hyperparameters are FlashSAC's GPU-simulator settings (paper Table 9) except
gamma, which is the PPO runs' 0.98. Each run goes to
acmpc/saved/race/SAC_<track>_<dynamics>_<n>/: config.json, RMS/fixed.npz,
Policy/step_<env steps>.pt (actor only, for evaluation), Agent/ (latest full
agent, for --resume), eval.json and TensorBoard logs. acmpc/eval_race.py and
acmpc/run_race_ros.py (race-sim) take FlashSAC runs like PPO ones, as RUN_DIR,
RUN_DIR:ENV_STEPS or RUN_DIR:best. Train on a GPU (flashsac/colab_flashsac_race.ipynb).
"""
import argparse
import glob
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

HERE = os.path.dirname(os.path.abspath(__file__))
ACMPC = os.path.join(HERE, "..", "acmpc")
sys.path.insert(0, ACMPC)
sys.path.insert(0, os.environ.get("FLASHSAC_PATH", os.path.join(HERE, "..", "src", "FlashSAC")))
import race_env as _re  # noqa: E402

# flash_rl imports jax and omegaconf, but the parts used here don't need them:
# jax.numpy.ndarray only appears in a type alias, OmegaConf only in create_agent
try:
    import jax.numpy  # noqa: F401
except ImportError:
    import types
    _jnp = types.ModuleType("jax.numpy")
    _jnp.ndarray = np.ndarray
    _jax = types.ModuleType("jax")
    _jax.numpy = _jnp
    sys.modules.update({"jax": _jax, "jax.numpy": _jnp})
try:
    import omegaconf  # noqa: F401
except ImportError:
    import types
    sys.modules["omegaconf"] = types.ModuleType("omegaconf")
    sys.modules["omegaconf"].OmegaConf = None

N_ACTOR = _re.OBS_DIM
# v (3), R (9), next gate corners (12), corners of the gate after (12)
OBS_SCALE = np.concatenate([np.full(3, 10.0), np.ones(9), np.full(24, 10.0)])

# FlashSAC paper Table 9 (GPU simulators); configs/agent/flashSAC.yaml
AGENT = dict(
    normalize_reward=True, normalized_G_max=5.0,
    buffer_min_length=100_000, sample_batch_size=2048,
    learning_rate_init=3e-4, learning_rate_peak=3e-4, learning_rate_end=1.5e-4,
    actor_num_blocks=2, actor_hidden_dim=128, actor_bc_alpha=0.0,
    actor_noise_zeta_mu=2.0, actor_noise_zeta_max=16, actor_update_period=2,
    critic_num_blocks=2, critic_hidden_dim=256, critic_num_bins=101, critic_min_v=-5.0, critic_max_v=5.0,
    critic_target_update_tau=0.01,
    temp_initial_value=0.01, temp_target_sigma=0.15,
    n_step=1, compile_mode="auto", load_optimizer=True, load_reward_normalizer=True,
)


class PrivilegedEnv(_re.RaceEnv):
    """Observation: the actor's 36 (normalized) values, then the critic's privileged ones."""

    def privileged(self):
        cmds = self.cmds[:, 1:].copy()                      # the two newest commands
        cmds[..., 0] = cmds[..., 0] / _re.C_HALF - 1.0
        cmds[..., 1:] /= _re.OMEGA_MAX
        return np.concatenate([
            self.p / 10.0, self.q, self.v / 10.0, self.w / 5.0, self.motors / 1000.0,
            self.delay[:, None] / self.cfg["ctrl_dt"], cmds.reshape(-1, 8),
        ], 1)

    def _obs(self, update):
        # RaceEnv calls this before resetting finished envs, so terminal
        # observations carry the privileged state of the terminal step
        obs = super()._obs(update)[:, :N_ACTOR]
        return np.concatenate([obs, self.privileged()], 1).astype(np.float32)


def write_fixed_rms(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mean=np.zeros(N_ACTOR), var=OBS_SCALE ** 2, count=1.0)


class RaceVectorEnv(gym.vector.VectorEnv):
    """RaceEnv (an SB3 VecEnv) behind the gymnasium VectorEnv API FlashSAC's training loop uses."""

    def __init__(self, env, rms_path):
        self.env = env
        env.load_rms(rms_path)
        env.training = False   # fixed normalization
        self.num_envs = env.num_envs
        self.privileged = isinstance(env, PrivilegedEnv)
        n_obs = N_ACTOR + (env.privileged().shape[1] if self.privileged else 0)
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.metadata = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}

    def _view(self, obs):
        return obs if self.privileged else obs[:, :N_ACTOR]

    def reset(self, *, seed=None, options=None):
        return self._view(self.env.reset()), {"actor_observation_size": [N_ACTOR]}

    def step(self, actions):
        self.env.step_async(actions)
        obs, rew, dones, infos = self.env.step_wait()
        trunc = np.array([i.get("TimeLimit.truncated", False) for i in infos]) & dones
        term = dones & ~trunc
        final = obs.copy()
        done_idx = np.flatnonzero(dones)
        for i in done_idx:
            final[i] = infos[i]["terminal_observation"]
        info = {"final_obs": self._view(final), "episodes": [infos[i] for i in done_idx]}
        return self._view(obs), rew, term, trunc, info


class FlashSACPolicy:
    """FlashSAC actor with SB3's predict(), so eval_race.py and run_race_ros.py can fly it."""

    def __init__(self, actor, device):
        self.actor = actor
        self.device = device

    def predict(self, obs, deterministic=True):
        x = torch.as_tensor(np.asarray(obs)[:, :N_ACTOR], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean, std = self.actor.get_mean_and_std(x, training=False)
            a = mean if deterministic else mean + std * torch.randn_like(mean)
        return torch.tanh(a).float().cpu().numpy(), None


def actor_state(agent):
    net = agent._actor.network
    return getattr(net, "_orig_mod", net).state_dict()


def load_run(spec, device="cpu"):
    """'RUN_DIR', 'RUN_DIR:STEP' or 'RUN_DIR:best' -> (policy, rms_path, config), as eval_race.load_run.

    best: the checkpoint with the highest success rate in the training-time
    evaluations (eval.json), then the shortest lap time. Its deterministic
    policy can collapse between checkpoints, so the latest is not always good.
    """
    from flash_rl.agents.flashSAC.network import FlashSACActor
    run_dir, _, step = spec.partition(":")
    run_dir = run_dir.rstrip("/")
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    steps = sorted(int(os.path.basename(p)[5:-3]) for p in glob.glob(os.path.join(run_dir, "Policy", "step_*.pt")))
    if step == "best":
        with open(os.path.join(run_dir, "eval.json")) as f:
            evals = [e for e in json.load(f) if e["env_steps"] in steps]
        step = max(evals, key=lambda e: (e["success_rate"], -np.nan_to_num(e["lap_time"], nan=1e9)))["env_steps"]
    step = int(step) if step else steps[-1]
    ckpt = torch.load(os.path.join(run_dir, "Policy", "step_{0:010d}.pt".format(step)), map_location=device)
    a = cfg["agent"]
    actor = FlashSACActor(a["actor_num_blocks"], N_ACTOR, a["actor_hidden_dim"], 4).to(device)
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()
    cfg["iteration"] = step
    cfg["name"] = "FlashSAC ({0} {1:.1f}M steps)".format(os.path.basename(run_dir), step / 1e6)
    return FlashSACPolicy(actor, device), os.path.join(run_dir, "RMS", "fixed.npz"), cfg


def evaluate(policy, rms_path, cfg, episodes):
    import eval_race
    return eval_race.evaluate(policy, rms_path, cfg, episodes, dynamics=cfg["env"]["dynamics"])[0]


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dynamics", default="dodgeros", choices=["dodgeros", "simple"])
    p.add_argument("--track", default="splits")
    p.add_argument("--laps", type=int, default=2)
    p.add_argument("--no_privileged", action="store_true", help="symmetric actor-critic")
    p.add_argument("--env_steps", type=float, default=50e6, help="total env steps (PPO: 2000 iterations = 50M)")
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--utd", type=float, default=2.0, help="gradient updates per vectorized env step")
    p.add_argument("--buffer", type=float, default=10e6, help="replay buffer size (transitions)")
    p.add_argument("--buffer_min", type=float, default=1e5)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--eval_every", type=float, default=2e6, help="env steps between evaluations / checkpoints")
    p.add_argument("--eval_episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--log_dir", default=os.path.join(ACMPC, "saved", "race"))
    p.add_argument("--resume", help="continue RUN_DIR from its Agent/ checkpoint (the replay buffer starts empty)")
    p.add_argument("--run_name", help="run directory name in --log_dir; an unfinished run of that name is resumed, "
                                      "a finished one is left alone")
    return p


def main():
    args = parser().parse_args()
    from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig
    from torch.utils.tensorboard import SummaryWriter

    if args.run_name and os.path.exists(os.path.join(args.log_dir, args.run_name, "progress.json")):
        args.resume = os.path.join(args.log_dir, args.run_name)
        with open(os.path.join(args.resume, "config.json")) as f:
            total = json.load(f)["num_interaction_steps"]
        with open(os.path.join(args.resume, "progress.json")) as f:
            if json.load(f)["interaction_step"] >= total:
                print(args.resume, "is finished")
                return

    if args.resume:
        run_dir = args.resume.rstrip("/")
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
    else:
        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        n_inter = int(args.env_steps // args.num_envs)
        gpu = args.device.startswith("cuda")
        agent_cfg = dict(
            AGENT, seed=args.seed, device_type=args.device, buffer_device_type=args.device,
            buffer_max_length=int(args.buffer), buffer_min_length=int(args.buffer_min),
            sample_batch_size=args.batch, asymmetric_observation=not args.no_privileged,
            learning_rate_warmup_rate=0.0, learning_rate_warmup_step=0,
            learning_rate_decay_rate=1.0, learning_rate_decay_step=int(n_inter * args.utd),
            temp_target_entropy=None, gamma=args.gamma,
            use_compile=gpu and not args.no_compile, use_amp=gpu,
        )
        env_cfg = dict(_re.DEFAULTS, track=args.track, laps=args.laps, dynamics=args.dynamics)
        cfg = dict(algo="flashsac", policy="flashsac", track=args.track, laps=args.laps,
                   privileged=not args.no_privileged, num_envs=args.num_envs, utd=args.utd,
                   num_interaction_steps=n_inter, eval_every=int(args.eval_every),
                   eval_episodes=args.eval_episodes, seed=args.seed,
                   device=torch.cuda.get_device_name() if gpu else "cpu", agent=agent_cfg, env=env_cfg)
        os.makedirs(args.log_dir, exist_ok=True)
        import train_race
        if args.run_name:
            run_dir = os.path.join(args.log_dir, args.run_name)
        else:
            run_dir = train_race.next_run_dir(args.log_dir, "SAC_{0}_{1}".format(os.path.basename(args.track),
                                                                                 args.dynamics))
        os.makedirs(run_dir, exist_ok=bool(args.run_name))   # a named run killed before its first checkpoint restarts
    rms_path = os.path.join(run_dir, "RMS", "fixed.npz")
    if not cfg["agent"]["device_type"].startswith("cuda"):
        # flash_rl torch.compiles some helpers regardless of use_compile, and
        # inductor's CPU backend fails to build them on macOS
        torch._dynamo.config.disable = True

    env_cls = PrivilegedEnv if cfg["privileged"] else _re.RaceEnv
    raw = env_cls(cfg["num_envs"], cfg["seed"], **cfg["env"])
    if not args.resume:
        cfg["track_path"] = raw.track.path
        write_fixed_rms(rms_path)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
    env = RaceVectorEnv(raw, rms_path)
    print("Run directory:", run_dir)

    torch.manual_seed(cfg["seed"])
    obs, env_info = env.reset()
    agent = FlashSACAgent(env.observation_space, env.action_space, env_info, FlashSACConfig(**cfg["agent"]))
    start, evals = 1, []
    if args.resume:
        agent.load(os.path.join(run_dir, "Agent"))
        with open(os.path.join(run_dir, "progress.json")) as f:
            start = json.load(f)["interaction_step"] + 1
        if os.path.exists(os.path.join(run_dir, "eval.json")):
            with open(os.path.join(run_dir, "eval.json")) as f:
                evals = json.load(f)

    writer = SummaryWriter(os.path.join(run_dir, "tb"))
    n = cfg["num_envs"]
    eval_every = max(cfg["eval_every"] // n, 1)
    log_every = max(eval_every // 10, 1)
    episodes, updates = [], []
    transition, update_counter = None, 0.0
    t0, t_env, t_upd = time.time(), 0.0, 0.0
    wall0 = evals[-1]["wall_s"] if evals else 0.0
    for it in range(start, cfg["num_interaction_steps"] + 1):
        t = time.time()
        if agent.can_start_training() and transition is not None:
            actions = agent.sample_actions(it, prev_transition=transition, training=True)
        else:
            actions = env.action_space.sample()
        next_obs, rew, term, trunc, info = env.step(np.asarray(actions))
        t_env += time.time() - t
        episodes += info["episodes"]
        transition = dict(observation=obs, action=actions, reward=rew, terminated=term, truncated=trunc,
                          next_observation=info["final_obs"])
        agent.process_transition(transition)
        transition["next_observation"] = next_obs
        obs = next_obs

        if agent.can_start_training():
            t = time.time()
            update_counter += cfg["utd"]
            while update_counter >= 1:
                updates.append(agent.update())
                update_counter -= 1
            t_upd += time.time() - t

        steps = it * n
        if it % log_every == 0:
            if episodes:
                for key in ("success", "crash", "gates", "speed"):
                    writer.add_scalar("race/" + key, np.mean([e[key] for e in episodes]), steps)
                writer.add_scalar("race/return", np.mean([e["episode"]["r"] for e in episodes]), steps)
                episodes = []
            for key in (updates[-1] if updates else {}):
                writer.add_scalar("train/" + key, np.mean([u[key] for u in updates if key in u]), steps)
            updates = []
            writer.add_scalar("time/wall_s", time.time() - t0, steps)
            writer.add_scalar("time/env_frac", t_env / max(time.time() - t0, 1e-9), steps)
        if it % eval_every == 0 or it == cfg["num_interaction_steps"]:
            os.makedirs(os.path.join(run_dir, "Policy"), exist_ok=True)
            torch.save({"state_dict": actor_state(agent)},
                       os.path.join(run_dir, "Policy", "step_{0:010d}.pt".format(steps)))
            policy, _, _ = load_run("{0}:{1}".format(run_dir, steps), device=agent._device)
            m = evaluate(policy, rms_path, cfg, cfg["eval_episodes"])
            m = {k: float(v) for k, v in m.items()}
            evals.append(dict(m, env_steps=steps, wall_s=wall0 + time.time() - t0))
            for k, v in m.items():
                writer.add_scalar("eval/" + k, v, steps)
            with open(os.path.join(run_dir, "eval.json"), "w") as f:
                json.dump(evals, f, indent=1)
            agent.save(os.path.join(run_dir, "Agent"))
            with open(os.path.join(run_dir, "progress.json"), "w") as f:
                json.dump({"interaction_step": it}, f)
            el = time.time() - t0
            print("{0:6.1f}M steps  {1:6.0f} s (env {2:.0%}, updates {3:.0%})  success {4:.0%}  crash {5:.0%}  "
                  "gates {6:.1f}  lap {7:.2f} s".format(steps / 1e6, el, t_env / el, t_upd / el, m["success_rate"],
                                                         m["crash_rate"], m["gates"], m["lap_time"]), flush=True)
    writer.close()


if __name__ == "__main__":
    main()
