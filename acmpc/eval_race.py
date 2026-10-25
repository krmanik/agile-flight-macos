#!/usr/bin/env python3
"""Evaluate and compare racing policies (AC-MPC vs AC-MLP) from train_race.py.

Like the paper (Fig. 12, Table I, Fig. 7): fly each policy from N start
positions spread over a 0.5 m cube, report success rate, lap time and speed,
and optionally sweep a model mismatch the policies never saw in training.

  python acmpc/eval_race.py acmpc/saved/race/MPC_splits_1 acmpc/saved/race/MLP_splits_1
  python acmpc/eval_race.py RUN_A RUN_B --plot race.png          # trajectories + gates
  python acmpc/eval_race.py RUN_A RUN_B --sweep mass_scale 0.8 1.0 1.2 1.4
  python acmpc/eval_race.py RUN_A --wind 0 0 -5                  # constant force (N)

A run is a run directory (latest checkpoint) or RUN_DIR:ITER.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_acmpc = os.environ.get("ACMPC_PATH", os.path.join(HERE, "..", "src", "acmpc_public"))
for sub in ("training_modules", "diff_mpc_drones", "mpc.pytorch"):
    sys.path.insert(0, os.path.join(_acmpc, sub))

from acmpc_policy import MlpMpcPolicy, MlpOnlyPolicy  # noqa: E402
from race_env import RaceEnv  # noqa: E402

POLICIES = {"mpc": MlpMpcPolicy, "mlp": MlpOnlyPolicy}
LABELS = {"mpc": "AC-MPC", "mlp": "AC-MLP"}


def load_run(spec):
    """'RUN_DIR' or 'RUN_DIR:ITER' -> (policy, rms_path, config)."""
    run_dir, _, it = spec.partition(":")
    run_dir = run_dir.rstrip("/")
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    if it:
        it = int(it)
    else:
        it = max(int(os.path.basename(p)[5:10]) for p in glob.glob(os.path.join(run_dir, "Policy", "iter_*.pth")))
    os.environ["ACMPC_T"] = str(cfg["T"])
    saved = torch.load(os.path.join(run_dir, "Policy", "iter_{0:05d}.pth".format(it)),
                       map_location="cpu", weights_only=False)
    policy = POLICIES[cfg["policy"]](**saved["data"])
    policy.load_state_dict(saved["state_dict"])
    policy.set_training_mode(False)
    cfg["iteration"] = it
    cfg["name"] = "{0} ({1} it {2})".format(LABELS[cfg["policy"]], os.path.basename(run_dir), it)
    return policy, os.path.join(run_dir, "RMS", "iter_{0:05d}.npz".format(it)), cfg


def evaluate(policy, rms_path, cfg, episodes=64, seed=1, **env_kwargs):
    """Fly one episode per environment; returns metrics and trajectories."""
    kw = dict(track=cfg["track_path"], laps=cfg["laps"], random_start_prob=0.0)
    kw.update(env_kwargs)
    env = RaceEnv(num_envs=episodes, seed=seed, **kw)
    env.load_rms(rms_path)
    env.training = False
    obs = env.reset()
    alive = np.ones(episodes, dtype=bool)
    trajs = [[env.p[i].copy()] for i in range(episodes)]
    speeds = [[0.0] for _ in range(episodes)]
    results = [None] * episodes
    while alive.any():
        action, _ = policy.predict(obs, deterministic=True)
        p_before, v_before = env.p.copy(), env.v.copy()
        obs, _, dones, infos = env.step(action)
        for i in np.flatnonzero(alive):
            # After a done the env has already reset, so keep the pre-reset state
            trajs[i].append(p_before[i] if dones[i] else env.p[i].copy())
            speeds[i].append(np.linalg.norm(v_before[i] if dones[i] else env.v[i]))
            if dones[i]:
                results[i] = infos[i]
                alive[i] = False
    success = np.array([r["success"] for r in results])
    times = np.array([r["episode"]["t"] for r in results])
    m = dict(
        success_rate=success.mean(),
        crash_rate=np.mean([r["crash"] for r in results]),
        gates=np.mean([r["gates"] for r in results]),
        lap_time=(times[success] / kw["laps"]).mean() if success.any() else np.nan,
        avg_speed=np.mean([r["speed"] for r, s in zip(results, success) if s]) if success.any() else np.nan,
        max_speed=max(max(s) for s in speeds),
    )
    return m, [np.array(t) for t in trajs], [np.array(s) for s in speeds], success, env.track


def print_table(rows, extra_cols=()):
    head = list(extra_cols) + ["policy", "success", "crash", "gates", "lap time [s]", "avg v [m/s]", "max v [m/s]"]
    print(" | ".join(head))
    print(" | ".join("---" for _ in head))
    for extra, name, m in rows:
        print(" | ".join(list(extra) + [
            name, "{0:.0%}".format(m["success_rate"]), "{0:.0%}".format(m["crash_rate"]),
            "{0:.1f}".format(m["gates"]), "{0:.2f}".format(m["lap_time"]),
            "{0:.2f}".format(m["avg_speed"]), "{0:.2f}".format(m["max_speed"])]))


def draw_gates(ax, track, view):
    """Gates as squares: view 'xy' (top), 'xz' (side) or '3d'."""
    for i, c in enumerate(track.corners):
        loop = np.vstack([c, c[:1]])
        if view == "3d":
            ax.plot(*loop.T, color="k", lw=2)
            ax.text(*track.centers[i] + [0, 0, 1.0], str(i + 1), fontsize=8)
        else:
            a, b = (0, 1) if view == "xy" else (0, 2)
            ax.plot(loop[:, a], loop[:, b], color="k", lw=2)
            ax.annotate(str(i + 1), track.centers[i][[a, b]] + 0.6, fontsize=8)


def plot(evals, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    n = len(evals)
    fig = plt.figure(figsize=(6 * n, 11))
    vmax = max(max(s.max() for s in e[2]) for e in evals)
    for k, (name, m, trajs, speeds, success, track) in enumerate(evals):
        ax = fig.add_subplot(2, n, k + 1)
        ax3 = fig.add_subplot(2, n, n + k + 1, projection="3d")
        for t, s, ok in zip(trajs, speeds, success):
            if ok:
                seg = np.stack([t[:-1, :2], t[1:, :2]], 1)
                lc = LineCollection(seg, cmap="viridis", norm=plt.Normalize(0, vmax), lw=1)
                lc.set_array(s[1:])
                ax.add_collection(lc)
                ax3.plot(*t.T, lw=0.6, color=plt.cm.viridis(s.mean() / vmax))
            else:
                ax.plot(t[:, 0], t[:, 1], color="r", lw=0.8, alpha=0.7)
                ax3.plot(*t.T, color="r", lw=0.6, alpha=0.7)
        draw_gates(ax, track, "xy")
        draw_gates(ax3, track, "3d")
        ax.set_title("{0}\nsuccess {1:.0%}, lap {2:.2f} s, v_max {3:.1f} m/s".format(
            name, m["success_rate"], m["lap_time"], m["max_speed"]), fontsize=9)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax3.set_box_aspect(np.ptp(track.bounds, 0) * [1, 1, 2])
        if k == n - 1:
            fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, vmax), "viridis"), ax=ax, label="speed [m/s] (crashes red)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("Saved", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="RUN_DIR or RUN_DIR:ITER")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--track", help="evaluate on another track than the training one")
    p.add_argument("--laps", type=int)
    p.add_argument("--mass_scale", type=float, default=1.0)
    p.add_argument("--omega_max_scale", type=float, default=1.0)
    p.add_argument("--drag", type=float, default=0.0)
    p.add_argument("--wind", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--sweep", nargs="+", metavar=("PARAM", "VALUES"),
                   help="e.g. --sweep mass_scale 0.8 1.0 1.2 (or omega_max_scale, drag)")
    p.add_argument("--plot", help="save trajectory plot (png)")
    args = p.parse_args()

    env_kwargs = dict(mass_scale=args.mass_scale, omega_max_scale=args.omega_max_scale,
                      drag=args.drag, wind=tuple(args.wind))
    if args.track:
        env_kwargs["track"] = args.track
    if args.laps:
        env_kwargs["laps"] = args.laps
    runs = [load_run(r) for r in args.runs]

    if args.sweep:
        param, values = args.sweep[0], [float(v) for v in args.sweep[1:]]
        rows = []
        for v in values:
            for policy, rms, cfg in runs:
                m = evaluate(policy, rms, cfg, args.episodes, **dict(env_kwargs, **{param: v}))[0]
                rows.append(([str(v)], cfg["name"], m))
        print_table(rows, [param])
        return

    evals = []
    for policy, rms, cfg in runs:
        m, trajs, speeds, success, track = evaluate(policy, rms, cfg, args.episodes, **env_kwargs)
        evals.append((cfg["name"], m, trajs, speeds, success, track))
    print_table([((), e[0], e[1]) for e in evals])
    if args.plot:
        plot(evals, args.plot)


if __name__ == "__main__":
    main()
