#!/usr/bin/env python3
"""Benchmark racing policies (PPO runs and FlashSAC runs) under model mismatch.

Each policy flies `--episodes` starts per condition with deterministic and with
sampled actions (eval_race.evaluate's rollouts):

  nominal        the training sim (dodgeros), starts in the 0.5 m cube
  random start   halfway to a random gate, up to 5 m/s (training's other half)
  mass x0.8/1.2/1.4, drag 0.1/0.3 1/s, delay 40 ms, wind 2 N sideways

  python flashsac/benchmark.py "AC-MLP=acmpc/saved/race/MLP_splits_1" \\
      "FlashSAC=acmpc/saved/race/SAC_splits_dodgeros_1" --out benchmark
  python flashsac/benchmark.py --checkpoints acmpc/saved/race/SAC_splits_dodgeros_1   # every checkpoint, nominal + random start

Writes <out>/benchmark.json (merged with earlier calls, so policies can run in
parallel processes), <out>/benchmark.md and plots.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "acmpc"))
import eval_race  # noqa: E402

CONDITIONS = [
    ("nominal", {}),
    ("random start", dict(random_start_prob=1.0)),
    ("mass x0.8", dict(mass_scale=0.8)),
    ("mass x1.2", dict(mass_scale=1.2)),
    ("mass x1.4", dict(mass_scale=1.4)),
    ("drag 0.1", dict(drag=0.1)),
    ("drag 0.3", dict(drag=0.3)),
    ("delay 40 ms", dict(cmd_delay=(0.04, 0.04))),
    ("wind 2 N", dict(wind=(0.0, 2.0, 0.0))),
]


class Sampled:
    """predict() with sampled actions, whatever the caller asks for."""

    def __init__(self, policy):
        self.policy = policy

    def predict(self, obs, deterministic=True):
        return self.policy.predict(obs, deterministic=False)


load = eval_race.load_run


def evaluate(policy, rms, cfg, episodes, seed, kw):
    kw = dict(kw)
    kw.setdefault("dynamics", "dodgeros")
    m, trajs, speeds, success, track = eval_race.evaluate(policy, rms, cfg, episodes, seed=seed, **kw)
    return {k: float(v) for k, v in m.items()}, (trajs, speeds, success, track)


def inference_ms(policy, cfg, n=200):
    env = eval_race.RaceEnv(num_envs=1, seed=0, track=cfg["track_path"], dynamics="dodgeros")
    obs = env.reset()
    policy.predict(obs, deterministic=True)
    t = time.time()
    for _ in range(n):
        policy.predict(obs, deterministic=True)
    return (time.time() - t) / n * 1e3


def merge(path, update):
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    for k, v in update.items():
        data.setdefault(k, {}).update(v)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    return data


def write_tables(data, out):
    lines = []
    for key, title in (("conditions", "Success rate per condition (deterministic / sampled actions)"),):
        names = list(data.get(key, {}))
        if not names:
            continue
        conds = [c for c, _ in CONDITIONS]
        lines += ["### " + title, "", "| condition | " + " | ".join(names) + " |",
                  "|---" * (len(names) + 1) + "|"]
        for c in conds:
            cells = []
            for n in names:
                r = data[key][n].get(c)
                cells.append("-" if r is None else "{0:.0%} / {1:.0%}".format(
                    r["deterministic"]["success_rate"], r["sampled"]["success_rate"]))
            lines.append("| {0} | {1} |".format(c, " | ".join(cells)))
        lines += ["", "### Lap time [s] and max speed [m/s] (deterministic, successful episodes)", "",
                  "| condition | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for c in conds:
            cells = []
            for n in names:
                r = data[key][n].get(c)
                d = r["deterministic"] if r else None
                cells.append("-" if not d or np.isnan(d["lap_time"]) else "{0:.2f} / {1:.1f}".format(
                    d["lap_time"], d["max_speed"]))
            lines.append("| {0} | {1} |".format(c, " | ".join(cells)))
    if data.get("inference"):
        lines += ["", "### Policy inference (1 env, CPU) [ms]", "",
                  "| policy | ms |", "|---|---|"]
        lines += ["| {0} | {1:.2f} |".format(n, v["ms"]) for n, v in data["inference"].items()]
    with open(os.path.join(out, "benchmark.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_conditions(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(data["conditions"])
    conds = [c for c, _ in CONDITIONS]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    w = 0.8 / len(names)
    x = np.arange(len(conds))
    for ax, mode in zip(axes, ("deterministic", "sampled")):
        for k, n in enumerate(names):
            y = [data["conditions"][n].get(c, {}).get(mode, {}).get("success_rate", np.nan) for c in conds]
            ax.bar(x + (k - (len(names) - 1) / 2) * w, y, w, label=n, color="C%d" % k)
        ax.set_ylabel("success rate")
        ax.set_title(mode + " actions", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, ncol=len(names))
    axes[1].set_xticks(x, conds, rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "robustness.png"), dpi=130)
    print("Saved", os.path.join(out, "robustness.png"))


def plot_checkpoints(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, rows in data["checkpoints"].items():
        steps = sorted(rows, key=int)
        x = [int(s) / 1e6 for s in steps]
        for ax, cond in zip(axes, ("nominal", "random start")):
            ax.plot(x, [rows[s][cond]["success_rate"] for s in steps], "-o", ms=3, label=name + " success")
            lap = [rows[s][cond]["lap_time"] for s in steps]
            ax2 = ax.twinx()
            ax2.plot(x, lap, ":", color="gray")
            ax2.set_ylabel("lap time [s] (dotted)")
            ax.set_title(cond + " (deterministic)", fontsize=10)
            ax.set_xlabel("env steps [M]")
            ax.set_ylabel("success rate")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "checkpoints.png"), dpi=130)
    print("Saved", os.path.join(out, "checkpoints.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*", help="LABEL=RUN_DIR[:ITER]")
    p.add_argument("--checkpoints", nargs="*", default=[], help="FlashSAC runs: evaluate every checkpoint")
    p.add_argument("--conditions", nargs="+", help="subset of the condition names")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--plot_traj", action="store_true", help="trajectory plot of the nominal condition")
    p.add_argument("--report", action="store_true", help="only rewrite tables and plots")
    p.add_argument("--out", default=os.path.join(HERE, "..", "acmpc", "saved", "benchmark"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "benchmark.json")

    conds = [(c, kw) for c, kw in CONDITIONS if not args.conditions or c in args.conditions]
    traj = []
    for spec in ([] if args.report else args.runs):
        label, _, run = spec.rpartition("=")
        policy, rms, cfg = load(run)
        label = label or cfg["name"]
        res = {}
        for c, kw in conds:
            t = time.time()
            det, tr = evaluate(policy, rms, cfg, args.episodes, args.seed, kw)
            smp, _ = evaluate(Sampled(policy), rms, cfg, args.episodes, args.seed, kw)
            res[c] = {"deterministic": det, "sampled": smp}
            print("{0:12s} {1:14s} det {2:4.0%}  sampled {3:4.0%}  lap {4:.2f}  ({5:.0f} s)".format(
                label, c, det["success_rate"], smp["success_rate"], det["lap_time"], time.time() - t), flush=True)
            if args.plot_traj and c == "nominal":
                traj.append(("{0}, {1}".format(label, c), det) + tr)
        merge(path, {"conditions": {label: res},
                     "inference": {label: {"ms": inference_ms(policy, cfg)}}})
    for run in ([] if args.report else args.checkpoints):
        _, rms, cfg = load(run)
        rows = {}
        for f in sorted(glob.glob(os.path.join(run, "Policy", "step_*.pt"))):
            step = int(os.path.basename(f)[5:-3])
            policy = load("{0}:{1}".format(run, step))[0]
            rows[str(step)] = {c: evaluate(policy, rms, cfg, args.episodes, args.seed + 1, kw)[0]
                               for c, kw in CONDITIONS[:2]}
            print("{0:.1f}M".format(step / 1e6), {c: "{0:.0%}".format(r["success_rate"]) for c, r in rows[str(step)].items()},
                  flush=True)
        merge(path, {"checkpoints": {os.path.basename(run.rstrip("/")): rows}})
    if traj:
        eval_race.plot([(n, m, t, s, ok, tr) for n, m, t, s, ok, tr in traj],
                       os.path.join(args.out, "trajectories_%d.png" % os.getpid()))

    with open(path) as f:
        data = json.load(f)
    write_tables(data, args.out)
    if data.get("conditions"):
        plot_conditions(data, args.out)
    if data.get("checkpoints"):
        plot_checkpoints(data, args.out)


if __name__ == "__main__":
    main()
