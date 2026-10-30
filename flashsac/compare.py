#!/usr/bin/env python3
"""Learning curves of PPO (acmpc/train_race.py) and FlashSAC (flashsac_race.py) runs.

Success rate of the training episodes against env steps and wall-clock time,
read from the runs' TensorBoard logs, and the steps / time each run needed to
reach a success rate:

  python flashsac/compare.py acmpc/saved/race/MLP_splits_1 acmpc/saved/race/SAC_splits_dodgeros_1 \\
      --out ppo_vs_flashsac.png
  python flashsac/compare.py "PPO=RUN_A" "PPO=RUN_B" "FlashSAC=RUN_C" ...   # seeds: mean, min-max band

Seed groups average the deterministic evaluation curves as well.

PPO's training episodes sample actions and half of them start before a random
gate, FlashSAC's sample actions with repeated noise; both are noisier than
the deterministic evaluation (FlashSAC's eval.json is drawn too, dashed).
Wall-clock times depend on the GPU each run had (config.json "device").
"""
import argparse
import glob
import json
import os

import numpy as np


def scalars(run_dir, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    steps, values, walls = [], [], []
    for d in sorted(set(os.path.dirname(p) for p in glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"),
                                                              recursive=True))):
        acc = EventAccumulator(d, size_guidance={"scalars": 0})
        acc.Reload()
        if tag in acc.Tags()["scalars"]:
            for e in acc.Scalars(tag):
                steps.append(e.step)
                values.append(e.value)
                walls.append(e.wall_time)
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(values)[order], np.array(walls)[order]


def load(run_dir):
    run_dir = run_dir.rstrip("/")
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    sac = cfg.get("algo") == "flashsac"
    steps, succ, wall = scalars(run_dir, "race/success" if sac else "race/success_rate")
    if sac:   # logged wall-clock time, which continues over --resume
        ws, wv, _ = scalars(run_dir, "time/wall_s")
        wall = np.interp(steps, ws, wv)
    else:     # PPO: time per iteration, continued over resumes
        it_s = scalars(run_dir, "time/iteration_s")[1]
        wall = np.cumsum(np.resize(it_s, len(steps))) if len(it_s) else wall - wall[0]
    name = "{0} {1}".format("FlashSAC" if sac else "PPO " + cfg["policy"].upper(), os.path.basename(run_dir))
    ev = None
    if sac and os.path.exists(os.path.join(run_dir, "eval.json")):
        with open(os.path.join(run_dir, "eval.json")) as f:
            ev = json.load(f)
    return dict(name=name, steps=steps, success=succ, wall=wall, eval=ev, device=cfg.get("device", "?"))


def mean_std(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return (x.mean(), x.std()) if len(x) else (np.nan, np.nan)


def smooth(y, k=10):
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k - 1, 0), mode="edge"), np.ones(k) / k, mode="valid")


def first_hit(r, t):
    hit = np.flatnonzero(smooth(r["success"]) >= t)
    return (r["steps"][hit[0]], r["wall"][hit[0]]) if len(hit) else (np.nan, np.nan)


def band(ax, runs, key, scale, color, label):
    """Mean of the smoothed curves over seeds, min-max shaded."""
    end = min(r[key][-1] for r in runs)
    x = np.linspace(0, end, 400)
    ys = np.array([np.interp(x, r[key], smooth(r["success"])) for r in runs])
    ax.plot(x / scale, ys.mean(0), color=color, label=label)
    if len(runs) > 1:
        ax.fill_between(x / scale, ys.min(0), ys.max(0), color=color, alpha=0.2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="RUN_DIR or LABEL=RUN_DIR; runs sharing a label are seeds of one group")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.8])
    p.add_argument("--out", default="ppo_vs_flashsac.png")
    args = p.parse_args()
    runs = []
    for spec in args.runs:
        label, _, run_dir = spec.rpartition("=")
        r = load(run_dir)
        r["label"] = label or r["name"]
        runs.append(r)
    groups = {}
    for r in runs:
        groups.setdefault(r["label"], []).append(r)

    head = ["run", "device", "env steps", "wall [min]", "final success"] + \
        ["to {0:.0%}: steps / min".format(t) for t in args.thresholds]
    print(" | ".join(head))
    print(" | ".join("---" for _ in head))
    for r in runs:
        s = smooth(r["success"])
        cells = [r["label"] if r["label"] == r["name"] else "{0} ({1})".format(r["label"], r["name"]), r["device"],
                 "{0:.1f}M".format(r["steps"][-1] / 1e6), "{0:.0f}".format(r["wall"][-1] / 60), "{0:.0%}".format(s[-1])]
        for t in args.thresholds:
            st, wl = first_hit(r, t)
            cells.append("-" if np.isnan(st) else "{0:.1f}M / {1:.0f}".format(st / 1e6, wl / 60))
        print(" | ".join(cells))
    if any(len(g) > 1 for g in groups.values()):
        print("\ngroup | seeds | final success | " +
              " | ".join("to {0:.0%}: steps [M]".format(t) for t in args.thresholds))
        print(" | ".join("---" for _ in range(3 + len(args.thresholds))))
        for label, g in groups.items():
            cells = [label, str(len(g)), "{0:.0%} ± {1:.0%}".format(*mean_std([smooth(r["success"])[-1] for r in g]))]
            for t in args.thresholds:
                st = [first_hit(r, t)[0] / 1e6 for r in g]
                cells.append("{0:.1f} ± {1:.1f}".format(*mean_std(st)) + (" (not all)" if np.isnan(st).any() else ""))
            print(" | ".join(cells))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, (label, g) in enumerate(groups.items()):
        color = "C%d" % k
        name = label + (" ({0} seeds)".format(len(g)) if len(g) > 1 else "")
        band(axes[0], g, "steps", 1e6, color, name + " (train episodes)")
        band(axes[1], g, "wall", 60, color, None)
        evs = [r["eval"] for r in g if r["eval"]]
        if evs:
            n = min(len(ev) for ev in evs)
            x = np.array([[e["env_steps"] for e in ev[:n]] for ev in evs]).mean(0)
            w = np.array([[e["wall_s"] for e in ev[:n]] for ev in evs]).mean(0)
            y = np.array([[e["success_rate"] for e in ev[:n]] for ev in evs]).mean(0)
            axes[0].plot(x / 1e6, y, "--o", ms=3, color=color, label=name + " (eval, deterministic)")
            axes[1].plot(w / 60, y, "--o", ms=3, color=color)
    axes[0].set_xlabel("env steps [M]")
    axes[1].set_xlabel("wall-clock [min]")
    for ax in axes:
        ax.set_ylabel("success rate")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
