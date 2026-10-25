"""Race tracks for the ACMPC gate-racing task.

Tracks use the format of agile_flight's envsim/parameters/tracks/*.yaml:
  start_pos: [x, y, z]
  gates:
    Gate1: {position: [x, y, z], rotation: [qw, qx, qy, qz], ...}
    ...
    N: <number of gates>
The gate is a square in its local x-z plane; the drone flies through it along
the local y axis (the thin axis in the upstream track files). The sign of
the flight direction comes from the gate order, so a yaw of 90 or -90 degrees
describes the same gate.
"""
import os
import re

import numpy as np
import yaml

TRACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracks")

# Inner opening of a racing gate (m), as in the RPG drone-racing papers
GATE_SIZE = 1.5
# Width of the gate frame around the opening (m): crossing the gate plane here
# is a crash
GATE_FRAME = 0.3


def quat_to_rot(q):
    """[qw, qx, qy, qz] (..., 4) -> rotation matrix (..., 3, 3)."""
    w, x, y, z = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


class Track:
    def __init__(self, path, gate_size=GATE_SIZE, gate_frame=GATE_FRAME):
        if not os.path.exists(path):
            # A track name, or a path from another machine (a Colab run's config.json)
            name = os.path.basename(path)
            path = os.path.join(TRACK_DIR, name if name.endswith(".yaml") else name + ".yaml")
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            cfg = yaml.safe_load(f)
        gates = cfg["gates"]
        n = int(gates["N"])
        self.centers = np.array([gates["Gate%d" % (i + 1)]["position"] for i in range(n)], dtype=np.float64)
        self.quats = np.array([gates["Gate%d" % (i + 1)]["rotation"] for i in range(n)], dtype=np.float64)
        self.scales = np.array([gates["Gate%d" % (i + 1)].get("scale", [1.0] * 3) for i in range(n)])
        self.start_pos = np.array(cfg["start_pos"], dtype=np.float64)
        self.num_gates = n
        self.gate_size = gate_size
        self.gate_frame = gate_frame

        rot = quat_to_rot(self.quats)
        # In-plane axes (horizontal, vertical) and the flight direction
        self.axis_x = rot[:, :, 0]
        self.axis_z = rot[:, :, 2]
        normal = rot[:, :, 1]
        for i in range(n):
            # The track is a loop: gate 0 follows the last gate
            through = self.centers[(i + 1) % n] - self.centers[i - 1]
            if n == 1:
                through = self.centers[0] - self.start_pos
            if normal[i] @ through < 0:
                normal[i] = -normal[i]
        self.normals = normal

        h = gate_size / 2
        corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        # (n, 4, 3)
        self.corners = (self.centers[:, None, :]
                        + corners[None, :, 0:1] * self.axis_x[:, None, :]
                        + corners[None, :, 1:2] * self.axis_z[:, None, :])

        lo = np.minimum(self.centers.min(0), self.start_pos) - 5.0
        hi = np.maximum(self.centers.max(0), self.start_pos) + 5.0
        lo[2] = 0.0
        # [xmin, ymin, zmin], [xmax, ymax, zmax]: leaving it is a crash
        self.bounds = np.stack([lo, hi])

    def shifted(self, offset):
        """Copy of the track moved by offset (e.g. to place it in the ROS sim)."""
        t = object.__new__(Track)
        t.__dict__.update(self.__dict__)
        offset = np.asarray(offset, dtype=np.float64)
        t.centers = self.centers + offset
        t.corners = self.corners + offset
        t.start_pos = self.start_pos + offset
        t.bounds = self.bounds + offset
        return t

    def crossing(self, p0, p1):
        """Check the segments p0 -> p1 (B, 3) against every gate.

        Returns (passed, hit), both (B, num_gates) bool:
          passed: crossed the opening in the flight direction
          hit:    crossed the gate plane through the frame (or the opening
                  backwards), i.e. a collision with the gate
        """
        d0 = np.einsum("bgk,gk->bg", p0[:, None, :] - self.centers[None], self.normals)
        d1 = np.einsum("bgk,gk->bg", p1[:, None, :] - self.centers[None], self.normals)
        forward = (d0 < 0) & (d1 >= 0)
        backward = (d0 >= 0) & (d1 < 0)
        t = np.clip(d0 / np.where(d0 == d1, 1.0, d0 - d1), 0.0, 1.0)
        x = p0[:, None, :] + t[..., None] * (p1 - p0)[:, None, :] - self.centers[None]
        u = np.abs(np.einsum("bgk,gk->bg", x, self.axis_x))
        v = np.abs(np.einsum("bgk,gk->bg", x, self.axis_z))
        h = self.gate_size / 2
        inside = (u < h) & (v < h)
        on_frame = (u < h + self.gate_frame) & (v < h + self.gate_frame) & ~inside
        passed = forward & inside
        hit = (forward | backward) & on_frame | backward & inside
        return passed, hit


# Unity scenes of the macOS renderer (sceneSchedule.cs ids). flightlib only
# accepts ids < 5, so the generated scenes (renderer/RaceScenes.cs) take the
# slots of upstream scenes that are not public: 0 Warehouse, 2 Mountain.
SCENES = {"warehouse": 0, "industrial": 1, "garage": 2, "arena": 4}


def write_unity_overlay(track, offset, flightmare_path, out_dir, scene="industrial"):
    """Write a FLIGHTMARE_PATH overlay whose obstacle set is the track's gates.

    visionsim_node reads $FLIGHTMARE_PATH/flightpy/configs/vision/config.yaml
    and spawns the static objects of <level>/<env_folder>/static_obstacles.csv
    in Unity. Pointing FLIGHTMARE_PATH at this overlay shows the gates instead
    of the DodgeDrone obstacles.
    """
    src = os.path.join(flightmare_path, "flightpy", "configs")
    dst = os.path.join(out_dir, "flightpy", "configs")
    env_dir = os.path.join(dst, "vision", "race", track.name)
    os.makedirs(env_dir, exist_ok=True)
    scene_id = SCENES[scene]
    with open(os.path.join(src, "scene.yaml")) as f:
        scenes = yaml.safe_load(f)
    if scene != "industrial":
        # The generated scenes are built around the origin
        scenes["scene_%d" % scene_id] = {"name": scene, "render_offset": [0.0, 0.0, 0.0]}
    path = os.path.join(dst, "scene.yaml")
    if os.path.islink(path):
        os.remove(path)
    with open(path, "w") as f:
        yaml.safe_dump(scenes, f, sort_keys=False)
    with open(os.path.join(src, "vision", "config.yaml")) as f:
        cfg = f.read()
    cfg = cfg.replace('level: "medium"', 'level: "race"').replace(
        'env_folder: "environment_0"', 'env_folder: "{0}"'.format(track.name))
    cfg = re.sub(r"scene_id: \d+", "scene_id: %d" % scene_id, cfg)
    with open(os.path.join(dst, "vision", "config.yaml"), "w") as f:
        f.write(cfg)
    with open(os.path.join(env_dir, "dynamic_obstacles.yaml"), "w") as f:
        f.write("N: 0\n")
    # Unity's loader (flightmare_unity ConfigObjects.ReadCSVFile) skips the first
    # row as a header and copies rotation and scale as given, while it swaps y/z
    # of the position itself. So: a dummy first row, and the rotation and scale
    # already in Unity's frame (flightlib quaternionRos2Unity / scalarRos2Unity).
    # visionsim_node's VisionEnv also reads this file, only for obstacle
    # observations that the race does not use.
    # gate_default is the UZH gate model (rpg_gate adds a 90 degree yaw). Both
    # prefabs carry a Rigidbody, which scripts/build_renderer.sh makes kinematic.
    rows = []
    for c, (w, x, y, z), (sx, sy, sz) in zip(track.centers + np.asarray(offset), track.quats, track.scales):
        rows.append(", ".join(["gate_default"] + ["%.6f" % v for v in [*c, w, -x, -z, -y, sx, sz, sy]]))
    with open(os.path.join(env_dir, "static_obstacles.csv"), "w") as f:
        f.write("\n".join(rows[:1] + rows) + "\n")
    return out_dir


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Write a Flightmare config overlay that renders a track's gates in Unity")
    p.add_argument("--track", default="splits")
    p.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--scene", choices=SCENES, default="industrial")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(write_unity_overlay(Track(a.track), a.offset, os.environ["FLIGHTMARE_PATH"], a.out, a.scene))
