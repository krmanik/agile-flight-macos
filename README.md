# agile-flight-macos

Run the [DodgeDrone / agile_flight](https://github.com/uzh-rpg/agile_flight) quadrotor simulator (UZH RPG, ICRA 2022) natively on **Apple Silicon macOS**. No VM, Docker or Ubuntu needed.

Upstream only supports Ubuntu 20.04 + ROS Noetic. This repo provides:

- an isolated ROS Noetic environment ([RoboStack](https://robostack.github.io/) via [pixi](https://pixi.sh)) that lives entirely inside this folder,
- small patches that make the upstream code build with clang/libc++ on macOS,
- one-command setup and demo scripts.

Upstream sources are fetched from the public `uzh-rpg` repositories at setup time. Nothing from them is vendored here.

## Requirements

- Mac with Apple Silicon (arm64), recent macOS
- Xcode Command Line Tools: `xcode-select --install`
- [pixi](https://pixi.sh): `brew install pixi` (or `curl -fsSL https://pixi.sh/install.sh | sh`)
- About 7 GB of disk space (ROS desktop-full + PyTorch)

## Quick start

```bash
git clone <this-repo-url> agile-flight-macos
cd agile-flight-macos
pixi run setup     # installs env, clones uzh-rpg/agile_flight, patches, builds (~5 min)
pixi run demo      # simulator + rqt pilot GUI + rviz, takes off and flies forward
```

`pixi run demo --headless` runs without any windows. Press Ctrl-C to stop.

The demo flies the example policy in `src/agile_flight/envtest/ros/user_code.py`, a constant 1 m/s forward command. Edit that file to plug in your own controller.

## Manual usage (upstream workflow)

Open a shell in the environment with `pixi shell`, then run `source devel/setup.bash` in each terminal:

```bash
# Terminal 1: simulator (+ GUI, rviz)
cd src/agile_flight/envtest
roslaunch envsim visionenv_sim.launch render:=False
# In the GUI: press "Arm Bridge", then "Start" to take off.

# Terminal 2: evaluation
cd src/agile_flight/envtest/ros && python evaluation_node.py

# Terminal 3: your policy
cd src/agile_flight/envtest/ros && python run_competition.py

# Terminal 4: go
rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" -1
```

Rebuild after code changes with `pixi run build` (or `catkin build` inside `pixi shell`).

## What works

| Component | Status |
|---|---|
| ROS Noetic core, rviz, rqt, Gazebo 11 (headless and GUI) | ✅ |
| DodgeDrone simulator (`envsim`), physics, obstacles, controllers | ✅ |
| rqt pilot GUI (Arm / Start / Land) | ✅ |
| Competition stack: `evaluation_node.py`, `run_competition.py`, state-based policies | ✅ |
| Camera images (RGB / depth), vision-based mode | ❌ needs the Unity renderer, see below |
| NVIDIA CUDA | ❌ PyTorch runs on CPU / MPS |

### Camera images (Unity renderer)

Images come from the Flightmare Unity standalone, which upstream only ships for Linux x86_64. Its source is public at [uzh-rpg/flightmare_unity](https://github.com/uzh-rpg/flightmare_unity), branch `dev/agile_flight` (Unity 2021.2.11f1). A macOS build of it should connect over ZMQ on localhost, but this hasn't been tested yet. Until then, run with `render:=False`. The image panels in rviz stay empty.

## What the setup changes

All changes are in [`patches/`](patches/), [`scripts/`](scripts/) and [`pixi.toml`](pixi.toml):

| Problem on macOS | Fix |
|---|---|
| RoboStack `librospack.dylib` uses CPython symbols without linking libpython, so rviz, Gazebo ROS plugins and every C++ node abort with `symbol not found in flat namespace '__Py_NoneStruct'` | [`scripts/fix_rospack.py`](scripts/fix_rospack.py) adds the missing `libpython` load command (task `pixi run fix-macos`; it runs during setup) |
| CMake 4 rejects `cmake_minimum_required(VERSION <3.5)` | `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` |
| `empy` 4 breaks ROS message generation (`em.RAW_OPT`) | pin `empy==3.3.4` |
| flightlib's bundled nlohmann/json 2.1.1 fails with libc++ (`std::vector<bool>`) | header shim to nlohmann/json 3.x (same JSON on the wire) |
| `std::string` passed to printf-style `%s` logger, which clang rejects because it aborts at runtime | `.c_str()` in flightlib (real bug, also on Linux) |
| `-Werror` plus newer clang warnings inside Eigen and generated headers | `-Wno-error` for clang in flightlib and dodgeros |
| flightlib downloads its own Eigen | `-DEIGEN_FROM_SYSTEM=ON` |
| Gazebo multicast fails, and the online model DB is gone | `GAZEBO_IP`/`IGN_IP=127.0.0.1`, empty `GAZEBO_MODEL_DATABASE_URI` |
| `FLIGHTMARE_PATH` unset, so the node segfaults | exported by [`activate.sh`](activate.sh) |
| Submodule URLs use SSH | rewritten to HTTPS during setup |

## Troubleshooting

- **`symbol not found in flat namespace '__Py_NoneStruct'`**: run `pixi run fix-macos`. Re-run it after anything that reinstalls the environment (`pixi install`, `pixi add`, `pixi update`).
- **"No valid state estimate, will not start!"**: you pressed Start too early. Wait a few seconds after launch and press it again.
- **Port 11311 already in use**: a previous `roscore` is still running. Stop it with `pkill -f rosmaster`.
- **Start over**: `rm -rf build devel logs .catkin_tools` and run `pixi run setup` again. Delete `.pixi` too for a completely fresh environment.

## Caveats

ROS Noetic (EOL May 2025) and Gazebo Classic (EOL January 2025) no longer receive updates, and RoboStack no longer rebuilds Noetic packages. `pixi.lock` pins a known-good set.

## Credits

- [agile_flight](https://github.com/uzh-rpg/agile_flight), [flightmare](https://github.com/uzh-rpg/flightmare) and [dodgedrone_simulation](https://github.com/uzh-rpg/dodgedrone_simulation): Robotics and Perception Group, University of Zurich. Upstream code is under its own licenses.
- [RoboStack](https://robostack.github.io/) for ROS on conda-forge.

The scripts and patches in this repository are MIT licensed (see [LICENSE](LICENSE)).
