# agile-flight-macos

Run the [DodgeDrone / agile_flight](https://github.com/uzh-rpg/agile_flight) quadrotor simulator (UZH RPG, ICRA 2022) natively on **Apple Silicon macOS**. No VM, Docker or Ubuntu needed.

Upstream only supports Ubuntu 20.04 + ROS Noetic. This repo provides:

- an isolated ROS Noetic environment ([RoboStack](https://robostack.github.io/) via [pixi](https://pixi.sh)) that lives entirely inside this folder,
- small patches that make the upstream code build with clang/libc++ on macOS,
- one-command setup and demo scripts.

Upstream sources are fetched from the public `uzh-rpg` repositories at setup time. Nothing from them is vendored here.

![pixi run demo --render on an Apple M4 Mac: rviz with RGB and depth camera panels, the DodgeDrone pilot GUI, and the Flightmare Unity renderer](docs/img/demo_windows.png)

*`pixi run demo --render` on an Apple M4: rviz with the onboard RGB and depth cameras (back), the DodgeDrone pilot GUI (left), and the natively built Flightmare Unity renderer (right). The drone flies through the obstacle field at 1 m/s.*

![Onboard camera: RGB (left) and depth (right)](docs/img/camera_rgb_depth.png)

*Onboard camera frames, 320×240 at 25 Hz: RGB (left) and depth (right).*

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
| Camera images (RGB / depth), vision-based mode | ✅ after building the Unity renderer (below) |
| Environments other than Industrial | ❌ not in the public Unity project |
| NVIDIA CUDA | ❌ PyTorch runs on CPU / MPS |

## Camera images (Unity renderer)

Images come from the Flightmare Unity standalone, which upstream only ships for Linux x86_64. `pixi run build-renderer` builds a native Apple Silicon (Metal) version from the public source, [uzh-rpg/flightmare_unity](https://github.com/uzh-rpg/flightmare_unity), branch `dev/agile_flight`.

1. Install Unity Hub (`brew install --cask unity-hub`) and Unity Editor **2021.3.45f2** (Apple silicon). Either use the Hub UI, or run:
   ```bash
   "/Applications/Unity Hub.app/Contents/MacOS/Unity Hub" -- --headless install \
     --version 2021.3.45f2 --changeset 88f88f591b2e --architecture arm64
   ```
2. Open Unity Hub, sign in, and add a license (Preferences → Licenses → Add → free Personal license).
3. Build and install the renderer. The first build imports all assets and takes a while:
   ```bash
   pixi run build-renderer
   ```
4. Run with images:
   ```bash
   pixi run demo --render
   ```
   The images appear on `/kingfisher/dodgeros_pilot/unity/image` and `.../unity/depth` (320×240, 25 Hz) and in rviz.

Notes:

- **Only the Industrial ("street") environment is public.** The Warehouse, Forest and Wasteland scenes use commercial assets that UZH keeps out of the public repo. The macOS patch therefore sets `scene_id: 1` in `flightmare/flightpy/configs/vision/config.yaml`. If you have the full asset set, put your project at `FLIGHTMARE_UNITY_PROJECT=/path/to/project`, rebuild, and change `scene_id` back.
- The renderer is built with 2021.3 LTS instead of the project's original 2021.2. The build script adds `com.unity.nuget.newtonsoft-json`, which the 2021.3 package upgrade otherwise drops.
- The Unity player log is at `~/Library/Logs/Robotics and Perception Group/RPG_Flightmare/Player.log`.

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
| Obstacle layouts (`static_obstacles.csv`) are a separate download | fetched by setup from UZH (~600 MB; skip with `SKIP_OBSTACLES=1`) |
| Unity renderer only released for Linux | built from source for macOS arm64 (`pixi run build-renderer`) |

## Troubleshooting

- **`symbol not found in flat namespace '__Py_NoneStruct'`**: run `pixi run fix-macos`. Re-run it after anything that reinstalls the environment (`pixi install`, `pixi add`, `pixi update`).
- **"No valid state estimate, will not start!"**: you pressed Start too early. Wait a few seconds after launch and press it again.
- **First `--render` run is slow**: the first launch of the freshly built app can take over a minute (macOS checks the new app, and Unity compiles Metal shaders). If the drone doesn't take off, stop with Ctrl-C and run the demo again.
- **`Unity Connection time out!` / empty images**: run `pixi run build-renderer` first. If it still fails, check `Player.log` (see above). A missing `static_obstacles.csv` makes the renderer error out, so re-run setup without `SKIP_OBSTACLES`.
- **Port 11311 already in use**: a previous `roscore` is still running. Stop it with `pkill -f rosmaster`.
- **Start over**: `rm -rf build devel logs .catkin_tools` and run `pixi run setup` again. Delete `.pixi` too for a completely fresh environment.

## Caveats

ROS Noetic (EOL May 2025) and Gazebo Classic (EOL January 2025) no longer receive updates, and RoboStack no longer rebuilds Noetic packages. `pixi.lock` pins a known-good set.

## Credits

- [agile_flight](https://github.com/uzh-rpg/agile_flight), [flightmare](https://github.com/uzh-rpg/flightmare) and [dodgedrone_simulation](https://github.com/uzh-rpg/dodgedrone_simulation): Robotics and Perception Group, University of Zurich. Upstream code is under its own licenses.
- [RoboStack](https://robostack.github.io/) for ROS on conda-forge.

The scripts and patches in this repository are MIT licensed (see [LICENSE](LICENSE)).
