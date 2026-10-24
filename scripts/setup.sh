#!/bin/bash
# Fetch upstream uzh-rpg/agile_flight (DodgeDrone), apply macOS patches, build.
# Run inside the workspace env:  pixi run setup
set -euo pipefail

WS="$PIXI_PROJECT_ROOT"
SRC="$WS/src"
AF="$SRC/agile_flight"
AGILE_FLIGHT_URL="${AGILE_FLIGHT_URL:-https://github.com/uzh-rpg/agile_flight.git}"
AGILE_FLIGHT_REF="${AGILE_FLIGHT_REF:-main}"

mkdir -p "$SRC"
if [ ! -d "$AF/.git" ]; then
  echo ">>> Cloning $AGILE_FLIGHT_URL ($AGILE_FLIGHT_REF)"
  git clone --branch "$AGILE_FLIGHT_REF" "$AGILE_FLIGHT_URL" "$AF"
fi

echo ">>> Submodules (SSH URLs rewritten to HTTPS, no GitHub SSH key needed)"
git -C "$AF" -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --recursive

apply_patch() {
  local repo="$1" patch="$2"
  if git -C "$repo" apply --reverse --check "$patch" 2>/dev/null; then
    echo ">>> $(basename "$patch"): already applied"
  else
    echo ">>> Applying $(basename "$patch")"
    git -C "$repo" apply "$patch"
  fi
}
apply_patch "$AF/flightmare" "$WS/patches/flightmare-macos.patch"
apply_patch "$AF/dodgedrone_simulation" "$WS/patches/dodgedrone_simulation-macos.patch"

# Bundled nlohmann/json 2.1.1 does not compile with libc++ (no to_json for
# std::vector<bool>); point it at the env's nlohmann/json 3.x.
cat > "$AF/flightmare/flightlib/include/flightlib/json/json.hpp" <<'EOF'
#pragma once
// macOS: bundled nlohmann/json 2.1.1 fails with libc++; use system 3.x
// (wire-compatible JSON for the Unity renderer).
#include <nlohmann/json.hpp>
EOF

# Obstacle layouts (static_obstacles.csv, dynamic trajectories) from upstream's
# setup_ros.bash; without them there are no obstacles and the renderer errors out.
VISION_CFG="$AF/flightmare/flightpy/configs/vision"
if [ -z "${SKIP_OBSTACLES:-}" ] && [ ! -d "$VISION_CFG/medium" ]; then
  echo ">>> Downloading obstacle trajectories (~600 MB)"
  curl -fL --retry 3 -o "$VISION_CFG/trajectories.zip" \
    https://download.ifi.uzh.ch/rpg/Flightmare/trajectories.zip
  unzip -q -o "$VISION_CFG/trajectories.zip" -d "$VISION_CFG"
  rm "$VISION_CFG/trajectories.zip"
fi

# flightros needs Gazebo/rotors packages not used by the DodgeDrone sim
touch "$AF/flightmare/flightros/CATKIN_IGNORE"

echo ">>> Patching librospack (missing libpython dependency)"
pixi run fix-macos

echo ">>> Configuring catkin"
cd "$WS"
catkin init >/dev/null
catkin config --merge-devel --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF \
  -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DEIGEN_FROM_SYSTEM=ON >/dev/null

echo ">>> Building"
catkin build --summarize

echo ">>> Done. Run the demo with: pixi run demo"
