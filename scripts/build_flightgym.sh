#!/bin/bash
# Build flightlib's Python binding (flightgym: VisionEnv_v1, QuadrotorEnv_v1)
# for the pixi Python. activate.sh puts the module and rpg_baselines on PYTHONPATH.
# Upstream builds it with `pip install .`, whose setup.py wipes flightlib/build
# and externals/ first; a plain CMake build directory avoids that.
# Run inside the workspace env:  pixi run build-flightgym
set -euo pipefail

WS="$PIXI_PROJECT_ROOT"
BUILD="$WS/build_flightgym"

cmake -S "$FLIGHTMARE_PATH/flightlib" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF \
  -DBUILD_UNITY_BRIDGE_TESTS=OFF \
  -DEIGEN_FROM_SYSTEM=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build "$BUILD" -j"$(sysctl -n hw.ncpu)"

python -c "import flightgym; print('>>> flightgym OK:', flightgym.__file__)"
