#!/bin/bash
# One-command DodgeDrone demo: simulator + rqt pilot GUI + rviz, then
# arm, take off, and fly the example state-based policy (envtest/ros/user_code.py).
# Run inside the workspace env:  pixi run demo   (or: pixi run bash scripts/demo_flight.sh)
# Flags: --headless (no GUI/rviz), --render (Unity camera images, needs
#        pixi run build-renderer), --vision_based (passed to run_competition.py)
set -e

GUI=True
RENDER=False
COMP_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --headless) GUI=False ;;
    --render) RENDER=True ;;
    *) COMP_ARGS+=("$arg") ;;
  esac
done

WS="$PIXI_PROJECT_ROOT"
source "$WS/devel/setup.bash"
ENVTEST="$WS/src/agile_flight/envtest"

pids=()
cleanup() {
  echo "Shutting down..."
  kill -INT "${pids[@]}" 2>/dev/null || true
  sleep 3
  kill "${pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

PILOT=/kingfisher/dodgeros_pilot
altitude() {
  timeout 5 rostopic echo -n1 $PILOT/state/pose/position/z 2>/dev/null | head -1
}

cd "$ENVTEST"
roslaunch envsim visionenv_sim.launch render:=$RENDER gui:=$GUI rviz:=$GUI &
pids+=($!)

echo ">>> Waiting for simulator state"
until [ -n "$(altitude)" ]; do sleep 1; done

cd "$ENVTEST/ros"
python -u evaluation_node.py &
pids+=($!)
python -u run_competition.py "${COMP_ARGS[@]}" &
pids+=($!)
sleep 6

# The pilot rejects Start until its state estimate is valid, so retry until airborne
echo ">>> Arm & Start (take off)"
for attempt in $(seq 1 10); do
  rostopic pub -1 $PILOT/enable std_msgs/Bool "data: true" >/dev/null
  rostopic pub -1 $PILOT/start std_msgs/Empty "{}" >/dev/null
  sleep 6
  if python -c "import sys; sys.exit(0 if float('$(altitude)' or 0) > 0.5 else 1)"; then
    break
  fi
  echo "    not airborne yet, retrying ($attempt)"
done

echo ">>> start_navigation"
rostopic pub -1 /kingfisher/start_navigation std_msgs/Empty "{}" >/dev/null

echo ">>> Flying. Ctrl-C to stop."
wait
