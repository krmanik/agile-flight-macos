# agile_flight reads configs via getenv("FLIGHTMARE_PATH"); NULL -> segfault
export FLIGHTMARE_PATH="$PIXI_PROJECT_ROOT/src/agile_flight/flightmare"
# flightgym (pixi run build-flightgym) and rpg_baselines; upstream mixes
# "rpg_baselines." and "flightrl.rpg_baselines." imports, so add both roots
export PYTHONPATH="$PIXI_PROJECT_ROOT/build_flightgym:$FLIGHTMARE_PATH/flightpy/flightrl:$FLIGHTMARE_PATH/flightpy${PYTHONPATH:+:$PYTHONPATH}"
# ACMPC (pixi run setup-acmpc): set ACMPC_PATH to use an existing checkout
export ACMPC_PATH="${ACMPC_PATH:-$PIXI_PROJECT_ROOT/src/acmpc_public}"
export PYTHONPATH="$PIXI_PROJECT_ROOT/acmpc:$ACMPC_PATH/training_modules:$ACMPC_PATH/diff_mpc_drones:$ACMPC_PATH/mpc.pytorch:$PYTHONPATH"
# FlashSAC (pixi run setup-flashsac): set FLASHSAC_PATH to use an existing checkout
export FLASHSAC_PATH="${FLASHSAC_PATH:-$PIXI_PROJECT_ROOT/src/FlashSAC}"
