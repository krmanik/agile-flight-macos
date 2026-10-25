#!/bin/bash
# Fetch uzh-rpg/acmpc_public (Actor-Critic MPC) and its mpc.pytorch submodule
# into $ACMPC_PATH (default: src/acmpc_public) and apply the gymnasium patch.
# Its stable-baselines3 fork (gym 0.21) is not needed: acmpc/acmpc_policy.py
# provides the same policy changes on stock stable-baselines3 2.x.
# Run inside the workspace env:  pixi run setup-acmpc
set -euo pipefail

WS="$PIXI_PROJECT_ROOT"

if [ ! -d "$ACMPC_PATH/.git" ]; then
  echo ">>> Cloning uzh-rpg/acmpc_public into $ACMPC_PATH"
  git clone https://github.com/uzh-rpg/acmpc_public.git "$ACMPC_PATH"
fi

echo ">>> Submodule mpc.pytorch (SSH URL rewritten to HTTPS)"
git -C "$ACMPC_PATH" -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init mpc.pytorch

if git -C "$ACMPC_PATH" apply --reverse --check "$WS/patches/acmpc_public.patch" 2>/dev/null; then
  echo ">>> acmpc_public.patch: already applied"
else
  echo ">>> Applying acmpc_public.patch"
  # Drop an older version of the patch first
  git -C "$ACMPC_PATH" checkout -- training_modules/mlp_mpc_policy.py training_modules/mlp_only_policy.py
  git -C "$ACMPC_PATH" apply "$WS/patches/acmpc_public.patch"
fi

# mpc.pytorch builds batch x batch diagonal matrices in its line searches
# (2.3 GB at the 25k-sample batches used on a GPU); scale rows instead
MPC="$ACMPC_PATH/mpc.pytorch"
if git -C "$MPC" apply --reverse --check "$WS/patches/mpc_pytorch.patch" 2>/dev/null; then
  echo ">>> mpc_pytorch.patch: already applied"
else
  echo ">>> Applying mpc_pytorch.patch"
  git -C "$MPC" checkout -- mpc
  git -C "$MPC" apply "$WS/patches/mpc_pytorch.patch"
fi

python -c "import mlp_mpc_policy, mpc; print('>>> ACMPC modules OK')"
