#!/usr/bin/env bash
# Install torch-mps-workarounds on a remote Mac over ssh (no uv or modern python needed).
# Usage: scripts/install_on_node.sh <ssh-host> [remote-dir]   (default ~/torch-mps-workarounds)
set -euo pipefail
host=${1:?usage: $0 <ssh-host> [remote-dir]}
dir=${2:-torch-mps-workarounds}   # relative to remote $HOME
root=$(cd "$(dirname "$0")/.." && pwd)

# Official uv installer (user-local, no sudo); skipped if uv is already present.
ssh "$host" 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'

ssh "$host" "mkdir -p '$dir'"
(cd "$root" && tar --exclude=__pycache__ --exclude='*.egg-info' -cf - pyproject.toml README.md LICENSE src tests examples scripts) \
  | ssh "$host" "tar -xf - -C '$dir'"

# uv fetches Python 3.12 (system python 3.9 is too old) and a torch wheel into a venv.
ssh "$host" "export PATH=\"\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH\"; cd '$dir' && \
  uv venv --python 3.12 --allow-existing .venv && \
  uv pip install --python .venv/bin/python torch . && \
  .venv/bin/python -c 'import torch, torch_mps_workarounds as t; print(\"torch\", torch.__version__, \"mps\", torch.backends.mps.is_available())'"
echo "Installed on $host:~/$dir (venv: .venv). Run tests there with:"
echo "  ssh $host 'cd $dir && TORCH_MPS_WORKAROUNDS_TEST_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m unittest discover -s tests'"
