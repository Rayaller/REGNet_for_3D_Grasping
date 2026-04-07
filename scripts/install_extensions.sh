#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PN2_DIR="${PROJECT_ROOT}/multi_model/utils/pn2_utils"
DGCNN_DIR="${PN2_DIR}/functions"

echo "[install_extensions] project root: ${PROJECT_ROOT}"

if ! command -v python >/dev/null 2>&1; then
  echo "[install_extensions] python was not found in PATH" >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[install_extensions] nvcc was not found in PATH; please install CUDA toolkit first" >&2
  exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
fi

echo "[install_extensions] using python: $(command -v python)"
echo "[install_extensions] using nvcc: $(command -v nvcc)"
echo "[install_extensions] CUDA_HOME: ${CUDA_HOME}"
echo "[install_extensions] TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"

python - <<'PY'
import torch
print(f"[install_extensions] torch: {torch.__version__}")
print(f"[install_extensions] torch cuda build: {torch.version.cuda}")
PY

python - <<'PY'
import importlib
import sys

for module_name in ("pip", "setuptools", "wheel"):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print(f"[install_extensions] missing build tool {module_name}: {exc}", file=sys.stderr)
        sys.exit(1)
    version = getattr(module, "__version__", "unknown")
    print(f"[install_extensions] {module_name}: {version}")
PY

echo "[install_extensions] building pn2_ext"
python -m pip install --force-reinstall --no-deps --no-build-isolation --disable-pip-version-check --verbose "${PN2_DIR}"

echo "[install_extensions] building dgcnn_ext"
python -m pip install --force-reinstall --no-deps --no-build-isolation --disable-pip-version-check --verbose "${DGCNN_DIR}"

echo "[install_extensions] done"
