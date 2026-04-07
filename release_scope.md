# v2 Release Scope

## Included in the publishable repository

- Source code under `dataset_utils/`, `multi_model/`, `vis/`
- Entry scripts such as `train.py`, `test.py`, `utils.py`
- CUDA extension source code under `multi_model/utils/pn2_utils/`
- Documentation files such as `README.md`, `plan.md`, and this file
- Static reference assets under `markdown/`
- Empty runtime directories:
  - `assets/log/`
  - `assets/models/`

## Excluded from the publishable repository

- Training datasets under `dataset/`
- Evaluation datasets under `eval_data/`
- Test input and prediction files under `test_file/`
- Generated checkpoints and logs under `assets/models/` and `assets/log/`
- Visualization outputs under `vis/output/`
- Local caches, compiled artifacts, and Python bytecode

## Why these boundaries exist

- The repository should stay lightweight enough for a server to clone quickly.
- Data, checkpoints, and logs are runtime artifacts, not source code.
- `assets/log/` and `assets/models/` are kept as empty directory stubs because the training code writes into them by default.

## Target effect for v2

A fresh server should be able to:

1. Clone the repository.
2. Install dependencies and compile extensions.
3. Prepare external datasets separately.
4. Launch training without first cleaning out bundled data or model artifacts.
