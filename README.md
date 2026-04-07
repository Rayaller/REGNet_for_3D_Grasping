# REGNet for 3D Grasping

This repository is a maintained, server-oriented version of the original
REGNet implementation for the ICRA 2021 paper "REGNet: REgion-based Grasp
Network for End-to-end Grasp Detection in Point Clouds".

The goal of this `v2` cleanup is practical reproducibility:

- clone the repo on a fresh server
- create the environment
- compile the CUDA extensions
- launch training and validation without editing source paths
- run single-file inference and visualization with documented commands

This README is written around commands that were actually re-checked during the
current cleanup work. Validation details are recorded in
[`validation_log.md`](./validation_log.md).

## What This Repository Includes

The publishable repository is intended to keep source code and documentation,
not datasets, checkpoints, logs, or prediction artifacts.

Included:

- source code under `dataset_utils/`, `multi_model/`, and `vis/`
- entry scripts such as `train.py`, `test.py`, and `utils.py`
- CUDA extension source under `multi_model/utils/pn2_utils/`
- helper scripts under `scripts/`
- runtime output directories:
  - `assets/log/`
  - `assets/models/`

Excluded from a clean release:

- training data under `dataset/`
- evaluation data under `eval_data/`
- sample inference inputs and outputs under `test_file/`
- generated checkpoints, logs, and visualization outputs

## Verified Environment

The following setup was verified during the latest cleanup pass:

- Linux
- Python `3.8.20`
- PyTorch `2.1.2`
- torchvision `0.16.2`
- CUDA toolkit detected by `nvcc`: `12.0`
- PyTorch CUDA build: `12.1`
- Open3D `0.18.0`

Important runtime constraint:

- `train.py` and `test.py` require a real CUDA-capable GPU at runtime.
- CPU-only execution is not supported by the PointNet++ / DGCNN CUDA
  operators used in this project.

## 1. Clone the Repository

```bash
git clone <your-repo-url> REGNet_for_3D_Grasping
cd REGNet_for_3D_Grasping
```

All commands below assume the current working directory is the project root.

## 2. Create the Environment

Using Conda is the most straightforward path on a server:

```bash
conda create -y -n regnet python=3.8
conda run -n regnet python -m pip install --upgrade pip
```

Install PyTorch and torchvision with a CUDA build that matches your server.
The validated local setup used CUDA 12.1 wheels:

```bash
conda run -n regnet python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining Python dependencies from `requirements.txt`:

```bash
conda run -n regnet python -m pip install -r requirements.txt
```

Current `requirements.txt` covers the non-PyTorch runtime dependencies:

- `numpy`
- `open3d`
- `tensorboardX`
- `tqdm`
- `transforms3d`

## 3. Compile the CUDA Extensions

This project depends on two CUDA extensions:

- `pn2_ext`
- `dgcnn_ext`

The repository includes a helper script that checks `nvcc`, sets `CUDA_HOME`,
and builds both extensions:

```bash
conda run --no-capture-output -n regnet bash scripts/install_extensions.sh
```

Then verify the environment and extension imports:

```bash
conda run -n regnet python scripts/check_runtime.py
```

Expected result:

- core Python dependencies import successfully
- `pn2_ext` imports successfully
- `dgcnn_ext` imports successfully
- the script prints `runtime check passed`

Notes:

- The build scripts automatically provide a default
  `TORCH_CUDA_ARCH_LIST` when it is unset.
- Minor CUDA version mismatch warnings such as CUDA `12.0` toolkit vs PyTorch
  CUDA `12.1` were observed during validation and did not prevent successful
  compilation.

## 4. Prepare the Data

### Training and Validation Data Layout

For the default training pipeline, point-cloud training data is expected under
a root directory like:

```text
dataset/0.08/
  training_data/
  training_data_test/
```

`ScoreDataset` splits the data this way:

- `training_data/`:
  - 80% used as train
  - 20% used as validate
- `training_data_test/`:
  - used as test

So you do not need a separate `validate/` directory for this code path.

If you keep the data elsewhere, pass it with `--data-path`.

Example:

```bash
DATA_ROOT=/path/to/dataset/0.08
```

### Dataset Source

The original project notes the dataset link below:

- Baidu Pan: `https://pan.baidu.com/s/1alwaKQZt0IGE11FFSClxOg`
- extraction code: `x79a`

Because this repository is intended to stay lightweight, datasets should be
prepared outside version control.

### Model Checkpoints

Validation, full refine inference, and `test.py` require pretrained model
checkpoints.

You can either:

- place them under `assets/models/<tag>/`
- or pass explicit paths with:
  - `--load-score-path`
  - `--load-region-path`

For `test.py`, if you omit both paths, the script tries to auto-discover the
latest matching `score_*.model` and `region_*.model` pair from:

- `assets/models/final/`
- `assets/models/regnet_train/`

## 5. Training Workflow

The code supports three main stages:

1. `pretrain_score`
2. `pretrain_region`
3. `train` (refine / joint stage)

The commands below are single-GPU examples, because that is the configuration
verified during the latest cleanup pass.

### 5.1 Pretrain ScoreNet

```bash
DATA_ROOT=/path/to/dataset/0.08

conda run --no-capture-output -n regnet python train.py \
  --cuda \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --batch-size 12 \
  --mode pretrain_score \
  --tag regnet_score \
  --data-path "${DATA_ROOT}"
```

Outputs:

- checkpoints under `assets/models/regnet_score/`
- TensorBoard logs under `assets/log/regnet_score/`

### 5.2 Pretrain Region Network

This stage needs a score-model checkpoint.

```bash
DATA_ROOT=/path/to/dataset/0.08
SCORE_MODEL=/path/to/score_checkpoint.model

conda run --no-capture-output -n regnet python train.py \
  --cuda \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --batch-size 12 \
  --mode pretrain_region \
  --tag regnet_region \
  --data-path "${DATA_ROOT}" \
  --load-score-path "${SCORE_MODEL}"
```

Outputs:

- checkpoints under `assets/models/regnet_region/`
- TensorBoard logs under `assets/log/regnet_region/`

### 5.3 Train the Full Refine Stage

This stage needs both a score checkpoint and a region checkpoint.

```bash
DATA_ROOT=/path/to/dataset/0.08
SCORE_MODEL=/path/to/score_checkpoint.model
REGION_MODEL=/path/to/region_checkpoint.model

conda run --no-capture-output -n regnet python train.py \
  --cuda \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --batch-size 12 \
  --mode train \
  --tag regnet_train \
  --data-path "${DATA_ROOT}" \
  --load-score-path "${SCORE_MODEL}" \
  --load-region-path "${REGION_MODEL}"
```

Outputs:

- checkpoints under `assets/models/regnet_train/`
- TensorBoard logs under `assets/log/regnet_train/`

### 5.4 TensorBoard

```bash
tensorboard --logdir assets/log --port 8080
```

## 6. Validation Commands

These validation commands were re-checked in a GPU environment during the
cleanup work.

### 6.1 Validate ScoreNet

```bash
DATA_ROOT=/path/to/dataset/0.08
SCORE_MODEL=/path/to/score_checkpoint.model

conda run --no-capture-output -n regnet python train.py \
  --cuda \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --mode validate_score \
  --tag smoke_json_metrics_tiny \
  --data-path "${DATA_ROOT}" \
  --load-score-path "${SCORE_MODEL}"
```

### 6.2 Validate Region Stage

```bash
DATA_ROOT=/path/to/dataset/0.08
SCORE_MODEL=/path/to/score_checkpoint.model
REGION_MODEL=/path/to/region_checkpoint.model

conda run --no-capture-output -n regnet python train.py \
  --cuda \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --mode validate_region \
  --tag smoke_region_tiny \
  --data-path "${DATA_ROOT}" \
  --load-score-path "${SCORE_MODEL}" \
  --load-region-path "${REGION_MODEL}"
```

### 6.3 Validate the Full Refine Pipeline

```bash
DATA_ROOT=/path/to/dataset/0.08
SCORE_MODEL=/path/to/score_checkpoint.model
REGION_MODEL=/path/to/region_checkpoint.model

conda run --no-capture-output -n regnet python train.py \
  --cuda \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --mode validate \
  --tag smoke_refine_tiny \
  --data-path "${DATA_ROOT}" \
  --load-score-path "${SCORE_MODEL}" \
  --load-region-path "${REGION_MODEL}"
```

The validation code writes TensorBoard scalars and also stores per-epoch JSON
metrics under:

- `assets/log/<tag>/epoch_metrics.json`

## 7. Test a Single Point Cloud File

`test.py` runs single-file inference and writes a prediction artifact next to
the input folder, for example:

- input:
  - `test_file/virtual_data/00001_view_1.p`
- output:
  - `test_file/virtual_data_predict/00001_view_1.p`

### 7.1 Virtual Data Example

This command was verified end-to-end on GPU:

```bash
SCORE_MODEL=/path/to/score_checkpoint.model
REGION_MODEL=/path/to/region_checkpoint.model

conda run --no-capture-output -n regnet python test.py \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --table-height 0.5 \
  --folder-name /path/to/virtual_data \
  --file-name 00001_view_1.p \
  --load-score-path "${SCORE_MODEL}" \
  --load-region-path "${REGION_MODEL}"
```

Why `--table-height 0.5` matters:

- many synthetic samples do not use the same table height as the training
  validation scenes
- for the verified sample `00001_view_1.p`, the default `0.75` filtered out
  all candidate grasps
- with `--table-height 0.5`, non-zero grasps were produced

### 7.2 Real Data Example

```bash
SCORE_MODEL=/path/to/score_checkpoint.model
REGION_MODEL=/path/to/region_checkpoint.model

conda run --no-capture-output -n regnet python test.py \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --table-height 0.7 \
  --folder-name /path/to/real_data \
  --file-name 0000_cloud.pcd \
  --load-score-path "${SCORE_MODEL}" \
  --load-region-path "${REGION_MODEL}"
```

If `--file-name` is omitted, `test.py` processes every supported file in the
folder.

## 8. Visualize a Prediction

After `test.py` generates a prediction file, you can inspect it with Open3D:

```bash
conda run --no-capture-output -n regnet python vis/vis_grasp.py \
  --path /path/to/prediction_file.p \
  --stage grasp_stage3 \
  --score-thre 0.5
```

Available stages:

- `grasp_stage2`
- `grasp_stage3_stage2`
- `grasp_stage3`
- `grasp_stage3_score`

Notes:

- `vis/vis_grasp.py` opens an Open3D window, so this step needs a GUI-capable
  session or remote display forwarding.
- For headless servers, run visualization locally after copying the prediction
  file if needed.

## 9. Common Problems

### `Mode 'validate' requires CUDA`

Cause:

- the project is running in an environment where `torch.cuda.is_available()`
  is false

Fix:

- use a GPU-enabled machine
- make sure the installed PyTorch build has CUDA support
- confirm with:

```bash
conda run -n regnet python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### `No module named 'pn2_ext'` or `No module named 'dgcnn_ext'`

Cause:

- the CUDA extensions were not built or not installed into the active
  environment

Fix:

```bash
conda run --no-capture-output -n regnet bash scripts/install_extensions.sh
conda run -n regnet python scripts/check_runtime.py
```

### `test.py` runs but all grasps are filtered out

Cause:

- the table height does not match the sample

Fix:

- adjust `--table-height`
- for the verified virtual sample, `--table-height 0.5` worked while the
  default `0.75` removed all grasps

### `torch.cuda.is_available()` changes unexpectedly inside custom wrappers

Observed during validation:

- direct `conda run --no-capture-output -n regnet python ...` was stable
- extra shell wrappers that manually changed runtime library variables caused
  CUDA visibility problems in one subprocess chain

Recommendation:

- prefer direct `conda run ... python ...` commands for automation
- or activate the environment first and then call `python` directly

## 10. Recommended End-to-End Server Checklist

On a fresh server, the intended order is:

1. Clone the repository.
2. Create the Conda environment.
3. Install PyTorch and `requirements.txt`.
4. Run `scripts/install_extensions.sh`.
5. Run `scripts/check_runtime.py`.
6. Prepare the external dataset root and model checkpoints.
7. Run `validate_score`, `validate_region`, or full `validate`.
8. Run `test.py` on a sample point cloud.
9. Optionally inspect the output with `vis/vis_grasp.py`.

If all steps above succeed, the server setup is in the intended reproducible
state for this repository.
