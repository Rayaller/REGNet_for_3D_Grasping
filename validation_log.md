# Validation Log

## 2026-04-07

This file records concrete validation work for the `v2` release plan so the
README can be updated from real commands instead of assumptions.

### Environment used

- Workspace: `/home/ray/work/regnet/REGNet_for_3D_Grasping`
- Python: `3.8.20`
- PyTorch: `2.1.2+cu121`
- CUDA toolkit detected by `nvcc`: `12.0`
- Open3D: `0.18.0`
- Tiny validation dataset: `/tmp/regnet_tiny_dataset`

### Issues found during validation

1. `scripts/check_runtime.py` initially failed because `pn2_ext` and
   `dgcnn_ext` were not installed in the Conda environment.
2. `scripts/install_extensions.sh` attempted to upgrade `pip`,
   `setuptools`, and `wheel` before building extensions. In an offline or
   restricted environment this caused repeated network retries before any
   local build step began.
3. Building the CUDA extensions in a GPU-less environment initially failed in
   PyTorch's extension helper with:

   `IndexError: list index out of range`

   The failure came from missing `TORCH_CUDA_ARCH_LIST`, so PyTorch could not
   infer CUDA architecture flags automatically.
4. `scripts/check_runtime.py` could not import package fallback modules when
   launched from `scripts/` because the project root was not on `sys.path`.

### Fixes applied

- `scripts/install_extensions.sh`
  - Removed the mandatory network-dependent build-tool upgrade step.
  - Added explicit local build-tool checks.
  - Added a default `TORCH_CUDA_ARCH_LIST` for GPU-less build hosts.
- `multi_model/utils/pn2_utils/setup.py`
  - Added a default `TORCH_CUDA_ARCH_LIST` when the environment variable is
    unset.
- `multi_model/utils/pn2_utils/functions/setup.py`
  - Added the same default `TORCH_CUDA_ARCH_LIST`.
- `multi_model/utils/pn2_utils/function.py`
  - Added fallback import logic so in-tree compiled extensions can be loaded.
- `multi_model/utils/pn2_utils/functions/gather_knn.py`
  - Added matching fallback import logic for `dgcnn_ext`.
- `scripts/check_runtime.py`
  - Added package fallback imports for the extensions.
  - Added project-root path setup so package fallback imports work reliably.

### Commands run

#### 1. Syntax check

```bash
/home/ray/miniconda3/bin/conda run -n regnet python -m py_compile \
  train.py test.py utils.py vis/vis_grasp.py \
  scripts/check_runtime.py \
  multi_model/utils/pn2_utils/function.py \
  multi_model/utils/pn2_utils/functions/gather_knn.py \
  multi_model/utils/pn2_utils/setup.py \
  multi_model/utils/pn2_utils/functions/setup.py
```

Result: passed for the Python files above.

#### 2. In-tree CUDA extension build

```bash
/home/ray/miniconda3/bin/conda run -n regnet bash -lc \
  'cd /home/ray/work/regnet/REGNet_for_3D_Grasping/multi_model/utils/pn2_utils && python setup.py build_ext --inplace'
```

Result: `pn2_ext` built successfully.

```bash
/home/ray/miniconda3/bin/conda run -n regnet bash -lc \
  'cd /home/ray/work/regnet/REGNet_for_3D_Grasping/multi_model/utils/pn2_utils/functions && python setup.py build_ext --inplace'
```

Result: `dgcnn_ext` built successfully.

Notes:

- The build emitted a CUDA 12.0 vs PyTorch CUDA 12.1 minor-version mismatch
  warning.
- The build still completed successfully.

#### 3. Runtime import verification

```bash
/home/ray/miniconda3/bin/conda run -n regnet python scripts/check_runtime.py
```

Result: passed.

#### 4. Smoke validation command

```bash
/home/ray/miniconda3/bin/conda run --no-capture-output -n regnet bash -lc \
  "export LD_LIBRARY_PATH=/home/ray/miniconda3/envs/regnet/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH && \
   export PYTHONPATH=/home/ray/work/regnet/REGNet_for_3D_Grasping:/home/ray/work/regnet/REGNet_for_3D_Grasping/multi_model/utils/pn2_utils:/home/ray/work/regnet/REGNet_for_3D_Grasping/multi_model/utils/pn2_utils/functions:$PYTHONPATH && \
   python /home/ray/work/regnet/REGNet_for_3D_Grasping/train.py \
     --cuda \
     --gpu 0 \
     --gpu-num 1 \
     --gpus 0 \
     --mode validate_score \
     --tag smoke_json_metrics_tiny \
     --data-path /tmp/regnet_tiny_dataset \
     --load-score-path /home/ray/work/regnet/REGNet_for_3D_Grasping/assets/models/regnet_train/score_119.model"
```

Observed output:

- Construct network successfully
- `validate_score` started successfully
- One validation batch completed
- Reported loss: `0.013489`

### Current validation status

- Dependency import check: passed
- CUDA extension compilation: passed
- Runtime extension import: passed
- Minimal `validate_score` smoke run: passed

### Still pending

- Re-run the extension install flow directly into a fresh environment instead
  of relying on in-tree builds
- Validate region/refine paths in the same staged way
- Validate `test.py` and `vis/vis_grasp.py` with publishable example inputs or
  documented external inputs
- Rewrite `README.md` so it reflects the verified commands above

### Follow-up check on train.py modes

After the initial smoke validation, `validate_region` and `validate` were
re-checked in the current session.

Observed issue:

- The project can compile CUDA extensions on a host without a visible GPU, but
  runtime execution of PointNet++ / DGCNN operators still requires CUDA
  tensors.
- Before the fix, `train.py` would silently fall back to CPU and then fail
  later inside the CUDA extension with:

  `RuntimeError: points must be a CUDA tensor`

Fix applied:

- `train.py` now fails fast for training / validation / test modes when
  `torch.cuda.is_available()` is false.
- `RefineModule.train_val` no longer swallows exceptions through a bare
  `except:` block. It now raises a contextual error that includes mode, batch,
  and sample path.

Current implication:

- Stage-3 hardening improved: the CLI now reports the real environment
  requirement at startup instead of failing deep inside the model.
- Stage-4 verification is still only partially complete in this session,
  because `validate_region` and `validate` require an actually GPU-enabled
  runtime to complete end-to-end.

### External GPU end-to-end verification

An external GPU runtime was later used to finish the remaining stage-4 checks.

Environment confirmation:

- `nvidia-smi -L` reported `NVIDIA GeForce RTX 4060 Laptop GPU`
- `conda run -n regnet python -c "import torch; ..."` reported:
  - `torch.cuda.is_available() == True`
  - `torch.cuda.device_count() == 1`

Important note:

- A `bash -lc` wrapper that manually exported `LD_LIBRARY_PATH` caused
  `torch.cuda.is_available()` to become false in one subprocess chain.
- Direct `conda run --no-capture-output -n regnet python ...` invocation was
  stable and is the preferred form for README commands.

#### 5. `validate_region` on GPU

Command:

```bash
/home/ray/miniconda3/bin/conda run --no-capture-output -n regnet python \
  /home/ray/work/regnet/REGNet_for_3D_Grasping/train.py \
  --cuda \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --mode validate_region \
  --tag smoke_region_tiny \
  --data-path /tmp/regnet_tiny_dataset \
  --load-score-path /home/ray/work/regnet/REGNet_for_3D_Grasping/assets/models/regnet_train/score_119.model \
  --load-region-path /home/ray/work/regnet/REGNet_for_3D_Grasping/assets/models/regnet_train/region_119.model
```

Result:

- Passed end-to-end on GPU
- Final reported metrics from the run included:
  - `stage2 total_vgr: 0.75`
  - `stage2 total_score: 0.18816813826560974`

#### 6. Full `validate` on GPU

First GPU run exposed a real refine-stage bug:

- File: `multi_model/gripper_region_network.py`
- Problem: the score-filtered refine metrics reused the `class_select` target
  length, which caused a tensor-size mismatch when `len(score_select)` was
  smaller than `len(class_select)`.
- Observed failure:

  `RuntimeError: The size of tensor a (35) must match the size of tensor b (7)`

Fix applied:

- Split the class-selected and score-selected metric computations so each uses
  its own target length.

Re-run command:

```bash
/home/ray/miniconda3/bin/conda run --no-capture-output -n regnet python \
  /home/ray/work/regnet/REGNet_for_3D_Grasping/train.py \
  --cuda \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --mode validate \
  --tag smoke_refine_tiny \
  --data-path /tmp/regnet_tiny_dataset \
  --load-score-path /home/ray/work/regnet/REGNet_for_3D_Grasping/assets/models/regnet_train/score_119.model \
  --load-region-path /home/ray/work/regnet/REGNet_for_3D_Grasping/assets/models/regnet_train/region_119.model
```

Result after the fix:

- Passed end-to-end on GPU
- Final reported metrics from the run included:
  - `stage2 total_vgr: 0.5`
  - `stage3_class total_vgr: 1.0`
  - `stage3_class_stage2 total_vgr: 0.3333333333333333`
  - `stage3_score total_vgr: 1.0`

#### 7. `test.py` on GPU

Command:

```bash
/home/ray/miniconda3/bin/conda run --no-capture-output -n regnet python \
  /home/ray/work/regnet/REGNet_for_3D_Grasping/test.py \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --folder-name /home/ray/work/regnet/REGNet_for_3D_Grasping/test_file/virtual_data \
  --file-name 00001_view_1.p \
  --load-score-path /home/ray/work/regnet/REGNet_for_3D_Grasping/assets/models/regnet_train/score_119.model \
  --load-region-path /home/ray/work/regnet/REGNet_for_3D_Grasping/assets/models/regnet_train/region_119.model
```

Result:

- Passed end-to-end
- Prediction file written to:
  `/home/ray/work/regnet/REGNet_for_3D_Grasping/test_file/virtual_data_predict/00001_view_1.p`
- With default `--table-height 0.75`, all grasps were filtered by the table
  height check for this virtual sample.

Follow-up command with a more suitable virtual-scene table height:

```bash
/home/ray/miniconda3/bin/conda run --no-capture-output -n regnet python \
  /home/ray/work/regnet/REGNet_for_3D_Grasping/test.py \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --table-height 0.5 \
  --folder-name /home/ray/work/regnet/REGNet_for_3D_Grasping/test_file/virtual_data \
  --file-name 00001_view_1.p \
  --load-score-path /home/ray/work/regnet/REGNet_for_3D_Grasping/assets/models/regnet_train/score_119.model \
  --load-region-path /home/ray/work/regnet/REGNet_for_3D_Grasping/assets/models/regnet_train/region_119.model
```

Result:

- Passed end-to-end
- Non-zero collision-filtered grasps were produced:
  - `stage2 grasp num: 7`
  - `stage3 grasp num: 23`
  - `stage3 grasp num (with scorethre): 1`

### Updated status

- Dependency import check: passed
- CUDA extension compilation: passed
- Runtime extension import: passed
- Minimal `validate_score` smoke run: passed
- `validate_region` end-to-end on GPU: passed
- Full `validate` end-to-end on GPU: passed after one refine-stage bug fix
- `test.py` end-to-end on GPU: passed

## 2026-04-07 Fresh-clone audit on `origin/v2`

Goal:

- Re-check the public `v2` branch from a fresh clone under `/tmp/regnet_v2_audit`
- Follow the README flow with external tiny data and pretrained checkpoints

### 1. Fresh clone

Command:

```bash
git clone --depth 1 --branch v2 https://github.com/Rayaller/REGNet_for_3D_Grasping.git /tmp/regnet_v2_audit
```

Result:

- Passed
- Fresh clone HEAD matched `8012bc8`

### 2. Extension install from README

Command:

```bash
cd /tmp/regnet_v2_audit
/home/ray/miniconda3/bin/conda run -n regnet bash scripts/install_extensions.sh
```

Initial result:

- Failed on the build-tool probe before compilation
- Root cause: `setuptools` import hit the `_distutils_hack` assertion in the
  validated Conda Python 3.8 environment, even though `pip install` itself
  could still work with `SETUPTOOLS_USE_DISTUTILS=stdlib`

Fix applied:

- `scripts/install_extensions.sh` now retries the build-tool probe with
  `SETUPTOOLS_USE_DISTUTILS=stdlib` and exports that compatibility mode for
  the subsequent extension build only when the initial probe fails

Result after the fix:

- Passed end-to-end from the fresh clone
- `pn2_ext` and `dgcnn_ext` were both rebuilt successfully

### 3. Runtime check from README

Command:

```bash
cd /tmp/regnet_v2_audit
/home/ray/miniconda3/bin/conda run -n regnet python scripts/check_runtime.py
```

Result:

- Passed
- `torch.cuda.is_available(): True`
- `pn2_ext` import: passed
- `dgcnn_ext` import: passed

### 4. Full validate from the fresh clone

Command:

```bash
cd /tmp/regnet_v2_audit
/home/ray/miniconda3/bin/conda run --no-capture-output -n regnet python train.py \
  --cuda \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --mode validate \
  --tag fresh_clone_validate_tiny \
  --data-path /tmp/regnet_tiny_dataset \
  --load-score-path /home/ray/work/regnet/REGNet_for_3D_Grasping/assets/models/regnet_train/score_119.model \
  --load-region-path /home/ray/work/regnet/REGNet_for_3D_Grasping/assets/models/regnet_train/region_119.model
```

Result:

- Passed end-to-end on GPU from the fresh clone
- Final reported metrics included:
  - `stage2 total_vgr: 0.5`
  - `stage3_class_stage2 total_vgr: 0.5`

### 5. `test.py` from the fresh clone

Setup:

- Copied `/home/ray/work/regnet/REGNet_for_3D_Grasping/test_file/virtual_data/00001_view_1.p`
  to `/tmp/regnet_v2_virtual_data/00001_view_1.p`

Command:

```bash
cd /tmp/regnet_v2_audit
/home/ray/miniconda3/bin/conda run --no-capture-output -n regnet python test.py \
  --gpu 0 \
  --gpu-num 1 \
  --gpus 0 \
  --table-height 0.5 \
  --folder-name /tmp/regnet_v2_virtual_data \
  --file-name 00001_view_1.p \
  --load-score-path /home/ray/work/regnet/REGNet_for_3D_Grasping/assets/models/regnet_train/score_119.model \
  --load-region-path /home/ray/work/regnet/REGNet_for_3D_Grasping/assets/models/regnet_train/region_119.model
```

Initial result:

- Inference ran, but saving failed with:
  `FileNotFoundError: /tmp/regnet_v2_virtual_data_predict/00001_view_1.p`
- Root cause: `utils.eval_notruth()` assumed the `_predict` output directory
  already existed

Fix applied:

- `utils.eval_notruth()` now creates the parent directory before writing the
  prediction artifact
- It now also prints a save confirmation with the final file size

Result after the fix:

- Passed end-to-end from the fresh clone
- Prediction file written to:
  `/tmp/regnet_v2_virtual_data_predict/00001_view_1.p`
- Saved file size reported by the run: `1186005 bytes`

### Fresh-clone audit status

- Fresh clone of `origin/v2`: passed
- README extension installation path: passed after compatibility fix
- README runtime check: passed
- README full validate path: passed
- README single-file inference path: passed after output-directory fix
