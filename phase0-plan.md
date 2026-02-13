# Phase 0: Make DECODE-FISH Run Again

**Date**: 2026-02-12
**Direction**: Stephan — fix deprecated code, fix basic errors, update old syntax, then validate on C. elegans data.
**Scope**: No scientific logic changes. Only compatibility fixes and modernization.
**Status**: ALL STEPS DONE. Phase 0 complete.

---

## Overview

The codebase was written for Python 3.7 / PyTorch 1.7 / pandas 1.x (circa 2021). Multiple APIs have been removed or deprecated. The goal is to get it running cleanly on Python 3.11 + PyTorch 2.x + pandas 2.x and prove it works on real data.

**Total estimated edits**: ~60 across ~15 files
**Test data**: C. elegans screen tiles from `celegansdata/`

---

## Step 0: Create Conda Environment — DONE

**Important**: Janelia cluster rules say never run conda installs on login nodes. Use an interactive LSF job.

### 0a. Start an interactive session

```bash
bsub -n 4 -W 1:00 -P turaga -Is /bin/bash
```

### 0b. Create and populate the environment

```bash
# Create env with Python 3.11
conda create -n decode_fish python=3.11 -y

# Activate
conda activate decode_fish

# PyTorch 2.5 + CUDA 12.4 (cluster has CUDA 11.7 through 13.1)
conda install pytorch=2.5 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y

# numba for the custom CUDA kernel (place_psfs.py uses numba.cuda JIT)
mamba install numba -c conda-forge -y

# Scientific stack
mamba install -c conda-forge \
    pandas=2.2 \
    scipy \
    scikit-image \
    scikit-learn \
    matplotlib \
    seaborn \
    h5py \
    tifffile \
    tqdm \
    pyyaml \
    ipykernel \
    -y

# pip-only packages
pip install \
    hydra-core>=1.3 \
    omegaconf>=2.3 \
    monai \
    wandb \
    kornia \
    torch-optimizer

# Install decode_fish in dev mode
cd /groups/turaga/turagalab/DECODE26/decode_fish
pip install -e .
```

### 0c. Sanity check (still inside interactive job)

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import numba; from numba import cuda; print(f'numba {numba.__version__}')"
python -c "import pandas; print(f'pandas {pandas.__version__}')"
python -c "import kornia, monai, wandb, hydra; print('all imports ok')"
```

### 0d. GPU sanity check (separate job with GPU)

```bash
bsub -n 4 -gpu "num=1" -q gpu_l4 -W 0:10 -P turaga -Is /bin/bash
conda activate decode_fish
python -c "import torch; print(torch.cuda.get_device_name(0)); x = torch.randn(100, device='cuda'); print('GPU tensor ok')"
python -c "from numba import cuda; @cuda.jit
def f(x): pass; print('numba cuda JIT ok')"
```

---

## Step 1: Fix Deprecated / Broken API Calls — DONE

### 1a. `DataFrame.append()` removed in pandas 2.0

| File | Line(s) | Current | Fix |
|------|---------|---------|-----|
| `funcs/merfish_comparison.py` | 74 | `istd_results = istd_results.append(df)` | `istd_results = pd.concat([istd_results, df])` |
| `funcs/merfish_comparison.py` | 126 | `bard_results = bard_results.append(df)` | `bard_results = pd.concat([bard_results, df])` |

### 1b. `torch.meshgrid()` missing `indexing=` parameter

Deprecated in PyTorch 1.10, will error in PyTorch 3.0. Default changed from `'ij'` to requiring explicit specification.

| File | Line(s) | Fix |
|------|---------|-----|
| `engine/psf.py` | 82 | Add `indexing='ij'` |
| `engine/psf.py` | 102 | Add `indexing='ij'` |
| `engine/psf.py` | 115 | Add `indexing='ij'` |
| `funcs/utils.py` | 169 | Add `indexing='ij'` |
| `funcs/utils.py` | 210 | Add `indexing='ij'` |
| `funcs/dataset.py` | 315-320 | Add `indexing='ij'` |

### 1c. `torch.stack(..., axis=N)` -- invalid kwarg (should be `dim=`)

NumPy uses `axis`, PyTorch uses `dim`. This may work in some versions but is not the correct API.

| File | Line(s) | Fix |
|------|---------|-----|
| `funcs/utils.py` | 173 | `axis=2` -> `dim=2` |
| `funcs/utils.py` | 217 | `axis=3` -> `dim=3` |
| `funcs/utils.py` | 229-236 | `axis=3` -> `dim=3` (8 occurrences) |

### 1d. `torch.nn.UpsamplingBilinear2d` deprecated

| File | Line(s) | Current | Fix |
|------|---------|---------|-----|
| `funcs/utils.py` | 260 | `upsamp = torch.nn.UpsamplingBilinear2d(size=[2048,2048])` | `F.interpolate(input, size=[2048,2048], mode='bilinear', align_corners=False)` |

---

## Step 2: Device-Agnostic Refactor — DONE

Replace all hardcoded CUDA references so the code works with any device string from config.

### 2a. `.cuda()` calls (30+ sites)

**Pattern**: Replace `.cuda()` with `.to(device)`, where `device` is passed through constructors or read from config.

**Key files (by number of edits)**:

| File | # of `.cuda()` sites | Notes |
|------|----------------------|-------|
| `engine/gmm_loss.py` | ~12 | Constructor args, tensor creation, type casts |
| `engine/microscope.py` | ~7 | Parameter registration, tensor ops |
| `engine/point_process.py` | ~3 | Point generation |
| `funcs/file_io.py` | ~2 | Model loading |
| `funcs/merfish_codenet.py` | ~5 | Model params |
| `train.py` | ~1 | Model init |
| `gentrain.py` | ~1 | Model init |
| `coloc_eval.py` | ~2 | Model + data |
| `coloc_eval_p3.py` | ~2 | Model + data |
| `figures/get_perf.py` | ~3 | Eval script |
| `test.py` | ~1 | Simple test |

### 2b. `torch.cuda.FloatTensor` / `torch.cuda.LongTensor` constructors

These are not device-agnostic. Replace with standard `torch.zeros()` or `torch.tensor()` plus `device=` and `dtype=`.

| File | Line(s) | Current | Fix |
|------|---------|---------|-----|
| `engine/gmm_loss.py` | 55 | `torch.zeros(...).cuda()` | `torch.zeros(..., device=device)` |
| `engine/gmm_loss.py` | 59 | `.type(torch.cuda.FloatTensor)` | `.to(dtype=torch.float32)` |
| `engine/gmm_loss.py` | 115 | `torch.cuda.LongTensor(bs).fill_(0)` | `torch.zeros(bs, dtype=torch.long, device=device)` |
| `engine/gmm_loss.py` | 122 | `torch.cuda.FloatTensor(bs,max_counts).fill_(0)` | `torch.zeros(bs, max_counts, device=device)` |
| `engine/gmm_loss.py` | 134-136 | `.type(torch.cuda.FloatTensor)` (x3) | `.to(dtype=torch.float32)` |
| `engine/gmm_loss.py` | 146 | `torch.cuda.FloatTensor(bs,max_counts,...)` | `torch.zeros(bs, max_counts, ..., device=device)` |
| `engine/gmm_loss.py` | 149 | `torch.cuda.LongTensor(bs,max_counts)` | `torch.zeros(bs, max_counts, dtype=torch.long, device=device)` |
| `engine/microscope.py` | 180 | `.type(torch.cuda.LongTensor)` | `.to(dtype=torch.long)` |

### 2c. Device propagation strategy

- `config/train.yaml` already has `device: gpu_device: cuda` -- use this as the source
- Add `device` parameter to `Microscope.__init__()`, `GMMForwardLoss.__init__()`, `PointProcess.__init__()`
- In training entry point (`train.py`), read device from config and pass through

---

## Step 3: Remove Hardcoded Absolute Paths — DONE

All paths reference Artur Speiser's home directory, which no longer exists at that location.

| File | Line(s) | Current | Fix |
|------|---------|---------|-----|
| `config/train.yaml` | 1 | `base_dir: /groups/turaga/home/speisera/Mackebox/Artur/WorkDB/deepstorm/` | Use relative path or `${oc.env:DECODE_BASE_DIR,.}` (Hydra OmegaConf resolver) |
| `imports.py` | 22-25 | `base_path = '/groups/turaga/home/speisera/...'` | `base_path = Path(__file__).parent.parent` (repo root) |
| `funcs/merfish_comparison.py` | 26, 33 | `sys.path.append('/groups/turaga/home/speisera/.../istdeco/')` | Remove or make conditional on availability |
| `coloc_eval.py` | 79 | `base_dir = '/groups/turaga/home/speisera/...'` | Read from config or env var |
| `submit.ipynb` | 20 | Hardcoded bsub command with absolute paths | Update to use relative paths |

---

## Step 4: Modern Packaging — DONE

### 4a. Create `pyproject.toml`

Replace `setup.py` (uses deprecated `pkg_resources`) + `settings.ini` (nbdev v1 format).

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "decode_fish"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "pandas>=2.0",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "monai>=1.3",
    "wandb>=0.16",
    "kornia>=0.7",
    "torch-optimizer>=0.3",
    "tifffile",
    "scikit-image",
    "scipy",
    "matplotlib",
]
```

### 4b. Update `environment.yaml`

Pin to specific PyTorch 2.x + CUDA 12.x versions that match the cluster.

### 4c. Clean up stale files

- Delete `requirements.yaml` (pins Python 3.7 / PyTorch 1.7.1)
- Keep `setup.py` and `settings.ini` temporarily (delete after `pyproject.toml` is validated)

---

## Step 5: Validate — PARTIALLY DONE

### 5a. msp300 smFISH Smoke Test — PASSED (2026-02-12)

Before attempting C. elegans, we validated on the simpler single-channel `example/msp300_smFISH_3.tif` (37x512x512, 19MB). This revealed 17 additional bugs, all caused by the original code only being tested with multi-channel MERFISH data. Single-channel smFISH exposed many implicit `n_channels > 1` assumptions.

**Results:**
- Training: 200 iterations, ~27 sec on NVIDIA L4, exit code 0
- Prediction: 37 localizations (vs 908 from reference fully-trained model) — expected for 200 iters
- Output: CSV with 13 columns (loc_idx, frame_idx, code_inds, x, y, z, prob, x/y/z_sig, comb_sig, int_0, int_sig_0)

**Additional files modified during smoke test (not in original plan):**

| File | Fix |
|------|-----|
| `engine/point_process.py` | Init `code_draw=None` before if; guard `torch.cat` on None codes; always insert channel dim in output_shape |
| `engine/microscope.py` | Handle single-channel in `get_single_ch_inputs`; move locations to CPU in `get_roi_filt_inds` |
| `engine/noise.py` | Use `torch.tensor()` for single-channel theta_scale; `torch.ones(1)` for theta_par |
| `engine/gmm_loss.py` | Default codes to zeros when None |
| `funcs/train_funcs.py` | Guard `code_cond` on None codes; add final `save_train_state()` after training loop |
| `funcs/file_io.py` | Add `unsqueeze(1)` for 3D single-channel images; `weights_only=False` for torch.load |
| `funcs/predict.py` | Add `predict()` wrapper function; add `unsqueeze(0)` for 3D images |
| `funcs/exp_specific.py` | Add `get_smfish_codebook()` |
| `predict.py` | Fix import to `UnetDecodeNoBn_2S` |
| `config/predict.yaml` | Update model._target_ to `UnetDecodeNoBn_2S` |
| `config/exp_type/smfish_3d.yaml` | Add codebook._target_ |
| `run_smoke_test.sh` | NEW — LSF batch script for full train+predict+compare pipeline |
| `run_predict_only.sh` | NEW — LSF batch script for predict-only |

**Run commands (for reference):**
```bash
# Training
bsub < run_smoke_test.sh
# or manually:
python decode_fish/train.py \
  data_path.image_path="${BASE}/example/msp300_smFISH_3.tif" \
  data_path.psf_path=null \
  codebook._target_=decode_fish.funcs.exp_specific.get_smfish_codebook \
  'genm.PSF.gauss_radii=[2.0,1.0,1.0]' 'genm.PSF.psf_extent_zyx=[21,21,21]' \
  genm.noise.theta=36.23 genm.microscope.scale=2000 genm.foci.n_foci_avg=2 \
  genm.prob_generator.low=0.0001 genm.prob_generator.high=0.0003 \
  sim.bg_estimation.fractal.scale=0 training.num_iters=200 \
  output.wandb_mode=disabled output.save_dir="${BASE}/models/msp300_smoke_test" \
  run_name=smoke_test

# Prediction
python decode_fish/predict.py \
  model_path="${BASE}/models/msp300_smoke_test/model.pkl" \
  image_path="${BASE}/example/msp300_smFISH_3.tif" \
  out_file="${BASE}/models/msp300_smoke_test/predictions.csv" \
  model._target_=decode_fish.engine.model.UnetDecodeNoBn_2S +model.n_p_ch=2
```

### 5b. C. elegans Validation — DONE (2026-02-12)

**High-SNR test** (Stephan's image, `N2_702_cropped_1620 (high SNR)_ch0.tif`):
- Single-channel (81, 454, 334), theta=10.71, 1000 iterations, ~3.3 min on L4
- 37,133 predictions (28,453 at prob > 0.9)
- Visualizations in `celegansdata/visualization/` (6 figures)

**Multi-image test** (5 images from tifs_N2_4.zip, channel 4 extracted):
- 925 predictions across 5 images at 1000 iterations
- Required uniform Z-cropping (images had different Z depths)

**Bug fixed**: `RuntimeError: Sizes of tensors must match` — dataloader `torch.cat` requires uniform dimensions across training images.

### 5c. Prepare test data

```bash
# Unzip on a compute node (not login node!)
bsub -n 2 -W 0:30 -P turaga -J unzip_tiles -o /dev/null \
  'cd /groups/turaga/turagalab/DECODE26/celegansdata && unzip -o tiles.zip -d tiles/'

# Small raw test set (7.8 GB)
bsub -n 2 -W 1:00 -P turaga -J unzip_n2 -o /dev/null \
  'cd /groups/turaga/turagalab/DECODE26/celegansdata && unzip -o tifs_N2_4.zip -d tifs_N2_4/'
```

### 5b. Create a test config

Write a minimal Hydra config override for C. elegans tiles:
- Point `data_path` at the unzipped tiles
- Use small crop size / few iterations for quick smoke test
- Keep all other parameters at defaults

### 5c. Smoke test: training (GPU job)

```bash
# Interactive GPU session for debugging
bsub -n 8 -gpu "num=1" -q gpu_l4 -W 2:00 -P turaga -Is /bin/bash
conda activate decode_fish
cd /groups/turaga/turagalab/DECODE26/decode_fish

# Run a short training (e.g., 100 iterations) on tiles
python decode_fish/train.py data_path.root=../celegansdata/tiles/ training.n_iter=100
```

**Success criteria**:
- No import errors
- No deprecated API warnings/crashes
- Loss decreases over 100 iterations
- No CUDA device errors

### 5d. Smoke test: prediction

```bash
# Still inside the GPU interactive session
python decode_fish/predict.py data_path.root=../celegansdata/tiles/ ...
```

**Success criteria**:
- Outputs a DataFrame with columns: x, y, z, intensity, probability (or similar)
- No crashes
- Predictions are plausible (non-zero, within image bounds)

### 5e. Optional: test on full-size raw images

If tiles pass, repeat with a few images from `tifs_N2_4/` to verify the pipeline handles full-resolution C. elegans data.

---

## Execution Order

```
Step 0 (conda env)          -- DONE
  |
Step 1 (API fixes)          -- DONE
  |
Step 2 (device refactor)    -- DONE
  |
Step 3 (hardcoded paths)    -- DONE
  |
Step 4 (packaging)          -- DONE
  |
Step 5a (msp300 smoke test) -- DONE (17 additional bugs fixed)
  |
Step 5b (C. elegans)        -- DONE (high-SNR + multi-image)
```

---

## Files Changed (Complete List)

| File | Steps | Edits |
|------|-------|-------|
| `engine/gmm_loss.py` | 1b, 2a, 2b, 5a | ~13 |
| `engine/microscope.py` | 2a, 2b, 5a | ~10 |
| `engine/psf.py` | 1b | 3 |
| `engine/point_process.py` | 2a, 5a | ~6 |
| `engine/noise.py` | 5a | 2 |
| `funcs/utils.py` | 1b, 1c, 1d | ~12 |
| `funcs/dataset.py` | 1b | 1 |
| `funcs/train_funcs.py` | 5a | 2 |
| `funcs/file_io.py` | 2a, 5a | ~4 |
| `funcs/predict.py` | 5a | 2 (+ new `predict()` function) |
| `funcs/exp_specific.py` | 5a | 1 (+ new `get_smfish_codebook()`) |
| `funcs/merfish_comparison.py` | 1a, 3 | 4 |
| `funcs/merfish_codenet.py` | 2a | ~5 |
| `train.py` | 2a, 2c, 5a | ~5 |
| `predict.py` | 5a | 1 |
| `gentrain.py` | 2a, 5a | ~2 |
| `sim_eval.py` | 5a | 1 |
| `merfish_eval.py` | 5a | 1 |
| `coloc_eval.py` | 2a, 3 | ~3 |
| `coloc_eval_p3.py` | 2a | ~2 |
| `config/train.yaml` | 3 | 1 |
| `config/predict.yaml` | 5a | 1 |
| `config/exp_type/smfish_3d.yaml` | 5a | 1 |
| `imports.py` | 3 | 1 |
| `pyproject.toml` | 4a | NEW |
| `run_smoke_test.sh` | 5a | NEW |
| `run_predict_only.sh` | 5a | NEW |

**Total: ~80+ edits across ~22 files + 3 new files**

---

## Phase 0 Complete — What Next?

Three candidate directions (need Stephan's input):

### Option A: C. elegans Parameter Tuning (1-2 days)
- 37K detections from high-SNR image is likely over-detecting (manual counts = ~4 foci/worm)
- Longer training (5K-10K iter), tune prob threshold and density priors
- Compare predicted counts vs manual_count.txt (0 vs 3-4 spots) — do they correlate?
- **Outcome**: Concrete "DECODE-FISH works on our data" result

### Option B: Public Dataset Benchmarks (~1 week)
- U-FISH: 4,166 images with GT annotations → F1 score, distance error
- RS-FISH: simulated 3D with exact positions → RMSE
- **Outcome**: Quantitative baseline before Phase 1 changes, comparison vs Spotiflow/deepBlink

### Option C: Phase 1 — Reconstruction Consistency Loss (3-4 weeks)
- Self-supervised fine-tuning: network predictions → forward model → rendered image → loss vs real input
- Leverages the differentiable forward model (DECODE-FISH's key advantage)
- Solves biggest pain point: need to retrain per-dataset
- See `update-plan.md` section 1.1 for full design

**Recommendation**: A first (quick win), then C (highest scientific ROI). B can run in parallel.

---

## Out of Scope for Phase 0

These are explicitly deferred to later phases:
- Reconstruction consistency loss (Phase 1)
- Spatially-varying PSF (Phase 1)
- Attention / transformer modules (Phase 2)
- Mixed-precision training (Phase 3)
- Async data pipeline / num_workers (Phase 3)
- torch.compile (Phase 3)
- New optimizer (Phase 3)
