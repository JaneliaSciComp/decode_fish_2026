# DECODE-FISH Modernization Plan

**Date**: 2026-02-07
**Author**: Max (with codebase audit by Claude)
**Scope**: Revitalize DECODE-FISH from a 2021 research prototype into a modern, maintainable tool

---

## Executive Summary

DECODE-FISH's core idea — **analysis by synthesis** using a differentiable physics simulator to train a neural network for the inverse problem — was ahead of its time and remains sound. The differentiable microscope forward model, the GMM loss, and the joint PSF/network optimization loop are genuine strengths that most newer methods still don't fully replicate.

However, the implementation has accumulated significant technical debt. The codebase has critical compatibility issues that prevent it from running on modern dependency stacks, the neural network architecture is dated, and the data pipeline has obvious throughput bottlenecks. Modernization should preserve the differentiable physics core while upgrading everything around it.

This plan is organized into three phases:
1. **Phase 0 — Triage** (make it run again, ~1-2 weeks)
2. **Phase 1 — Close the Reality Gap** (highest scientific ROI, ~3-4 weeks)
3. **Phase 2 — Architectural Modernization** (performance + accuracy, ~4-6 weeks)
4. **Phase 3 — Throughput & Usability** (production readiness, ~2-3 weeks)

---

## Current State: What's Solid vs. What's Broken

### Still Solid (Keep)

| Component | Assessment |
|---|---|
| **Differentiable forward model** (`engine/microscope.py`) | Fully differentiable end-to-end. Custom CUDA kernel (`CudaPlaceROI`) for PSF placement with correct gradients. This is the project's crown jewel. |
| **GMM loss** (`engine/gmm_loss.py`) | Elegant probabilistic loss combining count loss (Gaussian approx to sum of Bernoullis) + localization loss (weighted log-sum-exp over Gaussian likelihoods). Still state-of-the-art for this type of problem. |
| **Joint PSF/network optimization** | The alternating optimization of network params, microscope params, and PSF params is a genuine differentiator. Most competing methods don't do this. |
| **Learnable noise model** (`engine/noise.py`) | Gamma noise with learnable theta — correct physics for EMCCD cameras, differentiable. |
| **ISI post-processing** (`funcs/output_trafo.py`) | Iterative local-maxima detection handles overlapping emitters well. |
| **Hydra config system** | Hierarchical YAML configs are flexible and well-structured. |

### Outdated / Broken

| Component | Severity | Issue |
|---|---|---|
| **`pandas.DataFrame.append()`** | CRITICAL | Removed in pandas 2.0. Code will crash. (`funcs/merfish_comparison.py:74,126`) |
| **`torch.meshgrid()` without `indexing=`** | HIGH | Deprecated in PyTorch 2.0, will break in 3.0. (`engine/psf.py:82,102,115`, `funcs/utils.py:169,210`) |
| **`torch.stack(axis=N)` instead of `dim=N`** | HIGH | Invalid PyTorch API. (`funcs/utils.py:173,217,229-236`) |
| **Hardcoded `.cuda()` everywhere** | HIGH | Not device-agnostic. (`engine/microscope.py:85-88,132-133,221`, `engine/gmm_loss.py:31-33,55,59,115-116,146-149`) |
| **Hardcoded absolute paths** | HIGH | `/groups/turaga/home/speisera/...` in `config/train.yaml:1` and `imports.py:22-25`. Will break on any other machine. |
| **`DataLoader(num_workers=0)`** | MEDIUM | Synchronous data loading — the pipeline is CPU-bound during I/O. (`funcs/file_io.py:106`) |
| **No mixed-precision training** | MEDIUM | No `torch.cuda.amp` / `autocast` anywhere. Leaving ~2x throughput on the table. |
| **nbdev v1** | MEDIUM | Deprecated. CI/CD uses Python 3.6 with v1 commands. |
| **`setup.py` with `pkg_resources`** | MEDIUM | Deprecated packaging. No `pyproject.toml`. |
| **CUDA 11.8** | LOW | Works but 12.x is standard. |
| **No tests** | LOW | No test suite beyond nbdev notebook tests. |

### Architecture Limitations

| Limitation | Impact |
|---|---|
| **Spatially fixed PSF** | Same PSF used at every (x,y) position. Real microscopes have field-dependent aberrations, especially at FOV edges. Z-axis precision degrades. |
| **No temporal context** | Each channel processed independently. Misses blinking kinetics that span frames. |
| **Purely convolutional backbone** | Limited receptive field. No global context for background estimation or dense-emitter disambiguation. |
| **No reconstruction consistency loss** | Network is trained only on simulated data. No mechanism to fine-tune on real data without ground truth. |

---

## Phase 0 — Triage: Make It Run (Week 1-2)

**Goal**: Fix deprecated code, fix basic errors, update old syntax/grammar. Then validate end-to-end on a small C. elegans dataset. No scientific logic changes. See `phase0-plan.md` for the detailed execution plan.

**Stephan's direction (2026-02-12)**: Focus on making the code work — fix what's broken, modernize what's deprecated, then prove it runs on real data (C. elegans screen tiles).

### 0.1 Fix Deprecated / Broken API Calls (~30 edits across ~10 files)
- [ ] `DataFrame.append()` → `pd.concat()` (funcs/merfish_comparison.py)
- [ ] `torch.meshgrid()` → add `indexing='ij'` (engine/psf.py, funcs/utils.py, funcs/dataset.py)
- [ ] `torch.stack(..., axis=)` → `dim=` (funcs/utils.py, ~10 occurrences)
- [ ] `torch.nn.UpsamplingBilinear2d` → `F.interpolate()` (funcs/utils.py)

### 0.2 Device-Agnostic Refactor (30+ `.cuda()` call sites)
- [ ] `.cuda()` → `.to(device)` everywhere
- [ ] `torch.cuda.FloatTensor/LongTensor` → `torch.zeros(..., device=device)`
- [ ] Pass `device` from config through constructors

### 0.3 Remove Hardcoded Paths
- [ ] `config/train.yaml` base_dir → relative / env var
- [ ] `imports.py` absolute paths → relative
- [ ] `funcs/merfish_comparison.py` sys.path.append → package imports

### 0.4 Modern Packaging
- [ ] `pyproject.toml` replacing `setup.py` + `settings.ini`
- [ ] Pin dependencies in `environment.yaml`
- [ ] Remove stale `requirements.yaml`

### 0.5 Validate on C. elegans Data
- [ ] Unzip `tiles.zip` (1.2 GB, 337 pre-cropped tiles) as quick test set
- [ ] Unzip `tifs_N2_4.zip` (7.8 GB, 92 images) as small raw test set
- [ ] Run training on a small subset — verify loss converges
- [ ] Run prediction — verify output DataFrame format and basic sanity

---

## Phase 1 — Close the Reality Gap (Week 3-6)

**Goal**: The single highest-ROI improvement. The existing differentiable forward model is already 90% of what's needed — we just need to use it for a reconstruction consistency loss on real data.

### Why This First

Max's email correctly identifies that "running a dedicated simulation-training loop for every dataset is a bottleneck." The current DECODE-FISH trains purely on simulated data generated by its learned forward model. When the simulation doesn't match reality (aberrations, sample-induced scattering, non-uniform background), performance degrades. This is the #1 complaint in practice.

Modern approaches (FD-DeepLoc 2024, LUNAR-family) close this gap. DECODE-FISH already has the infrastructure — we just need to add a self-supervised loss path.

### 1.1 Reconstruction Consistency Loss

```
Files: funcs/train_funcs.py (modify), engine/microscope.py (minor)
New concept: After the network predicts emitter locations from a REAL image,
render those predictions through the forward model and compare to the input.
```

**Training loop modification:**
```
Current:  sim_data → network → predictions → GMM loss (vs. known GT from sim)
Add:      real_data → network → predictions → forward_model → rendered_image → recon_loss (vs. real_data)
```

- [ ] Add `reconstruction_loss()` function: takes network predictions, renders through `Microscope.forward()`, computes L1/L2 loss vs. input
- [ ] Add `training.recon_loss_scale` config parameter
- [ ] Implement alternating schedule: N iterations sim-supervised, M iterations self-supervised on real data
- [ ] Handle gradient flow: `network ← recon_loss` but `stop_gradient(microscope)` during recon pass (don't let reconstruction loss change the forward model — only the network)

### 1.2 Spatially-Varying PSF (Field-Dependent)

```
Files: engine/psf.py (major), engine/microscope.py (moderate)
Inspired by: FD-DeepLoc (2024), Max's "coordinate-aware layers" idea
```

The current `LinearInterpolatedPSF` stores a single `psf_volume` tensor applied identically everywhere. Real microscopes have position-dependent aberrations (coma, astigmatism, field curvature) that worsen toward FOV edges.

**Approach**: Represent the PSF as `PSF(x, y, z) = base_PSF(z) + Σ_k α_k(x,y) · basis_k(z)` where:
- `base_PSF` is the current learned PSF volume (center of FOV)
- `basis_k` are learned perturbation volumes (small, ~5-10 bases)
- `α_k(x,y)` are smooth spatial coefficient maps (learnable, low-resolution, Gaussian-blurred)

This extends the existing architecture minimally:
- [ ] Add `n_spatial_bases` parameter to `LinearInterpolatedPSF`
- [ ] Add `spatial_coefficients` as `nn.Parameter` of shape `[n_bases, H_ds, W_ds]`
- [ ] Add `basis_volumes` as `nn.Parameter` of shape `[n_bases, Z, Y, X]`
- [ ] In `forward()`: interpolate spatial coefficients at emitter (x,y), compute weighted sum of basis perturbations, add to base PSF
- [ ] Regularize: L2 penalty on `basis_volumes`, smoothness penalty on `spatial_coefficients`

**Alternative (simpler)**: Add a Zernike polynomial parameterization (interpretable, fewer parameters, built-in smoothness). But the learned-basis approach is more flexible and keeps the existing `grid_sample` pipeline.

### 1.3 Network Position Conditioning

```
Files: engine/model.py (moderate)
Max's idea: "coordinate-aware layers"
```

Give the network the spatial position within the FOV as additional input channels (CoordConv-style):

- [ ] Add 2-3 coordinate channels to input: normalized (x_pos, y_pos, r_from_center)
- [ ] Modify `UnetDecodeNoBn_2S.__init__()` to accept `n_coord_ch` extra input channels
- [ ] Generate coordinate grids in `forward()` and concatenate with image input
- [ ] This allows the network to learn position-dependent detection/localization behavior without explicit PSF modeling

---

## Phase 2 — Architectural Modernization (Week 7-12)

**Goal**: Bring the neural network architecture to 2025+ standards. This is where Max's transformer/attention ideas come in, but grounded in what the codebase actually needs.

### 2.1 Attention at the Bottleneck

```
Files: engine/model.py (major)
Rationale: Cheapest way to add global context; addresses dense-emitter disambiguation
```

The current `UnetDecodeNoBn_2S` is a depth-2 U-Net. At the bottleneck, feature maps are 4x spatially downsampled (e.g., 48→12 per dim for a crop of 48). A self-attention layer here is cheap and effective:

- [ ] Add `SelfAttention3D` module: reshape (B, C, D, H, W) → (B, D*H*W, C), apply multi-head self-attention, reshape back
- [ ] Insert between encoder and decoder at the deepest level
- [ ] Use 4-8 attention heads, same channel dim as existing f_maps
- [ ] Add positional encoding (sinusoidal or learned) to preserve spatial information
- [ ] **Do NOT replace the full U-Net with a ViT** — the conv encoder/decoder provides necessary locality bias for sub-pixel regression

### 2.2 Temporal Context via Cross-Attention

```
Files: engine/model.py (major), funcs/dataset.py (moderate)
Max's idea: "capture blinking kinetics that the 3-frame window misses"
```

Currently DECODE-FISH processes each channel independently. For SMLM (as opposed to smFISH), temporal blinking patterns across frames are informative.

**Approach**: Process each frame through the per-channel encoder independently, then use cross-attention to aggregate temporal information before the decoder:

- [ ] Modify `DecodeDataset` to return T consecutive frames (configurable, default T=5-9)
- [ ] Encode each frame independently through the shared encoder (weight sharing)
- [ ] At the bottleneck: flatten spatial dims, use cross-attention where the center frame queries all frames
- [ ] Decode from the attended features
- [ ] Fallback: for smFISH (single-channel, no temporal dynamics), this reduces to standard self-attention

### 2.3 Lightweight Inference Backbone (LiteLoc-inspired)

```
Files: engine/model.py (new class), funcs/predict.py (modify)
Rationale: 10-50x faster inference for large-scale datasets
```

- [ ] Add `LiteDecodeNet` class using depthwise-separable 3D convolutions
- [ ] Knowledge distillation: train `LiteDecodeNet` to match `UnetDecodeNoBn_2S` outputs on simulated data
- [ ] Offer as `network._target_: decode_fish.engine.model.LiteDecodeNet` in config
- [ ] Profile: target >100 fps at 256x256 on A100

### 2.4 (Exploratory) Diffusion-Based Denoising as Preprocessor

```
Rationale: Max's idea about "transforming noisy Z-stack into a noiseless probability map"
Assessment: Worth exploring but NOT as a replacement for DECODE's detection pipeline
```

The idea of using a diffusion model to denoise the input before running DECODE is appealing for smFISH where structured background is the main problem. However:

- **Pro**: DDPMs excel at removing structured noise/background while preserving signal. A denoised input would make DECODE's job easier.
- **Con**: Diffusion inference is slow (50-1000 steps). This conflicts with throughput goals.
- **Compromise**: Train a lightweight **flow-matching** model (1-4 steps) or a **consistency model** as a fast denoiser. Use it as an optional preprocessor.

- [ ] Prototype: Train a small U-Net-based consistency model on pairs of (noisy_real, synthetic_clean) smFISH patches
- [ ] Evaluate: Does DECODE performance improve when fed denoised input?
- [ ] If yes: integrate as optional `preprocessing.denoiser` config option

---

## Phase 3 — Throughput & Usability (Week 13-15)

### 3.1 Async Data Pipeline

```
Files: funcs/file_io.py, funcs/dataset.py, funcs/train_funcs.py
```

The current pipeline is synchronous (`num_workers=0`) and fetches data with `next(iter(dl))` inside the training loop. This is a major bottleneck.

- [ ] Set `num_workers=4` (or configurable) in DataLoader
- [ ] Add `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2`
- [ ] Refactor `__getitem__` to avoid GPU operations (move `.to(device)` out of dataset, into training loop)
- [ ] Profile CPU vs GPU utilization; ensure GPU is >90% utilized

### 3.2 Mixed-Precision Training

```
Files: funcs/train_funcs.py
```

- [ ] Wrap forward pass + loss computation in `torch.amp.autocast('cuda')`
- [ ] Use `GradScaler` for stable FP16 gradients
- [ ] Verify: PSF learning and GMM loss are numerically stable in FP16 (the `+ 0.00001` in gmm_loss.py:96 suggests sensitivity — may need to keep loss computation in FP32)

### 3.3 Modern Training Infrastructure

- [ ] Replace manual training loop with PyTorch Lightning or plain `torch.compile()`
- [ ] Add `torch.compile(model)` for inference (free 10-30% speedup on PyTorch 2.x)
- [ ] Replace QHAdam with AdamW (standard, well-understood, no extra dependency on `torch_optimizer`)
- [ ] Add cosine annealing scheduler option (better convergence than StepLR)
- [ ] Add gradient accumulation for effective larger batch sizes

### 3.4 Checkpoint & Reproducibility

- [ ] Save full training state in a single checkpoint (model + microscope + optimizer + scheduler + config + rng state)
- [ ] Add `torch.use_deterministic_algorithms(True)` option
- [ ] Log git hash and full resolved config to wandb

### 3.5 CLI & Documentation

- [ ] Add `--help` descriptions to Hydra configs
- [ ] Write a 1-page quickstart replacing the current sparse README
- [ ] Add type hints to public API functions
- [ ] Remove nbdev dependency (convert to standard Python package with pytest)

---

## Decision Framework: What's the Ultimate Goal?

Max's email asks: "is it for improved speed, more accuracy, or adaptivity across microscopes/tasks?" The answer determines priority ordering:

| Goal | Primary Phase | Key Deliverable |
|---|---|---|
| **Adaptivity** (generalize across microscopes without retraining) | Phase 1 | Reconstruction consistency loss + spatially-varying PSF |
| **Accuracy** (beat current DECODE on challenging high-density data) | Phase 2 | Attention mechanisms + temporal context |
| **Speed** (process massive datasets faster) | Phase 3 | Async pipeline + mixed precision + lightweight backbone |

**Recommendation**: Start with **Phase 0 + Phase 1**. The reconstruction consistency loss is the single highest-ROI change — it leverages the existing differentiable forward model (DECODE-FISH's key advantage) while solving the biggest practical pain point (per-dataset retraining). It also differentiates this project from simply using an off-the-shelf foundation model like UniFMIR.

---

## What NOT To Do

1. **Don't replace the forward model with a neural renderer.** The physics-based differentiable microscope is a genuine advantage. Neural renderers (NeRF-style) are trendy but lose the interpretability and physical guarantees.

2. **Don't replace the U-Net entirely with a Vision Transformer.** ViTs lack the locality bias needed for sub-pixel regression. Hybrid (conv encoder/decoder + attention bottleneck) is the right call.

3. **Don't add a diffusion model to the main detection pipeline.** Diffusion models are too slow for SMLM-scale data and output images, not coordinates. At most, use one as an optional preprocessor.

4. **Don't over-engineer the temporal model for smFISH.** smFISH doesn't have blinking dynamics. Temporal context only matters for SMLM/STORM. Make it configurable, not mandatory.

5. **Don't try to make DECODE-FISH a foundation model.** The analysis-by-synthesis approach is fundamentally different from (and complementary to) foundation models. Foundation models could provide better features; DECODE-FISH provides better physics.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Reconstruction loss destabilizes training | Medium | Warm-start with sim-only training for N iterations before adding recon loss. Tune `recon_loss_scale` carefully. |
| Spatially-varying PSF overfits | Medium | Strong regularization (smoothness + L2). Start with 3-5 bases. Validate on held-out FOV regions. |
| Attention layer OOM on large crops | Low | Only at bottleneck (4x downsampled). For crop_sz=48, bottleneck is 12^3 = 1728 tokens — trivial for attention. |
| Mixed precision breaks GMM loss numerics | Medium | Keep loss computation in FP32 via `autocast` exclusion. Test against FP32 baseline. |
| Breaking backward compat with saved models | High | Version checkpoint format. Write migration script for existing `.pkl` files. |

---

## Evaluation Strategy & Public Benchmark Datasets

A fundamental challenge for smFISH is the absence of ground truth — you never know the true positions of all molecules in a real sample. However, as of 2024-2025, several public datasets with labeled ground truth have become available. These should be used systematically to evaluate every improvement.

### Available Public Datasets

#### 1. U-FISH / FISH_spots (2025, Genome Biology) — PRIMARY BENCHMARK

The most comprehensive FISH spot detection benchmark to date.

- **Size**: 4,166 images, 1,638,108 manually verified spot annotations
- **Sources**: 7 different spatial-omics methods (smFISH, MERFISH, seqFISH, etc.)
- **Dimensionality**: 2D and 3D
- **Ground truth method**: Human annotation + model-assisted review (Napari-based per-image verification)
- **Includes**: Real experimental data + simulated data at multiple noise levels (300–20,300)
- **Format**: 512×512 image patches + CSV coordinate files, pre-defined train/val/test splits
- **Access**: [HuggingFace: GangCaoLab/FISH_spots](https://huggingface.co/datasets/GangCaoLab/FISH_spots)
- **Paper**: [U-FISH, Genome Biology 2025](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03736-x)
- **Code**: [github.com/UFISH-Team/U-FISH](https://github.com/UFISH-Team/U-FISH)

**Why this matters**: This is the first large-scale, multi-source, human-verified FISH benchmark. It enables direct comparison of DECODE-FISH against U-FISH, deepBlink, Spotiflow, and other methods on the same data with the same metrics (F1 score, distance error).

#### 2. RS-FISH Simulated Data (2022, Nature Methods)

- **Size**: 50 simulated images (256×256×32 voxels) with exact ground-truth coordinates
- **Densities**: Sparse (30 spots) and dense (300 spots)
- **Noise levels**: Multiple SNR conditions
- **3D**: Yes — diffraction-limited spots with Gaussian PSF + Poisson + Gaussian noise
- **Benchmarked methods**: BigFISH, FISH-quant, AIRLOCALIZE, Starfish, deepBlink
- **Access**: Downloadable from RS-FISH supplementary data
- **Paper**: [RS-FISH, Nature Methods 2022](https://www.nature.com/articles/s41592-022-01669-y)

**Best for**: Precise 3D localization error (RMSE) evaluation at controlled density/SNR levels.

#### 3. Spotiflow Benchmark (2025, Nature Methods)

- **Method**: Deep stereographic flow regression for spot detection
- **Benchmark**: Includes evaluation datasets and comparison against prior methods
- **Paper**: [Spotiflow, Nature Methods 2025](https://www.nature.com/articles/s41592-025-02662-x)

**Relevance**: Latest published method (2025) — represents the current state-of-the-art to beat.

#### 4. DeepSpot Experimental Dataset (2021)

- **Size**: 1,553 experimentally generated images
- **Ground truth**: High-confidence experimental annotations (not simulation)
- **Accuracy**: >97% reported
- **Paper**: [DeepSpot, Biological Imaging 2021](https://www.cambridge.org/core/journals/biological-imaging/article/deepspot-a-deep-neural-network-for-rna-spot-enhancement-in-singlemolecule-fluorescence-insitu-hybridization-microscopy-images/3D022F6E91BA5B101C9A019B4C2B1A96)

**Best for**: Validating on real (non-simulated) experimental data with reliable labels.

#### 5. deepBlink Datasets (2021, Nucleic Acids Research)

- **Size**: 6 public datasets (synthetic + smFISH + SunTag live-cell)
- **Performance**: Average detection efficiency >85%, localization error <0.5 px
- **Pre-trained models**: Available on Figshare
- **Paper**: [deepBlink, NAR 2021](https://academic.oup.com/nar/article/49/13/7292/6312733)

#### 6. EPFL SMLM Challenge (2016, ongoing)

- **Type**: Simulated SMLM data with exact ground truth
- **Modalities**: 2D, 3D astigmatism, biplane, double-helix
- **Densities**: Sparse (0.25 mol/μm²) and high-density (2.5 mol/μm²)
- **SNR levels**: N1 (high signal), N2 (low signal), N3 (high background)
- **Structures**: Microtubules, endoplasmic reticulum
- **Access**: [EPFL SRM Dataset Hub](https://srm.epfl.ch/srm/dataset.html)

**Best for**: Standard SMLM benchmark (different from smFISH but useful for validating the core localization engine).

### Evaluation Protocol

For every improvement in Phases 1-3, evaluate using a tiered strategy:

```
Tier 1 (Every PR):
  - U-FISH benchmark F1 score + distance error on test split
  - RS-FISH simulated 3D: RMSE at sparse and dense conditions
  - Reconstruction residual on held-out real images (no GT needed)

Tier 2 (Milestone releases):
  - DeepSpot experimental dataset: precision/recall
  - Cross-method comparison: run BigFISH + Spotiflow on same data
  - Per-cell mRNA count correlation with RNA-seq (biological consistency)

Tier 3 (Publication):
  - Full EPFL SMLM challenge leaderboard submission
  - Semi-synthetic data: real background + simulated spots via learned forward model
  - Ablation studies on each Phase 1/2 component
```

### Metrics

| Metric | What it measures | Dataset needed |
|---|---|---|
| **F1 score** | Detection accuracy (precision × recall balance) | Any labeled dataset |
| **Distance error (nm)** | Localization precision | Labeled dataset with sub-pixel GT |
| **RMSE (x,y,z)** | Per-axis localization error | Simulated 3D data |
| **Jaccard index** | TP / (TP + FP + FN) | Any labeled dataset |
| **Reconstruction residual** | How well detections explain the input image | No GT needed |
| **σ calibration** | Are predicted uncertainties accurate? | Simulated data |
| **Count linearity** | Detection count vs. probe concentration | Dilution experiment |

---

## Appendix: Files Requiring Changes by Phase

### Phase 0
- `funcs/merfish_comparison.py` — `DataFrame.append()` → `pd.concat()`
- `engine/psf.py` — `torch.meshgrid()` indexing
- `funcs/utils.py` — `torch.stack(axis=)`, `torch.meshgrid()`, `UpsamplingBilinear2d`
- `funcs/dataset.py` — `torch.meshgrid()` indexing
- `engine/microscope.py` — `.cuda()` → `.to(device)`
- `engine/gmm_loss.py` — `.cuda()` → `.to(device)`, `torch.cuda.*Tensor` → `torch.zeros`
- `engine/point_process.py` — device handling
- `config/train.yaml` — remove hardcoded `base_dir`
- `imports.py` — remove hardcoded paths
- `setup.py` / `settings.ini` → `pyproject.toml`
- `environment.yaml` — pin versions
- `.github/workflows/main.yml` — update to Python 3.11

### Phase 1
- `engine/psf.py` — spatially-varying PSF bases
- `engine/microscope.py` — position-dependent PSF dispatch
- `engine/model.py` — coordinate channel input (CoordConv)
- `funcs/train_funcs.py` — reconstruction consistency loss, alternating schedule
- `config/train.yaml` — new config keys for recon loss, spatial PSF

### Phase 2
- `engine/model.py` — attention modules, temporal cross-attention, LiteDecodeNet
- `funcs/dataset.py` — multi-frame loading
- `config/train.yaml` — temporal window config

### Phase 3
- `funcs/file_io.py` — DataLoader workers
- `funcs/train_funcs.py` — AMP, torch.compile, optimizer swap
- `funcs/predict.py` — torch.compile for inference
