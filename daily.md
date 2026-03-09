# Daily Work Log

## 2026-03-09 (continued): Automatic Theta Estimation

Created `estimate_noise_params.py` — automatic theta estimation using residual method (subtract smoothed background, compute var/mean per Z-slice). Validates against 4 test images. See project-level `daily.md` for full results.

Key finding: percentile-based var/mean gives pixel-level noise theta (~35 for msp300) which is too low. The residual method with window=crop_size/4 gives theta=83.4 for msp300, in the right ballpark of our manual estimate (139) and much better than the original config (36.23).

Validation: msp300 theta comparison confirms higher theta = better F1 (0.039→0.061→0.106 for theta 36→83→139). On simfish_43, theta=345 (manual) beats 541 (auto) with F1=0.578 — our new best. Convergence study shows 5K iterations is optimal for both datasets; longer training always degrades F1. See project-level `daily.md` for full tables.

---

## 2026-03-09: Over-detection Root Cause + Benchmarks

### Root cause analysis

Investigated why all models massively over-detect. Three issues found:

1. **Noise theta was 3.8x wrong** (CRITICAL). Estimated theta=139 from image background; we used 36.23. The model was told noise is much lower than reality, so it interprets real noise as signal.
2. **Loss function doesn't penalize false positives.** GMM loss only rewards finding GT spots + getting count right. No cost to predicting spots everywhere.
3. **Background modeling disabled** (`fractal.scale=0`).

### Bug fix: `AddPerlinNoise` crash for smFISH

`AddPerlinNoise` in `dataset.py` was written for MERFISH (5D tensor, multi-channel). For smFISH (4D tensor, 1 channel), it misinterprets Z=37 as batch size, generating 37 noise volumes and crashing on reshape. Fixed to handle both 4D and 5D cases.

### Fix comparison on msp300 (5K iterations)

| Run | Fix | F1 | Precision | Detections |
|-----|-----|------|-----------|------------|
| baseline | none | 0.048 | 2.4% | 37K |
| fix_lowdens | lower density prior | 0.054 | 2.8% | 33K |
| fix_fractal1 | enable background noise | 0.056 | 2.9% | 32K |
| fix_combined | theta + fractal + lowdens | 0.077 | 4.0% | 23K |
| **fix_theta139** | **theta 36→139 only** | **0.112** | **5.9%** | **15K** |

**Theta correction alone is the best fix** — better than all three combined. Fractal noise actually hurts when combined with correct theta (F1 drops from 0.112 to 0.077).

### msp300 convergence (all runs complete)

| Iters | F1 | Precision | Recall | RMSE | Detections |
|-------|------|-----------|--------|------|------------|
| 200 (smoke) | 0.078 | 100% | 4% | 128nm | 37 |
| 5K | 0.048 | 2.4% | 100% | 45nm | 37K |
| 10K | 0.033 | 1.7% | 100% | 41nm | 54K |
| 20K | 0.034 | 1.7% | 100% | 39nm | 52K |
| 40K | 0.031 | 1.6% | 100% | 37nm | 58K |

More training → worse F1 (more over-detection), but better RMSE on matched spots.

### Public dataset benchmarks (5K iterations, theta matched per image)

| Dataset | Image | GT spots | F1 | Precision | Recall | RMSE | Detections |
|---------|-------|----------|------|-----------|--------|------|------------|
| FISH_spots simfish_43 | 50×128×128 | 511 | **0.417** | 26.5% | 97.7% | 216nm | 1,885 |
| RS-FISH 300spots | 32×256×256 | 300 | 0.0 | — | — | — | 0 |
| RS-FISH N2_352 | 51×509×433 | (none) | — | — | — | — | 16K total |

- **simfish_43 (F1=0.42)** is a breakthrough — shows DECODE-FISH CAN work when noise model matches. This simulated image has theta≈345 (matched). 3.7x over-detection (1885 vs 511) but 98% recall.
- **rsfish_300 produced 0 predictions** — image has Poisson noise (values 190-224), fundamentally incompatible with gamma noise model at theta=1. Image statistics too different.
- **N2_352 (real C. elegans)** — 7.3K detections at t=0.99. Over-detects but less than our high-SNR image. Theta=404 used.

### Key insight

The over-detection severity depends on noise model alignment:
- simfish_43 (theta well-matched): F1=0.42, 3.7x over-detection
- msp300 (theta 3.8x off at 36, theta corrected to 139): F1=0.11, 17x over-detection
- msp300 (theta wrong): F1=0.05, 41x over-detection
- rsfish (Poisson noise, gamma model totally wrong): 0 predictions

### Scripts created
- `evaluate_benchmark.py` — Evaluation adapter for FISH_spots and RS-FISH GT formats
- `run_benchmark.sh` — Benchmark launcher for 3 public datasets
- `run_msp300_fixes.sh` — Diagnostic fix comparison (theta / fractal / density / combined)

### Next steps
- **For msp300**: Try higher theta (300-400, matching mid-slice variance) + early stopping (500-2000 iter)
- **For simfish_43**: Already decent — try 10K-20K iterations to see if it converges better
- **Longer term**: Add FP penalty to GMM loss (`gmm_loss.py`) to directly address over-detection
- **rsfish**: Would need a Poisson noise model instead of gamma — not a priority

---

## 2026-02-13 (afternoon): Leaving for the day

Pushed all work to GitHub (`JaneliaSciComp/decode_fish_2026`, commit `0b42bff`).

**Still running at the time:** `msp_40000` and 4 prediction-rerun jobs. All completed by 2026-03-09 (see results above).

---

## 2026-02-13: Parameter Tuning - First Batch

### What happened

Phase 0 is complete. Started systematic parameter tuning for msp300 (with ground truth) and C. elegans (no position GT).

**Scripts created:**
- `evaluate_msp300.py` -- GT-based evaluation: threshold sweep, matching(), PR curves, RMSE, XY overlays
- `evaluate_celegans.py` -- Count-based evaluation: threshold sweep, prob histogram, spatial density, multi-threshold overlays
- `run_msp300_train.sh` -- Convergence study: 5K/10K/20K/40K iterations
- `run_msp300_sweep.sh` -- Density prior sweep: `prob_generator.high` = 0.0001/0.0003/0.001/0.003 at 5K iters
- `run_celegans_train.sh` -- C. elegans high-SNR: 5K/10K iterations
- `summarize_results.py` -- Cross-run comparison (reads eval_summary.json from all model dirs)

**Bug found:** Reference predictions CSV (`example/msp300_predictions.csv`) has 4 frames: frame 0 = actual model predictions (1248 entries), frame 2 = GT positions embedded verbatim (908 entries). Evaluation script now auto-filters to GT frame indices.

### Submitted 10 GPU jobs (all `gpu_l4`)

| Job | Status | Result |
|-----|--------|--------|
| msp300_5000 (5K iter) | Training done, prediction resubmitted | Wall time killed before predict step |
| msp300_10000 (10K iter) | Training done, prediction resubmitted | Wall time killed before predict step |
| msp300_20000 (20K iter) | Training done, prediction resubmitted | Wall time killed before predict step |
| msp300_40000 (40K iter) | Still running (60min wall) | -- |
| sweep/dens_0001 (high=0.0001) | Done | F1=0.042, 105K preds, precision=2% |
| sweep/dens_0003 (high=0.0003) | Done | F1=0.038, 111K preds, precision=2% |
| sweep/dens_001 (high=0.001) | Done | F1=0.047, 128K preds, precision=2% |
| sweep/dens_003 (high=0.003) | Training done, prediction resubmitted | Wall time killed before predict step |
| celegans_5000 (5K iter) | Done | 60K preds; 24K at t=0.95 |
| celegans_10000 (10K iter) | Done | 59K preds; 22K at t=0.95 |

### Key findings so far

1. **Wall times were too short** for convergence jobs. Training finished but predict + eval didn't fit. Resubmitted prediction-only jobs for the 4 affected models.

2. **All models massively over-detect at samp_threshold=0.1.** The density sweep at 5K iterations produces 100K+ predictions. Even at t=0.975, there are 38K-47K predictions vs 908 GT emitters. Precision is ~2%.

3. **C. elegans still over-detects at 5K-10K.** Counts at t=0.99: 24K (5K iter) and 21K (10K iter). Was 27K at 1K iter, so it's decreasing but very slowly. Expected: 3-5 spots.

4. **Reference model baseline is poor.** F1=0.20 with 1248 frame-0 predictions vs 908 GT (only 215 match within 1000nm). RMSE of matches: 706nm.

5. **Smoke test (200 iter) had perfect precision** (37/37 correct) but 4% recall (37/908 found). RMSE=128nm. This is the opposite of the converged models which have high recall but terrible precision.

### Interpretation

The pattern is: short training = few high-confidence correct detections; longer training = model becomes overconfident and detects everything. This suggests the model is memorizing/overfitting the simulated data rather than learning to discriminate real spots from noise. The density prior (`prob_generator.high`) doesn't help much -- all 4 sweep values gave similar terrible results at 5K iterations.

### Next steps

- Wait for msp300_40000 and the 4 resubmitted prediction jobs
- Run `python summarize_results.py` once all complete
- **Likely need to investigate why the model over-detects** -- possible issues:
  - Noise model mismatch (`theta` parameter may not match actual image noise)
  - PSF model mismatch (Gaussian PSF vs real PSF)
  - Background estimation (`fractal.scale=0` disables it -- may need structured background)
  - The self-supervised approach may need a measured PSF (`psf_path`) for this data
- Consider trying with `psf_path` pointing to a measured PSF if available
- Consider re-running with `fractal.scale` > 0 to enable background modeling

### Job IDs for reference
- Convergence: 148256066 (5K), 148256067 (10K), 148256068 (20K), 148256069 (40K)
- Density sweep: 148256071-148256074
- C. elegans: 148256075-148256076
- Prediction reruns: 148256255-148256258
