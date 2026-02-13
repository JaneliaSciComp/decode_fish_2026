# Daily Work Log

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
