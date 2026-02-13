#!/bin/bash
#BSUB -J celegans_highsnr
#BSUB -P turaga
#BSUB -q gpu_l4
#BSUB -n 4
#BSUB -gpu "num=1"
#BSUB -o /groups/turaga/turagalab/DECODE26/decode_fish/models/celegans_highsnr/train.log
#BSUB -e /groups/turaga/turagalab/DECODE26/decode_fish/models/celegans_highsnr/train.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate decode_fish

BASE=/groups/turaga/turagalab/DECODE26/decode_fish
cd $BASE

echo "=== Starting C. elegans high-SNR smoke test ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "Image: N2_702_cropped_1620 (high SNR)_ch0.tif"
echo "  Shape: (81, 454, 334), single-channel, uint16"
echo "  Provided by Stephan as a clean test case"
echo ""

export DECODE_BASE_DIR=$BASE

# --- Training ---
# Parameters: N2_352 notebook PSF/density, theta re-estimated for this image
#   theta: 10.71 (estimated via estimate_noise_scale — lower = high SNR)
#   gauss_radii: [1.5, 1.0, 1.0] (N2 PSF from notebook)
#   prob_generator: 0.0002-0.001 (N2 density)
#   1000 iterations for meaningful convergence

python decode_fish/train.py \
  data_path.image_path="${BASE}/example/N2_702_highSNR.tif" \
  data_path.psf_path=null \
  codebook._target_=decode_fish.funcs.exp_specific.get_smfish_codebook \
  'genm.PSF.gauss_radii=[1.5,1.0,1.0]' \
  'genm.PSF.psf_extent_zyx=[21,21,21]' \
  genm.noise.theta=10.71 \
  genm.microscope.scale=2000 \
  genm.foci.n_foci_avg=2 \
  genm.prob_generator.low=0.0002 \
  genm.prob_generator.high=0.001 \
  sim.bg_estimation.fractal.scale=0 \
  training.num_iters=1000 \
  output.wandb_mode=disabled \
  output.save_dir="${BASE}/models/celegans_highsnr" \
  output.log_dir="${BASE}/models/celegans_highsnr" \
  run_name=celegans_highsnr

TRAIN_EXIT=$?
echo ""
echo "=== Training exit code: $TRAIN_EXIT ==="

# --- Prediction ---
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "=== Starting prediction ==="
    python decode_fish/predict.py \
      model_path="${BASE}/models/celegans_highsnr/model.pkl" \
      image_path="${BASE}/example/N2_702_highSNR.tif" \
      out_file="${BASE}/models/celegans_highsnr/predictions.csv" \
      model._target_=decode_fish.engine.model.UnetDecodeNoBn_2S \
      +model.n_p_ch=2

    PRED_EXIT=$?
    echo "=== Prediction exit code: $PRED_EXIT ==="

    if [ $PRED_EXIT -eq 0 ]; then
        echo "=== Quick summary ==="
        python -c "
import pandas as pd
pred = pd.read_csv('models/celegans_highsnr/predictions.csv')
print(f'Total predictions: {len(pred)}')
print(f'Columns: {list(pred.columns)}')
if len(pred) > 0:
    print()
    print(pred.describe())
else:
    print('WARNING: 0 predictions — model may need more iterations or parameter tuning')
"
    fi
fi

echo ""
echo "=== C. elegans high-SNR smoke test complete: $(date) ==="
