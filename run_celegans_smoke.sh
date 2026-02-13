#!/bin/bash
#BSUB -J celegans_smoke
#BSUB -P turaga
#BSUB -q gpu_l4
#BSUB -n 4
#BSUB -gpu "num=1"
#BSUB -o /groups/turaga/turagalab/DECODE26/decode_fish/models/celegans_smoke_test/train.log
#BSUB -e /groups/turaga/turagalab/DECODE26/decode_fish/models/celegans_smoke_test/train.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate decode_fish

BASE=/groups/turaga/turagalab/DECODE26/decode_fish
cd $BASE

echo "=== Starting C. elegans smoke test ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

export DECODE_BASE_DIR=$BASE

# --- Training ---
# Parameters from N2_352 notebook, adapted for channel 4 data:
#   gauss_radii: [1.5, 1.0, 1.0] (N2 PSF, slightly tighter than msp300)
#   theta: 20.31 (estimated from channel 4 via estimate_noise_scale)
#   prob_generator: higher density than msp300 (0.0002-0.001)
#   Data: 5 center-cropped 256x256 images, channel 4 extracted, spot counts 3-4

python decode_fish/train.py \
  "data_path.image_path=${BASE}/example/celegans_test/*.tif" \
  data_path.psf_path=null \
  codebook._target_=decode_fish.funcs.exp_specific.get_smfish_codebook \
  'genm.PSF.gauss_radii=[1.5,1.0,1.0]' \
  'genm.PSF.psf_extent_zyx=[21,21,21]' \
  genm.noise.theta=20.31 \
  genm.microscope.scale=2000 \
  genm.foci.n_foci_avg=2 \
  genm.prob_generator.low=0.0002 \
  genm.prob_generator.high=0.001 \
  sim.bg_estimation.fractal.scale=0 \
  training.num_iters=1000 \
  output.wandb_mode=disabled \
  output.save_dir="${BASE}/models/celegans_smoke_test" \
  output.log_dir="${BASE}/models/celegans_smoke_test" \
  run_name=celegans_smoke

TRAIN_EXIT=$?
echo ""
echo "=== Training exit code: $TRAIN_EXIT ==="

# --- Prediction ---
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "=== Starting prediction ==="

    # Predict on each test image
    python decode_fish/predict.py \
      model_path="${BASE}/models/celegans_smoke_test/model.pkl" \
      "image_path=${BASE}/example/celegans_test/*.tif" \
      out_file="${BASE}/models/celegans_smoke_test/predictions.csv" \
      model._target_=decode_fish.engine.model.UnetDecodeNoBn_2S \
      +model.n_p_ch=2

    PRED_EXIT=$?
    echo "=== Prediction exit code: $PRED_EXIT ==="

    if [ $PRED_EXIT -eq 0 ]; then
        echo "=== Quick summary ==="
        python -c "
import pandas as pd
pred = pd.read_csv('models/celegans_smoke_test/predictions.csv')
print(f'Total predictions: {len(pred)}')
print(f'Columns: {list(pred.columns)}')
if 'frame' in pred.columns or 'frame_idx' in pred.columns:
    frame_col = 'frame' if 'frame' in pred.columns else 'frame_idx'
    print(f'Per-image counts:')
    print(pred.groupby(frame_col).size())
print()
print(pred.describe())
"
    fi
fi

echo ""
echo "=== C. elegans smoke test complete: $(date) ==="
