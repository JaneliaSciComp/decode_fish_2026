#!/bin/bash
#BSUB -J smoke_test
#BSUB -P turaga
#BSUB -q gpu_l4
#BSUB -n 4
#BSUB -gpu "num=1"
#BSUB -o /groups/turaga/turagalab/DECODE26/decode_fish/models/msp300_smoke_test/train.log
#BSUB -e /groups/turaga/turagalab/DECODE26/decode_fish/models/msp300_smoke_test/train.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate decode_fish

BASE=/groups/turaga/turagalab/DECODE26/decode_fish
cd $BASE

echo "=== Starting training smoke test ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

export DECODE_BASE_DIR=$BASE

python decode_fish/train.py \
  data_path.image_path="${BASE}/example/msp300_smFISH_3.tif" \
  data_path.psf_path=null \
  codebook._target_=decode_fish.funcs.exp_specific.get_smfish_codebook \
  'genm.PSF.gauss_radii=[2.0,1.0,1.0]' \
  'genm.PSF.psf_extent_zyx=[21,21,21]' \
  genm.noise.theta=36.23 \
  genm.microscope.scale=2000 \
  genm.foci.n_foci_avg=2 \
  genm.prob_generator.low=0.0001 \
  genm.prob_generator.high=0.0003 \
  sim.bg_estimation.fractal.scale=0 \
  training.num_iters=200 \
  output.wandb_mode=disabled \
  output.save_dir="${BASE}/models/msp300_smoke_test" \
  output.log_dir="${BASE}/models/msp300_smoke_test" \
  run_name=smoke_test

TRAIN_EXIT=$?
echo ""
echo "=== Training exit code: $TRAIN_EXIT ==="

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "=== Starting prediction ==="
    python decode_fish/predict.py \
      model_path="${BASE}/models/msp300_smoke_test/model.pkl" \
      image_path="${BASE}/example/msp300_smFISH_3.tif" \
      out_file="${BASE}/models/msp300_smoke_test/predictions.csv" \
      model._target_=decode_fish.engine.model.UnetDecodeNoBn_2S \
      +model.n_p_ch=2

    PRED_EXIT=$?
    echo "=== Prediction exit code: $PRED_EXIT ==="

    if [ $PRED_EXIT -eq 0 ]; then
        echo "=== Quick comparison ==="
        cd $BASE
        python -c "
import pandas as pd, numpy as np
pred = pd.read_csv('models/msp300_smoke_test/predictions.csv')
ref_zyx = np.load('example/msp300_zyx.npy')
print(f'Our predictions: {len(pred)}, Reference: {len(ref_zyx)}')
print(f'Prediction columns: {list(pred.columns)}')
print(pred.describe())
"
    fi
fi

echo ""
echo "=== Smoke test complete: $(date) ==="
