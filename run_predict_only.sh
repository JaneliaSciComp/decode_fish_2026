#!/bin/bash
#BSUB -J predict_test
#BSUB -P turaga
#BSUB -q gpu_l4
#BSUB -n 4
#BSUB -gpu "num=1"
#BSUB -o /groups/turaga/turagalab/DECODE26/decode_fish/models/msp300_smoke_test/predict.log
#BSUB -e /groups/turaga/turagalab/DECODE26/decode_fish/models/msp300_smoke_test/predict.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate decode_fish

BASE=/groups/turaga/turagalab/DECODE26/decode_fish
cd $BASE

export DECODE_BASE_DIR=$BASE

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
    python -c "
import pandas as pd, numpy as np
pred = pd.read_csv('${BASE}/models/msp300_smoke_test/predictions.csv')
ref_zyx = np.load('${BASE}/example/msp300_zyx.npy')
print(f'Our predictions: {len(pred)}, Reference: {len(ref_zyx)}')
print(f'Prediction columns: {list(pred.columns)}')
print(pred.describe())
"
fi

echo "=== Done: $(date) ==="
