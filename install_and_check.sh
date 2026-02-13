#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate decode_fish
cd /groups/turaga/turagalab/DECODE26/decode_fish

echo "=== Installing einops ==="
pip install einops

echo ""
echo "=== Testing imports ==="
python -c "
from decode_fish.engine.gmm_loss import PointProcessGaussian
print('gmm_loss OK')

from decode_fish.funcs.train_funcs import train
print('train_funcs OK')

from decode_fish.funcs.predict import predict, window_predict
print('predict OK')

from decode_fish.funcs.exp_specific import get_smfish_codebook
cb, targets = get_smfish_codebook()
print(f'codebook OK: shape={cb.shape}, targets={targets}')

from decode_fish.funcs.file_io import get_dataloader, load_psf_noise_micro
print('file_io OK')

print()
print('All imports passed!')
"
