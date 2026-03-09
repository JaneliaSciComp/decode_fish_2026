#!/bin/bash
# Diagnostic runs to fix over-detection on msp300.
# Each run tests a different fix at 5K iterations.
#
# Baseline params (from run_msp300_train.sh):
#   theta=36.23, fractal.scale=0, prob_generator low=0.0001/high=0.0003
#
# Fixes:
#   A) theta=139 (re-estimated from background)
#   B) fractal.scale=1 (enable background variation)
#   C) lower density prior (low=0.00003, high=0.0001)
#   D) all three combined
#
# Usage: bash run_msp300_fixes.sh

BASE=/groups/turaga/turagalab/DECODE26/decode_fish
REF_CSV=${BASE}/example/msp300_predictions.csv

submit_job() {
    local NAME=$1
    local THETA=$2
    local FSCALE=$3
    local PLOW=$4
    local PHIGH=$5
    local OUTDIR="${BASE}/models/fix_${NAME}"

    mkdir -p "${OUTDIR}"

    bsub -J "fix_${NAME}" \
         -P turaga \
         -q gpu_l4 \
         -n 4 \
         -gpu "num=1" \
         -W 20 \
         -o "${OUTDIR}/train.log" \
         -e "${OUTDIR}/train.err" \
         <<ENDJOB
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate decode_fish
cd ${BASE}
export DECODE_BASE_DIR=${BASE}

echo "=== Fix run: ${NAME} ==="
echo "  theta=${THETA}, fractal.scale=${FSCALE}, prob low=${PLOW} high=${PHIGH}"
echo "Date: \$(date)"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python decode_fish/train.py \\
  data_path.image_path="${BASE}/example/msp300_smFISH_3.tif" \\
  data_path.psf_path=null \\
  codebook._target_=decode_fish.funcs.exp_specific.get_smfish_codebook \\
  'genm.PSF.gauss_radii=[2.0,1.0,1.0]' \\
  'genm.PSF.psf_extent_zyx=[21,21,21]' \\
  genm.noise.theta=${THETA} \\
  genm.microscope.scale=2000 \\
  genm.foci.n_foci_avg=2 \\
  genm.prob_generator.low=${PLOW} \\
  genm.prob_generator.high=${PHIGH} \\
  sim.bg_estimation.fractal.scale=${FSCALE} \\
  training.num_iters=5000 \\
  output.wandb_mode=disabled \\
  output.save_dir="${OUTDIR}" \\
  output.log_dir="${OUTDIR}" \\
  run_name=fix_${NAME}

TRAIN_EXIT=\$?
echo "=== Training exit: \$TRAIN_EXIT ==="

if [ \$TRAIN_EXIT -eq 0 ]; then
    echo "=== Prediction (samp_threshold=0.1) ==="
    python decode_fish/predict.py \\
      model_path="${OUTDIR}/model.pkl" \\
      image_path="${BASE}/example/msp300_smFISH_3.tif" \\
      out_file="${OUTDIR}/predictions.csv" \\
      model._target_=decode_fish.engine.model.UnetDecodeNoBn_2S \\
      +model.n_p_ch=2 \\
      post_proc_isi.samp_threshold=0.1

    if [ \$? -eq 0 ]; then
        echo "=== Evaluation ==="
        python evaluate_msp300.py \\
          --pred_csv "${OUTDIR}/predictions.csv" \\
          --ref_csv "${REF_CSV}" \\
          --output_dir "${OUTDIR}/eval"
    fi
fi
echo "=== Done: \$(date) ==="
ENDJOB

    echo "Submitted fix_${NAME}"
}

#                NAME            THETA  FSCALE  PLOW      PHIGH
submit_job       "theta139"      139    0       0.0001    0.0003    # Fix A: correct noise only
submit_job       "fractal1"      36.23  1       0.0001    0.0003    # Fix B: enable background only
submit_job       "lowdens"       36.23  0       0.00003   0.0001    # Fix C: lower density only
submit_job       "combined"      139    1       0.00003   0.0001    # Fix D: all three fixes

echo ""
echo "All fix jobs submitted. Monitor with: bjobs -w"
