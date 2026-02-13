#!/bin/bash
# Submit msp300 density prior sweep: 5K iterations, varying prob_generator.high
# Tests: 0.0001, 0.0003, 0.001, 0.003 (with low = high/3)
#
# Usage: bash run_msp300_sweep.sh

BASE=/groups/turaga/turagalab/DECODE26/decode_fish
REF_CSV=${BASE}/example/msp300_predictions.csv

for HIGH in 0.0001 0.0003 0.001 0.003; do
    # Compute low = high/3 using python
    LOW=$(python -c "print(f'{${HIGH}/3:.6f}')")

    # Clean name for directory (remove leading 0.)
    DNAME=$(python -c "print('${HIGH}'.replace('0.',''))")
    NAME="sweep_dens_${DNAME}"
    OUTDIR="${BASE}/models/sweep/dens_${DNAME}"

    mkdir -p "${OUTDIR}"

    bsub -J "${NAME}" \
         -P turaga \
         -q gpu_l4 \
         -n 4 \
         -gpu "num=1" \
         -W 15 \
         -o "${OUTDIR}/train.log" \
         -e "${OUTDIR}/train.err" \
         <<ENDJOB
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate decode_fish

cd ${BASE}
export DECODE_BASE_DIR=${BASE}

echo "=== msp300 density sweep: high=${HIGH}, low=${LOW} ==="
echo "Date: \$(date)"
echo "Host: \$(hostname)"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# --- Train ---
python decode_fish/train.py \\
  data_path.image_path="${BASE}/example/msp300_smFISH_3.tif" \\
  data_path.psf_path=null \\
  codebook._target_=decode_fish.funcs.exp_specific.get_smfish_codebook \\
  'genm.PSF.gauss_radii=[2.0,1.0,1.0]' \\
  'genm.PSF.psf_extent_zyx=[21,21,21]' \\
  genm.noise.theta=36.23 \\
  genm.microscope.scale=2000 \\
  genm.foci.n_foci_avg=2 \\
  genm.prob_generator.low=${LOW} \\
  genm.prob_generator.high=${HIGH} \\
  sim.bg_estimation.fractal.scale=0 \\
  training.num_iters=5000 \\
  output.wandb_mode=disabled \\
  output.save_dir="${OUTDIR}" \\
  output.log_dir="${OUTDIR}" \\
  run_name=${NAME}

TRAIN_EXIT=\$?
echo "=== Training exit code: \$TRAIN_EXIT ==="

if [ \$TRAIN_EXIT -eq 0 ]; then
    echo "=== Starting prediction (samp_threshold=0.1) ==="
    python decode_fish/predict.py \\
      model_path="${OUTDIR}/model.pkl" \\
      image_path="${BASE}/example/msp300_smFISH_3.tif" \\
      out_file="${OUTDIR}/predictions.csv" \\
      model._target_=decode_fish.engine.model.UnetDecodeNoBn_2S \\
      +model.n_p_ch=2 \\
      post_proc_isi.samp_threshold=0.1

    PRED_EXIT=\$?
    echo "=== Prediction exit code: \$PRED_EXIT ==="

    if [ \$PRED_EXIT -eq 0 ]; then
        echo "=== Running evaluation ==="
        python evaluate_msp300.py \\
          --pred_csv "${OUTDIR}/predictions.csv" \\
          --ref_csv "${REF_CSV}" \\
          --output_dir "${OUTDIR}/eval"
    fi
fi

echo "=== Done: \$(date) ==="
ENDJOB

    echo "Submitted ${NAME} (high=${HIGH}, low=${LOW})"
done

echo ""
echo "All density sweep jobs submitted. Monitor with: bjobs -w"
