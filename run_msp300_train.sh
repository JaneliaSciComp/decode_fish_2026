#!/bin/bash
# Submit msp300 convergence study: 5K, 10K, 20K, 40K iterations.
# Each job: train -> predict (samp_threshold=0.1) -> evaluate_msp300.py
#
# Usage: bash run_msp300_train.sh

BASE=/groups/turaga/turagalab/DECODE26/decode_fish
REF_CSV=${BASE}/example/msp300_predictions.csv

for ITERS in 5000 10000 20000 40000; do
    # Set wall time based on iterations
    case $ITERS in
        5000)  WALL="10" ;;
        10000) WALL="15" ;;
        20000) WALL="30" ;;
        40000) WALL="60" ;;
    esac

    NAME="msp300_${ITERS}"
    SHORT_NAME="msp_${ITERS}"
    OUTDIR="${BASE}/models/${NAME}"

    mkdir -p "${OUTDIR}"

    bsub -J "${SHORT_NAME}" \
         -P turaga \
         -q gpu_l4 \
         -n 4 \
         -gpu "num=1" \
         -W "${WALL}" \
         -o "${OUTDIR}/train.log" \
         -e "${OUTDIR}/train.err" \
         <<ENDJOB
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate decode_fish

cd ${BASE}
export DECODE_BASE_DIR=${BASE}

echo "=== msp300 convergence: ${ITERS} iterations ==="
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
  genm.prob_generator.low=0.0001 \\
  genm.prob_generator.high=0.0003 \\
  sim.bg_estimation.fractal.scale=0 \\
  training.num_iters=${ITERS} \\
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

    echo "Submitted ${SHORT_NAME} (${ITERS} iters, wall=${WALL}min)"
done

echo ""
echo "All msp300 convergence jobs submitted. Monitor with: bjobs -w"
