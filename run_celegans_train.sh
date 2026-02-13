#!/bin/bash
# Submit C. elegans high-SNR longer training: 5K and 10K iterations.
# Uses same params as run_celegans_highsnr.sh but with:
#   - More iterations
#   - samp_threshold=0.1 at prediction for post-hoc threshold sweep
#   - evaluate_celegans.py for analysis
#
# Usage: bash run_celegans_train.sh

BASE=/groups/turaga/turagalab/DECODE26/decode_fish
IMAGE="${BASE}/example/N2_702_highSNR.tif"

for ITERS in 5000 10000; do
    case $ITERS in
        5000)  WALL="20" ;;
        10000) WALL="40" ;;
    esac

    NAME="celegans_${ITERS}"
    SHORT_NAME="cel_${ITERS}"
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

echo "=== C. elegans high-SNR: ${ITERS} iterations ==="
echo "Date: \$(date)"
echo "Host: \$(hostname)"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Image: N2_702_highSNR.tif (81, 454, 334)"
echo ""

# --- Train ---
# Parameters from N2_352 notebook:
#   theta: 10.71 (re-estimated for this high-SNR image)
#   gauss_radii: [1.5, 1.0, 1.0] (N2 PSF)
#   prob_generator: 0.0002-0.001 (N2 density)

python decode_fish/train.py \\
  data_path.image_path="${IMAGE}" \\
  data_path.psf_path=null \\
  codebook._target_=decode_fish.funcs.exp_specific.get_smfish_codebook \\
  'genm.PSF.gauss_radii=[1.5,1.0,1.0]' \\
  'genm.PSF.psf_extent_zyx=[21,21,21]' \\
  genm.noise.theta=10.71 \\
  genm.microscope.scale=2000 \\
  genm.foci.n_foci_avg=2 \\
  genm.prob_generator.low=0.0002 \\
  genm.prob_generator.high=0.001 \\
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
      image_path="${IMAGE}" \\
      out_file="${OUTDIR}/predictions.csv" \\
      model._target_=decode_fish.engine.model.UnetDecodeNoBn_2S \\
      +model.n_p_ch=2 \\
      post_proc_isi.samp_threshold=0.1

    PRED_EXIT=\$?
    echo "=== Prediction exit code: \$PRED_EXIT ==="

    if [ \$PRED_EXIT -eq 0 ]; then
        echo "=== Running evaluation ==="
        python evaluate_celegans.py \\
          --pred_csv "${OUTDIR}/predictions.csv" \\
          --image_path "${IMAGE}" \\
          --output_dir "${OUTDIR}/eval"
    fi
fi

echo "=== Done: \$(date) ==="
ENDJOB

    echo "Submitted ${SHORT_NAME} (${ITERS} iters, wall=${WALL}min)"
done

echo ""
echo "All C. elegans jobs submitted. Monitor with: bjobs -w"
