#!/bin/bash
# Validate auto-estimated theta vs manual vs original.
#
# Experiments:
#   1. msp300 at theta=83 (auto), 139 (manual), 36 (original) — 5K iter each
#   2. simfish_43 at theta=541 (auto), 345 (manual) — 5K iter each
#   3. simfish_43 at theta=345 — 10K and 20K iter (convergence)
#
# All use samp_threshold=0.1 for post-hoc threshold sweeping.

BASE=/groups/turaga/turagalab/DECODE26/decode_fish
PUBDATA=/groups/turaga/turagalab/DECODE26/public_dataset

submit_msp300() {
    local NAME=$1
    local THETA=$2
    local ITERS=$3
    local OUTDIR="${BASE}/models/${NAME}"
    mkdir -p "${OUTDIR}"

    bsub -J "${NAME}" \
         -P turaga \
         -q gpu_l4 \
         -n 4 \
         -gpu "num=1" \
         -W 20 \
         -o "${OUTDIR}/train.log" \
         -e "${OUTDIR}/train.err" \
         <<ENDJOB
#!/bin/bash
source /groups/scicompsoft/home/gaos2/miniconda3/etc/profile.d/conda.sh
conda activate decode_fish
cd ${BASE}
export DECODE_BASE_DIR=${BASE}

echo "=== ${NAME}: msp300, theta=${THETA}, ${ITERS} iter ==="
echo "Date: \$(date)"

python decode_fish/train.py \\
  data_path.image_path="${BASE}/example/msp300_smFISH_3.tif" \\
  data_path.psf_path=null \\
  codebook._target_=decode_fish.funcs.exp_specific.get_smfish_codebook \\
  'genm.PSF.gauss_radii=[2.0,1.0,1.0]' \\
  'genm.PSF.psf_extent_zyx=[21,21,21]' \\
  genm.noise.theta=${THETA} \\
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
echo "=== Training exit: \$TRAIN_EXIT ==="

if [ \$TRAIN_EXIT -eq 0 ]; then
    python decode_fish/predict.py \\
      model_path="${OUTDIR}/model.pkl" \\
      image_path="${BASE}/example/msp300_smFISH_3.tif" \\
      out_file="${OUTDIR}/predictions.csv" \\
      model._target_=decode_fish.engine.model.UnetDecodeNoBn_2S \\
      +model.n_p_ch=2 \\
      post_proc_isi.samp_threshold=0.1

    if [ \$? -eq 0 ]; then
        python evaluate_msp300.py \\
          --pred_csv "${OUTDIR}/predictions.csv" \\
          --gt_npy "${BASE}/example/msp300_zyx.npy" \\
          --output_dir "${OUTDIR}/eval"
    fi
fi
echo "=== Done: \$(date) ==="
ENDJOB
    echo "Submitted ${NAME}"
}

submit_simfish() {
    local NAME=$1
    local THETA=$2
    local ITERS=$3
    local OUTDIR="${BASE}/models/${NAME}"
    mkdir -p "${OUTDIR}"

    bsub -J "${NAME}" \
         -P turaga \
         -q gpu_l4 \
         -n 4 \
         -gpu "num=1" \
         -W 30 \
         -o "${OUTDIR}/train.log" \
         -e "${OUTDIR}/train.err" \
         <<ENDJOB
#!/bin/bash
source /groups/scicompsoft/home/gaos2/miniconda3/etc/profile.d/conda.sh
conda activate decode_fish
cd ${BASE}
export DECODE_BASE_DIR=${BASE}

echo "=== ${NAME}: simfish_43, theta=${THETA}, ${ITERS} iter ==="
echo "Date: \$(date)"

python decode_fish/train.py \\
  data_path.image_path="${PUBDATA}/FISH_spots/3d/image/simfish_43.tif" \\
  data_path.psf_path=null \\
  codebook._target_=decode_fish.funcs.exp_specific.get_smfish_codebook \\
  'genm.PSF.gauss_radii=[2.0,1.0,1.0]' \\
  'genm.PSF.psf_extent_zyx=[21,21,21]' \\
  genm.noise.theta=${THETA} \\
  genm.microscope.scale=2000 \\
  genm.foci.n_foci_avg=2 \\
  genm.prob_generator.low=0.0001 \\
  genm.prob_generator.high=0.0005 \\
  sim.bg_estimation.fractal.scale=0 \\
  training.num_iters=${ITERS} \\
  output.wandb_mode=disabled \\
  output.save_dir="${OUTDIR}" \\
  output.log_dir="${OUTDIR}" \\
  run_name=${NAME}

TRAIN_EXIT=\$?
echo "=== Training exit: \$TRAIN_EXIT ==="

if [ \$TRAIN_EXIT -eq 0 ]; then
    python decode_fish/predict.py \\
      model_path="${OUTDIR}/model.pkl" \\
      image_path="${PUBDATA}/FISH_spots/3d/image/simfish_43.tif" \\
      out_file="${OUTDIR}/predictions.csv" \\
      model._target_=decode_fish.engine.model.UnetDecodeNoBn_2S \\
      +model.n_p_ch=2 \\
      post_proc_isi.samp_threshold=0.1

    if [ \$? -eq 0 ]; then
        python evaluate_benchmark.py \\
          --pred_csv "${OUTDIR}/predictions.csv" \\
          --gt_csv "${PUBDATA}/FISH_spots/3d/csv/simfish_43.csv" \\
          --gt_format fish_spots \\
          --output_dir "${OUTDIR}/eval" \\
          --name simfish_43
    fi
fi
echo "=== Done: \$(date) ==="
ENDJOB
    echo "Submitted ${NAME}"
}

# --- msp300 theta comparison (5K iter) ---
submit_msp300 "val_msp_theta83"  83  5000
submit_msp300 "val_msp_theta139" 139 5000
submit_msp300 "val_msp_theta36"  36  5000

# --- simfish_43 theta comparison (5K iter) ---
submit_simfish "val_sim_theta541" 541 5000
submit_simfish "val_sim_theta345" 345 5000

# --- simfish_43 convergence at theta=345 (10K, 20K) ---
submit_simfish "val_sim_t345_10k" 345 10000
submit_simfish "val_sim_t345_20k" 345 20000

# --- simfish_43 convergence at theta=541 (10K) ---
submit_simfish "val_sim_t541_10k" 541 10000

echo ""
echo "Submitted 8 validation jobs. Monitor with: bjobs -w"
