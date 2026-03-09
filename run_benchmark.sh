#!/bin/bash
# Benchmark DECODE-FISH on public datasets with GT.
#
# Benchmarks:
#   1. simfish_43 (FISH_spots/3d): 50x128x128, 511 spots, realistic noise (theta~345)
#   2. rsfish_300 (RS-FISH): 32x256x256, 300 spots, Poisson noise (theta~1)
#   3. N2_352 (RS-FISH): 51x509x433, real C. elegans smFISH (no position GT, qualitative)
#
# Each benchmark: estimate params -> train 5K -> predict -> evaluate
#
# Usage: bash run_benchmark.sh

BASE=/groups/turaga/turagalab/DECODE26/decode_fish
PUBDATA=/groups/turaga/turagalab/DECODE26/public_dataset

# --- Benchmark 1: simfish_43 ---
NAME="bench_simfish43"
OUTDIR="${BASE}/models/${NAME}"
mkdir -p "${OUTDIR}"

bsub -J "${NAME}" \
     -P turaga \
     -q gpu_l4 \
     -n 4 \
     -gpu "num=1" \
     -W 20 \
     -o "${OUTDIR}/train.log" \
     -e "${OUTDIR}/train.err" \
     <<'ENDJOB'
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate decode_fish

BASE=/groups/turaga/turagalab/DECODE26/decode_fish
PUBDATA=/groups/turaga/turagalab/DECODE26/public_dataset
OUTDIR=${BASE}/models/bench_simfish43
cd ${BASE}
export DECODE_BASE_DIR=${BASE}

echo "=== Benchmark: simfish_43 (50x128x128, 511 spots) ==="
echo "Date: $(date)"

# theta estimated from background: ~345
python decode_fish/train.py \
  data_path.image_path="${PUBDATA}/FISH_spots/3d/image/simfish_43.tif" \
  data_path.psf_path=null \
  codebook._target_=decode_fish.funcs.exp_specific.get_smfish_codebook \
  'genm.PSF.gauss_radii=[2.0,1.0,1.0]' \
  'genm.PSF.psf_extent_zyx=[21,21,21]' \
  genm.noise.theta=345 \
  genm.microscope.scale=2000 \
  genm.foci.n_foci_avg=2 \
  genm.prob_generator.low=0.0001 \
  genm.prob_generator.high=0.0005 \
  sim.bg_estimation.fractal.scale=0 \
  training.num_iters=5000 \
  output.wandb_mode=disabled \
  output.save_dir="${OUTDIR}" \
  output.log_dir="${OUTDIR}" \
  run_name=bench_simfish43

TRAIN_EXIT=$?
echo "=== Training exit: $TRAIN_EXIT ==="

if [ $TRAIN_EXIT -eq 0 ]; then
    python decode_fish/predict.py \
      model_path="${OUTDIR}/model.pkl" \
      image_path="${PUBDATA}/FISH_spots/3d/image/simfish_43.tif" \
      out_file="${OUTDIR}/predictions.csv" \
      model._target_=decode_fish.engine.model.UnetDecodeNoBn_2S \
      +model.n_p_ch=2 \
      post_proc_isi.samp_threshold=0.1

    if [ $? -eq 0 ]; then
        python evaluate_benchmark.py \
          --pred_csv "${OUTDIR}/predictions.csv" \
          --gt_csv "${PUBDATA}/FISH_spots/3d/csv/simfish_43.csv" \
          --gt_format fish_spots \
          --output_dir "${OUTDIR}/eval" \
          --name simfish_43
    fi
fi
echo "=== Done: $(date) ==="
ENDJOB
echo "Submitted ${NAME}"


# --- Benchmark 2: RS-FISH 300 spots ---
NAME="bench_rsfish300"
OUTDIR="${BASE}/models/${NAME}"
RSFISH_DIR="${PUBDATA}/RS-FISH/RadialSymmetryLocalization/documents/Simulation_of_data/Simulated data/Empty Bg Density Range Sigxy 1pt5 SigZ 2"
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
source ~/miniconda3/etc/profile.d/conda.sh
conda activate decode_fish

BASE=/groups/turaga/turagalab/DECODE26/decode_fish
OUTDIR=\${BASE}/models/bench_rsfish300
RSFISH_DIR="${RSFISH_DIR}"
cd \${BASE}
export DECODE_BASE_DIR=\${BASE}

echo "=== Benchmark: RS-FISH 300 spots (32x256x256, Poisson noise) ==="
echo "Date: \$(date)"

# RS-FISH: Poisson noise with bg=200, theta~1 (nearly noiseless)
# PSF: sigma_xy=1.5, sigma_z=2 (from dataset name)
python decode_fish/train.py \\
  data_path.image_path="\${RSFISH_DIR}/Poiss_300spots_bg_200_2_I_300_0_img0.tif" \\
  data_path.psf_path=null \\
  codebook._target_=decode_fish.funcs.exp_specific.get_smfish_codebook \\
  'genm.PSF.gauss_radii=[2.0,1.5,1.5]' \\
  'genm.PSF.psf_extent_zyx=[21,21,21]' \\
  genm.noise.theta=1.0 \\
  genm.microscope.scale=2000 \\
  genm.foci.n_foci_avg=2 \\
  genm.prob_generator.low=0.00005 \\
  genm.prob_generator.high=0.0002 \\
  sim.bg_estimation.fractal.scale=0 \\
  training.num_iters=5000 \\
  output.wandb_mode=disabled \\
  output.save_dir="\${OUTDIR}" \\
  output.log_dir="\${OUTDIR}" \\
  run_name=bench_rsfish300

TRAIN_EXIT=\$?
echo "=== Training exit: \$TRAIN_EXIT ==="

if [ \$TRAIN_EXIT -eq 0 ]; then
    python decode_fish/predict.py \\
      model_path="\${OUTDIR}/model.pkl" \\
      image_path="\${RSFISH_DIR}/Poiss_300spots_bg_200_2_I_300_0_img0.tif" \\
      out_file="\${OUTDIR}/predictions.csv" \\
      model._target_=decode_fish.engine.model.UnetDecodeNoBn_2S \\
      +model.n_p_ch=2 \\
      post_proc_isi.samp_threshold=0.1

    if [ \$? -eq 0 ]; then
        python evaluate_benchmark.py \\
          --pred_csv "\${OUTDIR}/predictions.csv" \\
          --gt_loc "\${RSFISH_DIR}/Poiss_300spots_bg_200_2_I_300_0_img0.loc" \\
          --gt_format rsfish \\
          --output_dir "\${OUTDIR}/eval" \\
          --name rsfish_300
    fi
fi
echo "=== Done: \$(date) ==="
ENDJOB
echo "Submitted ${NAME}"


# --- Benchmark 3: N2_352 (qualitative, no position GT) ---
NAME="bench_n2_352"
OUTDIR="${BASE}/models/${NAME}"
N2_IMAGE="${PUBDATA}/RS-FISH/RadialSymmetryLocalization/documents/Example_smFISH_images/N2_352-1.tif"
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
source ~/miniconda3/etc/profile.d/conda.sh
conda activate decode_fish

BASE=/groups/turaga/turagalab/DECODE26/decode_fish
OUTDIR=\${BASE}/models/bench_n2_352
cd \${BASE}
export DECODE_BASE_DIR=\${BASE}

echo "=== Benchmark: N2_352-1 real C. elegans smFISH (51x509x433) ==="
echo "Date: \$(date)"

# Real data, theta~404, same gene as DECODE-FISH paper
python decode_fish/train.py \\
  data_path.image_path="${N2_IMAGE}" \\
  data_path.psf_path=null \\
  codebook._target_=decode_fish.funcs.exp_specific.get_smfish_codebook \\
  'genm.PSF.gauss_radii=[1.5,1.0,1.0]' \\
  'genm.PSF.psf_extent_zyx=[21,21,21]' \\
  genm.noise.theta=404 \\
  genm.microscope.scale=2000 \\
  genm.foci.n_foci_avg=2 \\
  genm.prob_generator.low=0.0002 \\
  genm.prob_generator.high=0.001 \\
  sim.bg_estimation.fractal.scale=0 \\
  training.num_iters=5000 \\
  output.wandb_mode=disabled \\
  output.save_dir="\${OUTDIR}" \\
  output.log_dir="\${OUTDIR}" \\
  run_name=bench_n2_352

TRAIN_EXIT=\$?
echo "=== Training exit: \$TRAIN_EXIT ==="

if [ \$TRAIN_EXIT -eq 0 ]; then
    python decode_fish/predict.py \\
      model_path="\${OUTDIR}/model.pkl" \\
      image_path="${N2_IMAGE}" \\
      out_file="\${OUTDIR}/predictions.csv" \\
      model._target_=decode_fish.engine.model.UnetDecodeNoBn_2S \\
      +model.n_p_ch=2 \\
      post_proc_isi.samp_threshold=0.1

    if [ \$? -eq 0 ]; then
        python evaluate_celegans.py \\
          --pred_csv "\${OUTDIR}/predictions.csv" \\
          --image_path "${N2_IMAGE}" \\
          --output_dir "\${OUTDIR}/eval"
    fi
fi
echo "=== Done: \$(date) ==="
ENDJOB
echo "Submitted ${NAME}"

echo ""
echo "All benchmark jobs submitted. Monitor with: bjobs -w"
