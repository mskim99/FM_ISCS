#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ISCS score-model training launcher
# ------------------------------------------------------------
# Usage examples:
#   bash train.sh
#   TASK=CT bash train.sh
#   TASK=MRI bash train.sh
#   GPU=1 TRAIN_DIR=/data/AAPM/train VAL_DIR=/data/AAPM/val bash train.sh
#   TASK=MRI TRAIN_DIR=/data/BMR/train VAL_DIR=/data/BMR/val WORKDIR=./experiments/bmr_256 bash train.sh
#
# Assumptions:
# - Run this script from the ISCS repository root.
# - train_iscs_score_model.py is located in the repository root.
# - configs/ve/*.yaml from the ISCS repo are available.
# ============================================================

# -----------------------------
# Common settings
# -----------------------------
PYTHON_BIN=${PYTHON_BIN:-python}
SCRIPT=${SCRIPT:-train_iscs_score_model.py}
TASK=${TASK:-CT}                 # CT or MRI
GPU=${GPU:-0}                    # set -1 for CPU
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}
SPLIT_SEED=${SPLIT_SEED:-1234}

# Optimization
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
N_ITERS=${N_ITERS:-200000}
LR=${LR:-2e-4}
BETA1=${BETA1:-0.9}
ADAM_EPS=${ADAM_EPS:-1e-8}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP=${WARMUP:-5000}
GRAD_CLIP=${GRAD_CLIP:-1.0}

# Logging / checkpointing
LOG_FREQ=${LOG_FREQ:-100}
EVAL_FREQ=${EVAL_FREQ:-2000}
SNAPSHOT_FREQ=${SNAPSHOT_FREQ:-5000}
SNAPSHOT_SAMPLING=${SNAPSHOT_SAMPLING:-1}   # 1 to enable, 0 to disable
LIKELIHOOD_WEIGHTING=${LIKELIHOOD_WEIGHTING:-0}

# Dataset behavior
PLANE=${PLANE:-axial}            # axial / coronal / sagittal
MIN_SLICE_STD=${MIN_SLICE_STD:-0.0}
RANDOM_FLIP=${RANDOM_FLIP:-1}    # 1 to enable, 0 to disable
CACHE_IN_MEMORY=${CACHE_IN_MEMORY:-0}
VAL_RATIO=${VAL_RATIO:-0.05}     # used only if VAL_DIR is empty

# -----------------------------
# Task-specific defaults
# -----------------------------
case "${TASK}" in
  CT|ct)
    CONFIG_PATH=${CONFIG_PATH:-configs/ve/AAPM_256_ncsnpp_Chung.yaml}
    TRAIN_DIR=${TRAIN_DIR:-/data/AAPM/train}
    VAL_DIR=${VAL_DIR:-/data/AAPM/val}
    WORKDIR=${WORKDIR:-./experiments/aapm_256_ncsnpp}
    NORMALIZATION=${NORMALIZATION:-ct}
    HU_MIN=${HU_MIN:--1024}
    HU_MAX=${HU_MAX:-3072}
    ;;
  MRI|mri)
    CONFIG_PATH=${CONFIG_PATH:-configs/ve/BMR_ZSR_256.yaml}
    TRAIN_DIR=${TRAIN_DIR:-/data/jionkim/ISCS-main/train/}
    VAL_DIR=${VAL_DIR:-/data/jionkim/ISCS-main/val}
    WORKDIR=${WORKDIR:-./experiments/bmr_256_ncsnpp}
    NORMALIZATION=${NORMALIZATION:-minmax}
    HU_MIN=${HU_MIN:--1024}
    HU_MAX=${HU_MAX:-3072}
    ;;
  *)
    echo "[ERROR] Unsupported TASK='${TASK}'. Use CT or MRI."
    exit 1
    ;;
esac

# -----------------------------
# Validation
# -----------------------------
if [[ ! -f "${SCRIPT}" ]]; then
  echo "[ERROR] Cannot find ${SCRIPT}. Run this from the ISCS repo root or set SCRIPT=/path/to/train_iscs_score_model.py"
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Cannot find config: ${CONFIG_PATH}"
  exit 1
fi

if [[ ! -d "${TRAIN_DIR}" ]]; then
  echo "[ERROR] TRAIN_DIR does not exist: ${TRAIN_DIR}"
  exit 1
fi

mkdir -p "${WORKDIR}"

# -----------------------------
# Optional flags
# -----------------------------
EXTRA_ARGS=()

if [[ -n "${VAL_DIR}" ]]; then
  EXTRA_ARGS+=(--val-dir "${VAL_DIR}")
fi

if [[ "${SNAPSHOT_SAMPLING}" == "1" ]]; then
  EXTRA_ARGS+=(--snapshot-sampling)
fi

if [[ "${LIKELIHOOD_WEIGHTING}" == "1" ]]; then
  EXTRA_ARGS+=(--likelihood-weighting)
fi

if [[ "${RANDOM_FLIP}" == "1" ]]; then
  EXTRA_ARGS+=(--random-flip)
fi

if [[ "${CACHE_IN_MEMORY}" == "1" ]]; then
  EXTRA_ARGS+=(--cache-in-memory)
fi

# -----------------------------
# Run
# -----------------------------
echo "[INFO] TASK              : ${TASK}"
echo "[INFO] CONFIG_PATH       : ${CONFIG_PATH}"
echo "[INFO] TRAIN_DIR         : ${TRAIN_DIR}"
echo "[INFO] VAL_DIR           : ${VAL_DIR:-<split from train>}"
echo "[INFO] WORKDIR           : ${WORKDIR}"
echo "[INFO] GPU               : ${GPU}"
echo "[INFO] BATCH_SIZE        : ${BATCH_SIZE}"
echo "[INFO] N_ITERS           : ${N_ITERS}"
echo "[INFO] NORMALIZATION     : ${NORMALIZATION}"
echo "[INFO] PLANE             : ${PLANE}"

set -x
"${PYTHON_BIN}" "${SCRIPT}" \
  --config-path "${CONFIG_PATH}" \
  --train-dir "${TRAIN_DIR}" \
  --workdir "${WORKDIR}" \
  --gpu "${GPU}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --n-iters "${N_ITERS}" \
  --lr "${LR}" \
  --beta1 "${BETA1}" \
  --adam-eps "${ADAM_EPS}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --warmup "${WARMUP}" \
  --grad-clip "${GRAD_CLIP}" \
  --log-freq "${LOG_FREQ}" \
  --eval-freq "${EVAL_FREQ}" \
  --snapshot-freq "${SNAPSHOT_FREQ}" \
  --plane "${PLANE}" \
  --normalization "${NORMALIZATION}" \
  --hu-min "${HU_MIN}" \
  --hu-max "${HU_MAX}" \
  --min-slice-std "${MIN_SLICE_STD}" \
  --val-ratio "${VAL_RATIO}" \
  --split-seed "${SPLIT_SEED}" \
  --seed "${SEED}" \
  --num-workers "${NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}"
set +x

echo "[DONE] Training command finished. Outputs are under: ${WORKDIR}"
