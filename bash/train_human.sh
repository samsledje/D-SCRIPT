#!/bin/bash
#SBATCH --job-name=human
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100-80G"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%A_bernett_train_human_%a.out
#SBATCH --error=logs/%A_bernett_train_human_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu
module purge
module load ngc/1.0
module load pytorch/2.5.1-cuda12.1-cudnn9

cd /cluster/tufts/cowenlab/wlou01/D-SCRIPT
export PYTHONNOUSERSITE=1

# module python
PY=/cluster/tufts/ngc/tools/pytorch/2.5.1-cuda12.1-cudnn9/bin/python


# import torch_optimizer from conda env
# export PYTHONPATH=/cluster/tufts/cowenlab/wlou01/condaenv/dscript/lib/python3.11/site-packages:$PYTHONPATH
DEPS=/cluster/tufts/cowenlab/wlou01/pymoddeps
#mkdir -p "$DEPS"
export PYTHONPATH="$DEPS:/cluster/tufts/cowenlab/wlou01/D-SCRIPT:$PYTHONPATH"

# Install missing deps (safe to rerun)
#$PY -m pip install --upgrade --no-deps --target "$DEPS" protobuf

$PY - <<'EOF'
import sys, torch
import torch_optimizer
print('torch_optimizer OK')
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
sys.exit(0 if ok else 1)
EOF


if [ $? -ne 0 ]; then
    echo "ERROR: CUDA not available. Exiting."
    exit 1
fi



TOPSY_TURVY=
TRAIN=/cluster/tufts/cowenlab/tt3d+/data/pair_files/human_train.tsv
TEST=/cluster/tufts/cowenlab/tt3d+/data/pair_files/human_test.tsv

# These are *paths* and numeric dims, not full flags
EMBEDDING=/cluster/tufts/cowenlab/tt3d+/data/esm2_mapped/human
EMBEDDING_DIM=1280

OUTPUT_BASE=/cluster/tufts/cowenlab/wlou01/D-SCRIPT/results
OUTPUT_FOLDER=${OUTPUT_BASE}/human_esm2_train
OUTPUT_PREFIX=bernett
FOLDSEEK_FASTA=/cluster/tufts/cowenlab/tt3d+/data/foldseek_files/human.fasta

DEVICE=0

usage() {
    echo "Usage: ./train.sh [-d DEVICE] [-v] [-f] [-F foldseek_fasta_file] [-e EMB_DIR] [-E EMB_DIM]

    -d: CUDA device index (default: 0)
    -v: When set, trains a Topsy-Turvy model
    -f: When set, trains a TT3D model (enables foldseek features)
    -F: Foldseek 3di sequences fasta file (used only when -f is set)
    -e: Embedding h5py file or directory (default: $EMBEDDING)
    -E: Embedding dimension (default: $EMBEDDING_DIM)
    "
}

while getopts "d:t:fF:T:e:E:vo:p:" args; do
    case $args in
        d) DEVICE=${OPTARG} ;;
        t) TRAIN=${OPTARG} ;;
        T) TEST=${OPTARG} ;;
        e) EMBEDDING=${OPTARG} ;;
        E) EMBEDDING_DIM=${OPTARG} ;;
        v) TOPSY_TURVY="--topsy-turvy --glider-weight 0.2 --glider-thres 0.925" ;;
        o) OUTPUT_FOLDER=${OUTPUT_BASE}/${OPTARG} ;;
        p) OUTPUT_PREFIX=${OPTARG} ;;
        f) FOLDSEEK="" ;;
        F) FOLDSEEK_FASTA=${OPTARG} ;;
        *) usage; exit 1 ;;
    esac
done

# Build flags from the path / dim variables
EMBEDDING_FLAG="--embedding ${EMBEDDING}"
EMBEDDING_DIM_FLAG="--input-dim ${EMBEDDING_DIM}"

BACKBONE_CMD=""
if [ -n "$BACKBONE" ]; then
    BACKBONE_CMD="--allow_backbone3di --backbone3di_fasta ${FOLDSEEK_FASTA}"
fi

FOLDSEEK_CMD=""
if [ -n "$FOLDSEEK" ]; then
    FOLDSEEK_CMD="--allow_foldseek --foldseek_fasta ${FOLDSEEK_FASTA}" # --add_foldseek_after_projection"
fi

if [ ! -d "${OUTPUT_FOLDER}" ]; then
    mkdir -p "${OUTPUT_FOLDER}"
fi

$PY -m dscript.commands.train \
    --train "${TRAIN}" \
    --test "${TEST}" \
    ${EMBEDDING_FLAG} \
    ${EMBEDDING_DIM_FLAG} \
    ${TOPSY_TURVY} \
    --outfile "${OUTPUT_FOLDER}/${OUTPUT_PREFIX}_results.log" \
    --save-prefix "${OUTPUT_FOLDER}/${OUTPUT_PREFIX}" \
    --device "${DEVICE}" \
    --lr 0.0005 \
    --lambda 0.99 \
    --num-epoch 20 \
    --weight-decay 0.001 \
    --batch-size 32 \
    --pool-width 9 \
    --kernel-width 7 \
    --dropout-p 0.2 \
    --projection-dim 100 \
    --hidden-dim 50 \
    ${BACKBONE_CMD} \
    ${FOLDSEEK_CMD} \
    #--log_wandb \
    #--wandb-entity bergerlab-mit \
    #--wandb-project tt3d_backbone 