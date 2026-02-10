#!/bin/bash
#SBATCH --job-name=q1b
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
# --constraint="a100-80G"
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%A_bernett_train_q1b_%a.out
#SBATCH --error=logs/%A_bernett_train_q1b_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu

module purge
module load ngc/1.0
module load anaconda/2024.10
#module load pytorch/2.5.1-cuda12.1-cudnn9


#cd /cluster/tufts/cowenlab/wlou01/D-SCRIPT

module purge   
hash -r

# Activate conda properly
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate dscript



export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r


TOPSY_TURVY=
TRAIN=/cluster/tufts/cowenlab/tt3d+/data/gold_data/bernett_train.tsv
TEST=/cluster/tufts/cowenlab/tt3d+/data/gold_data/bernett_validation.tsv

# These are *paths* and numeric dims, not full flags
EMBEDDING=/cluster/tufts/cowenlab/tt3d+/data/esm2/bernett
EMBEDDING_DIM=1280

OUTPUT_BASE=/cluster/tufts/cowenlab/wlou01/D-SCRIPT/results
OUTPUT_FOLDER=${OUTPUT_BASE}/bernett_esm2_train_q1b
OUTPUT_PREFIX=bernett
FOLDSEEK_FASTA=/cluster/tufts/cowenlab/tt3d+/data/foldseek_files/bernett.fasta

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

python -m dscript.commands.train_q1 \
    --train "${TRAIN}" \
    --test "${TEST}" \
    ${EMBEDDING_FLAG} \
    ${EMBEDDING_DIM_FLAG} \
    ${TOPSY_TURVY} \
    --outfile "${OUTPUT_FOLDER}/${OUTPUT_PREFIX}_results.log" \
    --save-prefix "${OUTPUT_FOLDER}/${OUTPUT_PREFIX}_q1" \
    --device "${DEVICE}" \
    --lr 0.0005 \
    --lambda 0.05 \
    --num-epoch 10 \
    --weight-decay 0 \
    --batch-size 16 \
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