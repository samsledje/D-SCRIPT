#!/bin/bash

TRAIN="seqs-pairs/pairs/human_train.tsv"
TEST="seqs-pairs/pairs/human_test.tsv"
EMBEDDING="embeddings/human.h5"
OUTPUT_BASE="results"
OUTPUT_FOLDER="${OUTPUT_BASE}/tt"
OUTPUT_PREFIX="results-"

FOLDSEEK_FASTA="human.fa"
BACKBONE_FASTA="human.12st.fa"

TOPSY_TURVY=""
EMBEDDING_DIM=""

usage() {
    echo "
Usage: ./train.sh [options]

General:
    -d DEVICE            GPU/CPU device to use
    -t TRAIN_FILE        Training pairs file
    -T TEST_FILE         Testing pairs file
    -e EMBEDDING_FILE    Input embedding (.h5)
    -E DIM               Override input embedding dimension
    -o OUTPUT_SUBDIR     Folder name under OUTPUT_BASE
    -p OUTPUT_PREFIX     Prefix for saved model files

Models:
    -v                   Enable Topsy-Turvy mode

Foldseek:
    -f                   Enable Foldseek features (TT3D)
    -F FOLDSEEK_FASTA    Foldseek 3Di FASTA file

Backbone:
    -b                   Enable Backbone features (TT3D+)
    -B BACKBONE_FASTA     Backbone 3Di FASTA file
"
}

#########################################
# Parse Options
#########################################

while getopts "d:t:T:e:E:o:p:fF:bB:vh" args; do
    case $args in
        d) DEVICE=${OPTARG} ;;
        t) TRAIN=${OPTARG} ;;
        T) TEST=${OPTARG} ;;
        e) EMBEDDING=${OPTARG} ;;
        E) EMBEDDING_DIM="--input-dim ${OPTARG}" ;;
        o) OUTPUT_FOLDER="${OUTPUT_BASE}/${OPTARG}" ;;
        p) OUTPUT_PREFIX=${OPTARG} ;;

        # Topsy-Turvy
        v) TOPSY_TURVY="--topsy-turvy --glider-weight 0.2 --glider-thres 0.925" ;;

        # Foldseek
        f) FOLDSEEK=1 ;;
        F) FOLDSEEK_FASTA=${OPTARG} ;;

        # Backbone
        b) BACKBONE=1 ;;
        B) BACKBONE_FASTA=${OPTARG} ;;

        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

FOLDSEEK_CMD=""
if [ ! -z "$FOLDSEEK" ]; then
    FOLDSEEK_CMD="--allow_foldseek --foldseek_fasta ${FOLDSEEK_FASTA} --add_foldseek_after_projection"
fi

BACKBONE_CMD=""
if [ ! -z "$BACKBONE" ]; then
    BACKBONE_CMD="--allow_backbone3di --backbone3di_fasta ${BACKBONE_FASTA}"
fi

mkdir -p "${OUTPUT_FOLDER}"

echo "Running DSCRIPT with:"
echo "  TRAIN=${TRAIN}"
echo "  TEST=${TEST}"
echo "  EMBEDDING=${EMBEDDING}"
echo "  OUTPUT=${OUTPUT_FOLDER}"
echo ""

dscript train --train $TRAIN --test $TEST --embedding $EMBEDDING $TOPSY_TURVY \
              --outfile ${OUTPUT_FOLDER}/results.log \
              --save-prefix ${OUTPUT_FOLDER}/${OUTPUT_PREFIX} \
              --lr 0.0005 --lambda 0.05 --num-epoch 10 ${EMBEDDING_DIM} \
              --weight-decay 0 --batch-size 25 --pool-width 9 \
              --kernel-width 7 --dropout-p 0.2 --projection-dim 100 \
              --hidden-dim 50 --kernel-width 7 --device $DEVICE \
              $FOLDSEEK_CMD $BACKBONE_CMD

