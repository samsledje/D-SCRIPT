#!/bin/bash

TOPSY_TURVY=
TRAIN=seqs-pairs/pairs/human_train.tsv
TEST=seqs-pairs/pairs/human_test.tsv
EMBEDDING=/afs/csail.mit.edu/u/s/samsl/Work/DSCRIPT_Dev_and_Testing/FoldSeek_BriefCommunication/embeddings/human.h5
OUTPUT_BASE=/afs/csail.mit.edu/u/s/samsl/Work/DSCRIPT_Dev_and_Testing/FoldSeek_BriefCommunication/results
OUTPUT_FOLDER=${OUTPUT_BASE}/tt
OUTPUT_PREFIX=results-
FOLDSEEK_FASTA=../data/r1_foldseekrep_seq.fa

usage() {
    echo "Usage ./train.sh [-d DEVICE] [-v] [-f] [-F foldseek_fasta_file]

    -v: When set, returns a Topsy-Turvy model
    -f: When set, returns a TT3D model
    -F: Used only when -f is set. When -f is set, use this argument to pass in the foldseek 3di sequences in the fasta format
    "
}

while getopts "d:t:fF:T:e:E:vo:p:" args; do
    case $args in
        d) DEVICE=${OPTARG}
        ;;
        t) TRAIN=${OPTARG}
        ;;
        T) TEST=${OPTARG}
        ;;
        e) EMBEDDING=${OPTARG}
        ;;
        E) EMBEDDING_DIM="--input-dim ${OPTARG}"
        ;;
        v) TOPSY_TURVY="--topsy-turvy --glider-weight 0.2 --glider-thres 0.925";
        ;;
        o) OUTPUT_FOLDER=${OUTPUT_BASE}/${OPTARG}
        ;;
        p) OUTPUT_PREFIX=${OPTARG}
        ;;
        f) FOLDSEEK=1
        ;;
        F) FOLDSEEK_FASTA=${OPTARG}
        ;;
        *) usage
        exit 1;;
    esac
done


FOLDSEEK_CMD=""
if [ ! -z $FOLDSEEK ]; then FOLDSEEK_CMD="--allow_foldseek --foldseek_fasta ${FOLDSEEK_FASTA} --add_foldseek_after_projection"; fi

if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir -p $OUTPUT_FOLDER; fi



dscript train --train $TRAIN --test $TEST --embedding $EMBEDDING $TOPSY_TURVY \
              --outfile ${OUTPUT_FOLDER}/results.log \
              --save-prefix  ${OUTPUT_FOLDER}/ep_ \
              --lr 0.0005 --lambda 0.05 --num-epoch 10 ${EMBEDDING_DIM}\
              --weight-decay 0 --batch-size 25 --pool-width 9 \
              --kernel-width 7 --dropout-p 0.2 --projection-dim 100 \
              --hidden-dim 50 --kernel-width 7 --device $DEVICE $FOLDSEEK_CMD
