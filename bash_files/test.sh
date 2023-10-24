#!/bin/bash

ORGS=( ecoli yeast worm mouse fly )

TOPSY_TURVY=
EMBEDDING_DIR=/afs/csail.mit.edu/u/s/samsl/Work/TT3D/FoldSeek_BriefCommunication/embeddings/
SEQ_DIR=seqs-pairs/pairs
FOLDSEEK_FASTA=../data/r3-ALLSPECIES_foldseekrep_seq.fa
FOLDSEEK_VOCAB=../data/foldseek_vocab.json
MODEL_PARAMS=""
DEVICE=0

usage() {
    echo "USAGE: ./test.sh [-d DEVICE] [-m MODEL] [-T MODEL_TYPE]

    OPTIONS:

    -d DEVICE: Device used
    -m MODEL: The model sav file
    -T: Set this flag if if the model passed by the '-m MODEL' command is a TT3D Model. Unset this for Topsy-Turvy/D-SCRIPT
    "
}


while getopts "d:m:T" args; do
    case $args in
        d) DEVICE=${OPTARG}
        ;;
        m) MODEL=${OPTARG}
        ;;
        T) MODEL_PARAMS="${MODEL_PARAMS} --add_foldseek_after_projection --foldseek_vocab ${FOLDSEEK_VOCAB} --foldseek_fasta ${FOLDSEEK_FASTA} --allow_foldseek"
        ;;
        *) usage
        exit 1;
    esac
done

# Construct the folder
OUTPUT_FLD=${MODEL%/*}
OUTPUT_FILE=${MODEL##*/}
OUTPUT_FILE_PREF=${OUTPUT_FILE%.*}
OUTPUT_FOLDER=${OUTPUT_FLD}/eval-${OUTPUT_FILE_PREF}

echo "Output folder: ${OUTPUT_FOLDER}, model: ${MODEL}, DEVICE: ${DEVICE}"
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir $OUTPUT_FOLDER; fi

for ORG in ${ORGS[@]}
do
    EMBEDDING=${EMBEDDING_DIR}/${ORG}.h5
    TEST=${SEQ_DIR}/${ORG}_test.tsv
    OP_FOLDER_ORG=${OUTPUT_FOLDER}/${ORG}
    if [ ! -d ${OP_FOLDER_ORG} ]; then mkdir -p ${OP_FOLDER_ORG}; fi
    OP_FILE=${OP_FOLDER_ORG}/${OUTPUT_FILE}
    dscript evaluate --model ${MODEL} --embedding ${EMBEDDING} --test ${TEST} -d $DEVICE ${MODEL_PARAMS} -o $OP_FILE
done
