#!/bin/bash

ORGS=( ecoli yeast worm mouse fly )

TOPSY_TURVY=
EMBEDDING_DIR=/afs/csail.mit.edu/u/s/samsl/Work/DSCRIPT_Dev_and_Testing/FoldSeek_BriefCommunication/embeddings/
SEQ_DIR=seqs-pairs/pairs
FOLDSEEK_FASTA=../data/r1_foldseekrep_seq.fa
FOLDSEEK_VOCAB=../data/foldseek_vocab.json
MODEL_PARAMS=""
DEVICE=0
while getopts "d:m:T:tf:" args; do
    case $args in
        d) DEVICE=${OPTARG}
        ;;
        m) MODEL=${OPTARG}
        ;;
        T) if [ ${OPTARG} = "fseek_after" ]; then MODEL_PARAMS="${MODEL_PARAMS} --add_foldseek_after_projection --foldseek_vocab ${FOLDSEEK_VOCAB} --foldseek_fasta ${FOLDSEEK_FASTA} --allow_foldseek"; else MODEL_PARAMS=""; fi
        ;;
        t) TOPSY_TURVY="--topsy-turvy --glider-weight 0.2 --glider-thres 0.925"
        ;;
        f) OUTPUT_FILE=${OPTARG}
        ;;
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

#AFTER: ./test.sh -d 1 -m fseek_after_human_model_dscript/ep__epoch07.sav -T fseek_after -D fseek_after_human_model_dscript/eval -f results

#BEFORE: ./test.sh -d 2 -m fseek_before_human_model_dscript/ep__epoch01.sav -T fseek_before -D fseek_before_human_model_dscript/eval -f results

#DSCRIPT: ./test.sh -d 3 -m original_human_model_dscript/ep__epoch03.sav -T dscript -D original_human_model_dscript/eval -f results

#Topsyturvy AFTER: ./test.sh -d 1 -m fseek_after_human_model_dscript/ep__epoch07.sav -T fseek_after -D fseek_after_human_model_dscript/eval -f results -t

#Topsyturvy BEFORE: ./test.sh -d 2 -m fseek_before_human_model_dscript/ep__epoch01.sav -T fseek_before -D fseek_before_human_model_dscript/eval -f results -t

#Topsyturvy DSCRIPT: ./test.sh -d 3 -m original_human_model_dscript/ep__epoch03.sav -T dscript -D original_human_model_dscript/eval -f results -t

## m = x^y, log_x(m) = y, log_2(m) = log_2(2^log_2(m)) = log_2(x^y) = y log_2(x)
## log_2(x) log_x(m) = log_2(m)