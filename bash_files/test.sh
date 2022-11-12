#!/bin/bash

ORGS=( worm mouse fly ) #ecoli yeast worm mouse fly )

TOPSY_TURVY=
EMBEDDING_DIR=embeddings/
SEQ_DIR=seqs-pairs/pairs
OUTPUT_FOLDER=fseek_after_human_model_dscript
OUTPUT_PREFIX=results-
FOLDSEEK_FASTA=../../foldseek_emb/r1_foldseekrep_seq.fa
FOLDSEEK_VOCAB=../data/foldseek_vocab.json
MODEL_PARAMS="--allow_foldseek --foldseek_fasta ${FOLDSEEK_FASTA} --foldseek_vocab ${FOLDSEEK_VOCAB}"
DEVICE=0
OUTPUT_FILE="results.txt"
while getopts "d:m:T:tD:f:" args; do
    case $args in
        d) DEVICE=${OPTARG}
        ;;
        m) MODEL=${OPTARG}
        ;;
        T) if [ ${OPTARG} = "fseek_before" ]; then MODEL_PARAMS=$MODEL_PARAMS; elif [ ${OPTARG} = "fseek_after" ]; then MODEL_PARAMS="${MODEL_PARAMS} --add_foldseek_after_projection"; else MODEL_PARAMS=""; fi
        ;;
        t) TOPSY_TURVY="--topsy-turvy --glider-weight 0.2 --glider-thres 0.925"
        ;;
        D) OUTPUT_FOLDER=${OPTARG}
        ;;
        f) OUTPUT_FILE=${OPTARG}
        ;;
    esac
done
    
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir -p $OUTPUT_FOLDER; fi

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
