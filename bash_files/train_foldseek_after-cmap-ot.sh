#!/bin/sh

TOPSY_TURVY=
DEBUG=                      #_debug
TRAIN=../data/pairs/human_train${DEBUG}
TEST=../data/pairs/human_test${DEBUG}
EMBEDDING=/net/scratch3.mit.edu/scratch3-3/kdevko01/Project/coral_PPI/data/embeddings/human.h5
OUTPUT_FOLDER=dscript${DEBUG}
#fseek_after_human_model_dscript_cmap
OUTPUT_PREFIX=results-
FOLDSEEK_FASTA=../data/foldseek_emb/r2_foldseekrep_seq.fa
FOLDSEEK_VOCAB=../data/foldseek_vocab.json
SAMPLER="../data/models/sampler/sampler-run-Mon-26-Dec-2022-12:07:02-PM-EST/iter_999.sav"
CMAP_TRAIN="../data/pairs/lynntao_pdbseqs_TRAIN-SET_cmap-filtered-lt400${DEBUG}.tsv"
CMAP_TEST="../data/pairs/lynntao_pdbseqs_TEST-SET_cmap-filtered-lt400${DEBUG}.tsv"
CMAP_LANG_EMB=../lynnfiles/new_cmap_embed
CMAP_EMB=../data/embeddings/cmap-latest.h5
CMAP_MODE=ot
FOLDSEEK_CMD=
CMAP_CMD=
CHECKPOINT=
NUM_EPOCH=10
CMAP_LR=0.0001
ESM_CMD=

while getopts "d:t:T:e:Evo:p:s:Xc:C:l:m:fF:M:k:i:L:" args; do
    case $args in
        d) DEVICE=${OPTARG}
        ;;
        t) TRAIN=${OPTARG}
        ;;
        T) TEST=${OPTARG}
        ;;
        e) EMBEDDING=${OPTARG}
        ;;
        E) ESM=_esm
        ;;
        v) TOPSY_TURVY="--topsy-turvy --glider-weight 0.2 --glider-thres 0.925"; OUTPUT_FOLDER=topsyturvy${DEBUG}; #fseek_after_human_model_topsyturvy_cmap;
        ;;
        o) OUTPUT_FOLDER=${OPTARG}
        ;;
        p) OUTPUT_PREFIX=${OPTARG}
        ;;
        s) SAMPLER=${OPTARG}
        ;;
        X) CMAP_CMD="--run-cmap"
        ;;
        c) CMAP_TRAIN=${OPTARG}
        ;;
        C) CMAP_TEST=${OPTARG}
        ;;
        l) CMAP_LANG_EMB=${OPTARG}
        ;;
        m) CMAP_EMB=${OPTARG}
        ;;
        M) CMAP_MODE=${OPTARG}
        ;;
        f) FOLDSEEK_CMD="--allow_foldseek" 
        ;;
        F) FOLDSEEK_FASTA=${OPTARG}
        ;;
        k) CHECKPOINT="--checkpoint ${OPTARG}"
        ;;
        i) NUM_EPOCH=${OPTARG}
        ;;
        L) CMAP_LR=${OPTARG}
        ;;
    esac
done

# Use the ESM language model
if [ ! -z ${ESM} ]
then
    TRAIN=${TRAIN}_esm
    TEST=${TEST}_esm
    EMBEDDING=../data/embeddings/esm_emb.h5
    ESM_CMD="--input-dim 1280"
    OUTPUT_FOLDER=${OUTPUT_FOLDER}_ESM
fi

TRAIN=${TRAIN}.tsv
TEST=${TEST}.tsv

# Setup the foldseek command
if [ ! -z ${FOLDSEEK_CMD} ]; then FOLDSEEK_CMD="${FOLDSEEK_CMD} --foldseek_fasta ${FOLDSEEK_FASTA} --foldseek_vocab ${FOLDSEEK_VOCAB} --add_foldseek_after_projection"; OUTPUT_FOLDER="${OUTPUT_FOLDER}_fseek-after" ;fi


# Setup the cmap command
if [ ! -z ${CMAP_CMD} ] 
then 
    CMAP_CMD="${CMAP_CMD} --contact-map-train ${CMAP_TRAIN} --contact-map-test ${CMAP_TEST} --contact-map-mode ${CMAP_MODE} --contact-map-embedding ${CMAP_LANG_EMB} --contact-maps ${CMAP_EMB} --contact-map-lr ${CMAP_LR} --contact-map-lambda 0.1"
    if [ ${CMAP_MODE} = "ot" ]; then CMAP_CMD="${CMAP_CMD} --contact-map-sampler ${SAMPLER} --ot-cmap-nsamples 100"; fi
    OUTPUT_FOLDER="${OUTPUT_FOLDER}_cmap-${CMAP_MODE}_lr-${CMAP_LR}"
fi

# Create the output folder
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir -p $OUTPUT_FOLDER; fi

#./train_foldseek_after-cmap-ot.sh -v -s ../data/models/sampler/sampler-run-Mon-26-Dec-2022-12\:07\:02-PM-EST/iter_999.sav -d 3

dscript train --train $TRAIN --test $TEST --embedding $EMBEDDING $TOPSY_TURVY \
              -o ${OUTPUT_FOLDER}/results.log \
              --save-prefix  ${OUTPUT_FOLDER}/ep_ \
              --lr 0.0005 --lambda 0.05 --num-epoch ${NUM_EPOCH} --n-jobs 40 \
              --weight-decay 0 --batch-size 25 --pool-width 9 \
              --kernel-width 7 --dropout-p 0.2 --projection-dim 100 ${CHECKPOINT} \
              --hidden-dim 50 --kernel-width 7 --device $DEVICE ${CMAP_CMD} ${FOLDSEEK_CMD} ${ESM_CMD}


# Training CMAP commands: Without FOLDSEEK
## OT
# Topsy turvy: ./train_foldseek_after-cmap-ot.sh -v -X -M ot  -d 3
# D-SCRIPT: ./train_foldseek_after-cmap-ot.sh -v -X -M ot  -d 3

## Regression
# Topsy turvy: ./train_foldseek_after-cmap-ot.sh -v -X -M regression -d 3
# D-SCRIPT: ./train_foldseek_after-cmap-ot.sh -v -X -M regression -d 3

# Without CMAP, Without FOLDSEEK
# Topsy turvy: ./train_foldseek_after-cmap-ot.sh -v -d 3
# D-SCRIPT: ./train_foldseek_after-cmap-ot.sh -v -X -M regression -d 3


#With foldseek, just add -f