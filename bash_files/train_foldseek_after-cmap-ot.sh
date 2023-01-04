#!/bin/sh

TOPSY_TURVY=
TRAIN=seqs-pairs/pairs/human_train.tsv
TEST=seqs-pairs/pairs/human_test.tsv
EMBEDDING=embeddings/human.h5
OUTPUT_FOLDER=dscript
#fseek_after_human_model_dscript_cmap
OUTPUT_PREFIX=results-
FOLDSEEK_FASTA=../../foldseek_emb/r1_foldseekrep_seq.fa
FOLDSEEK_VOCAB=../data/foldseek_vocab.json
SAMPLER="../data/models/sampler/sampler-run-Mon-26-Dec-2022-12:07:02-PM-EST/iter_999.sav"
CMAP_TRAIN=../data/pairs/cmap_train_lt_400.tsv
CMAP_TEST=../data/pairs/cmap_test_lt_400.tsv
CMAP_LANG_EMB=../lynnfiles/new_cmap_embed
CMAP_EMB=../data/embeddings/cmap_filtered_lt_400.h5
CMAP_MODE=ot
FOLDSEEK_CMD=
CMAP_CMD=
while getopts "d:t:T:e:vo:p:s:Xc:C:l:m:fF:M:" args; do
    case $args in
        d) DEVICE=${OPTARG}
        ;;
        t) TRAIN=${OPTARG}
        ;;
        T) TEST=${OPTARG}
        ;;
        e) EMBEDDING=${OPTARG}
        ;;
        v) TOPSY_TURVY="--topsy-turvy --glider-weight 0.2 --glider-thres 0.925"; OUTPUT_FOLDER=topsyturvy; #fseek_after_human_model_topsyturvy_cmap;
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
    esac
done

# Setup the foldseek command
if [ ! -z ${FOLDSEEK_CMD} ]; then FOLDSEEK_CMD="${FOLDSEEK_CMD} --foldseek_fasta ${FOLDSEEK_FASTA} --foldseek_vocab ${FOLDSEEK_VOCAB} --add_foldseek_after_projection"; OUTPUT_FOLDER="${OUTPUT_FOLDER}_fseek-after" ;fi


# Setup the cmap command
if [ ! -z ${CMAP_CMD} ] 
then 
    CMAP_CMD="${CMAP_CMD} --contact-map-train ${CMAP_TRAIN} --contact-map-test ${CMAP_TEST} --contact-map-mode ${CMAP_MODE} --contact-map-embedding ${CMAP_LANG_EMB} --contact-maps ${CMAP_EMB} --contact-map-lr 0.0001 --contact-map-lambda 0.1"
    if [ ${CMAP_MODE} = "ot" ]; then CMAP_CMD="${CMAP_CMD} --contact-map-sampler ${SAMPLER} --ot-cmap-nsamples 100"; fi
    OUTPUT_FOLDER="${OUTPUT_FOLDER}_cmap-${CMAP_MODE}"
fi

# Create the output folder
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir -p $OUTPUT_FOLDER; fi

#./train_foldseek_after-cmap-ot.sh -v -s ../data/models/sampler/sampler-run-Mon-26-Dec-2022-12\:07\:02-PM-EST/iter_999.sav -d 3

dscript train --train $TRAIN --test $TEST --embedding $EMBEDDING $TOPSY_TURVY \
              -o ${OUTPUT_FOLDER}/results.log \
              --save-prefix  ${OUTPUT_FOLDER}/ep_ \
              --lr 0.0005 --lambda 0.05 --num-epoch 10 \
              --weight-decay 0 --batch-size 25 --pool-width 9 \
              --kernel-width 7 --dropout-p 0.2 --projection-dim 100 \
              --hidden-dim 50 --kernel-width 7 --device $DEVICE ${CMAP_COMMANDS} ${FOLDSEEK_CMD}

# Training CMAP commands: Without FOLDSEEK
## OT
# Topsy turvy: ./train_foldseek_after-cmap-ot.sh -v -X -M ot  -d 3
# D-SCRIPT: ./train_foldseek_after-cmap-ot.sh -v -X -M ot  -d 3

## Regression
# Topsy turvy: ./train_foldseek_after-cmap-ot.sh -v -X -M regression -d 3
# D-SCRIPT: ./train_foldseek_after-cmap-ot.sh -v -X -M regression -d 3

