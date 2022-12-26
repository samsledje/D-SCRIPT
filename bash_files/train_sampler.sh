#!/bin/bash

EMBEDDING=../data/embeddings/cmap_filtered_lt_400.h5
CHECKPOINT=
ITER=10

OUTPUT=../data/models/sampler/sampler-run
IFS= read -a args <<< $(date); for a in ${args[@]}; do OUTPUT=${OUTPUT}-${a};done

SAVE_AT_ITER=1
LR=1
DEVICE=1
MAX_DATA=1000
while getopts "e:o:c:i:s:l:d:m:" args; do
    case $args in 
        e) EMBEDDING=${OPTARG}
        ;;
        o) OUTPUT=${OPTARG}
        ;;
        c) CHECKPOINT="--checkpoint ${OPTARG}"
        ;;
        i) ITER=${OPTARG}
        ;;
        s) SAVE_AT_ITER=${OPTARG}
        ;;
        l) LR=${OPTARG}
        ;;
        d) DEVICE=${OPTARG}
        ;;
        m) MAX_DATA=${OPTARG}
    esac
done

echo "MAX_DATA=${MAX_DATA}, ITER=${ITER}, DEVICE=${DEVICE}, OUTPUT=${OUTPUT}, LR=${LR}"

if [ ! -d $OUTPUT ]; then mkdir $OUTPUT; fi

dscript sampler  --embedding $EMBEDDING --output $OUTPUT --device $DEVICE $CHECKPOINT --lr $LR --max-data ${MAX_DATA} --save-at-iter ${SAVE_AT_ITER}