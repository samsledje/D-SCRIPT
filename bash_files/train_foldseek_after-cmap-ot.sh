#!/bin/sh

TOPSY_TURVY=
TRAIN=seqs-pairs/pairs/human_train.tsv
TEST=seqs-pairs/pairs/human_test.tsv
EMBEDDING=embeddings/human.h5
OUTPUT_FOLDER=fseek_after_human_model_dscript_cmap_ot
OUTPUT_PREFIX=results-
FOLDSEEK_FASTA=../../foldseek_emb/r1_foldseekrep_seq.fa
FOLDSEEK_VOCAB=../data/foldseek_vocab.json
SAMPLER=../data/models/sampler/iter_9.sav
CMAP_TRAIN=../data/pairs/cmap_train_lt_400.tsv
CMAP_TEST=../data/pairs/cmap_test_lt_400.tsv
CMAP_LANG_EMB=../lynnfiles/new_cmap_embed
CMAP_EMB=../data/embeddings/cmap_filtered_lt_400.h5
CMAP_MODE=ot
while getopts "d:t:T:e:vo:p:s:c:C:l:m:" args; do
    case $args in
        d) DEVICE=${OPTARG}
        ;;
        t) TRAIN=${OPTARG}
        ;;
        T) TEST=${OPTARG}
        ;;
        e) EMBEDDING=${OPTARG}
        ;;
        v) TOPSY_TURVY="--topsy-turvy --glider-weight 0.2 --glider-thres 0.925"; OUTPUT_FOLDER=fseek_after_human_model_topsyturvy;
        ;;
        o) OUTPUT_FOLDER=${OPTARG}
        ;;
        p) OUTPUT_PREFIX=${OPTARG}
        ;;
        s) SAMPLER=${OPTARG}
        ;;
        c) CMAP_TRAIN=${OPTARG}
        ;;
        C) CMAP_TEST=${OPTARG}
        ;;
        l) CMAP_LANG_EMB=${OPTARG}
        ;;
        m) CMAP_EMB=${OPTARG}
        ;;
    esac
done
    
if [ ! -d ${OUTPUT_FOLDER} ]; then mkdir -p $OUTPUT_FOLDER; fi



dscript train --train $TRAIN --test $TEST --embedding $EMBEDDING $TOPSY_TURVY \
              -o ${OUTPUT_FOLDER}/results.log \
              --save-prefix  ${OUTPUT_FOLDER}/ep_ \
              --lr 0.0005 --lambda 0.05 --num-epoch 10 \
              --weight-decay 0 --batch-size 25 --pool-width 9 \
              --kernel-width 7 --dropout-p 0.2 --projection-dim 100 \
              --hidden-dim 50 --kernel-width 7 --device $DEVICE --run-cmap --contact-map-train ${CMAP_TRAIN} --contact-map-test ${CMAP_TEST} --contact-map-mode ot --contact-map-embedding ${CMAP_LANG_EMB} --contact-maps ${CMAP_EMB} --contact-map-sampler ${SAMPLER} --ot-cmap-nsamples 100 --contact-map-lr 1 --contact-map-lambda 0.2 # --allow_foldseek --foldseek_fasta ${FOLDSEEK_FASTA} --foldseek_vocab ${FOLDSEEK_VOCAB} --add_foldseek_after_projection  ## need to add the foldseek part
