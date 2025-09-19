#!/bin/bash

# custom config
DATA=/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/
DATASET=derm7pt
SHOTS=$1
MODEL=BiomedCLIP
NCTX=4
CSC=False
CTP=end

METHOD=BiomedCoOp
TRAINER=BiomedCoOp_${MODEL}

for SEED in 1
do
        DIR=output/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        # if [ -d "$DIR" ]; then
        #     echo "Oops! The results exist at ${DIR} (so skip this job)"
        # else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml  \
        --output-dir ${DIR} \
        TRAINER.BIOMEDCOOP.N_CTX ${NCTX} \
        TRAINER.BIOMEDCOOP.CSC ${CSC} \
        TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
        # fi
done