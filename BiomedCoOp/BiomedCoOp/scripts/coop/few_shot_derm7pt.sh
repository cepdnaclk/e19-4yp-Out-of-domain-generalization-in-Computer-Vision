#!/bin/bash

# custom config
DATA=/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/
DATASET=derm7pt
SHOTS=16
MODEL=BiomedCLIP
NCTX=4
CSC=False
CTP=end
CFG=vit_b16

METHOD=CoOp
TRAINER=CoOp_${MODEL}

for SEED in 1 2 3
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
            --config-file configs/trainers/${METHOD}/${CFG}.yaml \
            --output-dir ${DIR} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
        # fi
done