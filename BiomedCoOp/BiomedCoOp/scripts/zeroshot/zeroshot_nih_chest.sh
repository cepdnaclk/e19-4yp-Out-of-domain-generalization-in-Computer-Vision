CFG=vit_b16
DATA=/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/NIH_Chest/
DATASET=nihchest
MODEL=BiomedCLIP
METHOD=Zeroshot
TRAINER=Zeroshot${MODEL}

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${METHOD}/${CFG}.yaml \
--output-dir output/${DATASET}/${TRAINER}/${CFG} \
--eval-only