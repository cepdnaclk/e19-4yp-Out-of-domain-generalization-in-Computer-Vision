# training script: an example
DATA=/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/wbc_att/
DATASET=wbc_att
SHOTS=$1
OUTPUT_DIR=./output_${DATASET}_shot${SHOTS}

python train.py \
 --config-file configs/XCoOp/vit_b16_c4_batch32_wbc_att.yaml\
 --dataset-config-file configs/datasets/wbc_att.yaml \
 --trainer XCoOp \
 --root ${ROOT} \
 --output-dir ${OUTPUT_DIR} \
 --seed 3407 \
 --resume false \
 --shots ${SHOTS}

