# training script: an example
DATA=/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/
DATASET=derm7pt
SHOTS=$1
OUTPUT_DIR=./output_${DATASET}_shot${SHOTS}

python train.py \
 --config-file configs/XCoOp/vit_b16_c4_batch32_derm7pt.yaml \
 --dataset-config-file configs/datasets/derm7pt.yaml \
 --trainer XCoOp \
 --root ${ROOT} \
 --output-dir ${OUTPUT_DIR} \
 --seed 3407 \
 --resume false \
 --shots ${SHOTS}

