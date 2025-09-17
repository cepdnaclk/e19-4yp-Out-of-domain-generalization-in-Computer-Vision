# training script: an example
DATASET=camelyon17
ROOT=/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/
SHOTS=1
OUTPUT_DIR=./output_${DATASET}_shot${SHOTS}

python train.py \
 --config-file configs/XCoOp/vit_b16_c4_batch32_camelyon17.yaml \
 --dataset-config-file configs/datasets/camelyon17.yaml \
 --trainer XCoOp \
 --root ${ROOT} \
 --output-dir ${OUTPUT_DIR} \
 --seed 3407 \
 --resume false \
 --shots ${SHOTS}

