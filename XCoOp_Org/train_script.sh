# training script: an example
DATASET=cam17
ROOT=/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17
OUTPUT_DIR=./output

XCoOp_Org\configs\XCoOp\vit_b16_c4_batch32_camelyon17.yaml
if [ ${DATASET} == "camelyon17" ];
then
python train.py \
 --config-file configs/XCoOp/vit_b16_c4_batch32_camelyon17.yaml \
 --dataset-config-file configs/datasets/camelyon17.yaml \
 --trainer XCoOp \
 --root ${ROOT} \
 --output-dir ${OUTPUT_DIR} \
 --seed 3407 \
 --resume false
fi

