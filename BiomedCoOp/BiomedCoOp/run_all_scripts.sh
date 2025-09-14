#!/bin/bash

# wrapper script to run few_shot_camelyon17.sh for multiple shot settings

SCRIPT=./scripts/cocoop/few_shot_camelyon17.sh

# run with different shot counts
for SHOTS in 1 2 4 8 16
do
    echo ">>> Running with ${SHOTS} shots"
    bash $SCRIPT ${SHOTS}
done

# run with full dataset (no shot argument)
echo ">>> Running with full dataset"
bash $SCRIPT None
