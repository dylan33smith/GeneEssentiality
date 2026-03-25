#!/bin/bash
# Runner script for training experiments

EXP_NAME=$1

cd /home/ds85/projects/GeneEssentiality/autoresearch_regression
source /home/ds85/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

LOG_FILE="run_${EXP_NAME}.log"

echo "Starting $EXP_NAME at $(date)" >> "$LOG_FILE"
python3 "train_${EXP_NAME}.py" >> "$LOG_FILE" 2>&1
echo "Finished $EXP_NAME at $(date)" >> "$LOG_FILE"
