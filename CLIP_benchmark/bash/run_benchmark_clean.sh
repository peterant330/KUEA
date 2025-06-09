#!/bin/bash
set -e # stop on error
# add parent to python path
export PYTHONPATH="../":"${PYTHONPATH}"

SECONDS=0
SAMPLES=-1
BS=64

SAVE_DIR=../CLIP_benchmark/result  # TODO
mkdir -p "$SAVE_DIR"
python -m clip_benchmark.cli eval --dataset_root "../CLIP_benchmark/datasets/wds_{dataset_cleaned}" --dataset benchmark/datasets.txt \
--pretrained_model benchmark/models.txt \
--output "${SAVE_DIR}/clean_{model}_{pretrained}_beta{beta}_{dataset}_{n_samples}_bs{bs}_{attack}_{eps}_{iterations}.json" \
--attack none --eps 1 \
--batch_size $BS --n_samples $SAMPLES \


hours=$((SECONDS / 3600))
minutes=$(( (SECONDS % 3600) / 60 ))
echo "[Runtime] $hours h $minutes min"
