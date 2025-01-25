#!/bin/bash

TASK="ma"
TAG="phi_normal"
ALIAS="phi"
MODE="normal"
SAVE_PATH='outputs/phi/ma/test'

echo "=== Evaluation of MA init ==="
# in-domain
python src/evaluate.py \
--raw_file "data/MA/id_test.jsonl" \
--pred_file "${SAVE_PATH}/${TAG}_ma_idtest_${ALIAS}_normal_preds.jsonl" \
--run_name "${TASK}_idtest_${TAG}" \
--tag ${TASK} \
--mode ${MODE}

# ood-easy
python src/evaluate.py \
--raw_file "data/MA/ood_easy.jsonl" \
--pred_file "${SAVE_PATH}/${TAG}_ma_oodeasy_${ALIAS}_normal_preds.jsonl" \
--run_name "${TASK}_oodeasy_${TAG}" \
--tag ${TASK} \
--mode ${MODE}

# ood-hard
python src/evaluate.py \
--raw_file "data/MA/ood_hard.jsonl" \
--pred_file  "${SAVE_PATH}/${TAG}_ma_oodhard_${ALIAS}_normal_preds.jsonl" \
--run_name "${TASK}_oodhard_${TAG}" \
--tag ${TASK} \
--mode ${MODE}

