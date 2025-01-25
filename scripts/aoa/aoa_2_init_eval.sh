#!/bin/bash

TASK="aoa"
TAG="phi_normal"
ALIAS="phi"
MODE="normal"
SAVE_PATH='outputs/phi/aoa/test'

echo "=== Evaluation of AOA init ==="
# in-domain
python src/evaluate_aoa.py \
--raw_file "data/AOA/id_test.jsonl" \
--pred_file "${SAVE_PATH}/${TAG}_aoa_idtest_${ALIAS}_normal_preds.jsonl" \
--run_name "${TASK}_idtest_${TAG}" \
--tag ${TASK} \
--mode ${MODE}

# ood-easy
python src/evaluate_aoa.py \
--raw_file "data/AOA/ood_easy.jsonl" \
--pred_file "${SAVE_PATH}/${TAG}_aoa_oodeasy_${ALIAS}_normal_preds.jsonl" \
--run_name "${TASK}_oodeasy_${TAG}" \
--tag ${TASK} \
--mode ${MODE}

# ood-hard
python src/evaluate_aoa.py \
--raw_file "data/AOA/ood_hard.jsonl" \
--pred_file "${SAVE_PATH}/${TAG}_aoa_oodhard_${ALIAS}_normal_preds.jsonl" \
--run_name "${TASK}_oodhard_${TAG}" \
--tag ${TASK} \
--mode ${MODE}
