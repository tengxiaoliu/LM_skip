#!/bin/bash

TASK="dr"
TAG="phi_normal"
ALIAS="phi"
MODE="normal"
SAVE_PATH='outputs/phi/dr/test'

echo "=== Evaluation of DR init ==="
# in-domain
python src/evaluate.py \
  --raw_file "data/DRid_test.jsonl" \
  --pred_file "${SAVE_PATH}/${TAG}_dr_idtest_${ALIAS}_normal_preds.jsonl" \
  --run_name "${TASK}_idtest_${TAG}" \
  --tag ${TASK} \
  --mode ${MODE}

# ood-easy
python src/evaluate.py \
  --raw_file "data/DRood_easy.jsonl" \
  --pred_file "${SAVE_PATH}/${TAG}_dr_oodeasy_${ALIAS}_normal_preds.jsonl" \
  --run_name "${TASK}_oodeasy_${TAG}" \
  --tag ${TASK} \
  --mode ${MODE}

# ood-hard
python src/evaluate.py \
  --raw_file "data/DRood_hard.jsonl" \
  --pred_file "${SAVE_PATH}/${TAG}_dr_oodhard_${ALIAS}_normal_preds.jsonl" \
  --run_name "${TASK}_oodhard_${TAG}" \
  --tag ${TASK} \
  --mode ${MODE}
