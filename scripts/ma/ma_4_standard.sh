#!/bin/bash

for ITER_NUM in {0..4}
do

  echo "========================================================================"
  echo "=== Iteration for ma post iter standard model sft - iter${ITER_NUM}  ==="
  echo ">>> 1. Inference on train set"

  MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"
  TOKENIZER_NAME="microsoft/Phi-3-mini-4k-instruct"
  SAVE_PATH='outputs/phi/ma/standard'
  ALIAS="phi"
  SEED=200


  WANDB_PROJECT="lm_skip" \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml --num_processes 8 --main_process_port 12306 src/sft.py \
      --do_train \
      --train_dataset "outputs/phi/ma/data/phi_iter${ITER_NUM}_ma_train_${ALIAS}_skip-2_new_sft_normal_data.jsonl" \
      --max_seq_length 4000 \
      --model_name_or_path ${MODEL_NAME} \
      --tokenizer_name ${TOKENIZER_NAME} \
      --per_device_train_batch_size 4 \
      --learning_rate 5e-6 \
      --warmup_ratio 0.05 \
      --num_train_epochs 2 \
      --gradient_accumulation_steps 2 \
      --fp16 \
      --logging_steps 400 \
      --output_dir ${SAVE_PATH} \
      --save_only_model True \
      --seed ${SEED} \
      --report_to "wandb" \
      --logging_first_step \
      --run_name "phi_normal_iter${ITER_NUM}_seed${SEED}"\
      --gradient_checkpointing \
      --mode "normal"

  # test
  MODEL_PATH="${SAVE_PATH}/phi_normal_iter${ITER_NUM}_seed${SEED}"
  TAG="phi_normal_iter${ITER_NUM}_seed${SEED}"
  DELTA=0
  MODE='normal'
  TASK="ma"

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python src/vllm_offline_inference_multi.py \
    --data_path "data/MA/train.jsonl" \
    --model_name ${MODEL_PATH} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --dataset_name ma_idtest \
    --tag ${TAG} \
    --max_new_tokens 8000 \
    --world_size 8 \
    --save_path "${SAVE_PATH}/test" \
    --delta ${DELTA} \
    --mode ${MODE} \
    --model_alias ${ALIAS}

  # eval on the fly
  python src/general_evaluate.py \
    --raw_file "data/MA/id_test.jsonl" \
    --pred_file "${SAVE_PATH}/test/${TAG}_ma_idtest_${ALIAS}_normal_preds.jsonl" \
    --run_name "${TASK}_idtest_${TAG}" \
    --tag ${TASK} \
    --mode ${MODE}

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python src/vllm_offline_inference_multi.py \
    --data_path "data/MA/ood_easy.jsonl" \
    --model_name ${MODEL_PATH} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --dataset_name ma_oodeasy \
    --tag ${TAG} \
    --max_new_tokens 8000 \
    --world_size 8 \
    --save_path "${SAVE_PATH}/test" \
    --delta ${DELTA} \
    --mode ${MODE} \
    --model_alias ${ALIAS}

  # eval on the fly
  python src/general_evaluate.py \
    --raw_file "data/MA/ood_easy.jsonl" \
    --pred_file "${SAVE_PATH}/test/${TAG}_ma_oodeasy_${ALIAS}_normal_preds.jsonl" \
    --run_name "${TASK}_oodeasy_${TAG}" \
    --tag ${TASK} \
    --mode ${MODE}

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python src/vllm_offline_inference_multi.py \
    --data_path 'data/MA/ood_hard.jsonl' \
    --model_name ${MODEL_PATH} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --dataset_name ma_oodhard \
    --tag ${TAG} \
    --max_new_tokens 8000 \
    --world_size 8 \
    --save_path "${SAVE_PATH}/test" \
    --delta ${DELTA} \
    --mode ${MODE} \
    --model_alias ${ALIAS}

  # eval on the fly
  python src/general_evaluate.py \
    --raw_file "data/MA/ood_hard.jsonl" \
    --pred_file "${SAVE_PATH}/test/${TAG}_ma_oodhard_${ALIAS}_normal_preds.jsonl" \
    --run_name "${TASK}_oodhard_${TAG}" \
    --tag ${TASK} \
    --mode ${MODE}

  rm -r ${MODEL_PATH}

done
