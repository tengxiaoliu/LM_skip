#!/bin/bash

for ITER_NUM in {0..4}
do

  echo "========================================================================"
  echo "=== Iteration for ma iter${ITER_NUM} -> iter$((ITER_NUM + 1)) ==="
  echo ">>> 1. Inference on train set"

  export VLLM_WORKER_MULTIPROC_METHOD=spawn

  MODEL_DIR="outputs/phi/ma/model/phi_num_iter${ITER_NUM}"
  MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"
  TOKENIZER_NAME="microsoft/Phi-3-mini-4k-instruct"
  ALIAS="phi"
  DATA_SAVE_PATH="outputs/phi/ma/data"

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python src/vllm_offline_inference.py \
      --data_path  "data/MA/train_cold.jsonl" \
      --model_name ${MODEL_DIR} \
      --save_path ${DATA_SAVE_PATH} \
      --tokenizer_name ${TOKENIZER_NAME} \
      --dataset_name ma_train \
      --tag "phi_iter${ITER_NUM}" \
      --max_new_tokens 8000 \
      --world_size 8 \
      --delta -1 \
      --mode 'num' \
      --model_alias ${ALIAS}

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python src/vllm_offline_inference.py \
      --data_path  "data/MA/train_cold.jsonl" \
      --model_name ${MODEL_DIR} \
      --save_path ${DATA_SAVE_PATH} \
      --tokenizer_name ${TOKENIZER_NAME} \
      --dataset_name addsp_train \
      --tag "phi_iter${ITER_NUM}" \
      --max_new_tokens 8000 \
      --world_size 8 \
      --delta -2 \
      --mode 'num' \
      --model_alias ${ALIAS}

  echo ">>> 2. Prepare SFT data"

  python src/prepare_sft_data_general.py \
      --raw_file "data/MA/train_cold.jsonl" \
      --pred_files "${DATA_SAVE_PATH}/phi_iter${ITER_NUM}_ma_train_${ALIAS}_skip-1_preds.jsonl" \
                   "${DATA_SAVE_PATH}/phi_iter${ITER_NUM}_ma_train_${ALIAS}_skip-2_preds.jsonl" \
      --output_dir ${DATA_SAVE_PATH} \
      --mode new

  if [ $ITER_NUM -gt 0 ]; then
    rm -r ${MODEL_DIR}
    rm -r *checkpoint*
  fi

  echo ">>> 3. SFT new model"

  WANDB_PROJECT="lm_skip" \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml --num_processes 8 --main_process_port 12306 src/sft.py \
      --do_train \
      --train_dataset "${DATA_SAVE_PATH}/phi_iter${ITER_NUM}_ma_train_${ALIAS}_skip-2_new_sft_data.jsonl" \
      --max_seq_length 8000 \
      --model_name_or_path ${MODEL_NAME} \
      --tokenizer_name ${TOKENIZER_NAME} \
      --per_device_train_batch_size 4 \
      --learning_rate 5e-6 \
      --warmup_ratio 0.05 \
      --num_train_epochs 2 \
      --gradient_accumulation_steps 2 \
      --fp16 \
      --logging_steps 20 \
      --output_dir "outputs/phi/ma/model" \
      --save_only_model True \
      --seed 200 \
      --report_to "wandb" \
      --logging_first_step \
      --run_name "phi_num_iter$((ITER_NUM + 1))" \
      --gradient_checkpointing \
      --mode "num"


  echo ">>> 4. Inference under mode num with the new iter model"
  
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  
  MODEL_PATH="outputs/phi/ma/model/phi_num_iter$((ITER_NUM + 1))"
  OUTPUT_PATH="outputs/phi/ma/test"
  TAG="phi_num_iter$((ITER_NUM + 1))"
  DELTA=-1
  MODE='num'

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python src/vllm_offline_inference.py \
      --data_path  "data/MA/id_test.jsonl" \
      --model_name ${MODEL_PATH} \
      --tokenizer_name ${TOKENIZER_NAME} \
      --dataset_name ma_idtest \
      --tag ${TAG} \
      --max_new_tokens 8000 \
      --world_size 8 \
      --save_path ${OUTPUT_PATH} \
      --delta ${DELTA} \
      --mode ${MODE} \
      --model_alias ${ALIAS}
  
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python src/vllm_offline_inference.py \
      --data_path "data/MA/ood_easy.jsonl" \
      --model_name ${MODEL_PATH} \
      --tokenizer_name ${TOKENIZER_NAME} \
      --dataset_name ma_oodeasy \
      --tag ${TAG} \
      --max_new_tokens 8000 \
      --world_size 8 \
      --save_path ${OUTPUT_PATH} \
      --delta ${DELTA} \
      --mode ${MODE} \
      --model_alias ${ALIAS}
  
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python src/vllm_offline_inference.py \
      --data_path "data/MA/ood_hard.jsonl" \
      --model_name ${MODEL_PATH} \
      --tokenizer_name ${TOKENIZER_NAME} \
      --dataset_name ma_oodhard \
      --tag ${TAG} \
      --max_new_tokens 8000 \
      --world_size 8 \
      --save_path ${OUTPUT_PATH} \
      --delta ${DELTA} \
      --mode ${MODE} \
      --model_alias ${ALIAS}

  echo "=== DONE Iteration for ma iter${ITER_NUM} -> iter$((ITER_NUM + 1)) ==="
  echo "=========================================================================DONE"

done
