set -x
MODEL_PATH="outputs/phi/aoa/model/phi_normal"
TOKENIZER_NAME="microsoft/Phi-3-mini-4k-instruct"
TAG='phi_normal'
ALIAS='phi'
PROMPT_PATH='data/prompts/train_normal.txt'
DELTA=0
SAVE_PATH='outputs/phi/aoa/test'
MODE='normal'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/vllm_offline_inference.py \
    --data_path "data/AOA/id_test.jsonl" \
    --demonstration_path ${PROMPT_PATH} \
    --model_name ${MODEL_PATH} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --dataset_name aoa_idtest \
    --tag ${TAG} \
    --max_new_tokens 8000 \
    --world_size 8 \
    --delta ${DELTA} \
    --save_path ${SAVE_PATH} \
    --mode ${MODE} \
    --model_alias ${ALIAS}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/vllm_offline_inference.py \
    --data_path "data/AOA/ood_easy.jsonl" \
    --demonstration_path ${PROMPT_PATH} \
    --model_name ${MODEL_PATH} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --dataset_name aoa_oodeasy \
    --tag ${TAG} \
    --max_new_tokens 8000 \
    --world_size 8 \
    --delta ${DELTA} \
    --save_path ${SAVE_PATH} \
    --mode ${MODE} \
    --model_alias ${ALIAS}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/vllm_offline_inference.py \
    --data_path "data/AOA/ood_hard.jsonl" \
    --demonstration_path ${PROMPT_PATH} \
    --model_name ${MODEL_PATH} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --dataset_name aoa_oodhard \
    --tag ${TAG} \
    --max_new_tokens 8000 \
    --world_size 8 \
    --delta ${DELTA} \
    --save_path ${SAVE_PATH} \
    --mode ${MODE} \
    --model_alias ${ALIAS}