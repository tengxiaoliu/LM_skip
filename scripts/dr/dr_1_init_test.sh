set -x
MODEL_PATH="outputs/phi/dr/model/phi_normal"
TOKENIZER_PATH="microsoft/Phi-3-mini-4k-instruct"
TAG='phi_normal'
ALIAS='phi'
SAVE_PATH='outputs/phi/dr/test'
MODE='normal'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/vllm_offline_inference.py \
    --data_path  "data/DR/id_test.jsonl" \
    --model_name ${MODEL_PATH} \
    --tokenizer_name ${TOKENIZER_PATH} \
    --dataset_name dr_idtest \
    --tag ${TAG} \
    --max_new_tokens 8000 \
    --world_size 8 \
    --save_path ${SAVE_PATH} \
    --mode ${MODE} \
    --model_alias ${ALIAS}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/vllm_offline_inference.py \
    --data_path "data/DR/ood_easy.jsonl" \
    --model_name ${MODEL_PATH} \
    --tokenizer_name ${TOKENIZER_PATH} \
    --dataset_name dr_oodeasy \
    --tag ${TAG} \
    --max_new_tokens 8000 \
    --world_size 8 \
    --save_path ${SAVE_PATH} \
    --mode ${MODE} \
    --model_alias ${ALIAS}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/vllm_offline_inference.py \
    --data_path  "data/DR/ood_hard.jsonl" \
    --model_name ${MODEL_PATH} \
    --tokenizer_name ${TOKENIZER_PATH} \
    --dataset_name dr_oodhard \
    --tag ${TAG} \
    --max_new_tokens 8000 \
    --world_size 8 \
    --save_path ${SAVE_PATH} \
    --mode ${MODE} \
    --model_alias ${ALIAS}