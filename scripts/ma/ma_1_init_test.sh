set -x
MODEL_PATH="outputs/phi/ma/model/phi_normal"
TOKENIZER_PATH="microsoft/Phi-3-mini-4k-instruct"
TAG='phi_normal'
ALIAS='phi'
SAVE_PATH='outputs/phi/ma/test'
MODE='normal'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/vllm_offline_inference.py \
    --data_path  "data/MA/id_test.jsonl" \
    --model_name ${MODEL_PATH} \
    --tokenizer_name ${TOKENIZER_PATH} \
    --dataset_name ma_idtest \
    --tag ${TAG} \
    --max_new_tokens 8000 \
    --world_size 8 \
    --save_path ${SAVE_PATH} \
    --mode ${MODE} \
    --model_alias ${ALIAS}


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/vllm_offline_inference.py \
    --data_path  "data/MA/ood_easy.jsonl" \
    --model_name ${MODEL_PATH} \
    --tokenizer_name ${TOKENIZER_PATH} \
    --dataset_name ma_oodeasy \
    --tag ${TAG} \
    --max_new_tokens 8000 \
    --world_size 8 \
    --save_path ${SAVE_PATH} \
    --mode ${MODE} \
    --model_alias ${ALIAS}


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/vllm_offline_inference.py \
    --data_path  "data/MA/ood_hard.jsonl" \
    --model_name ${MODEL_PATH} \
    --tokenizer_name ${TOKENIZER_PATH} \
    --dataset_name ma_oodhard \
    --tag ${TAG} \
    --max_new_tokens 8000 \
    --world_size 8 \
    --save_path ${SAVE_PATH} \
    --mode ${MODE} \
    --model_alias ${ALIAS}