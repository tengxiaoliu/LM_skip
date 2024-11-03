
OUTPUT_DIR="outputs/phi/aoa/model"

# standard model (normal training)
WANDB_PROJECT="trl_sft" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml --num_processes 8 --main_process_port 12312 src/sft.py \
    --do_train \
    --train_dataset "data/AOA/train.jsonl" \
    --max_seq_length 4000 \
    --model_name_or_path "microsoft/Phi-3-mini-4k-instruct" \
    --tokenizer_name "microsoft/Phi-3-mini-4k-instruct" \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --logging_steps 20 \
    --output_dir ${OUTPUT_DIR} \
    --save_only_model True \
    --seed 200 \
    --report_to "wandb" \
    --logging_first_step \
    --run_name "phi_normal" \
    --gradient_checkpointing \
    --mode "normal"

# number
WANDB_PROJECT="trl_sft" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml --num_processes 8 --main_process_port 12312 src/sft.py \
    --do_train \
    --train_dataset "data/AOA/train.jsonl" \
    --max_seq_length 4000 \
    --model_name_or_path "microsoft/Phi-3-mini-4k-instruct" \
    --tokenizer_name "microsoft/Phi-3-mini-4k-instruct" \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --logging_steps 20 \
    --output_dir ${OUTPUT_DIR} \
    --save_only_model True \
    --seed 200 \
    --report_to "wandb" \
    --logging_first_step \
    --run_name "phi_num_iter0" \
    --gradient_checkpointing \
    --mode "num"

