# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch

from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments
import pandas as pd
from datasets import Dataset
from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
    DataCollatorForCompletionOnlyLM,
    set_seed
)

from utils import *

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


def load_aoa_dataset(path_dict, tokenizer, mode):
    data_bundle = {}

    if mode == 'normal':
        prompt_path = 'v3/prompts/train_normal.txt'
    elif mode == 'num':
        prompt_path = 'v3/prompts/train_num.txt'
    elif mode == 'num_in_gen':
        prompt_path = 'v3/prompts/train_num_in_gen.txt'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    prompt = load_txt(prompt_path)

    for name, path in path_dict.items():
        # print(f"[Data] Loading {name} from {path}")
        input_name = 'input'
        output_name = 'output'

        raw_dataset = load_data_from_file(path)
        processed_dataset = []
        for inst in raw_dataset:

            if 'iter' in path and 'prompt' in inst and 'chosen' in inst:
                if mode == 'num' or mode == 'normal':
                    one_input = inst['prompt'].strip()
                    one_output = inst['chosen'].strip()
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            else:
                cot_answer = ''
                for step in inst['steps']:
                    # cot_answer += f"Step {step['step']}: {step['desc']}.\n"
                    cot_answer += step['equation'] + '\n'

                cot_answer += f"Thus, the answer is {inst['answer']}"

                output_text = cot_answer
                one_input = prompt.replace('[[QUESTION]]', inst['question']).strip()
                one_output = output_text.strip()
                if mode == 'num':
                    one_input = one_input.replace('[[NUM]]', str(inst['num_steps']))
                elif mode == 'num_in_gen':
                    raise NotImplementedError("Need to implement the num_in_gen mode.")

            new_inst = {
                'id': inst['id'],
                input_name: one_input.strip() + '\n',
                output_name: one_output.strip() + tokenizer.eos_token
            }
            processed_dataset.append(new_inst)

        one_dataset = Dataset.from_pandas(pd.DataFrame(data=processed_dataset))
        data_bundle[name] = one_dataset

    if is_main_process():
        for name, one_dataset in data_bundle.items():
            print(f"[Data] Loaded {len(one_dataset)} {name} samples from {path_dict[name]}")

    return data_bundle


def load_multiplication_dataset(path_dict, tokenizer, mode):
    data_bundle = {}

    for name, path in path_dict.items():
        # print(f"[Data] Loading {name} from {path}")
        input_name = 'input'
        output_name = 'output'

        raw_dataset = load_data_from_file(path)
        processed_dataset = []
        for inst in raw_dataset:
            if mode == 'num':
                src = inst['prompt'].strip() + f"\nLet's solve it in {inst['step_num']} steps.\nAnswer:\n"
                tgt = inst['completion'].strip() + tokenizer.eos_token
            elif mode == 'num_in_gen':
                raise NotImplementedError
            elif mode == 'normal':
                src = inst['prompt'].strip() + '\nAnswer:\n'
                tgt = inst['completion'].strip() + tokenizer.eos_token
            else:
                raise ValueError(f"Unknown mode: {mode}")

            new_inst = {
                'id': inst['id'],
                input_name: src,
                output_name: tgt
            }
            processed_dataset.append(new_inst)

        one_dataset = Dataset.from_pandas(pd.DataFrame(data=processed_dataset))
        data_bundle[name] = one_dataset

    if is_main_process():
        for name, one_dataset in data_bundle.items():
            print(f"[Data] Loaded {len(one_dataset)} {name} samples from {path_dict[name]}")

    return data_bundle

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f"{example['input'][i]}{example['output'][i]}"
        output_texts.append(text)
    return output_texts


@dataclass
class MyScriptArguments(SFTScriptArguments):
    """
    The arguments for the DPO training script.
    """

    tokenizer_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "The tokenizer checkpoint for weights initialization."},
    )
    train_dataset: Optional[str] = field(
        default="",
        metadata={"help": "The name of the dataset to be loaded."},
    )
    test_dataset: Optional[str] = field(
        default="",
        metadata={"help": "The name of the dataset to be loaded."},
    )
    prompt_path: Optional[str] = field(
        default="",
        metadata={"help": "The path of the prompt file."},
    )
    data_num: Optional[int] = field(
        default=0,
        metadata={"help": "The number of data to use."},
    )
    mode: Optional[str] = field(
        default="normal",
        metadata={"help": "whether to include step num"},
    )
    max_seq_length: Optional[int] = field(
        default=4096,
        metadata={"help": "The maximum length of the input sequence."},
    )


if __name__ == "__main__":
    parser = TrlParser((MyScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    training_args.save_steps = 500000
    training_args.save_strategy = "no"

    set_seed(training_args.seed)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # print(f"Using tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, padding_side="left")
    tokenizer.pad_token = tokenizer.unk_token  # Make sure to have a pad_token_id which is different from
    # eos_token_id which can result in the model not properly predicting EOS (End of Sentence) tokens during
    # generation. https://huggingface.co/docs/trl/en/sft_trainer#train-on-completions-only

    ################
    # Dataset
    ################
    path_dict = {
        'train': args.train_dataset,
        'test': args.test_dataset
    }
    if '/aoa/' in args.train_dataset:
        raw_datasets = load_aoa_dataset(path_dict, tokenizer, args.mode)
    elif '/ma/' in args.train_dataset or '/dr/' in args.train_dataset or '/gsm/' in args.train_dataset:
        path_dict.pop('test')
        raw_datasets = load_multiplication_dataset(path_dict, tokenizer, args.mode)
    else:
        raise ValueError(f"Unknown dataset: {args.train_dataset}")

    if args.data_num > 0:
        raw_datasets['train'] = raw_datasets['train'].select(range(args.data_num))
    train_dataset = raw_datasets["train"]
    # eval_dataset = raw_datasets["test"]

    response_template_with_context = "\nAnswer:\n"  # We added context here: "\n". This is enough for this tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[
                            2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=eval_dataset,
            # dataset_text_field=args.dataset_text_field,
            packing=False,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        os.makedirs(training_args.output_dir, exist_ok=True)
        save_path = os.path.join(training_args.output_dir,
                                 f"{training_args.run_name}")
        # if os.path.exists(save_path):
        #     # remove the directory
        #     os.removedirs(save_path)
        #     print(f"Model existed. Will overwrite by removing {save_path}")
        trainer.save_model(save_path)
    if is_main_process():
        print(f"Model saved to {save_path}")
