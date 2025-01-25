from vllm import LLM, SamplingParams

import json
import os
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple
import argparse
import torch
from transformers import AutoTokenizer
from evaluate_aoa import sympy_equal

import logging

# Set the logging level to WARNING to suppress INFO and DEBUG messages
logging.basicConfig(level=logging.WARNING)

# Your script's code here

class InferencePipe:
    def __init__(self, args, sampling_params: SamplingParams):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.tokenizer_name = args.tokenizer_name
        self.save_path = args.save_path
        self.demonstration_path = args.demonstration_path
        self.delta = args.delta
        print(f"===== using delta {self.delta} =====")

        self.model = LLM(model=self.model_name,
                         tokenizer=self.tokenizer_name,
                         tensor_parallel_size=args.world_size,
                         load_format='auto')

        self.sampling_params = sampling_params

        self.input_name = None
        self.target_name = None
        self.global_id = 0

        os.makedirs(self.save_path, exist_ok=True)

        if args.mode == 'normal':
            self.prompt_creator = self.prompt_num_in_gen
            self.save_path_file = os.path.join(self.save_path,
                                               f'{args.tag}_{self.dataset_name}_{args.model_alias}_normal_preds.jsonl')
            print("Using prompt_normal")
        elif args.mode == 'num_in_gen':
            self.prompt_creator = self.prompt_num_in_gen
            self.save_path_file = os.path.join(self.save_path,
                                               f'{args.tag}_{self.dataset_name}_{args.model_alias}_num_in_gen_preds.jsonl')
            print("Using prompt_num_in_gen")
        elif args.mode == 'num':
            self.prompt_creator = self.prompt_num
            self.save_path_file = os.path.join(self.save_path,
                                               f'{args.tag}_{self.dataset_name}_{args.model_alias}_skip{self.args.delta}_preds.jsonl')
            print("Using prompt_num")
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

        print(f"Will save results to {self.save_path_file}")
        if os.path.exists(self.save_path_file):
            os.remove(self.save_path_file)
            print(f"Result file existed. Will overwrite by removing {self.save_path_file}")

        self.eval_path_file = os.path.join(self.save_path,
                                           f'{args.tag}_{self.dataset_name}_{args.model_alias}_preds_eval.jsonl')
        if os.path.exists(self.eval_path_file):
            os.remove(self.eval_path_file)
            print(f"Eval Result file existed. Will overwrite by removing {self.eval_path_file}")

    def set_sampling_params(self, params):
        self.sampling_params = params

    def prompt_normal(self, in_context_example, test_example):
        full_prompt = in_context_example
        step_num = test_example['num_steps']

        question = test_example['question'].strip()
        full_prompt = full_prompt.replace('[[QUESTION]]', question) + '\n'
        return full_prompt, step_num

    def prompt_num(self, in_context_example, test_example):
        full_prompt = in_context_example
        step_num = test_example['num_steps']
        if step_num + self.delta > 0:
            step_num += self.delta

        # if step_num - 2 > 0:
        #     step_num -= 2
        # step_num += 1
        full_prompt = full_prompt.replace('[[NUM]]', str(step_num))

        question = test_example['question'].strip()
        full_prompt = full_prompt.replace('[[QUESTION]]', question) + '\n'
        return full_prompt, step_num

    def prompt_num_in_gen(self, in_context_example, test_example):
        full_prompt = in_context_example
        step_num = test_example['num_steps']

        # if step_num - 2 > 0:
        #     step_num -= 2
        # step_num += 1
        # full_prompt = full_prompt.replace('[[NUM]]', str(step_num))

        question = test_example['question'].strip()
        full_prompt = full_prompt.replace('[[QUESTION]]', question) + '\n'
        return full_prompt, step_num

    def load_in_context_examples(self):
        with open(self.demonstration_path, 'r', encoding='utf-8') as f:
            in_context_examples = f.read()
        print(f"Loaded in-context examples from {self.demonstration_path}")
        return in_context_examples

    def load_raw_dataset(self):
        if 'jsonl' in self.data_path:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                raw_dataset = [json.loads(line) for line in f]
        else:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                raw_dataset = json.load(f)
        print(f"Loaded {len(raw_dataset)} examples from {self.data_path}")
        return raw_dataset

    def run_inference_one_go(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset()

        if 'input' in raw_dataset[0]:
            self.input_name = 'input'
        elif 'question' in raw_dataset[0]:
            self.input_name = 'question'
        else:
            raise ValueError("Cannot find input in raw dataset")

        if 'target' in raw_dataset[0]:
            self.target_name = 'target'
        elif 'answer' in raw_dataset[0]:
            self.target_name = 'answer'
        elif 'gold_answer' in raw_dataset[0]:
            self.target_name = 'gold_answer'
        else:
            raise ValueError("Cannot find target in raw dataset")

        # load in-context examples
        in_context_examples = self.load_in_context_examples()

        all_prompts = []
        step_num_list = []
        for example in tqdm(raw_dataset):
            # create prompt
            full_prompt, step_num = self.prompt_creator(in_context_examples, example)
            step_num_list.append(step_num)

            all_prompts.append(full_prompt)
            # print(full_prompt)

        all_outputs = self.model.generate(all_prompts, self.sampling_params)
        # get the answer
        total_score = 0.0
        cnt = 0
        # for example, one_output, one_prompt in zip(raw_dataset, all_outputs, all_prompts):
        for i in tqdm(range(len(raw_dataset))):
            example = raw_dataset[i]
            one_output = all_outputs[i]
            one_prompt = all_prompts[i]

            # create output
            assert one_output.prompt == one_prompt
            response = one_output.outputs[0].text

            dict_output = self.exact_match_metric(example, response, idx=cnt)
            dict_output['eval_instruct_step_num'] = step_num_list[i]
            total_score += dict_output['score']

            cnt += 1
            # save outputs
            with open(self.save_path_file, 'a') as f:
                json.dump(dict_output, f)
                f.write('\n')

        print(f"Average score: {total_score}/{len(raw_dataset)}, {round(total_score/len(raw_dataset), 5)}")
        print(f"Saved results to {self.save_path_file}")

    def exact_match_metric(self, sample, response, idx):

        # if self.args.mode == 'cot' or self.args.mode == 'skip':
        #     pred = self.extract_answer(response).strip()
        # else:
        #     pred = response.strip()
        if self.args.mode == 'direct':
            pred = response.strip()
        else:
            pred = self.extract_answer(response).strip()

        # if idx % 200 == 0:
        #     print(pred)
        if pred is not None:
            # score = int(pred == sample[self.target_name].strip())
            sympy_eq = sympy_equal(pred, sample[self.target_name].strip())
            score = int(sympy_eq)
        else:
            score = 0.0

        dict_output = {'id': sample['id'] if 'id' in sample else self.global_id,
                       'question': sample[self.input_name],
                       'answer': sample[self.target_name],
                       'pred': pred,
                       'score': score,
                       'response': response,
                       }
        self.global_id += 1
        return dict_output

    def extract_answer(self, response):
        indicators = ['the answer is', 'the answer is:',
                      'The answer is', 'The answer is:',
                      'Thus, the answer is']
        answer_format_flag = False
        for indicator in indicators:
            if response.find(indicator) >= 0:
                answer_format_flag = True
                answer_str = response.split(indicator)[-1].strip('.').replace(',', '').strip()
                break
        if not answer_format_flag:
            answer_str = response.strip('.').replace(',', '').strip()

        # print(f"response: {response}")
        # print(f"answer_str: {answer_str}")

        return answer_str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--dataset_name', type=str, default='folio')
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_path', type=str, default='./outputs/inference')
    parser.add_argument('--demonstration_path', type=str, required=True)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_alias', type=str, default='llama2')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--tokenizer_name', type=str, default='hf-internal-testing/llama-tokenizer')
    parser.add_argument('--stop_words', type=str, default='\n\n\n')
    parser.add_argument('--tag', type=str, default='debug')
    parser.add_argument('--max_new_tokens', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--beam_search', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_in_gen', action='store_true')
    parser.add_argument('--delta', type=int, default=0)
    parser.add_argument('--mode', type=str, default='cot')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    model_name_dict = {
        'llama-7b': 'huggyllama/llama-7b',
        'llama-13b': 'huggyllama/llama-13b',
        'llama-30b': 'huggyllama/llama-30b',
        'llama2-7b': 'meta-llama/Llama-2-7b-hf'
    }
    if len(args.model_name) == 0:
        args.model_name = model_name_dict[args.model_alias]

    # get world size
    # world_size = torch.cuda.device_count()
    # print(f"World size: {world_size}")
    # args.world_size = world_size
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_eos_token=False)

    if args.beam_search:
        sampling_params = SamplingParams(temperature=0.0,
                                         top_p=1.0,
                                         top_k=-1,
                                         stop_token_ids=[tokenizer.eos_token_id],
                                         max_tokens=args.max_new_tokens,
                                         n=1,
                                         best_of=3,  # beam size
                                         use_beam_search=True,
                                         length_penalty=0.2,
                                         early_stopping=True,
                                         )
    else:
        sampling_params = SamplingParams(temperature=0.0,
                                         top_p=1.0,
                                         top_k=-1,
                                         stop_token_ids=[tokenizer.eos_token_id],
                                         max_tokens=args.max_new_tokens,)

    inference_pipe = InferencePipe(args, sampling_params)
    inference_pipe.run_inference_one_go()
    # gpt3_problem_reduction.batch_run_inference(batch_size=args.batch_size)
