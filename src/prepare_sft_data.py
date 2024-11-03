import os
import re
import copy
import json
import random
from tqdm import tqdm
from collections import OrderedDict
from fractions import Fraction
from typing import Dict, List, Tuple
from sympy import symbols, Eq, solve, simplify, sympify
from utils import *
import argparse



def parse_args():
    # new_sft: pred_valid + raw_all
    # cont_sft: pred_valid + raw_(len(pred_valid))
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_file', type=str, default='')
    parser.add_argument('--pred_files', nargs='+', type=str, default=[])
    parser.add_argument('--output_dir', type=str, default='outputs/debug')
    parser.add_argument('--mode', type=str, default='new', choices=['new', 'cont', 'onlyskip'])
    args = parser.parse_args()
    return args


def prepare_data_altogether(raw_data, pred_data, save_path, mode='new'):
    r"""
    Prepare sft data from skipped predictions
    :param raw_data:
    :param pred_data:
    :return: {'id', 'prompt', 'chosen', 'reject'}
    """
    raw_prompt = load_txt('v3/prompts/train_num.txt')
    save_data_list = []

    for i in tqdm(range(len(pred_data))):

        raw_inst = raw_data[i]
        pred_inst = pred_data[i]
        assert raw_inst['id'] == pred_inst['id'], f"{raw_inst['id']} != {pred_inst['id']}"

        # print(f"gold: {raw_inst['answer']}, pred: {pred_inst['pred']}")
        # exact_match_score += 1.0 if pred_inst['answer'] == pred_inst['pred'] else 0.0
        sympy_eq_score = sympy_equal(raw_inst['answer'], pred_inst['pred'])
        if sympy_eq_score != 1.0:
            # this could build raw prompt, this serves as a negative sample
            continue
        # eval step num
        eval_instruct_step_num = pred_inst['eval_instruct_step_num']
        full_step_num = raw_inst['num_steps']

        pred_steps_list = pred_inst['response'].split('\n')
        pred_steps_list = [s for s in pred_steps_list if s.strip() != '' and 'Thus' not in s]
        step_num = len(pred_steps_list)
        step_eqs = [s for s in pred_steps_list if 'Step' not in s and 'Move' not in s and 'Swap' not in s]
        assert len(step_eqs) == step_num, f"{len(step_eqs)} != {step_num}"

        if eval_instruct_step_num == step_num and step_num != full_step_num:
            one_prompt = raw_prompt.replace('[[NUM]]', str(int(step_num)))\
                             .replace('[[QUESTION]]', pred_inst['question']).strip() + '\n'
            chosen_response = pred_inst['response'].strip()

            one_save_data = {
                'id': raw_inst['id'],
                'prompt': one_prompt,
                'chosen': chosen_response,
                'line_id': i
            }
            save_data_list.append(one_save_data)
            # for k, v in one_save_data.items():
            #     print(f"{k}: {v}")
            # print()
    print(f"Valid data num: {len(save_data_list)} from prediction")
    valid_data_cnt = len(save_data_list)
    if mode != 'onlyskip':
        if mode == 'new':
            sample_train = raw_data
        elif mode == 'cont':
            sample_train = random.sample(raw_data, k=len(save_data_list))
        else:
            raise ValueError(f"Invalid mode: {mode}")

        for inst in sample_train:
            one_prompt = raw_prompt.replace('[[NUM]]', str(inst['num_steps']))\
                             .replace('[[QUESTION]]', inst['question']).strip() + '\n'
            cot_answer = ""
            for step in inst['steps']:
                cot_answer += step['equation'] + '\n'
            cot_answer += f"Thus, the answer is {inst['answer']}"

            one_save_data = {
                'id': inst['id'],
                'prompt': one_prompt,
                'chosen': cot_answer.strip(),
                'line_id': -1
            }
            save_data_list.append(one_save_data)
        print(f"[{mode}] Mix data num: {len(save_data_list) - valid_data_cnt} from training set")

    random.shuffle(save_data_list)
    with open(save_path, 'w', encoding='utf-8') as f:
        for one_save_data in save_data_list:
            f.write(json.dumps(one_save_data) + '\n')
    print(f"Save data num: {len(save_data_list)} to {save_path}")

def prepare_data(raw_data, pred_data, raw_prompt):
    r"""
    Prepare sft data from skipped predictions
    :param raw_data:
    :param pred_data:
    :return: {'id', 'prompt', 'chosen', 'reject'}
    """
    save_data_list = []

    for i in tqdm(range(len(pred_data))):

        raw_inst = raw_data[i]
        pred_inst = pred_data[i]
        assert raw_inst['id'] == pred_inst['id'], f"{raw_inst['id']} != {pred_inst['id']}"

        # print(f"gold: {raw_inst['answer']}, pred: {pred_inst['pred']}")
        # exact_match_score += 1.0 if pred_inst['answer'] == pred_inst['pred'] else 0.0
        sympy_eq_score = sympy_equal(raw_inst['answer'], pred_inst['pred'])
        if sympy_eq_score != 1.0:
            # this could build raw prompt, this serves as a negative sample
            continue
        # eval step num
        eval_instruct_step_num = pred_inst['eval_instruct_step_num']
        full_step_num = raw_inst['num_steps']

        pred_steps_list = pred_inst['response'].split('\n')
        pred_steps_list = [s for s in pred_steps_list if s.strip() != '' and 'Thus' not in s]
        step_num = len(pred_steps_list)
        step_eqs = [s for s in pred_steps_list if 'Step' not in s and 'Move' not in s and 'Swap' not in s]
        assert len(step_eqs) == step_num, f"{len(step_eqs)} != {step_num}"

        if eval_instruct_step_num == step_num and step_num != full_step_num:
            one_prompt = raw_prompt.replace('[[NUM]]', str(int(step_num)))\
                             .replace('[[QUESTION]]', pred_inst['question']).strip() + '\n'
            chosen_response = pred_inst['response'].strip()

            one_save_data = {
                'id': raw_inst['id'],
                'prompt': one_prompt,
                'chosen': chosen_response,
                'line_id': i
            }
            save_data_list.append(one_save_data)
            # for k, v in one_save_data.items():
            #     print(f"{k}: {v}")
            # print()
    print(f"Valid data num: {len(save_data_list)} from prediction")
    return save_data_list

def prepare_raw_data(raw_data, raw_prompt, mode='new', valid_data_cnt=0):

    save_train_list = []
    if mode != 'onlyskip':
        if mode == 'new':
            sample_train = raw_data
        elif mode == 'cont':
            sample_train = random.sample(raw_data, k=valid_data_cnt)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        for inst in sample_train:
            one_prompt = raw_prompt.replace('[[NUM]]', str(inst['num_steps']))\
                             .replace('[[QUESTION]]', inst['question']).strip() + '\n'
            cot_answer = ""
            for step in inst['steps']:
                cot_answer += step['equation'] + '\n'
            cot_answer += f"Thus, the answer is {inst['answer']}"

            one_save_data = {
                'id': inst['id'],
                'prompt': one_prompt,
                'chosen': cot_answer.strip(),
                'line_id': -1
            }
            save_train_list.append(one_save_data)
        print(f"[{mode}] Mix data num: {len(save_train_list)} from training set")
    return save_train_list


if __name__ == '__main__':
    args = parse_args()
    random.seed(1)

    raw_prompt = load_txt('data/prompts/train_num.txt')
    raw_data = load_data_from_file(args.raw_file)

    skip_data = []
    file_list = [f for f in args.pred_files if len(f) > 0]
    for one_file in file_list:
        pred_data = load_data_from_file(one_file)
        assert len(raw_data) == len(pred_data), f"{len(raw_data)} != {len(pred_data)}"
        one_skip_data = prepare_data(raw_data, pred_data, raw_prompt)
        skip_data += one_skip_data

    mix_train_data = prepare_raw_data(raw_data, raw_prompt, mode=args.mode, valid_data_cnt=len(skip_data))

    save_data_list = skip_data + mix_train_data
    random.shuffle(save_data_list)

    file_name = os.path.split(file_list[-1])[-1].replace('preds.jsonl', f'{args.mode}_sft_data.jsonl')
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, file_name)
    if os.path.exists(save_path):
        print(f"Result file {save_path} already exists. Will overwrite.")
        os.remove(save_path)

    with open(save_path, 'w', encoding='utf-8') as f:
        for one_save_data in save_data_list:
            f.write(json.dumps(one_save_data) + '\n')
    print(f"Save data num: {len(save_data_list)} to {save_path}")