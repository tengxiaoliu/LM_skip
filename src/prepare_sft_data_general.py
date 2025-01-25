import os
import re
import copy
import json
import random
from utils import *
import argparse


def parse_args():
    # new_sft: pred_valid + raw_all
    # cont_sft: pred_valid + raw_(len(pred_valid))
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_file', type=str, default='')
    parser.add_argument('--mix_file', type=str, default='')
    parser.add_argument('--pred_files', nargs='+', type=str, default=[])
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--mode', type=str, default='new', choices=['new', 'cont', 'onlyskip'])
    args = parser.parse_args()
    return args


def prepare_data(raw_data, pred_data):
    r"""
    Prepare sft data from skipped predictions
    :param raw_data:
    :param pred_data:
    :return: {'id', 'prompt', 'chosen', 'reject'}
    """
    save_data_list = []
    acc = 0
    step_num_consistent_cnt = 0

    for i in tqdm(range(len(pred_data))):

        raw_inst = raw_data[i]
        pred_inst = pred_data[i]
        assert raw_inst['id'] == pred_inst['id'], f"{raw_inst['id']} != {pred_inst['id']}"

        acc += pred_inst['score']
        if pred_inst['score'] != 1.0:
            # todo: this could build raw prompt, this serves as a negative sample
            continue
        # eval step num
        eval_instruct_step_num = pred_inst['eval_instruct_step_num']
        full_step_num = raw_inst['step_num']

        response = pred_inst['response']
        response = response.split('Thus, the')[0].strip()
        pred_step_num = len(response.split('\n'))
        if pred_step_num == pred_inst['eval_instruct_step_num']:
            step_num_consistent_cnt += 1

        if eval_instruct_step_num == pred_step_num and pred_step_num != full_step_num:
            one_prompt = raw_inst['prompt']
            one_completion = pred_inst['response'].strip()

            one_save_data = copy.deepcopy(raw_inst)
            one_save_data['id'] = f"{raw_inst['id']}/skip"
            one_save_data['prompt'] = one_prompt
            one_save_data['completion'] = one_completion
            one_save_data['step_num'] = pred_step_num
            save_data_list.append(one_save_data)
    print(f"Average: {round(acc / len(pred_data), 5)}, {acc} / {len(pred_data)}")
    print(f"Valid data num: {len(save_data_list)} from prediction")
    return save_data_list

def prepare_raw_data(raw_data, mode='new', valid_data_cnt=0):
    sample_train = []
    if mode != 'onlyskip':
        if mode == 'new':
            sample_train = raw_data
        elif mode == 'cont':
            sample_train = random.sample(raw_data, k=valid_data_cnt)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    print(f"[{mode}] Mix data num: {len(sample_train)} from training set")
    return sample_train


if __name__ == '__main__':
    args = parse_args()
    random.seed(1)

    raw_data = load_data_from_file(args.raw_file)

    skip_data = []
    file_list = [f for f in args.pred_files if len(f) > 0]
    for one_file in file_list:
        pred_data = load_data_from_file(one_file)
        assert len(raw_data) == len(pred_data), f"{len(raw_data)} != {len(pred_data)}"
        one_skip_data = prepare_data(raw_data, pred_data)
        skip_data += one_skip_data

    if len(args.mix_file) > 0:
        mix_train_data = load_data_from_file(args.mix_file)
        file_name = os.path.split(file_list[-1])[-1].replace('preds.jsonl', f'{args.mode}_mix_sft_data.jsonl')
    else:
        mix_train_data = prepare_raw_data(raw_data, mode=args.mode, valid_data_cnt=len(skip_data))
        file_name = os.path.split(file_list[-1])[-1].replace('preds.jsonl', f'{args.mode}_sft_data.jsonl')

    save_data_list = skip_data + mix_train_data
    random.shuffle(save_data_list)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, file_name)
    if os.path.exists(save_path):
        print(f"Result file {save_path} already exists. Will overwrite.")
        os.remove(save_path)

    with open(save_path, 'w', encoding='utf-8') as f:
        for one_save_data in save_data_list:
            f.write(json.dumps(one_save_data) + '\n')
    print(f"Save data num: {len(save_data_list)} to {save_path}")