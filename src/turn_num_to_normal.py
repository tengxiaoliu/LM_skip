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
    parser.add_argument('--dataset', type=str, default='eq')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    random.seed(1)
    raw_data = load_data_from_file(args.raw_file)

    save_path = args.raw_file.replace('data.jsonl', f'normal_data.jsonl')
    if os.path.exists(save_path):
        print(f"Result file {save_path} already exists. Will overwrite.")
        os.remove(save_path)

    if args.dataset == 'eq':
        for inst in raw_data:
            response = inst['prompt']
            normal_response, remain = response.split('Solve')
            normal_question = remain.split('steps.')[1].strip()
            inst['prompt'] = normal_response.strip() + '\n\n' + normal_question.strip()

    else:
        raise NotImplementedError

    with open(save_path, 'w', encoding='utf-8') as f:
        for one_save_data in raw_data:
            f.write(json.dumps(one_save_data) + '\n')
    print(f"Save data num: {len(raw_data)} to {save_path}")