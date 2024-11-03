import os
import re
import copy
import json
import torch
from torch import distributed as dst
from tqdm import tqdm
from collections import OrderedDict
from fractions import Fraction
from typing import Dict, List, Tuple
from sympy import symbols, Eq, solve, simplify, sympify
import argparse

def is_main_process():
    # Check if the distributed environment is initialized
    if dst.is_initialized():
        # In a distributed environment, the main process usually has rank 0
        return dst.get_rank() == 0
    else:
        # If the distributed environment is not initialized, it's likely a single-process scenario,
        # so we can consider this as the main process.
        return True

def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data


def load_data_from_file(file_path):
    if 'jsonl' in file_path or 'txt' in file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]
    elif 'json' in file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    print(f"[Data] Loaded {len(data)} samples from {file_path}")
    return data


def replace_with_symbols_reverse(in_str, add_space=True):
    if not add_space:
        ret_s = in_str.split(' ')
    else:
        ret_s = [s for s in in_str if s.strip() != '']
    ret_s = ' '.join(ret_s)
    symbol_map = {
        'x': u'\u2764',
        'A': u'\u264B',
        'B': u'\u2659',
        'C': u'\u264F',
        'D': u'\u269B',
        'F': u'\u263A',
        'G': u'\u26ED',
        'H': u'\u2668',
        '+': u'\u2730',
        '-': u'\u2740',
        '*': u'\u273E',
        '/': u'\u266A',
        '=': u'\u2194',
        '(': u'\u2772',
        ')': u'\u2773',
        '0': u'\u25EF'
    }
    reverse_symbol_map = {v: k for k, v in symbol_map.items()}
    for k, v in reverse_symbol_map.items():
        ret_s = ret_s.replace(k, v)
    return ret_s

def solve_equation(equation_str, solve_for='x'):
    # Parse the string into a SymPy equation
    lhs, rhs = equation_str.split('=')
    equation = Eq(sympify(lhs), sympify(rhs))

    # Solve the equation
    solutions = solve(equation, symbols(solve_for))

    return solutions

def sympy_equal(eq1, eq2, check_final_answer=True):
    if eq1.strip() == eq2.strip():
        return True
    eq1 = replace_with_symbols_reverse(eq1)
    eq2 = replace_with_symbols_reverse(eq2)
    try:
        eq1_lhs = eq1.split('=')[0].strip()
        eq2_lhs = eq2.split('=')[0].strip()
        if check_final_answer:
            assert eq1_lhs == 'x', f"{eq1_lhs} != x"
            assert eq2_lhs == 'x', f"{eq2_lhs} != x"
        eq1_ans = solve_equation(eq1)[0]
        eq2_ans = solve_equation(eq2)[0]
        diff = simplify(eq1_ans - eq2_ans)
    except Exception as e:
        # print(f"Exception in sympy eq: {e}")
        return False
    return diff == 0


# tx: define the reward function based on heuristic rules, the reward is assigned to a whole sequence
def get_skip_rewards(query, gold, response):
    # print(f"query: {len(query)}, {query[:3]}")

    assert len(query) == len(gold) == len(response), f"The length of query, gold, and response should be the same, but get {len(query)}, {len(gold)}, {len(response)}"

    all_scores = []
    for i in range(len(query)):
        one_query, one_gold, one_pred = query[i], gold[i], response[i]
        # remove </s>
        one_query = one_query.replace("</s>", "").strip()

        # extract from query
        # try:
        question = one_query.split("Question:")[1].strip().split("\nAnswer:")[0].strip()
        instruct_step_num = int(one_query.split("Solve it in")[1].split("steps.")[0].strip())
        # except:
        #     print(f"Idx{i}: Error in query {len(one_query)}: {one_query}")
        #     raise ValueError("Error in query")
        # extract from gold
        gold_ans = one_gold.split("Thus, the answer is")[1].strip()
        assert sympy_equal(question, gold_ans, check_final_answer=False), f"Question and gold answer should be consistent, but get {question} and {gold_ans}"

        # evaluate response
        # 1. format correctness
        format_score = 1.0 if "Thus, the answer is" in one_pred else 0.0
        if format_score == 0.0:
            all_scores.append(0.0)
            continue
        # 2. answer correctness
        pred_ans = one_pred.split("Thus, the answer is")[1].strip()
        answer_score = sympy_equal(pred_ans, gold_ans)
        # 3. step correctness
        pred_steps = one_pred.split("Thus, the answer is")[0].strip().split("\n")
        step_num_score = 1.0 if len(pred_steps) == instruct_step_num else 0.0
        # 3.1 step inner consistency, check if each step is consistent with the gold answer
        step_inner_score = 1.0
        for i in range(len(pred_steps) - 1):
            if sympy_equal(pred_steps[i], gold_ans, check_final_answer=False) == 0:
                step_inner_score = 0.0
                break
        if answer_score and step_num_score and step_inner_score:
            one_reward = 1.0
        elif answer_score or (step_num_score and step_inner_score):
            one_reward = 0.8
        else:
            one_reward = 0.3

        all_scores.append(one_reward)
    # to a list of tensor
    all_scores = [torch.tensor(score) for score in all_scores]
    return all_scores

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# evaluate output answer

def extract_answer(response):
    indicators = ['Thus, the answer is', 'the answer is', 'the answer is:',
                  'The answer is', 'The answer is:',
                  ]
    answer_format_flag = False
    for indicator in indicators:
        if response.find(indicator) >= 0:
            answer_format_flag = True
            answer_str = response.split(indicator)[-1].strip('.').replace(',', '').strip()
            break
    if not answer_format_flag:
        answer_str = response.strip('.').replace(',', '').strip()
    return answer_str

def exact_match_metric(sample, response):

    pred = extract_answer(response).strip()

    if pred is not None:
        sympy_eq = sympy_equal(pred, sample['answer'].strip())
        score = int(sympy_eq)
    else:
        score = 0.0

    dict_output = {'id': sample['id'],
                   'question': sample['question'],
                   'answer': sample['answer'],
                   'pred': pred,
                   'score': score,
                   'response': response,
                   }
    return dict_output