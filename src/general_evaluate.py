import os
import re
import copy
import json
import random
from tqdm import tqdm
from collections import OrderedDict
from fractions import Fraction
from typing import Dict, List, Tuple
import argparse
from sympy import symbols, Eq, solve, simplify, sympify
import wandb


def load_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    print(f"Loaded {len(data)} data from {path}")
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
        'I': u'\u2618',  # new add begins
        'J': u'\u270A',
        'K': u'\u2615',
        'L': u'\u2708',
        'M': u'\u25A3',
        'P': u'\u22A4',
        'R': u'\u2704',  # new add ends
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

# def total_score(data):
#     score = 0
#     for d in data:
#         score += d['score']
#     print(f"Average: {round(score / len(data), 5)}, {score} / {len(data)}")
#     return score

def solve_equation(equation_str, solve_for='x'):
    # Parse the string into a SymPy equation
    lhs, rhs = equation_str.split('=')
    equation = Eq(sympify(lhs), sympify(rhs))

    # Solve the equation
    solutions = solve(equation, symbols(solve_for))

    return solutions


import signal

# Define a custom exception for timeout
class TimeoutException(Exception):
    pass

# Define a handler function that raises the TimeoutException
def timeout_handler(signum, frame):
    raise TimeoutException("Function execution exceeded the time limit of 1 minute")

# Decorator to apply the timeout to functions
def time_limit(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler for the alarm signal
            signal.signal(signal.SIGALRM, timeout_handler)
            # Schedule the alarm to go off after a given number of seconds
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the alarm after the function executes
                signal.alarm(0)
            return result
        return wrapper
    return decorator

@time_limit(30)
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
        # Optionally, you could handle specific exceptions here if needed
        return False
    return diff == 0

def evaluate_turn_num(raw_data, pred_data):
    acc_cnt = 0

    pred_correct_step_num_sum = 0
    pred_step_num_sum = 0
    full_step_num_sum = 0
    instruct_step_num_sum = 0

    eval_skip_cnt = 0  # number of eval_instruct < full_step_num
    eval_step_eq = 0  # whether eval_instruct step num equals to the actual step num
    skip_eval_step_eq = 0  # whether the eval_instruct is actually a skipping
    instruct_step_num_delta_dict = {}
    step_num_delta_dict = {}  # step num delta distribution

    eval_step_eq_acc = 0  # step consistent, and answer correct
    skip_eval_step_eq_acc = 0  # skipping, step consistent, and answer correct

    for i in tqdm(range(len(raw_data)), desc="Evaluating"):
        raw_inst = raw_data[i]
        pred_inst = pred_data[i]
        assert raw_inst['id'] == pred_inst['id'], f"{raw_inst['id']} != {pred_inst['id']}"
        score = pred_inst['score']
        assert score == (1 if pred_inst['answer'].lower() == pred_inst[
            'pred'].lower() else 0), f"{score} != {pred_inst['answer'] == pred_inst['pred']}"
        acc_cnt += score

        # eval step num
        full_step_num = raw_inst['step_num']
        full_step_num_sum += full_step_num
        eval_instruct_step_num = pred_inst['eval_instruct_step_num']
        instruct_step_num_sum += eval_instruct_step_num

        response = pred_inst['response']
        response = response.split('Thus, the')[0].strip()
        pred_step_num = len(response.split('\n'))

        # print(f"response: {response}")
        # print(f"pred_step_num: {pred_step_num}")

        pred_step_num_sum += pred_step_num
        if pred_inst['score'] == 1.0:
            pred_correct_step_num_sum += pred_step_num

        if eval_instruct_step_num < full_step_num:
            eval_skip_cnt += 1

        if eval_instruct_step_num == pred_step_num:
            eval_step_eq += 1
            eval_step_eq_acc += pred_inst['score']

            if eval_instruct_step_num < full_step_num:
                skip_eval_step_eq += 1
                skip_eval_step_eq_acc += pred_inst['score']
        # log distribution
        one_instruct_delta = eval_instruct_step_num - full_step_num
        eval_delta = pred_step_num - full_step_num
        instruct_step_num_delta_dict[one_instruct_delta] = instruct_step_num_delta_dict.get(one_instruct_delta, 0) + 1
        step_num_delta_dict[eval_delta] = step_num_delta_dict.get(eval_delta, 0) + 1

    results = {'score_accuracy': round(acc_cnt / len(pred_data), 5),
               'detail/score_accuracy_stat': [acc_cnt, len(pred_data)],
               'avg_full_step_num': round(full_step_num_sum / len(pred_data), 5),
               'detail/avg_full_step_num_stat': [full_step_num_sum, len(pred_data)],
               'avg_instruct_step_num': round(instruct_step_num_sum / len(pred_data), 5),
               'detail/avg_instruct_step_num_stat': [instruct_step_num_sum, len(pred_data)],
               'avg_pred_step_num_all': round(pred_step_num_sum / len(pred_data), 5),
               'detail/avg_pred_step_num_all_stat': [pred_step_num_sum, len(pred_data)],
               'avg_pred_step_num_correct': round(pred_correct_step_num_sum / acc_cnt, 5) if acc_cnt > 0 else 0,
               'detail/avg_pred_step_num_correct_stat': [pred_correct_step_num_sum, acc_cnt],
               'step_consistent': round(eval_step_eq / len(pred_data), 5),
               'detail/step_consistent_stat': [eval_step_eq, len(pred_data)],
               'step_consistent_real_skip': round(skip_eval_step_eq / eval_skip_cnt, 5) if eval_skip_cnt > 0 else 0,
               'detail/step_consistent_real_skip_stat': [skip_eval_step_eq, eval_skip_cnt],
               'step_consistent_answer_correct': round(eval_step_eq_acc / eval_step_eq,
                                                       5) if eval_step_eq > 0 else 0,
               'detail/step_consistent_answer_correct_stat': [eval_step_eq_acc, eval_step_eq],
               'step_consistent_answer_correct_real_skip': round(skip_eval_step_eq_acc / eval_skip_cnt,
                                                                 5) if eval_skip_cnt > 0 else 0,
               'detail/step_consistent_answer_correct_real_skip_stat': [skip_eval_step_eq_acc, eval_skip_cnt]}
    wandb.log(results)
    # step consistency analysis

    # if pred_step_num == pred_inst['eval_instruct_step_num']:
    #     step_num_correct += 1
    #     step_num_correct_score += score
    #     if score == 1 or score == 0:
    #         print(f"SCORE={score}\nQ: {pred_inst['question']}")
    #         print(f"instruct step num {pred_inst['eval_instruct_step_num']}, full step num {full_step_num}")
    #         steps = response.split('\n')
    #         prev_num = None
    #         for one_step in steps:
    #             print(one_step)
    #             # count the number of , in one_step
    #             one_num = one_step.count(',')
    #             if prev_num is not None and prev_num - one_num > 1:
    #                 print(f"[[diff: {prev_num - one_num}]]")
    #             prev_num = one_num
    #         print(f"GOLD: {raw_inst['answer']}")

    # print(f"Pred:\n{pred_inst['response']}")


def evaluate_add_num(raw_data, pred_data):
    acc_cnt = 0
    cnt_incomplete_answer = 0
    pred_correct_step_num_sum = 0
    pred_step_num_sum = 0
    full_step_num_sum = 0
    instruct_step_num_sum = 0

    eval_skip_cnt = 0  # number of eval_instruct < full_step_num
    eval_step_eq = 0  # whether eval_instruct step num equals to the actual step num
    skip_eval_step_eq = 0  # whether the eval_instruct is actually a skipping
    instruct_step_num_delta_dict = {}
    step_num_delta_dict = {}  # step num delta distribution

    eval_step_eq_acc = 0  # step consistent, and answer correct
    skip_eval_step_eq_acc = 0  # skipping, step consistent, and answer correct

    for i in tqdm(range(len(raw_data)), desc="Evaluating"):
        raw_inst = raw_data[i]
        pred_inst = pred_data[i]
        assert raw_inst['id'] == pred_inst['id'], f"{raw_inst['id']} != {pred_inst['id']}"
        score = pred_inst['score']
        try:
            pred_ans = int(pred_inst['pred'])
        except:
            # pred = -inf
            pred_ans = -1000000
        assert score == (1 if int(pred_inst['answer']) == pred_ans else 0), f"{score} != {pred_inst['answer'] == pred_inst['pred']}"

        acc_cnt += score

        # eval step num
        full_step_num = raw_inst['step_num']
        full_step_num_sum += full_step_num
        eval_instruct_step_num = pred_inst['eval_instruct_step_num']
        instruct_step_num_sum += eval_instruct_step_num

        response = pred_inst['response']
        cnt_incomplete_answer += 1 if "Thus, the answer is" not in response else 0

        response = response.split('Thus, the')[0].strip()
        pred_step_num = len(response.split('\n'))

        # print(f"raw: {pred_inst['response']}")
        # print(f"response: {response}")
        # print(f"pred_step_num: {pred_step_num}")

        pred_step_num_sum += pred_step_num
        if pred_inst['score'] == 1.0:
            pred_correct_step_num_sum += pred_step_num

        if eval_instruct_step_num < full_step_num:
            eval_skip_cnt += 1

        if eval_instruct_step_num == pred_step_num:
            eval_step_eq += 1
            eval_step_eq_acc += pred_inst['score']

            if eval_instruct_step_num < full_step_num:
                skip_eval_step_eq += 1
                skip_eval_step_eq_acc += pred_inst['score']
        # log distribution
        one_instruct_delta = eval_instruct_step_num - full_step_num
        eval_delta = pred_step_num - full_step_num
        instruct_step_num_delta_dict[one_instruct_delta] = instruct_step_num_delta_dict.get(one_instruct_delta, 0) + 1
        step_num_delta_dict[eval_delta] = step_num_delta_dict.get(eval_delta, 0) + 1

    results = {'score_accuracy': round(acc_cnt / len(pred_data), 5),
               'detail/score_accuracy_stat': [acc_cnt, len(pred_data)],
               'cnt_incomplete_answer': round(cnt_incomplete_answer / len(pred_data), 5),
               'detail/cnt_incomplete_answer_stat': [cnt_incomplete_answer, len(pred_data)],
               'avg_full_step_num': round(full_step_num_sum / len(pred_data), 5),
               'detail/avg_full_step_num_stat': [full_step_num_sum, len(pred_data)],
               'avg_instruct_step_num': round(instruct_step_num_sum / len(pred_data), 5),
               'detail/avg_instruct_step_num_stat': [instruct_step_num_sum, len(pred_data)],
               'avg_pred_step_num_all': round(pred_step_num_sum / len(pred_data), 5),
               'detail/avg_pred_step_num_all_stat': [pred_step_num_sum, len(pred_data)],
               'avg_pred_step_num_correct': round(pred_correct_step_num_sum / acc_cnt, 5) if acc_cnt > 0 else 0,
               'detail/avg_pred_step_num_correct_stat': [pred_correct_step_num_sum, acc_cnt],
               'step_consistent': round(eval_step_eq / len(pred_data), 5),
               'detail/step_consistent_stat': [eval_step_eq, len(pred_data)],
               'step_consistent_real_skip': round(skip_eval_step_eq / eval_skip_cnt, 5) if eval_skip_cnt > 0 else 0,
               'detail/step_consistent_real_skip_stat': [skip_eval_step_eq, eval_skip_cnt],
               'step_consistent_answer_correct': round(eval_step_eq_acc / eval_step_eq,
                                                       5) if eval_step_eq > 0 else 0,
               'detail/step_consistent_answer_correct_stat': [eval_step_eq_acc, eval_step_eq],
               'step_consistent_answer_correct_real_skip': round(skip_eval_step_eq_acc / eval_skip_cnt,
                                                                 5) if eval_skip_cnt > 0 else 0,
               'detail/step_consistent_answer_correct_real_skip_stat': [skip_eval_step_eq_acc, eval_skip_cnt]}
    wandb.log(results)


def evaluate_eq_num(raw_data, pred_data):
    acc_cnt = 0
    cnt_incomplete_answer = 0
    pred_correct_step_num_sum = 0
    pred_step_num_sum = 0
    full_step_num_sum = 0
    instruct_step_num_sum = 0

    eval_skip_cnt = 0  # number of eval_instruct < full_step_num
    eval_step_eq = 0  # whether eval_instruct step num equals to the actual step num
    skip_eval_step_eq = 0  # whether the eval_instruct is actually a skipping
    instruct_step_num_delta_dict = {}
    step_num_delta_dict = {}  # step num delta distribution

    eval_step_eq_acc = 0  # step consistent, and answer correct
    skip_eval_step_eq_acc = 0  # skipping, step consistent, and answer correct

    for i in tqdm(range(len(raw_data)), desc="Evaluating"):
        raw_inst = raw_data[i]
        pred_inst = pred_data[i]
        assert raw_inst['id'] == pred_inst['id'], f"{raw_inst['id']} != {pred_inst['id']}"
        score = pred_inst['score']

        sympy_eq_score = sympy_equal(raw_inst['answer'], pred_inst['pred'])
        assert pred_inst[
                   'score'] == sympy_eq_score, f"not consistent with eval {pred_inst['score']} != {sympy_eq_score}"
        acc_cnt += score

        # eval step num
        full_step_num = raw_inst['step_num'] if "step_num" in raw_inst else raw_inst['num_steps']
        full_step_num_sum += full_step_num
        eval_instruct_step_num = pred_inst['eval_instruct_step_num']
        instruct_step_num_sum += eval_instruct_step_num

        response = pred_inst['response']
        cnt_incomplete_answer += 1 if "Thus, the answer is" not in response else 0

        response = response.split('Thus, the')[0].strip()
        pred_step_num = len(response.split('\n'))

        # print(f"raw: {pred_inst['response']}")
        # print(f"response: {response}")
        # print(f"pred_step_num: {pred_step_num}")

        pred_step_num_sum += pred_step_num
        if pred_inst['score'] == 1.0:
            pred_correct_step_num_sum += pred_step_num

        if eval_instruct_step_num < full_step_num:
            eval_skip_cnt += 1

        if eval_instruct_step_num == pred_step_num:
            eval_step_eq += 1
            eval_step_eq_acc += pred_inst['score']

            if eval_instruct_step_num < full_step_num:
                skip_eval_step_eq += 1
                skip_eval_step_eq_acc += pred_inst['score']
        # log distribution
        one_instruct_delta = eval_instruct_step_num - full_step_num
        eval_delta = pred_step_num - full_step_num
        instruct_step_num_delta_dict[one_instruct_delta] = instruct_step_num_delta_dict.get(one_instruct_delta, 0) + 1
        step_num_delta_dict[eval_delta] = step_num_delta_dict.get(eval_delta, 0) + 1

    results = {'score_accuracy': round(acc_cnt / len(pred_data), 5),
               'detail/score_accuracy_stat': [acc_cnt, len(pred_data)],
               'cnt_incomplete_answer': round(cnt_incomplete_answer / len(pred_data), 5),
               'detail/cnt_incomplete_answer_stat': [cnt_incomplete_answer, len(pred_data)],
               'avg_full_step_num': round(full_step_num_sum / len(pred_data), 5),
               'detail/avg_full_step_num_stat': [full_step_num_sum, len(pred_data)],
               'avg_instruct_step_num': round(instruct_step_num_sum / len(pred_data), 5),
               'detail/avg_instruct_step_num_stat': [instruct_step_num_sum, len(pred_data)],
               'avg_pred_step_num_all': round(pred_step_num_sum / len(pred_data), 5),
               'detail/avg_pred_step_num_all_stat': [pred_step_num_sum, len(pred_data)],
               'avg_pred_step_num_correct': round(pred_correct_step_num_sum / acc_cnt, 5) if acc_cnt > 0 else 0,
               'detail/avg_pred_step_num_correct_stat': [pred_correct_step_num_sum, acc_cnt],
               'step_consistent': round(eval_step_eq / len(pred_data), 5),
               'detail/step_consistent_stat': [eval_step_eq, len(pred_data)],
               'step_consistent_real_skip': round(skip_eval_step_eq / eval_skip_cnt, 5) if eval_skip_cnt > 0 else 0,
               'detail/step_consistent_real_skip_stat': [skip_eval_step_eq, eval_skip_cnt],
               'step_consistent_answer_correct': round(eval_step_eq_acc / eval_step_eq,
                                                       5) if eval_step_eq > 0 else 0,
               'detail/step_consistent_answer_correct_stat': [eval_step_eq_acc, eval_step_eq],
               'step_consistent_answer_correct_real_skip': round(skip_eval_step_eq_acc / eval_skip_cnt,
                                                                 5) if eval_skip_cnt > 0 else 0,
               'detail/step_consistent_answer_correct_real_skip_stat': [skip_eval_step_eq_acc, eval_skip_cnt]}
    wandb.log(results)



def analyze_normal(raw_data, pred_data, task='add'):
    acc_cnt = 0
    cnt_incomplete_answer = 0
    pred_correct_step_num_sum = 0
    pred_step_num_sum = 0
    full_step_num_sum = 0
    instruct_step_num_sum = 0

    eval_skip_cnt = 0  # number of eval_instruct < full_step_num
    eval_step_eq = 0  # whether eval_instruct step num equals to the actual step num
    skip_eval_step_eq = 0  # whether the eval_instruct is actually a skipping
    instruct_step_num_delta_dict = {}
    step_num_delta_dict = {}  # step num delta distribution

    eval_step_eq_acc = 0  # step consistent, and answer correct
    skip_eval_step_eq_acc = 0  # skipping, step consistent, and answer correct

    acc_map = {}  # delta: [acc, total]

    for i in tqdm(range(len(raw_data)), desc="Evaluating"):
        raw_inst = raw_data[i]
        pred_inst = pred_data[i]
        assert raw_inst['id'] == pred_inst['id'], f"{raw_inst['id']} != {pred_inst['id']}"
        score = pred_inst['score']
        if task == 'add':
            try:
                assert score == (1 if int(pred_inst['answer']) == int(
                    pred_inst['pred']) else 0), f"{score} != {pred_inst['answer'] == pred_inst['pred']}"
            except:
                score = 0.0
        elif task == 'turn':
            assert score == (1 if pred_inst['answer'].lower() == pred_inst[
                'pred'].lower() else 0), f"{score} != {pred_inst['answer'] == pred_inst['pred']}"
        else:
            raise NotImplementedError(f"Invalid task")
        acc_cnt += score

        # eval step num
        full_step_num = raw_inst['step_num']
        full_step_num_sum += full_step_num

        response = pred_inst['response']
        cnt_incomplete_answer += 1 if "Thus, the answer is" not in response else 0

        response = response.split('Thus, the')[0].strip()
        pred_step_num = len(response.split('\n'))
        one_delta = pred_step_num - full_step_num
        if one_delta not in acc_map:
            acc_map[one_delta] = [0, 0]
        acc_map[one_delta][0] += score
        acc_map[one_delta][1] += 1

        pred_step_num_sum += pred_step_num
        if pred_inst['score'] == 1.0:
            pred_correct_step_num_sum += pred_step_num

    fewer_step_acc = sum([v[0] for k, v in acc_map.items() if k < 0])
    fewer_step_total = sum([v[1] for k, v in acc_map.items() if k < 0])
    full_step_acc = sum([v[0] for k, v in acc_map.items() if k == 0])
    full_step_total = sum([v[1] for k, v in acc_map.items() if k == 0])
    more_step_acc = sum([v[0] for k, v in acc_map.items() if k > 0])
    more_step_total = sum([v[1] for k, v in acc_map.items() if k > 0])

    results = {
        'score_accuracy': round(acc_cnt / len(pred_data), 5),
        'detail/score_accuracy_stat': [acc_cnt, len(pred_data)],
        'cnt_incomplete_answer': round(cnt_incomplete_answer / len(pred_data), 5),
        'detail/cnt_incomplete_answer_stat': [cnt_incomplete_answer, len(pred_data)],
        'avg_full_step_num': round(full_step_num_sum / len(pred_data), 5),
        'detail/avg_full_step_num_stat': [full_step_num_sum, len(pred_data)],
        'avg_pred_step_num_all': round(pred_step_num_sum / len(pred_data), 5),
        'detail/avg_pred_step_num_all_stat': [pred_step_num_sum, len(pred_data)],
        'avg_pred_step_num_correct': round(pred_correct_step_num_sum / acc_cnt, 5) if acc_cnt > 0 else 0,
        'detail/avg_pred_step_num_correct_stat': [pred_correct_step_num_sum, acc_cnt],
        'normal/fewer_cnt': round(fewer_step_total / len(pred_data), 5),
        'normal/fewer_cnt_stat': [fewer_step_total, len(pred_data)],
        'normal/full_cnt': round(full_step_total / len(pred_data), 5),
        'normal/full_cnt_stat': [full_step_total, len(pred_data)],
        'normal/more_cnt': round(more_step_total / len(pred_data), 5),
        'normal/more_cnt_stat': [more_step_total, len(pred_data)],
        'normal/fewer_acc': round(fewer_step_acc / fewer_step_total, 5) if fewer_step_total > 0 else 0,
        'normal/fewer_acc_stat': [fewer_step_acc, fewer_step_total],
        'normal/full_acc': round(full_step_acc / full_step_total, 5) if full_step_total > 0 else 0,
        'normal/full_acc_stat': [full_step_acc, full_step_total],
        'normal/more_acc': round(more_step_acc / more_step_total, 5) if more_step_total > 0 else 0,
        'normal/more_acc_stat': [more_step_acc, more_step_total],
    }

    wandb.log(results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_file', type=str, default='')
    parser.add_argument('--pred_file', type=str, default='')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--tag', type=str, default='turn')
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--delta', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print("============================\n============================")

    if args.mode == "num":
        args.run_name = f"{args.run_name}_skip{args.delta}"

    wandb.init(
        project="skip_eval",
        name=args.run_name,
        tags=[args.tag, args.mode],
        config=vars(args),
    )

    raw_data = load_jsonl(args.raw_file)
    pred_data = load_jsonl(args.pred_file)
    # full_step_eval(raw_data, pred_data, args.mode)
    if args.mode == 'normal':
        analyze_normal(raw_data, pred_data, task=args.tag)
    elif args.mode == "num":
        if args.tag == 'eq':
            evaluate_eq_num(raw_data, pred_data)
        elif args.tag == 'turn':
            evaluate_turn_num(raw_data, pred_data)
        elif args.tag == 'add':
            evaluate_add_num(raw_data, pred_data)
    else:
        raise NotImplementedError(f"Invalid dataset")

    wandb.finish()
