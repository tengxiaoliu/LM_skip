#!/bin/bash

echo "=====Init====="
bash scripts/ma/ma_0_init.sh

echo "=====Eval init====="
bash scripts/ma/ma_1_init_testl.sh
bash scripts/ma/ma_2_init_eval.sh

echo "=====Iteration phase====="
bash scripts/ma/ma_3_iter.sh

echo "=====Standard model====="
bash scripts/ma/ma_4_standard.sh
