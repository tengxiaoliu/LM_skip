#!/bin/bash

echo "=====Init====="
bash scripts/dr/dr_0_init.sh

echo "=====Eval init====="
bash scripts/dr/dr_1_init_testl.sh
bash scripts/dr/dr_2_init_eval.sh

echo "=====Iteration phase====="
bash scripts/dr/dr_3_iter.sh

echo "=====Standard model====="
bash scripts/dr/dr_4_standard.sh

