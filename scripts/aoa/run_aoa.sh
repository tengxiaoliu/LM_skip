#!/bin/bash

echo "=====Init====="
bash scripts/aoa/aoa_0_init.sh

echo "=====Eval init====="
bash scripts/aoa/aoa_1_init_testl.sh
bash scripts/aoa/aoa_2_init_eval.sh

echo "=====Iteration phase====="
bash scripts/aoa/aoa_3_iter.sh

echo "=====Standard model====="
bash scripts/aoa/aoa_4_standard.sh
