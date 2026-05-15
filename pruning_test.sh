#!/bin/bash
#
# Distributed Transformer Inference - Pruning Test
# This script is a starting point for testing model pruning.
#


for i in {0..1}; do
    echo "Iteration $i"
    python train.py "+run=ett" "wandb_log=false" "pruning_step=$i" "do_finetuning=false"
done
