#!/bin/bash
# Direct Preference Optimization execution script

export CUDA_VISIBLE_DEVICES=0,1,2,3

deepspeed --num_gpus=4 src/trainers/dpo_trainer.py \
    --deepspeed configs/ds_config_stage3.json \
    --config configs/dpo_config.yaml \
    --peft_config configs/peft_config.yaml