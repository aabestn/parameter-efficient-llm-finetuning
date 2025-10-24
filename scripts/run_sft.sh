#!/bin/bash
# Multi-GPU Supervised Fine-Tuning execution script

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4

deepspeed --num_gpus=4 src/trainers/sft_trainer.py \
    --deepspeed configs/ds_config_stage3.json \
    --config configs/sft_config.yaml \
    --peft_config configs/peft_config.yaml