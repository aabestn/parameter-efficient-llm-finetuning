#!/bin/bash
# Lighteval & LLM-as-a-Judge benchmark execution script

echo "Running Lighteval Benchmarks..."
lighteval accelerate \
    --tasks configs/lighteval_config.yaml \
    --output_dir eval_results/

echo "Running LLM-as-a-Judge Alignment Evaluation..."
python eval/llm_judge.py \
    --input eval/sample_pairs.jsonl \
    --output eval_results/judge_results.json