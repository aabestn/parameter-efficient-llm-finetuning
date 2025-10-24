# Parameter-Efficient LLM Fine-Tuning Engine

An end-to-end LLM fine-tuning toolkit supporting SFT, DPO, and ORPO via Hugging Face TRL, optimized with QLoRA 4-bit quantization, DeepSpeed Stage-3, and FlashAttention-2.

## Quickstart

1. **Install Dependencies**:
pip install -r requirements.txt
2. **Prepare Datasets**:
python data/data_processor.py --input data/raw/sft_raw.jsonl
3. **Run Supervised Fine-Tuning (SFT)**:
bash scripts/run_sft.sh
4. **Run Preference Alignment (DPO)**:
bash scripts/run_dpo.sh
5. **Execute Evaluation Benchmarks**:
bash scripts/run_eval.sh