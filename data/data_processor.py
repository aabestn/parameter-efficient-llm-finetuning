import json
import argparse
from pathlib import Path


def format_sft_sample(raw_sample: dict) -> dict:
    """Formats raw instruction-input-output JSON into chat template messages."""
    user_content = raw_sample["instruction"]
    if raw_sample.get("input"):
        user_content += f"\nContext: {raw_sample['input']}"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": raw_sample["output"]},
        ]
    }


def process_sft_dataset(input_file: str, output_file: str):
    """Reads raw JSONL, transforms schema, and writes processed output."""
    path_in = Path(input_file)
    path_out = Path(output_file)
    path_out.parent.mkdir(parents=True, exist_ok=True)

    with open(path_in, "r", encoding="utf-8") as fin, open(
        path_out, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if line.strip():
                raw = json.loads(line)
                processed = format_sft_sample(raw)
                fout.write(json.dumps(processed) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SFT dataset")
    parser.add_argument(
        "--input", default="data/raw/sft_raw.jsonl", help="Input JSONL path"
    )
    parser.add_argument(
        "--output",
        default="data/processed/sft_data.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    process_sft_dataset(args.input, args.output)
    print(f"Data preprocessing completed -> {args.output}")