from typing import Dict, List, Any
from transformers import AutoTokenizer


def apply_chat_template(
    sample: Dict[str, List[Dict[str, str]]],
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    """Applies model specific chat template to conversation messages."""
    messages = sample["messages"]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": formatted_prompt}


def format_preference_pair(sample: Dict[str, str]) -> Dict[str, str]:
    """Ensures DPO/ORPO dataset keys align with TRL Trainer requirements."""
    return {
        "prompt": sample["prompt"],
        "chosen": sample["chosen"],
        "rejected": sample["rejected"],
    }