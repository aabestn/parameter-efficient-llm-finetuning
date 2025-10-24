import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_quantized_model(
    model_name_or_path: str,
    peft_kwargs: dict,
    quantization_kwargs: dict,
    use_flash_attention: bool = True,
):
    """Loads a base LLM with BitsAndBytes 4-bit NF4 quantization,

    FlashAttention-2, and injects trainable LoRA adapters.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_kwargs.get("load_in_4bit", True),
        bnb_4bit_quant_type=quantization_kwargs.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quantization_kwargs.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=getattr(
            torch, quantization_kwargs.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
    )

    attn_implementation = "flash_attention_2" if use_flash_attention else "eager"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        attn_implementation=attn_implementation,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=peft_kwargs.get("r", 16),
        lora_alpha=peft_kwargs.get("lora_alpha", 32),
        lora_dropout=peft_kwargs.get("lora_dropout", 0.05),
        bias=peft_kwargs.get("bias", "none"),
        task_type=peft_kwargs.get("task_type", "CAUSAL_LM"),
        target_modules=peft_kwargs.get("target_modules", None),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer