from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
from src.models.loader import load_quantized_model


def run_sft(config: dict, peft_config: dict):
    model, tokenizer = load_quantized_model(
        config["model_name_or_path"],
        peft_kwargs=peft_config["lora"],
        quantization_kwargs=peft_config["quantization"],
        use_flash_attention=config.get("use_flash_attention_2", True),
    )

    dataset = Dataset.from_json(config["dataset_path"])

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        bf16=config.get("bf16", True),
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args,
        packing=config.get("packing", True),
    )

    trainer.train()
    trainer.save_model(config["output_dir"])