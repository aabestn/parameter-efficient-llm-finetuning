from trl import ORPOTrainer, ORPOConfig
from datasets import Dataset
from src.models.loader import load_quantized_model


def run_orpo(config: dict, peft_config: dict):
    model, tokenizer = load_quantized_model(
        config["model_name_or_path"],
        peft_kwargs=peft_config["lora"],
        quantization_kwargs=peft_config["quantization"],
        use_flash_attention=config.get("use_flash_attention_2", True),
    )

    dataset = Dataset.from_json(config["dataset_path"])

    orpo_args = ORPOConfig(
        output_dir=config["output_dir"],
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        bf16=config.get("bf16", True),
        max_length=config["max_length"],
        max_prompt_length=config["max_prompt_length"],
        beta=config.get("beta", 0.1),
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(config["output_dir"])