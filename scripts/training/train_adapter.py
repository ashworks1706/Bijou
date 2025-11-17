"""
Train Omi-specific LoRA adapter on top of base Qwen 2.5 1.5B.

This creates a device-specific adapter that can be hot-swapped at runtime.
"""

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import argparse
from pathlib import Path


def load_training_data(data_path: str):
    """Load Omi training data from JSONL files."""
    print(f"Loading Omi training data from: {data_path}")

    dataset = load_dataset('json', data_files={
        'train': str(Path(data_path) / 'omi_synthetic_train.jsonl'),
        'validation': str(Path(data_path) / 'omi_synthetic_val.jsonl')
    })

    print(f"Train examples: {len(dataset['train'])}")
    print(f"Val examples: {len(dataset['validation'])}")

    return dataset


def setup_model(
    base_model: str = "Qwen/Qwen2.5-1.5B",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 16,
):
    """Load base model and add LoRA adapters."""
    print(f"Loading base model: {base_model}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"✓ Model loaded with LoRA adapters")
    return model, tokenizer


def train(
    model,
    tokenizer,
    dataset,
    output_dir: str = "adapters/omi",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
):
    """Fine-tune LoRA adapter."""
    print(f"\nTraining Omi LoRA adapter...")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=5,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        max_length=2048,
        packing=False,
        dataset_text_field="text",
        eos_token=tokenizer.eos_token,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        args=training_args,
    )

    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    trainer.train()

    print("\n✓ Training complete!")

    # Save adapter
    final_path = Path(output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving Omi LoRA adapter to: {final_path}")
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print("✓ Adapter saved!")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train Omi LoRA adapter")

    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--data_path", type=str, default="data/omi_training")
    parser.add_argument("--output_dir", type=str, default="adapters/omi")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)

    args = parser.parse_args()

    # Load data
    dataset = load_training_data(args.data_path)

    # Setup model
    model, tokenizer = setup_model(
        base_model=args.base_model,
        lora_r=args.lora_r,
    )

    # Train
    train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    print("\n✓ Omi LoRA adapter training complete!")
    print(f"\nAdapter saved to: {Path(args.output_dir) / 'final'}")
    print("\nNext steps:")
    print("  1. Test the adapter with hot-swap demo")
    print("  2. Train AirPods adapter")
    print("  3. Compare performance")


if __name__ == "__main__":
    main()
