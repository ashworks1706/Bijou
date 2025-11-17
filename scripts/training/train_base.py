"""
Fine-tune Qwen 2.5 1.5B on function calling using Unsloth.

This creates the BASE model for Bijou-Core that understands general tool calling.
Device-specific LoRA adapters will be trained separately on top of this base.
"""

import torch
from unsloth import FastLanguageModel  # Import unsloth FIRST for optimizations
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
import argparse
from pathlib import Path
import os
from training_utils import convert_to_gguf, upload_to_huggingface


def load_training_data(data_path: str):
    """Load formatted training data from JSONL file."""
    print(f"Loading training data from: {data_path}")

    # Load dataset
    dataset = load_dataset('json', data_files={
        'train': str(Path(data_path) / '*_train.jsonl'),
        'validation': str(Path(data_path) / '*_val.jsonl')
    })

    print(f"Train examples: {len(dataset['train'])}")
    print(f"Val examples: {len(dataset['validation'])}")

    return dataset


def setup_model(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
):
    """
    Load Qwen model with Unsloth optimizations and add LoRA adapters.

    Args:
        model_name: HuggingFace model ID
        max_seq_length: Maximum sequence length
        load_in_4bit: Use 4-bit quantization for memory efficiency
        lora_r: LoRA rank (higher = more parameters, better accuracy, slower)
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout for LoRA layers
    """
    print(f"Loading model: {model_name}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  4-bit quantization: {load_in_4bit}")
    print(f"  LoRA rank: {lora_r}, alpha: {lora_alpha}")

    # Load model and tokenizer with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect best dtype
        load_in_4bit=load_in_4bit,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=42,
    )

    print(f"✓ Model loaded with LoRA adapters")

    return model, tokenizer


def train(
    model,
    tokenizer,
    dataset,
    output_dir: str = "models/bijou-base",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 10,
    logging_steps: int = 10,
    save_steps: int = 100,
):
    """
    Fine-tune the model using SFTTrainer.

    Args:
        model: The model to train
        tokenizer: The tokenizer
        dataset: Training dataset
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Accumulate gradients over N steps
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
    """
    print(f"\nTraining configuration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps: {warmup_steps}")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_steps=save_steps,
        save_total_limit=3,  # Keep only last 3 checkpoints
        optim="adamw_8bit",  # 8-bit optimizer for memory efficiency
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",  # Disable wandb/tensorboard for now
        max_length=2048,
        packing=False,  # Don't pack multiple examples together
        dataset_text_field="text",  # Our formatted data has 'text' field
        eos_token=tokenizer.eos_token,
    )

    # SFT Trainer for supervised fine-tuning
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

    # Train
    trainer.train()

    print("\n✓ Training complete!")

    # Save final model
    final_model_path = Path(output_dir) / "final"
    final_model_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving final model to: {final_model_path}")
    model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    print("✓ Model saved!")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 1.5B for function calling")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B", help="Base model to fine-tune")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")

    # Data arguments
    parser.add_argument("--data_path", type=str, default="data/training", help="Path to training data directory")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="models/bijou-base", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")

    # Upload arguments
    parser.add_argument("--hf_repo_id", type=str, default=None, help="HuggingFace repo ID to upload to (e.g., 'username/bijou-core-base')")
    parser.add_argument("--hf_private", action="store_true", help="Make HuggingFace repo private")
    parser.add_argument("--convert_gguf", action="store_true", help="Convert model to GGUF format after training")
    parser.add_argument("--gguf_methods", type=str, default="q4_k_m,q8_0", help="Comma-separated GGUF quantization methods")

    args = parser.parse_args()

    # Load data
    dataset = load_training_data(args.data_path)

    # Setup model
    model, tokenizer = setup_model(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # Train
    trainer = train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    final_model_path = Path(args.output_dir) / "final"

    # Convert to GGUF if requested
    gguf_dir = None
    if args.convert_gguf:
        gguf_dir = Path(args.output_dir) / "gguf"
        quant_methods = [m.strip() for m in args.gguf_methods.split(',')]
        convert_to_gguf(
            model_path=str(final_model_path),
            output_dir=str(gguf_dir),
            quantization_methods=quant_methods
        )

    # Upload to HuggingFace if repo ID provided
    if args.hf_repo_id:
        upload_to_huggingface(
            model_path=str(final_model_path),
            repo_id=args.hf_repo_id,
            commit_message=f"Upload Bijou-Core base model (trained on {len(dataset['train'])} examples)",
            private=args.hf_private,
            gguf_dir=str(gguf_dir) if gguf_dir else None
        )

    print(f"\nFinal model saved to: {final_model_path}")
    if args.hf_repo_id:
        print(f"Uploaded to: https://huggingface.co/{args.hf_repo_id}")

    print("\nNext steps:")
    print("  1. Evaluate the base model on benchmark")
    print("  2. Train device-specific LoRA adapters (e.g., Omi, AirPods)")
    print("  3. Evaluate base + adapter performance")


if __name__ == "__main__":
    main()
