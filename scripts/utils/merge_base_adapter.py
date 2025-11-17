#!/usr/bin/env python3
"""
Merge bijou-base LoRA adapter with Qwen 2.5 1.5B to create a full model for vLLM.
"""

from unsloth import FastLanguageModel
from pathlib import Path


def merge_base_adapter(
    adapter_path: str = "models/bijou-base/final",
    output_path: str = "models/bijou-base-merged",
):
    """Merge bijou-base adapter with Qwen to create full servable model."""

    print("="*80)
    print("Merging bijou-base LoRA adapter with Qwen 2.5 1.5B")
    print("="*80)
    print(f"\nAdapter path: {adapter_path}")
    print(f"Output path: {output_path}")

    print("\nLoading Qwen 2.5 1.5B with bijou-base adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,  # Full precision for merging
    )

    print("\nMerging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"\nSaving merged model to: {output_path}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("\n" + "="*80)
    print("✓ Merge complete!")
    print("="*80)
    print(f"\nMerged model saved to: {output_path}")
    print("\nYou can now serve with vLLM:")
    print(f"  vllm serve {output_path} \\")
    print("    --enable-lora \\")
    print("    --max-loras 4 \\")
    print("    --max-lora-rank 16 \\")
    print("    --gpu-memory-utilization 0.9")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="models/bijou-base/final",
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/bijou-base-merged",
        help="Output path for merged model"
    )

    args = parser.parse_args()

    merge_base_adapter(
        adapter_path=args.adapter_path,
        output_path=args.output_path,
    )
