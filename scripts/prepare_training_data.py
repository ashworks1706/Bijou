"""
Generic dataset loader for function-calling datasets.

Loads datasets from HuggingFace and formats them using dataset-specific formatters.
"""

from datasets import load_dataset, Dataset
import json
from pathlib import Path
from typing import Optional
import argparse

from data_formatters import get_formatter


def load_and_format_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    formatter_name: Optional[str] = None
) -> Dataset:
    """
    Load a dataset from HuggingFace and format it for training.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "Salesforce/xlam-function-calling-60k")
        split: Dataset split to load (train, test, validation)
        max_samples: Optional limit on number of samples
        formatter_name: Override auto-detected formatter (e.g., "xlam", "glaive")

    Returns:
        Formatted dataset ready for training
    """
    print(f"Loading dataset: {dataset_name} (split: {split})")

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Loaded {len(dataset)} examples")

    # Show sample
    print("\nSample raw example:")
    print(json.dumps(dataset[0], indent=2, default=str))

    # Get formatter
    formatter_key = formatter_name or dataset_name
    formatter = get_formatter(formatter_key)

    print(f"\nUsing formatter: {formatter.__name__}")

    # Convert dataset
    print("Converting to Qwen format...")
    converted_dataset = dataset.map(
        formatter,
        remove_columns=dataset.column_names,
        desc="Formatting dataset"
    )

    # Remove None values (failed conversions)
    original_len = len(converted_dataset)
    converted_dataset = converted_dataset.filter(lambda x: x is not None and x.get('text'))
    filtered_count = original_len - len(converted_dataset)

    if filtered_count > 0:
        print(f"⚠️  Filtered out {filtered_count} failed conversions")

    print(f"✓ Successfully converted {len(converted_dataset)} examples")

    # Show sample converted
    print("\nSample converted example:")
    print(converted_dataset[0]['text'][:800] + "...\n")

    return converted_dataset


def save_dataset(dataset: Dataset, output_dir: str, dataset_name: str, val_split: float = 0.05):
    """
    Save formatted dataset to disk as JSONL files.

    Args:
        dataset: Formatted dataset to save
        output_dir: Directory to save files
        dataset_name: Name for the output files
        val_split: Fraction to use for validation (default 5%)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Split into train/val
    split_dataset = dataset.train_test_split(test_size=val_split, seed=42)

    # Clean name for filename
    clean_name = dataset_name.replace('/', '_').replace('-', '_')

    # Save as jsonl
    train_path = output_path / f"{clean_name}_train.jsonl"
    val_path = output_path / f"{clean_name}_val.jsonl"

    print(f"Saving to {output_path}...")

    with open(train_path, 'w') as f:
        for example in split_dataset['train']:
            f.write(json.dumps(example) + '\n')

    with open(val_path, 'w') as f:
        for example in split_dataset['test']:
            f.write(json.dumps(example) + '\n')

    print(f"✓ Saved training data:")
    print(f"  Train: {train_path} ({len(split_dataset['train'])} examples)")
    print(f"  Val:   {val_path} ({len(split_dataset['test'])} examples)")

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from HuggingFace datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Salesforce/xlam-function-calling-60k",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--formatter",
        type=str,
        default=None,
        help="Override formatter name (xlam, glaive, hermes)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/training",
        help="Output directory for formatted data"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Validation split fraction (default: 0.05)"
    )

    args = parser.parse_args()

    # Load and format
    dataset = load_and_format_dataset(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        formatter_name=args.formatter
    )

    # Save
    save_dataset(
        dataset=dataset,
        output_dir=args.output_dir,
        dataset_name=args.dataset.split('/')[-1],  # Use last part of name
        val_split=args.val_split
    )

    print("\n✓ Data preparation complete!")


if __name__ == "__main__":
    main()
