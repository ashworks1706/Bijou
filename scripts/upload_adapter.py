"""
Upload LoRA adapter to HuggingFace Hub.

Adapters are much smaller than full models (typically <100MB) and store only
the LoRA weights that can be merged with the base model.
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def upload_adapter(
    adapter_path: str,
    repo_id: str,
    base_model_id: str = None,
    commit_message: str = "Upload LoRA adapter",
    private: bool = False,
):
    """
    Upload LoRA adapter to HuggingFace Hub.

    Args:
        adapter_path: Path to the adapter directory
        repo_id: HuggingFace repo ID (e.g., "username/bijou-omi-adapter")
        base_model_id: Base model this adapter works with
        commit_message: Commit message for upload
        private: Whether to make repo private
    """
    print("="*80)
    print(f"Uploading adapter to HuggingFace: {repo_id}")
    print("="*80)

    adapter_path = Path(adapter_path)

    if not adapter_path.exists():
        print(f"Error: Adapter path does not exist: {adapter_path}")
        return False

    # Check for adapter files
    adapter_config = adapter_path / "adapter_config.json"
    adapter_model = adapter_path / "adapter_model.safetensors"

    if not adapter_config.exists() or not adapter_model.exists():
        print("Error: Missing adapter files (adapter_config.json or adapter_model.safetensors)")
        return False

    try:
        api = HfApi()

        # Create repo
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True
            )
            print(f"Repository created/verified: {repo_id}")
        except Exception as e:
            print(f"Note: {e}")

        # Create a README with adapter info
        readme_content = f"""---
library_name: peft
base_model: {base_model_id or 'unknown'}
tags:
- lora
- function-calling
- bijou-core
---

# Bijou-Core LoRA Adapter

This is a LoRA adapter for Bijou-Core, designed for device-specific function calling.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{base_model_id or 'base-model-id'}")
tokenizer = AutoTokenizer.from_pretrained("{base_model_id or 'base-model-id'}")

# Load adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")

# Generate
inputs = tokenizer("Your command here", return_tensors="pt")
outputs = model.generate(**inputs)
```

## Model Details

- **Base Model**: {base_model_id or 'unknown'}
- **Adapter Type**: LoRA
- **Task**: Function Calling
- **Framework**: Unsloth + PEFT

## Training

Trained using the Bijou-Core framework on device-specific function calling data.
"""

        readme_path = adapter_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)

        # Upload adapter files
        print("\nUploading adapter files...")
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )

        print("\nUpload complete!")
        print(f"View at: https://huggingface.co/{repo_id}")

        return True

    except Exception as e:
        print(f"Error during upload: {e}")
        print("Make sure you're logged in: huggingface-cli login")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload LoRA adapter to HuggingFace")

    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to adapter directory"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., 'username/bijou-omi-adapter')"
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default=None,
        help="Base model ID this adapter works with"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload LoRA adapter",
        help="Commit message"
    )

    args = parser.parse_args()

    success = upload_adapter(
        adapter_path=args.adapter_path,
        repo_id=args.repo_id,
        base_model_id=args.base_model_id,
        commit_message=args.commit_message,
        private=args.private,
    )

    if success:
        print("\nAdapter uploaded successfully!")
    else:
        print("\nAdapter upload failed.")
        exit(1)


if __name__ == "__main__":
    main()
