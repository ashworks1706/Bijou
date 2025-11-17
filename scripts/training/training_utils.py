"""
Common utilities for training models and LoRA adapters.
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo
from unsloth import FastLanguageModel


def convert_to_gguf(
    model_path: str,
    output_dir: str,
    quantization_methods: list = ["q4_k_m", "q8_0"]
):
    """
    Convert model to GGUF format for fast inference.

    Args:
        model_path: Path to the saved model
        output_dir: Directory to save GGUF files
        quantization_methods: List of quantization methods to use
            - q4_k_m: 4-bit quantization (recommended, good balance)
            - q8_0: 8-bit quantization (higher quality)
            - f16: 16-bit float (highest quality, largest size)
    """
    print("\n" + "="*80)
    print("Converting to GGUF format...")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load the model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,  # Load full precision for conversion
        )

        for quant_method in quantization_methods:
            print(f"\nCreating {quant_method.upper()} quantized version...")

            # Save in GGUF format
            gguf_path = output_path / f"model-{quant_method}.gguf"

            model.save_pretrained_gguf(
                str(gguf_path.parent),
                tokenizer,
                quantization_method=quant_method
            )

            print(f"✓ Saved {quant_method} GGUF to: {gguf_path}")

        print("\n✓ GGUF conversion complete!")
        return True

    except Exception as e:
        print(f"✗ GGUF conversion failed: {e}")
        print("Continuing without GGUF conversion...")
        return False


def upload_to_huggingface(
    model_path: str,
    repo_id: str,
    commit_message: str = "Upload model",
    private: bool = False,
    gguf_dir: str = None
):
    """
    Upload model/adapter to HuggingFace Hub.

    Args:
        model_path: Path to the model/adapter directory
        repo_id: HuggingFace repo ID (e.g., "username/omi-adapter")
        commit_message: Commit message for the upload
        private: Whether to make the repo private
        gguf_dir: Optional path to GGUF files directory
    """
    print("\n" + "="*80)
    print(f"Uploading to HuggingFace: {repo_id}")
    print("="*80)

    try:
        api = HfApi()
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True
            )
            print(f"✓ Repository created/verified: {repo_id}")
        except Exception as e:
            print(f"Note: {e}")

        # Upload model files
        print("\nUploading model files...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        print("✓ Model files uploaded!")

        # Upload GGUF files if provided
        if gguf_dir and Path(gguf_dir).exists():
            print("\nUploading GGUF files...")
            gguf_files = list(Path(gguf_dir).glob("*.gguf"))

            for gguf_file in gguf_files:
                api.upload_file(
                    path_or_fileobj=str(gguf_file),
                    path_in_repo=f"gguf/{gguf_file.name}",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add {gguf_file.name}"
                )
                print(f"  ✓ Uploaded {gguf_file.name}")

        print(f"\n✓ Upload complete!")
        print(f"View at: https://huggingface.co/{repo_id}")

        return True

    except Exception as e:
        print(f"✗ Upload failed: {e}")
        print("Make sure you're logged in: huggingface-cli login")
        return False


def get_adapter_size(adapter_path: str) -> float:
    """Get total size of adapter in MB."""
    path = Path(adapter_path)
    total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    return total_size / (1024 * 1024)  # Convert to MB
