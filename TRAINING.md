# Bijou-Core Training Guide

This guide walks through fine-tuning the Bijou-Core base model and training device-specific LoRA adapters.

## Architecture

```
┌─────────────────────────────────────┐
│  Bijou-Core Base Model              │
│  (Qwen 2.5 1.5B + LoRA)             │
│  Trained on 60k function calls      │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       │  LoRA Adapter   │ (Hot-swappable)
       │  Device-specific │
       └───────┬────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼───┐  ┌──▼────┐
│  Omi  │  │AirPods│ │RayBan │
│       │  │  Pro  │ │Stories│
└───────┘  └───────┘ └───────┘
```

## Step 1: Prepare Training Data

Download and format the XLAM dataset:

```bash
# Install dependencies
pip install -r scripts/requirements.txt

# Login to HuggingFace (required for xlam dataset)
huggingface-cli login

# Prepare training data
python scripts/prepare_training_data.py \
  --dataset "Salesforce/xlam-function-calling-60k" \
  --output_dir data/training \
  --max_samples 10000  # Optional: limit for testing
```

This will create:
- `data/training/xlam_function_calling_60k_train.jsonl`
- `data/training/xlam_function_calling_60k_val.jsonl`

## Step 2: Fine-tune Base Model

Train the base Bijou-Core model with automatic upload to HuggingFace:

```bash
python scripts/train_base_model.py \
  --model_name "Qwen/Qwen2.5-1.5B" \
  --data_path data/training \
  --output_dir models/bijou-base \
  --num_epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --hf_repo_id "yourusername/bijou-core-base" \
  --convert_gguf \
  --gguf_methods "q4_k_m,q8_0"
```

**Upload Options:**
- `--hf_repo_id`: Your HuggingFace repo (creates if doesn't exist)
- `--hf_private`: Make repo private
- `--convert_gguf`: Convert to GGUF format for fast inference
- `--gguf_methods`: Quantization methods (q4_k_m, q8_0, f16)

**GPU Requirements:**
- Minimum: 16GB VRAM (RTX 4080/A4000)
- Recommended: 24GB+ VRAM (RTX 4090/A5000)
- With 4-bit quantization, can run on 12GB VRAM

**Training time:**
- ~2-4 hours on RTX 4090 for 10k examples
- ~6-8 hours on RTX 4080 for 60k examples

The final model will be saved to `models/bijou-base/final/`

## Step 3: Generate Device-Specific Data

Create synthetic training data for a specific device (e.g., Omi):

```bash
python scripts/generate_synthetic_data.py \
  --schema schemas/omi/tools.json \
  --output data/synthetic/omi_train.jsonl \
  --num_examples 500
```

## Step 4: Train Device LoRA Adapter

Train a device-specific adapter on top of the base model:

```bash
python scripts/train_adapter.py \
  --base_model models/bijou-base/final \
  --data_path data/synthetic/omi_train.jsonl \
  --output_dir adapters/omi \
  --num_epochs 2 \
  --lora_r 8  # Smaller rank for adapter
```

## Step 5: Evaluate

Evaluate base model vs base+adapter:

```bash
# Evaluate base model only
python scripts/evaluate_model.py \
  --model_name models/bijou-base/final \
  --dataset data/eval_omi_commands.jsonl \
  --schema schemas/omi/tools.json

# Evaluate base + Omi adapter
python scripts/evaluate_model.py \
  --model_name models/bijou-base/final \
  --adapter adapters/omi \
  --dataset data/eval_omi_commands.jsonl \
  --schema schemas/omi/tools.json
```

## Expected Performance

**Base Model (after training on XLAM):**
- Valid JSON: ~95%
- Function Match: ~70-80%
- Exact Match: ~50-60%

**Base + Omi Adapter:**
- Valid JSON: ~98%
- Function Match: ~90%+
- Exact Match: ~80%+

## Quick Test

```bash
# Quick test on 1000 examples (faster iteration)
python scripts/prepare_training_data.py \
  --dataset "Salesforce/xlam-function-calling-60k" \
  --max_samples 1000 \
  --output_dir data/training

python scripts/train_base_model.py \
  --data_path data/training \
  --output_dir models/bijou-base-test \
  --num_epochs 1 \
  --save_steps 50
```

## Troubleshooting

**Out of Memory:**
- Reduce `--batch_size` to 2 or 1
- Increase `--gradient_accumulation_steps` to maintain effective batch size
- Use `--max_seq_length 1024` for shorter sequences

**Slow Training:**
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Use `--batch_size 8` if you have >24GB VRAM
- Reduce `--max_samples` for faster iteration

**Dataset Not Found:**
- Run `huggingface-cli login` first
- Accept dataset terms on HuggingFace website
- Check internet connection

## Next Steps

After training:
1. Quantize model to GGUF for fast inference
2. Build terminal demo with hot-swapping
3. Measure latency on target hardware
4. Train additional adapters (AirPods, AR glasses)
