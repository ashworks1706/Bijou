# Bijou Scripts

Organized scripts for training, evaluation, and deployment.

## Structure

```
scripts/
├── training/          # Model training
│   ├── train_base.py       - Train base model on function-calling data
│   ├── train_adapter.py    - Train device-specific LoRA adapters
│   └── requirements.txt    - Python dependencies
├── data/              # Data preparation
│   ├── prepare.py          - Prepare training data from HuggingFace
│   ├── formatters.py       - Dataset formatters (xlam, glaive, etc.)
│   └── generate_synthetic.py - Generate synthetic device data
├── evaluation/        # Model evaluation
│   └── evaluate.py         - Evaluate models on benchmarks
├── demo/              # Interactive demos
│   └── interactive.py      - Interactive testing
├── utils/             # Utilities
│   └── upload.py           - Upload models to HuggingFace
└── runs/              # Training run logs
```

## Quick Start

### 1. Train Omi LoRA Adapter
```bash
python scripts/training/train_adapter.py \
  --base_model Qwen/Qwen2.5-1.5B \
  --data_path data/omi_training \
  --output_dir adapters/omi \
  --num_epochs 3
```

### 2. Evaluate Model
```bash
python scripts/evaluation/evaluate.py \
  --model_name Qwen/Qwen2.5-1.5B \
  --dataset data/eval_dataset_small.jsonl
```

### 3. Interactive Testing
```bash
python scripts/demo/interactive.py
```

## Data Flow

1. **Prepare data**: `scripts/data/prepare.py` → `data/training/`
2. **Train adapter**: `scripts/training/train_adapter.py` → `adapters/device_name/`
3. **Evaluate**: `scripts/evaluation/evaluate.py` → `results/`
