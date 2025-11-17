#!/bin/bash
# Reorganize scripts directory

echo "Reorganizing scripts..."

# Create new structure
mkdir -p scripts/{training,data,evaluation,demo,utils}

# Move files
mv scripts/train_base_model.py scripts/training/train_base.py
mv scripts/train_omi_adapter.py scripts/training/train_adapter.py
mv scripts/requirements.txt scripts/training/requirements.txt

mv scripts/prepare_training_data.py scripts/data/prepare.py
mv scripts/data_formatters.py scripts/data/formatters.py
mv scripts/generate_omi_synthetic_data.py scripts/data/generate_synthetic.py

mv scripts/evaluate_model.py scripts/evaluation/evaluate.py

mv scripts/interactive_test.py scripts/demo/interactive.py

mv scripts/upload_adapter.py scripts/utils/upload.py

echo "✓ Reorganization complete!"
echo ""
echo "New structure:"
find scripts -type f -name "*.py" | sort
