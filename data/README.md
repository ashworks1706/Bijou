# Dataset layout for Bijou

This folder contains the raw and processed datasets used for training and conditioning Bijou-Core.

Structure created by `scripts/prepare_datasets.py`:

- data/
  - raw/                # raw downloads saved per dataset (Hugging Face / Kaggle / manual)
    - common_voice/
    - librispeech_asr/
    - urbansound8k/
  - processed/          # generated or preprocessed items (OEM synthetic JSONL etc.)
    - oem_synthetic_<...>.jsonl
  - dataset_manifest.json

Notes:
- Large datasets like AudioSet, CHiME-6, or some Alexa datasets may require manual download or API keys. The script will print instructions when manual steps are required.
- For Kaggle downloads (UrbanSound8K), set `KAGGLE_USERNAME` and `KAGGLE_KEY` in your environment.
- Use `python scripts/prepare_datasets.py --help` for usage details.
# Data

this repo is for preparing finetuning dataset