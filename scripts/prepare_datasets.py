#!/usr/bin/env python3
"""
Robust dataset downloader for Bijou.
Downloads only verified HF datasets that work with current API.
"""

import json
from pathlib import Path

from datasets import load_dataset, Dataset, DatasetDict


def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save(ds, out_dir: Path):
    """Save using save_to_disk, fallback to JSONL."""
    try:
        ds.save_to_disk(str(out_dir))
        print(f"✔ Saved to {out_dir}")
        return
    except Exception:
        print("⚠ save_to_disk failed, falling back to JSONL")

    if isinstance(ds, DatasetDict):
        for split, d in ds.items():
            split_dir = out_dir / split
            ensure(split_dir)
            with (split_dir / "data.jsonl").open("w") as f:
                for rec in d:
                    f.write(json.dumps(rec) + "\n")
    else:
        with (out_dir / "data.jsonl").open("w") as f:
            for rec in ds:
                f.write(json.dumps(rec) + "\n")


def download(id: str, out_root: Path):
    print(f"➡ Downloading: {id}")
    try:
        ds = load_dataset(id)
    except Exception as e:
        print(f"❌ Failed: {id} → {e}")
        return False

    out = out_root / "raw" / id.replace("/", "_")
    ensure(out)
    save(ds, out)
    return True


def main():
    out_root = Path("data")
    ensure(out_root / "raw")

    HF_DATASETS = {
    "superb": "superb_ks",                         # keyword spotting
    "PolyAI/minds14": "minds14",                   # intent classification
    "speech_commands": "speech_commands",          # speech → command
    "fluent_speech_commands": "fluent_speech",     # spoken command dataset
    "google/gnaturalquestions-short": "nq_short",  # short NL commands/questions
    "HuggingFaceH4/ultrachat_200k": "ultrachat",   # already working
    "Open-Orca/OpenOrca": "openorca",              # instruction following
    "Trelis/jsonformer-data": "jsonformer_data",   # JSON target
    "RikL/jsondataset": "jsondataset",             # more JSON samples
}



    MANUAL = {
        "topv2": "Alexa TOPv2 requires academic mirror",
        "openhermes": "Manual download",
        "t0_mix": "Manual",
        "jsonformer": "Manual",
        "gorilla": "Manual",
        "voxtrees": "Manual",
    }

    results = {}

    for hf_id, name in HF_DATASETS.items():
        ok = download(hf_id, out_root)
        results[name] = ok

    for name, note in MANUAL.items():
        print(f"ℹ {name}: manual download required → {note}")
        results[name] = False

    with (out_root / "dataset_manifest.json").open("w") as mf:
        json.dump(results, mf, indent=2)

    print("\n✨ Done!")


if __name__ == "__main__":
    main()
