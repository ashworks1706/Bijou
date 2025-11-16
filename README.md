<img width="1825" height="270" alt="image" src="https://github.com/user-attachments/assets/99625444-ebc7-48e1-af4b-203ce4c82374" />
<p align="center">
    <b>Tiny language models for tiny devices</b>
</p>

Bijou is a framework and series tiny, on-device language models designed to work for minute expert tasks.

It runs fully offline, optimized for  **headphones, wearables, AR glasses, and low-power hardware** .

This repository contains:

* synthetic dataset generation
* tool-schema definitions
* small-model fine-tuning
* schema-constrained decoding
* quantized inference engine
* browser-based demo (mic → STT → model → action)

---

# 🚀 **What We're Building**

Wearables today rely on cloud LLMs → slow, wrong, delayed.

Bijou-Core fixes this by using a **tiny, specialized model** that only does one thing:

> **Understand a user’s command and trigger the correct function.**

Example:

User says:

> “turn noise cancelling to high”

Bijou-Core outputs:

```json
{
  "function": "set_anc_mode",
  "mode": "high"
}
```

Zero hallucination.

Zero chit-chat.

Just actions.

---

# 🛠️ **Repository Structure**

```
bijou/
│
├── adapters/           # LoRA adapters & skill-pack modules for extending model capabilities
│   └── README.md
│
├── api/                # Public API interfaces (Python/JS) for calling the model + schema engine
│   └── README.md
│
├── data/               # Synthetic + processed datasets used for fine-tuning Bijou-Core
│   └── README.md
│
├── demo/               # Browser/desktop demo (mic → STT → model → action simulator)
│   └── README.md
│
├── inference/             # On-device inference engine (quantized models, kernels, runtime)
│   └── README.md
|
├── models/             # Base, fine-tuned, and quantized model checkpoints
│   ├── base/           # Original downloaded SLMs (Qwen, Phi, Gemma, etc.)
│   ├── finetuned/      # Command-specialized models trained for tool-calling
│   ├── quantized/      # int4/int8 optimized exports for on-device inference
│   └── README.md
│
├── OEMs/               # OEM-specific configs (schemas, notes, device constraints)
│   ├── omi/            # Example target OEM folder with tools.json + integration notes
│   └── README.md
│
├── scripts/            # Training, dataset generation, quantization, and evaluation scripts
│   └── README.md
│
└── utils/              # Shared utilities (tokenization, schema validation, helpers)
    └── README.md

```

---


# 🤝 **License**

MIT (MVP) — subject to change for OEM licensing.
