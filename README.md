# **Bijou-Core v1 — On-Device Command Model for Wearables**

Bijou-Core is a tiny, on-device language model designed to convert **speech → device actions** instantly.

It runs fully offline, optimized for  **headphones, wearables, AR glasses, and low-power hardware** .

This repository contains the  **MVP implementation** , including:

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

# 🧠 **Architecture Overview**

```
Microphone
    ↓
Audio Preprocessing (VAD, noise filtering)
    ↓
Speech-to-Text (Whisper Tiny / Bijou-STT)
    ↓
Bijou-Core (Small Command Model)
    ↓
Tool-Calling Schema Engine
    ↓
Device Action Layer (simulator or OEM SDK)
```

---

# 🔧 **Features in the MVP**

### ✔ Synthetic dataset generator

Generates command → function-call pairs using a teacher model (Qwen/Phi/etc.).

### ✔ Tool-schema definition (`tools.json`)

Defines the full list of actions a target device supports.

### ✔ Fine-tuning for tool-calling

Train small models (1–4B) to output  **structured JSON only** .

### ✔ Schema-constrained decoding

Ensures every output is valid, typed, and deterministic.

### ✔ Quantized inference

Export to **int8/int4** for fast local inference.

### ✔ Wearable Simulator (Browser Demo)

Mic → STT → LLM → JSON → simulated device UI

(Used for testing and demos).

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

# 📦 **Installation (MVP)**

Clone repo:

```bash
git clone https://github.com/your-org/bijou-core
cd bijou-core
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download base model:

```bash
python scripts/download_base_model.py --model qwen2.5-1.5b
```

---

# 🔨 **How to Run the MVP**

## **1. Define device actions**

Edit `tools/tools.json`:

```json
{
  "tools": [
    {"name": "set_volume", "params": {"level": "int"}},
    {"name": "set_anc_mode", "params": {"mode": ["off","low","high"]}}
  ]
}
```

---

## **2. Generate synthetic dataset**

```bash
python scripts/generate_data.py \
    --tools tools/tools.json \
    --output data/processed/omi_dataset.jsonl
```

---

## **3. Fine-tune model**

```bash
python scripts/finetune.py \
    --model models/base/qwen2.5-1.5b \
    --data data/processed/omi_dataset.jsonl
```

---

## **4. Quantize to int4**

```bash
python scripts/quantize.py \
    --model models/finetuned/bijou-core-mvp \
    --output models/quantized/bijou-core-int4
```

---

## **5. Run local browser demo**

```bash
cd demo/web
npm install
npm run dev
```

Open the UI, speak into your microphone, and watch the model:

* detect your command
* output structured JSON
* trigger simulated device actions

---

# 🧪 **Evaluation**

Run:

```bash
python scripts/evaluate.py \
    --model models/quantized/bijou-core-int4
```

Evaluates:

* tool-calling accuracy
* schema validity
* noise robustness
* latency

---

# 🗺️ **Roadmap**

### **v1 (MVP)**

* STT → Bijou-Core → JSON output
* Web simulator
* OEM-targeted dataset
* Fine-tuning small base models
* int4 quantization

### **v2 (Production Candidate)**

* Distilled <700M Bijou-Core
* On-device DSP/NNAPI acceleration
* Skill Packs (LoRA)
* Multilingual command support

### **v3 (OEM Release)**

* Partner integrations
* Offline multimodal conditioning
* Hybrid cloud fallback
* Full embedded SDK

---

# 🤝 **License**

MIT (MVP) — subject to change for OEM licensing.
