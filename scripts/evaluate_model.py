"""
Evaluate a small language model on tool-calling tasks for wearable devices.

Usage:
    # Evaluate a single model by HuggingFace ID
    python scripts/evaluate_model.py --model_name Qwen/Qwen2.5-1.5B --dataset data/eval_dataset_small.jsonl

    # Evaluate all enabled models from config
    python scripts/evaluate_model.py --config models/models_config.json

    # Evaluate a specific model from config by name
    python scripts/evaluate_model.py --config models/models_config.json --model_name "Qwen 2.5 1.5B"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime


def load_eval_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset from JSONL file."""
    dataset = []
    with open(dataset_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset


def load_models_config(config_path: str) -> Dict[str, Any]:
    """Load models configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_model_from_config(config: Dict[str, Any], model_identifier: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get model(s) from config.

    Args:
        config: The loaded config dictionary
        model_identifier: Optional model name or ID. If None, returns all enabled models.

    Returns:
        List of model dictionaries to evaluate
    """
    models = config.get("models", [])

    if model_identifier is None:
        return [m for m in models if m.get("enabled", True)]

    for model in models:
        if model.get("name") == model_identifier or model.get("model_id") == model_identifier:
            return [model]

    return [{
        "name": model_identifier,
        "model_id": model_identifier,
        "description": "Direct HuggingFace model ID",
        "enabled": True
    }]


def load_model_and_tokenizer(model_name: str, device: str = "auto"):
    """Load HuggingFace model and tokenizer."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    print(f"Model loaded successfully on device: {model.device}")
    return model, tokenizer


def create_prompt(command: str) -> str:
    """
    Create a prompt for the model to convert a command into a function call.

    This is a basic prompt template. You'll want to fine-tune this based on
    how you train your model.
    """
    prompt = f"""Convert the following voice command into a JSON function call.

Available functions:
- set_volume(level: int) - Set volume from 0-100
- set_anc_mode(mode: str) - Set ANC mode: "off", "low", "medium", "high"
- play_music() - Start playing music
- pause_music() - Pause music
- skip_track(direction: str) - Skip track: "forward" or "backward"
- answer_call() - Answer incoming call
- end_call() - End current call
- set_eq_preset(preset: str) - Set EQ: "bass_boost", "treble_boost", "vocal", "flat", "custom"
- set_transparency_mode(enabled: bool) - Enable/disable transparency mode
- check_battery() - Check battery level

Command: "{command}"

Output only valid JSON in this format:
{{"function": "function_name", "arguments": {{"param": "value"}}}}

JSON:"""

    return prompt


def generate_function_call(model, tokenizer, command: str, max_new_tokens: int = 100) -> str:
    """Generate function call JSON from a command."""
    prompt = create_prompt(command)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for more deterministic outputs
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return generated_text.strip()


def parse_json_output(output: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from model output."""
    try:
        start_idx = output.find('{')
        end_idx = output.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            return None

        json_str = output[start_idx:end_idx]
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None


def normalize_json(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize JSON for comparison (sort keys, handle empty dicts)."""
    if not obj:
        return {}
    normalized = {
        "function": obj.get("function", ""),
        "arguments": obj.get("arguments", {})
    }

    return normalized


def compare_outputs(predicted: Optional[Dict[str, Any]], expected: Dict[str, Any]) -> Dict[str, bool]:
    """
    Compare predicted and expected outputs.

    Returns a dict with:
        - exact_match: True if outputs are identical
        - function_match: True if function names match
        - args_match: True if arguments match
    """
    pred_norm = normalize_json(predicted) if predicted else {}
    exp_norm = normalize_json(expected)
    function_match = pred_norm.get("function") == exp_norm.get("function")
    args_match = pred_norm.get("arguments") == exp_norm.get("arguments")
    exact_match = function_match and args_match

    return {
        "exact_match": exact_match,
        "function_match": function_match,
        "args_match": args_match,
        "valid_json": predicted is not None
    }


def evaluate(model, tokenizer, dataset: List[Dict[str, Any]], verbose: bool = False, model_name: str = "model"):
    """Run evaluation on the dataset."""
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "total": len(dataset),
        "exact_match": 0,
        "function_match": 0,
        "args_match": 0,
        "valid_json": 0,
        "details": []
    }

    print(f"\nEvaluating {model_name} on {results['total']} examples...")
    print("=" * 80)

    for i, example in enumerate(dataset):
        command = example["command"]
        expected = example["expected_output"]
        raw_output = generate_function_call(model, tokenizer, command)
        predicted = parse_json_output(raw_output)
        comparison = compare_outputs(predicted, expected)

        if comparison["exact_match"]:
            results["exact_match"] += 1
        if comparison["function_match"]:
            results["function_match"] += 1
        if comparison["args_match"]:
            results["args_match"] += 1
        if comparison["valid_json"]:
            results["valid_json"] += 1

        detail = {
            "command": command,
            "expected": expected,
            "predicted": predicted,
            "raw_output": raw_output,
            "comparison": comparison
        }
        results["details"].append(detail)

        if verbose or not comparison["exact_match"]:
            status = "✓" if comparison["exact_match"] else "✗"
            print(f"{status} [{i+1}/{results['total']}] {command}")
            if not comparison["exact_match"]:
                print(f"  Expected:  {json.dumps(expected)}")
                print(f"  Predicted: {json.dumps(predicted)}")
                print(f"  Raw:       {raw_output[:100]}...")
                print()

    print("=" * 80)
    print("\nResults:")
    print(f"  Valid JSON:       {results['valid_json']}/{results['total']} ({results['valid_json']/results['total']*100:.1f}%)")
    print(f"  Function Match:   {results['function_match']}/{results['total']} ({results['function_match']/results['total']*100:.1f}%)")
    print(f"  Arguments Match:  {results['args_match']}/{results['total']} ({results['args_match']/results['total']*100:.1f}%)")
    print(f"  Exact Match:      {results['exact_match']}/{results['total']} ({results['exact_match']/results['total']*100:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on tool-calling tasks")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., Qwen/Qwen2.5-1.5B, microsoft/phi-2)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval_dataset_small.jsonl",
        help="Path to evaluation dataset (JSONL format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed results (JSON format)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on (auto, cpu, cuda, cuda:0, etc.)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all examples (not just failures)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )

    args = parser.parse_args()
    print(f"Loading dataset from: {args.dataset}")
    dataset = load_eval_dataset(args.dataset)
    print(f"Loaded {len(dataset)} examples")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)
    results = evaluate(model, tokenizer, dataset, verbose=args.verbose)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
