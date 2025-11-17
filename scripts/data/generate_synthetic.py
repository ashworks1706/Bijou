"""
Generate 1000 synthetic Omi training examples using Anthropic API.

Calls Anthropic 100 times to generate 10 examples each = 1000 total examples.
"""

import json
from pathlib import Path
from typing import List, Dict
from anthropic import Anthropic


def load_omi_tools():
    """Load Omi tool definitions."""
    tools_path = Path("OEMs/omi/tools.json")
    with open(tools_path) as f:
        data = json.load(f)
    return data['tools']


def format_training_example(query: str, tool_name: str, arguments: Dict, tools: List[Dict]) -> Dict:
    """Format a single training example in Qwen chat format."""
    tools_text = "\n\nAvailable functions:\n"
    for tool in tools:
        name = tool['name']
        desc = tool['description']
        params = tool['parameters'].get('properties', {})
        required = tool['parameters'].get('required', [])

        param_strs = []
        for param_name, param_info in params.items():
            param_type = param_info.get('type', 'string')
            is_req = param_name in required
            marker = "" if is_req else "?"
            param_strs.append(f"{param_name}{marker}: {param_type}")

        params_str = ", ".join(param_strs) if param_strs else ""
        tools_text += f"- {name}({params_str})\n  {desc}\n"

    system_msg = f"You are a helpful assistant with access to functions. Use them when appropriate.{tools_text}\n\nOutput format: {{\"function\": \"function_name\", \"arguments\": {{...}}}}"
    assistant_msg = json.dumps({
        "function": tool_name,
        "arguments": arguments
    })
    text = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"

    return {"text": text}


def generate_batch_with_anthropic(tools: List[Dict], batch_num: int, client: Anthropic) -> List[Dict]:
    """Generate 10 examples using Anthropic API."""
    tools_json = json.dumps(tools, indent=2)
    prompt = f"""You are generating synthetic training data for an Omi wearable AI assistant.

The Omi device records conversations and helps users recall information, manage tasks, and search their memories.

Here are the available tools:
{tools_json}

Generate 10 diverse, realistic user queries that would use these Omi tools. Include a variety of:
- Different time periods (today, last week, yesterday, this month, etc.)
- Different topics (work, family, health, projects, etc.)
- Different tools (spread across all available tools)
- Natural, conversational language
- Realistic arguments with proper dates, limits, etc.

Output ONLY a valid JSON array with this exact format (no markdown, no explanations, no code blocks):
[
  {{"query": "user question here", "tool": "tool_name", "arguments": {{"param": "value"}}}},
  ...
]

Make sure the output is valid JSON that can be parsed directly."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        temperature=0.9,  # High creativity for diversity
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    content = response.content[0].text.strip()
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()
    elif content.startswith("```"):
        content = content.replace("```", "").strip()

    data = json.loads(content)
    if isinstance(data, dict):
        # Try to find the array
        for key in ['examples', 'data', 'queries', 'results']:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        print(f"Warning: Unexpected response format in batch {batch_num}")
        return []

    # Convert to training format
    examples = []
    for item in data:
        try:
            example = format_training_example(
                query=item['query'],
                tool_name=item['tool'],
                arguments=item['arguments'],
                tools=tools
            )
            examples.append(example)
        except Exception as e:
            print(f"Warning: Failed to format example in batch {batch_num}: {e}")
            continue

    return examples


def generate_all_examples(num_batches: int = 100, examples_per_batch: int = 10) -> List[Dict]:
    """Generate all examples by calling Anthropic multiple times."""
    client = Anthropic(api_key="fake_key_lol")
    tools = load_omi_tools()
    all_examples = []

    print(f"Generating {num_batches * examples_per_batch} examples in {num_batches} batches...")
    print("=" * 60)

    for i in range(num_batches):
        print(f"Batch {i+1}/{num_batches}...", end=" ", flush=True)

        try:
            batch_examples = generate_batch_with_anthropic(tools, i+1, client)
            all_examples.extend(batch_examples)
            print(f"✓ ({len(batch_examples)} examples)")
        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    print(f"\n✓ Generated {len(all_examples)} total examples")
    return all_examples


def save_dataset(examples: List[Dict], output_dir: str = "data/omi_training"):
    """Save to JSONL files with train/val split."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    split_idx = int(len(examples) * 0.95)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    train_path = output_path / "omi_synthetic_train.jsonl"
    val_path = output_path / "omi_synthetic_val.jsonl"

    with open(train_path, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')

    with open(val_path, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\n✓ Saved Omi training data:")
    print(f"  Train: {train_path} ({len(train_examples)} examples)")
    print(f"  Val:   {val_path} ({len(val_examples)} examples)")
    print(f"\nSample examples:")
    for i in range(2):
        ex = train_examples[i]['text']
        user_q = ex.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
        assistant = ex.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]
        print(f"\n{i+1}. User: {user_q}")
        print(f"   Assistant: {assistant}")


def main():
    examples = generate_all_examples(num_batches=100, examples_per_batch=10)
    save_dataset(examples)
    print("\n✓ Data generation complete!")


if __name__ == "__main__":
    main()
