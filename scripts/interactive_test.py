"""
Interactive testing script for Bijou base model.
Uses Unsloth for fast GPU inference.
"""

from unsloth import FastLanguageModel

# Load model
print("Loading model on GPU...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="haidangung/bijou-core-base",  # Your HF repo
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # Enable inference mode
print("✓ Model loaded!\n")

def test_function_calling(user_query, tools=None):
    """Test the model on a function calling prompt."""

    # Default tools if none provided
    if tools is None:
        tools = [
            {
                "name": "send_email",
                "description": "Send an email to a recipient",
                "parameters": {
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"}
                    },
                    "required": ["to", "subject", "body"]
                }
            }
        ]

    # Format prompt
    tools_text = "\n\nAvailable functions:\n"
    for tool in tools:
        name = tool['name']
        desc = tool['description']
        params = tool['parameters']['properties']
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

    prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

    # Generate
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        use_cache=True,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract just the assistant response
    assistant_response = result.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]

    return assistant_response

# Interactive loop
print("=" * 60)
print("Interactive Function Calling Test")
print("=" * 60)
print("Type your query and press Enter. Type 'quit' to exit.\n")

while True:
    try:
        user_input = input(">>> ")

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        # Test the model
        response = test_function_calling(user_input)
        print(f"\n{response}\n")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}\n")
