#!/usr/bin/env python3
"""
Interactive CLI for testing vLLM server with LoRA hot-swapping.

Usage:
    python scripts/demo/vllm_chat.py --schema OEMs/omi/tools.json

Commands:
    /model <name>  - Switch model (e.g., /model omi, /model Qwen/Qwen2.5-1.5B)
    /help          - Show help
    /quit          - Exit
"""

import requests
import json
import sys
from pathlib import Path
from typing import Optional


def load_tools_schema(schema_path: str) -> str:
    """Load and format tools schema for system prompt."""
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    tools = schema.get("tools", [])
    tool_descriptions = []

    for tool in tools:
        name = tool["name"]
        description = tool["description"]
        params = tool["parameters"]["properties"]
        required = tool["parameters"].get("required", [])

        # Format parameters
        param_strs = []
        for param_name, param_info in params.items():
            param_type = param_info.get("type", "string")
            is_required = param_name in required
            param_strs.append(f"{param_name}{'?' if not is_required else ''}: {param_type}")

        params_str = ", ".join(param_strs)
        tool_descriptions.append(f"- {name}({params_str})\n  {description}")

    return "\n".join(tool_descriptions)


class VLLMChatClient:
    def __init__(self, base_url: str = "http://localhost:8000", schema_path: Optional[str] = None):
        self.base_url = base_url
        self.current_model = "Qwen/Qwen2.5-1.5B"
        self.api_url = f"{base_url}/v1/chat/completions"

        # Build system prompt
        if schema_path and Path(schema_path).exists():
            tools_desc = load_tools_schema(schema_path)
            self.system_prompt = f"""You are a helpful assistant with access to functions. Use them when appropriate.

Available functions:
{tools_desc}

Output format: {{"function": "function_name", "arguments": {{...}}}}"""
        else:
            self.system_prompt = "You are a helpful assistant."

    def send_message(self, content: str, model: Optional[str] = None) -> dict:
        """Send a message to the vLLM server."""
        model = model or self.current_model

        # Define JSON schema for function calling
        json_schema = {
            "type": "object",
            "properties": {
                "function": {"type": "string"},
                "arguments": {"type": "object"}
            },
            "required": ["function", "arguments"]
        }

        payload = {
            "model": "models/bijou-base-merged",  # Always use base model
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ],
            "temperature": 0.1,
            "max_tokens": 256,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "guided_json": json_schema  # Force JSON output
        }

        # If using LoRA adapter, add lora_request
        if model == "omi":
            payload["extra_body"] = {
                "lora_request": {
                    "lora_name": "omi",
                    "lora_int_id": 1
                }
            }

        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def format_response(self, response: dict) -> str:
        """Format the API response for display."""
        if "error" in response:
            return f"\n❌ Error: {response['error']}\n"

        if "choices" in response and len(response["choices"]) > 0:
            message = response["choices"][0].get("message", {})
            content = message.get("content", "")

            # Try to parse as JSON for pretty printing
            try:
                parsed = json.loads(content)
                return f"\n✓ Response:\n{json.dumps(parsed, indent=2)}\n"
            except json.JSONDecodeError:
                return f"\n✓ Response:\n{content}\n"

        return f"\n❓ Unexpected response:\n{json.dumps(response, indent=2)}\n"

    def run(self):
        """Run the interactive chat loop."""
        print("="*80)
        print("vLLM Interactive Chat Client")
        print("="*80)
        print(f"Connected to: {self.base_url}")
        print(f"Current model: {self.current_model}")
        print("\nCommands:")
        print("  /model <name>  - Switch model (e.g., /model omi)")
        print("  /help          - Show this help")
        print("  /quit          - Exit")
        print("\nType your message and press Enter to send.\n")
        print("="*80 + "\n")

        while True:
            try:
                user_input = input(f"[{self.current_model}] You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    cmd_parts = user_input.split(maxsplit=1)
                    cmd = cmd_parts[0].lower()

                    if cmd == "/quit" or cmd == "/exit":
                        print("\nGoodbye!")
                        break

                    elif cmd == "/help":
                        print("\nCommands:")
                        print("  /model <name>  - Switch model")
                        print("  /help          - Show this help")
                        print("  /quit          - Exit")
                        print()
                        continue

                    elif cmd == "/model":
                        if len(cmd_parts) > 1:
                            new_model = cmd_parts[1]
                            self.current_model = new_model
                            print(f"\n✓ Switched to model: {new_model}\n")
                        else:
                            print("\n❌ Usage: /model <name>\n")
                        continue

                    else:
                        print(f"\n❌ Unknown command: {cmd}\n")
                        continue

                # Send message
                print(f"\n Sending to {self.current_model}...")
                response = self.send_message(user_input)
                print(self.format_response(response))

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interactive vLLM chat client")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/bijou-base-merged",
        help="Initial model to use (default: models/bijou-base-merged)"
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Path to tools schema JSON file (e.g., OEMs/omi/tools.json)"
    )

    args = parser.parse_args()

    client = VLLMChatClient(base_url=args.url, schema_path=args.schema)
    client.current_model = args.model
    client.run()


if __name__ == "__main__":
    main()
