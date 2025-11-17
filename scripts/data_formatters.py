"""
Dataset formatters for converting different function-calling datasets to Qwen format.

Each formatter takes a dataset example and converts it to Qwen's chat template format.
"""

import json
from typing import Dict, Any, Optional


def format_tools_for_prompt(tools: list) -> str:
    """Format tools list into a human-readable prompt string."""
    if not tools:
        return ""

    tools_text = "\n\nAvailable functions:\n"
    for tool in tools:
        name = tool.get('name', '')
        desc = tool.get('description', '')
        params = tool.get('parameters', {}).get('properties', {})
        required = tool.get('parameters', {}).get('required', [])
        param_strs = []
        for param_name, param_info in params.items():
            param_type = param_info.get('type', 'string')
            is_req = param_name in required
            marker = "" if is_req else "?"
            param_strs.append(f"{param_name}{marker}: {param_type}")

        params_str = ", ".join(param_strs) if param_strs else ""
        tools_text += f"- {name}({params_str})\n  {desc}\n"

    return tools_text


def xlam_formatter(example: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Format xlam-function-calling-60k dataset to Qwen chat format.

    xlam format:
        {
            "query": "user question",
            "tools": [...],
            "answers": "[{name: ..., arguments: {...}}]"
        }

    Qwen format:
        {
            "text": "<|im_start|>system\\n...\\n<|im_end|>..."
        }
    """
    try:
        # Debug counter
        if not hasattr(xlam_formatter, '_debug_count'):
            xlam_formatter._debug_count = 0

        if not isinstance(example, dict):
            if xlam_formatter._debug_count < 1:
                print("DEBUG: example is not a dict")
                xlam_formatter._debug_count += 1
            return None

        answers = example.get('answers', example.get('answer', []))
        if isinstance(answers, str):
            try:
                answers = json.loads(answers)
            except Exception as parse_err:
                if xlam_formatter._debug_count < 1:
                    print(f"DEBUG: Failed to parse answers JSON: {parse_err}")
                    print(f"  Answers string: {answers[:200]}")
                    xlam_formatter._debug_count += 1
                return None

        if not isinstance(answers, list):
            answers = [answers] if answers else []
        function_call = answers[0] if answers else None

        if not function_call:
            if xlam_formatter._debug_count < 1:
                print(f"DEBUG: No function_call found. Answers: {answers}")
                xlam_formatter._debug_count += 1
            return None
        tools = example.get('tools', [])
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except:
                tools = []

        tools_text = format_tools_for_prompt(tools)
        system_msg = f"You are a helpful assistant with access to functions. Use them when appropriate.{tools_text}\n\nOutput format: {{\"function\": \"function_name\", \"arguments\": {{...}}}}"
        user_msg = example.get('query', example.get('instruction', ''))

        if not user_msg:
            return None

        assistant_msg = json.dumps({
            "function": function_call.get('name', function_call.get('function', '')),
            "arguments": function_call.get('arguments', function_call.get('parameters', {}))
        })

        text = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"

        return {"text": text}

    except Exception as e:
        # Always print first error to debug
        if not hasattr(xlam_formatter, '_error_count'):
            xlam_formatter._error_count = 0

        if xlam_formatter._error_count < 3:  # Print first 3 errors
            print(f"\nDEBUG Error #{xlam_formatter._error_count + 1}: {e}")
            print(f"Example type: {type(example)}")
            if isinstance(example, dict):
                print(f"Keys: {list(example.keys())}")
                print(f"Answers type: {type(example.get('answers'))}")
                print(f"Answers value: {str(example.get('answers'))[:200]}")
            import traceback
            traceback.print_exc()
            xlam_formatter._error_count += 1

        return None


def glaive_formatter(example: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Format glaive-function-calling-v2 dataset to Qwen chat format.

    Glaive format:
        {
            "system": "You are a helpful assistant...",
            "chat": "USER: ...\nASSISTANT: ...\n<|endoftext|>"
        }

    Qwen format:
        {
            "text": "<|im_start|>system\\n...\\n<|im_end|>..."
        }
    """
    try:
        system = example.get('system', 'You are a helpful assistant.')
        chat = example.get('chat', '')
        turns = chat.replace('<|endoftext|>', '').strip().split('\n')
        text = f"<|im_start|>system\n{system}<|im_end|>\n"
        current_role = None
        current_content = []

        for turn in turns:
            if turn.startswith('USER:'):
                if current_role and current_content:
                    content = '\n'.join(current_content)
                    text += f"<|im_start|>{current_role}\n{content}<|im_end|>\n"
                current_role = 'user'
                current_content = [turn[5:].strip()]
            elif turn.startswith('ASSISTANT:'):
                if current_role and current_content:
                    content = '\n'.join(current_content)
                    text += f"<|im_start|>{current_role}\n{content}<|im_end|>\n"
                current_role = 'assistant'
                current_content = [turn[10:].strip()]
            elif turn.strip():
                current_content.append(turn.strip())

        if current_role and current_content:
            content = '\n'.join(current_content)
            text += f"<|im_start|>{current_role}\n{content}<|im_end|>"

        return {"text": text}

    except Exception as e:
        print(f"Error formatting glaive example: {e}")
        return None


def hermes_formatter(example: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Format NousResearch/hermes-function-calling-v1 dataset to Qwen format.

    Hermes uses conversations with tool/function definitions.
    """
    try:
        # Hermes uses 'conversations' key
        conversations = example.get('conversations', [])

        text = ""
        for msg in conversations:
            role = msg.get('from', 'user')
            content = msg.get('value', '')

            # Map roles
            if role == 'human':
                role = 'user'
            elif role == 'gpt':
                role = 'assistant'

            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        return {"text": text.strip()}

    except Exception as e:
        print(f"Error formatting hermes example: {e}")
        return None


# Registry of formatters
# Note: Only use formatters that produce PURE function calling (no conversational responses)
# mixing how the dataset format handles the function calling response will confuse model me thinks
FORMATTERS = {
    'xlam': xlam_formatter,
    # 'glaive': glaive_formatter,  # Disabled: conversational format, not pure function calling
    # 'hermes': hermes_formatter,  # Disabled: conversational format, not pure function calling
}


def get_formatter(dataset_name: str):
    """Get the appropriate formatter for a dataset."""
    for key, formatter in FORMATTERS.items():
        if key in dataset_name.lower():
            return formatter

    raise ValueError(f"No formatter found for dataset: {dataset_name}. Available: {list(FORMATTERS.keys())}")
