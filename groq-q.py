
import json
import os
from groq import Groq

# Initialize Groq client
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Set the GROQ_API_KEY environment variable.")

client = Groq(api_key=api_key)

MODEL_NAME = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = """
You are an echo assistant.
Use the echo tool whenever the user asks you to echo text.
For other requests, respond normally.
"""

# Python tool
def echo_fun(content: str):
    return content

# Groq native tool definition
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo_fun",
            "description": "Echo the supplied text exactly as provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Text to echo"
                    }
                },
                "required": ["content"],
                "additionalProperties": False
            }
        }
    }
]

# Map tool names to Python functions
TOOL_FUNCTIONS = {
    "echo_fun": echo_fun
}


def run_agent():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("Groq Echo Agent")
    print("Type 'exit' to quit.")

    while True:
        prompt = input("\nEnter message: ").strip()

        if prompt.lower() == "exit":
            break

        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})

        while True:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message

            # Preserve the assistant message and its tool calls
            messages.append(assistant_message.model_dump(exclude_none=True))

            if not assistant_message.tool_calls:
                print("\nAgent:", assistant_message.content or "")
                break

            # Execute each requested tool
            for call in assistant_message.tool_calls:
                name = call.function.name

                try:
                    args = json.loads(call.function.arguments)
                    tool = TOOL_FUNCTIONS.get(name)

                    if tool is None:
                        raise ValueError(f"Unknown tool: {name}")

                    result = tool(**args)

                except Exception as exc:
                    result = f"Error: {exc}"

                print(f"\n[Tool: {name}]")
                print(result)

                # Return the tool result to the model
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result)
                })


if __name__ == "__main__":
    run_agent()
