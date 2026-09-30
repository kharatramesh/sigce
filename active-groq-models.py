import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("=== YOUR ACTIVE GROQ MODELS ===")
for model in client.models.list().data:
    print(f"- {model.id}")
