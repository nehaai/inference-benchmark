"""One-off script: list free-tier models your OpenRouter key actually
has access to, and confirm with a real test call -- given today's
pattern of docs/blogs being wrong or account-specific, verify before
wiring anything into benchmark.py."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

print("=== Free models (id ends in ':free') ===")
models = client.models.list()
free_models = [m.id for m in models.data if m.id.endswith(":free")]
for m in free_models:
    print(m)

if free_models:
    test_model = free_models[0]
    print(f"\n=== Testing a real call against {test_model} ===")
    response = client.chat.completions.create(
        model=test_model,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    print("Response:", response.choices[0].message.content)