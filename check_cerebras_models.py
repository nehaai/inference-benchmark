"""One-off script: list what models your Cerebras API key actually
has access to, rather than guessing from docs/blogs that keep
turning out to be wrong or account-specific."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["CEREBRAS_API_KEY"],
    base_url="https://api.cerebras.ai/v1",
)

models = client.models.list()
for m in models.data:
    print(m.id)