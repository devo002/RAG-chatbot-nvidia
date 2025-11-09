import os
from openai import OpenAI

from dotenv import load_dotenv
from pathlib import Path

# Load .env from the project root (rag_api/.env)
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)


class NvidiaBackend:
    """
    Simple wrapper for NVIDIA-hosted models using OpenAI-compatible API.
    """

    def __init__(self, model: str = None):
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY not set in environment")

        # Create NVIDIA client
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
        )

        self.model = model or os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
