from dotenv import load_dotenv
import os

# explicitly load .env from current directory
load_dotenv(dotenv_path=".env")

print("DEBUG:", os.getenv("NVIDIA_API_KEY"))

from src.rag_app.llm_backends import NvidiaBackend

llm = NvidiaBackend()
response = llm.generate("Explain retrieval-augmented generation (RAG) simply.")
print(response)
