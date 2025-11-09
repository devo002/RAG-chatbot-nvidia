import argparse
from typing import Optional, List, Dict
from pathlib import Path
from pydantic import BaseModel
from collections import defaultdict
from pydantic import Field
from .llm_backends import NvidiaBackend
from .chat_memory import save_message, load_history

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, Path as PathParam, Query, File, UploadFile
from pydantic import BaseModel, Field

from .load_data import load_split_pdf_file, load_split_html_file, initialize_splitter
from .load_llm import load_lamma_cpp
from .vector_db import create_vector_db, load_local_db
from .prompts import create_prompt
from .utils import read_file, load_yaml_file
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fake_output(x: float):
    return "Answer to this query is 42"

ml_models = {}
db_name = {}

class ChatTurn(BaseModel):
    user: str
    assistant: str


class ChatRequest(BaseModel):
    session_id: str
    collection_name: str
    message: str
    n_results: int = 4


class ChatResponse(BaseModel):
    answer: str
    history: List[ChatTurn]
    sources: List[str]


# session_id -> list of ChatTurn
SESSION_HISTORY: Dict[str, List[ChatTurn]] = defaultdict(list)




config_path = Path(__file__).resolve().parent / "llama2_config.yaml"
model_args = load_yaml_file(config_path)
#model_args = load_yaml_file("llama2_config.yaml")
text_splitter = initialize_splitter(chunk_size = 1000, chunk_overlap = 100)
vector_db_model_name = "all-MiniLM-L6-v2"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Use NVIDIA LLM instead of local llama.cpp ---
    nvidia_llm = NvidiaBackend()  # uses NVIDIA_API_KEY + NVIDIA_MODEL from .env
    # store the generate() method as our callable
    ml_models["answer_to_query"] = nvidia_llm.generate

    yield

    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(
    title="RAG_APP",
    description="Retrival Augmented Generation APP which let's user upload a file and get the answer for the question using LLMs",
    lifespan=lifespan
)

@app.get("/")
def index():
    return {"message": "Hello World"}



# the model initialized when the app gets loaded but we can configure it if we want
@app.get("/init_llm")
def init_llama_llm(n_gpu_layers: int = Query(500, description="Number of layers to load in GPU"),
                n_batch: int = Query(32, description="Number of tokens to process in parallel. Should be a number between 1 and n_ctx."),
                max_tokens: int = Query(300, description="The maximum number of tokens to generate."),
                n_ctx: int = Query(4096, description="Token context window."),
                temperature: int = Query(0, description="Temperature for sampling. Higher values means more random samples.")):
    model_path = model_args["model_path"]
    model_args = {'model_path' : model_path,
                  'n_gpu_layers': n_gpu_layers,
                  'n_batch': n_batch,
                  'max_tokens': max_tokens,
                  'n_ctx': n_ctx,
                  'temperature': temperature,
                  'device': device}
    llm = load_lamma_cpp(model_args)
    ml_models["answer_to_query"] = llm
    return {"message": "LLM initialized"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...), collection_name : Optional[str] = "test_collection"):
    try:
        contents = file.file.read()
        with open(f'../data/{file.filename}', 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    
    if file.filename.endswith('.pdf'):
        data = load_split_pdf_file(f'../data/{file.filename}', text_splitter)
    elif file.filename.endswith('.html'):
        data = load_split_html_file(f'../data/{file.filename}', text_splitter)
    else:
        return {"message": "Only pdf and html files are supported"}
    
    db = create_vector_db(data, vector_db_model_name, collection_name)


    return {"message": f"Successfully uploaded {file.filename}", 
            "num_splits" : len(data)}


# ---------- RAG helper functions (shared by /query and /chat) ----------

def get_collection_names() -> List[str]:
    """Read collection names from COLLECTIONS.txt (same as /query)."""
    try:
        collection_list = read_file('COLLECTIONS.txt').split("\n")
        return [c for c in collection_list if c]
    except Exception:
        return []


def get_rag_results(query: str, collection_name: str, n_results: int):
    """Run the vector DB query and return the raw results dict."""
    collection_list = get_collection_names()
    if not collection_list:
        raise FileNotFoundError("No collections found. Upload documents first.")

    if collection_name not in collection_list:
        raise ValueError(
            f"There is no collection with name {collection_name}. "
            f"Available: {collection_list}"
        )

    collection = load_local_db(collection_name)
    results = collection.query(query_texts=[query], n_results=n_results)
    return results


def results_to_context(results) -> str:
    """Turn vector DB 'results' into a single context string for the LLM."""
    # Chroma's .query usually returns {"documents": [[...]], "metadatas": [[...]], ...}
    docs_lists = results.get("documents") or []
    if not docs_lists or not docs_lists[0]:
        return "[no relevant context found]"

    docs = docs_lists[0]          # first query's docs
    return "\n\n".join(docs)






@app.get("/query")
def query(query : str, n_results : Optional[int] = 2, collection_name : Optional[str] = "test_collection"):
    try:
        collection_list = read_file('COLLECTIONS.txt')
        collection_list = collection_list.split("\n")[:-1]
    except Exception:
        return {"message": "No collections found uplaod some documents first"}

    if collection_name not in collection_list:
        return {"message": f"There is no collection with name {collection_name}",
                "available_collections" : collection_list}
    collection = load_local_db(collection_name)
    results = collection.query(query_texts=[query], n_results = n_results)
    prompt = create_prompt(query, results)
    output = ml_models["answer_to_query"](prompt)
    return {"message": f"Query is {query}",
            "relavent_docs" : results,
            "llm_output" : output}


# ---------- Conversational RAG endpoint ----------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Conversational RAG endpoint with persistent memory (SQLite).
    """

    # 1. Load previous history from the database
    history_rows = load_history(request.session_id)
    history = [
        ChatTurn(user=row[1], assistant="") if row[0] == "user" else ChatTurn(user="", assistant=row[1])
        for row in history_rows
    ]

    # 2. Retrieve docs using the same RAG pipeline
    try:
        results = get_rag_results(
            query=request.message,
            collection_name=request.collection_name,
            n_results=request.n_results,
        )
    except FileNotFoundError as e:
        # No collections at all yet
        return ChatResponse(answer=str(e), history=history, sources=[])
    except ValueError as e:
        # Bad collection name
        return ChatResponse(answer=str(e), history=history, sources=[])

    context_text = results_to_context(results)

    # 3. Turn history into text
    if history:
        history_text = "\n".join(
            f"User: {turn.user}\nAssistant: {turn.assistant}" for turn in history
        )
    else:
        history_text = "[no previous turns]"

    # 4. Build the prompt for the LLM (your NVIDIA backend)
    prompt = f"""
You are a helpful assistant answering questions about the user's documents.

Conversation so far:
{history_text}

Retrieved context from the documents:
{context_text}

User's new question:
{request.message}

Answer based ONLY on the retrieved context when possible.
If the answer is not in the context, say you don't know.
Keep the answer concise.
"""

    # 5. Call the current LLM (NVIDIA) through the existing ml_models dict
    answer = ml_models["answer_to_query"](prompt)

    # 6. Update and store history
    save_message(request.session_id, "user", request.message)
    save_message(request.session_id, "assistant", answer)

    # 7. Try to collect source file names (if metadatas are present)
    sources: List[str] = []
    try:
        metadatas_lists = results.get("metadatas") or []
        if metadatas_lists:
            for meta in metadatas_lists[0]:
                src = meta.get("source")
                if src:
                    sources.append(src)
    except Exception:
        # If structure is different, just skip sources instead of crashing
        pass

    return ChatResponse(answer=answer, history=history, sources=sources)




if __name__ == "__main__":
    pass

