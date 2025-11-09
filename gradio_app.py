import uuid
import requests
import gradio as gr

# Your FastAPI backend URL
API_URL = "http://127.0.0.1:8000/chat"


def chat_with_backend(message, history, session_id, collection_name):
    """Send the user message + session_id to the FastAPI /chat endpoint."""
    if not collection_name:
        collection_name = "test_collection"

    # If this is a brand new session, generate a random session_id
    if session_id is None:
        session_id = str(uuid.uuid4())

    payload = {
        "session_id": session_id,
        "collection_name": collection_name,
        "message": message,
        "n_results": 4,
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "[No 'answer' field in response]")
    except Exception as e:
        answer = f"Error talking to backend: {e}"

    # Update Gradio chat history: list of [user, assistant]
    history = history + [[message, answer]]
    return history, session_id


def clear_chat():
    """Clear the chat and start a fresh session_id."""
    new_session_id = str(uuid.uuid4())
    return [], new_session_id


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 📚 RAG Chatbot (FastAPI + NVIDIA + Gradio)
        1. Make sure your FastAPI backend is running on <code>http://127.0.0.1:8000</code>.
        2. Upload documents via the <code>/upload</code> endpoint (Swagger UI).
        3. Then chat here using the same <code>collection_name</code>.
        """
    )

    with gr.Row():
        collection_name = gr.Textbox(
            value="test_collection",
            label="Collection name",
            info="Must match the collection name you used when uploading documents.",
        )

    chatbot = gr.Chatbot(label="Chat about your documents")
    msg = gr.Textbox(
        label="Your message",
        placeholder="Ask a question about your uploaded documents...",
    )
    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear chat & new session")

    # Gradio state to hold session_id across turns
    session_state = gr.State(value=str(uuid.uuid4()))

    # When user hits Enter in the textbox
    msg.submit(
        chat_with_backend,
        inputs=[msg, chatbot, session_state, collection_name],
        outputs=[chatbot, session_state],
    )

    # When user clicks Send button
    send_btn.click(
        chat_with_backend,
        inputs=[msg, chatbot, session_state, collection_name],
        outputs=[chatbot, session_state],
    )

    # Clear button: clears history and creates new session_id
    clear_btn.click(
        clear_chat,
        inputs=[],
        outputs=[chatbot, session_state],
    )

demo.launch()
