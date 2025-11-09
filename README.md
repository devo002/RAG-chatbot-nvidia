# Retrieval-Augmented Generation (RAG) Chatbot with FastAPI, NVIDIA LLMs, and Gradio


Welcome to the Retrieval Augmented Generation (RAG) repository! This project empowers users to perform Question-Answering (QnA) tasks over their own documents using the state-of-the-art RAG technique. By combining open-sourced Large Language Models (LLMs), Langchain and FastAPI, we provide a powerful and user-friendly platform for handling document-based QnA.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/LLM-HuggingFace_Transformers-green?logo=huggingface)](https://huggingface.co/models)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.2-teal?logo=fastapi)](https://fastapi.tiangolo.com/)
[![SwaggerUI](https://img.shields.io/badge/SwaggerUI-Interactive-orange?logo=swagger)](https://swagger.io/)
[![Langchain](https://img.shields.io/badge/Langchain-0.0.312-red?logo=custom)](https://www.example.com/langchain)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-0.2.11-red?logo=custom)](https://github.com/ggerganov/llama.cpp)

![RAG Pipeline](docs/img/jumpstart-fm-rag.jpg)
<p class = 'image-caption' align = 'center'>
<i>RAG Pipeline <a href = "https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html">source</a>.</i>
</p>

## Table of Content
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
    - [Starting the Server](#starting-the-server)
    - [Upload Document](#upload-document)
    - [QnA](#query-document)
- [Advanced Configuration](#advanced-configuration)
    - [Configure LLm Endpoint](#configure-llm-parameters)


## Getting Started
In this section, we'll guide you through setting up and running RAG for your document-based QnA. Follow these steps to get started:

### Prerequisites
Create a vertual python env in your local directory and activate it.
```bash
python3.9 -m venv llm_env/
source activate llm_env/bin/activate
```

### Installation 
1. Clone this repository to your local machine.
```bash
https://github.com/AshishSinha5/rag_api.git
cd rag_api
```
2. Install the required Python packages.
```bash
pip install -r requirements.txt
```


## Usage
We'll be using the [SwaggerUI](https://swagger.io/tools/swagger-ui/) (that comes bundled with the FastAPI library) to interact with our API interface.
### Starting the server 
```bash
cd src/rag_app
uvicorn main:app
```
### Opening SwaggerUI
In your favorite browser, go to the following link - 
```text
http://127.0.0.1:8000/docs
``` 


