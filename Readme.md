# RAG Chatbot (PDF Question Answering)

This project is a simple **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload a PDF and ask questions about its content.

The system chunks the document, generates embeddings, stores them in a FAISS vector database, retrieves relevant context, and generates answers using a **local LLM**.

---

## Features
- Upload and process PDF files
- Text chunking for efficient retrieval
- Embeddings using **Ollama (`nomic-embed-text`)**
- Vector storage and retrieval using **FAISS**
- Question answering via a **Gradio UI**
- Runs fully locally (no cloud API required)

---

## Tech Stack
- Python
- LangChain
- FAISS
- Ollama
- Gradio

---
### Notes & Future Improvements

Add evaluation metrics for retrieval quality
Persist vector database to disk
Improve chunking strategy
Add conversation memory
Optimize retrieval latency
Support multiple PDF uploads

## Setup & Installation

### 1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
2. Install dependencies
pip install -r requirements.txt
3. Install and run Ollama models
Make sure Ollama is running, then pull the required models:

ollama pull llama3
ollama pull nomic-embed-text
4. Run the application
python RAG_CHATBOT.py
The Gradio interface will open in your browser.

