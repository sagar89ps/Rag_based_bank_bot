# 🏦  Bank RAG Q&A Chatbot

An intelligent, conversational chatbot for **Bank** built with **Retrieval-Augmented Generation (RAG)**.  
It answers customer queries about loans, accounts, FDs, credit cards, and more — grounded in official documents.

---

## 🏗️ Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │              User (Browser / API)            │
                 └───────────────────┬─────────────────────────┘
                                     │  Question
                                     ▼
                 ┌─────────────────────────────────────────────┐
                 │         Streamlit UI  /  FastAPI             │
                 └───────────────────┬─────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   Conversational RAG Chain        │
                    │  (LangChain + Memory Window)      │
                    └───┬──────────────────────────┬───┘
                        │ Retrieval                 │ Generation
              ┌─────────▼──────────┐     ┌─────────▼──────────┐
              │   FAISS / Chroma   │     │   OpenAI GPT-3.5/4  │
              │  Vector Store      │     │   LLM               │
              └─────────┬──────────┘     └────────────────────┘
                        │ Embeddings
              ┌─────────▼──────────┐
              │  Bank Documents    │
              │  TXT / PDF / DOCX  │
              └────────────────────┘
```

---

## 📁 Project Structure

```
bank-rag-chatbot/
│
├── app.py                    # Streamlit frontend
├── ingest.py                 # Document ingestion script
├── config.py                 # Centralised config (env vars)
├── requirements.txt
├── .env.example              # Copy → .env and fill API key
│
├── src/
│   ├── document_loader.py    # Load & chunk TXT/PDF/DOCX
│   ├── vector_store.py       # FAISS / Chroma build & load
│   ├── rag_chain.py          # RAG chain + BankChatSession
│   └── prompts.py            # Prompt templates
│
├── api/
│   └── main.py               # FastAPI REST API
│
├── data/
│   └── sample_docs/
│       ├── bank_faq.txt      # SBI FAQ document
│       └── loan_policy.txt   # Loan & account policies
│
├── vectorstore/              # Auto-created by ingest.py
│   └── faiss_index/
│
└── tests/
    └── test_pipeline.py      # Pytest unit tests
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourname/bank-rag-chatbot.git
cd bank-rag-chatbot
pip install -r requirements.txt
```

### 2. Set up API Key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Ingest Documents

```bash
python ingest.py
```
This loads all documents from `data/sample_docs/`, splits them into chunks, creates embeddings, and saves the FAISS index.

### 4a. Run Streamlit App

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501)

### 4b. Run FastAPI

```bash
uvicorn api.main:app --reload --port 8000
```
Open [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

---

## 🔌 API Reference

### `POST /chat`
```json
// Request
{
  "question": "What is the SBI home loan interest rate?",
  "session_id": "user-123"   // optional; auto-generated if omitted
}

// Response
{
  "session_id": "user-123",
  "answer": "SBI home loan interest rates start from 8.50% per annum...",
  "sources": ["loan_policy.txt"],
  "bot_name": "SBI Assist"
}
```

### `POST /reset/{session_id}`
Clears conversation memory for a session.

### `GET /health`
Returns `{ "status": "ok", "bank": "SBI Bank", "bot": "SBI Assist" }`

---

## ➕ Adding Your Own Documents

1. Drop `.txt`, `.pdf`, or `.docx` files into `data/sample_docs/`
2. Re-run `python ingest.py`
3. Restart the app

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-3.5-turbo / GPT-4 |
| RAG Framework | LangChain |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS (default) / ChromaDB |
| Memory | ConversationBufferWindowMemory |
| Document Loaders | PyMuPDF, docx2txt, TextLoader |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Testing | Pytest |
| Config | python-dotenv |

---

## 🔧 Configuration

All settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `LLM_MODEL` | `gpt-3.5-turbo` | LLM model name |
| `LLM_TEMPERATURE` | `0.2` | Lower = more factual |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `VECTORSTORE_TYPE` | `faiss` | `faiss` or `chroma` |
| `CHUNK_SIZE` | `500` | Chars per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K_DOCS` | `4` | Chunks retrieved per query |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📜 License

MIT — free to use and extend for educational / commercial purposes.

---

*Built for SBI Bank customer service automation using Retrieval-Augmented Generation.*
