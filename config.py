import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL        = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS       = int(os.getenv("MAX_TOKENS", "512"))

# ── Embeddings ────────────────────────────────────────
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Vector Store ──────────────────────────────────────
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/faiss_index")
VECTORSTORE_TYPE = os.getenv("VECTORSTORE_TYPE", "faiss")   # faiss | chroma

# ── Document Ingestion ────────────────────────────────
DOCS_DIR         = os.getenv("DOCS_DIR", "data/sample_docs")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── Retrieval ─────────────────────────────────────────
TOP_K_DOCS       = int(os.getenv("TOP_K_DOCS", "4"))

# ── API ───────────────────────────────────────────────
API_HOST         = os.getenv("API_HOST", "0.0.0.0")
API_PORT         = int(os.getenv("API_PORT", "8000"))

# ── Bank Branding ─────────────────────────────────────
BANK_NAME        = "SBI Bank"
BOT_NAME         = "SBI Assist"
