"""
ingest.py
One-time script to load, chunk, embed, and persist all bank documents.
Run this before starting the API or Streamlit app.

Usage:
    python ingest.py
    python ingest.py --docs_dir data/my_docs --store_type faiss
"""

import argparse
import logging
import sys
import os

# ── Make sure project root is on path ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.document_loader import load_and_split
from src.vector_store import build_vector_store, save_faiss_store
from config import DOCS_DIR, VECTORSTORE_PATH, VECTORSTORE_TYPE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ingest(docs_dir: str = DOCS_DIR):
    logger.info("=" * 60)
    logger.info("SBI Bank RAG Chatbot — Document Ingestion")
    logger.info("=" * 60)

    # 1. Load & chunk documents
    logger.info(f"Loading documents from: {docs_dir}")
    chunks = load_and_split(docs_dir)
    if not chunks:
        logger.error("No documents found. Please add .txt / .pdf / .docx files to the docs directory.")
        sys.exit(1)
    logger.info(f"Total chunks ready for embedding: {len(chunks)}")

    # 2. Build vector store
    logger.info(f"Building {VECTORSTORE_TYPE.upper()} vector store...")
    store = build_vector_store(chunks)

    # 3. Persist (FAISS needs explicit save; Chroma auto-persists)
    if VECTORSTORE_TYPE == "faiss":
        save_faiss_store(store, VECTORSTORE_PATH)

    logger.info("=" * 60)
    logger.info(f"✅ Ingestion complete! Index saved to: {VECTORSTORE_PATH}")
    logger.info("You can now run: streamlit run app.py  OR  uvicorn api.main:app")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest bank documents into vector store.")
    parser.add_argument("--docs_dir", default=DOCS_DIR, help="Directory containing documents")
    args = parser.parse_args()
    ingest(args.docs_dir)
