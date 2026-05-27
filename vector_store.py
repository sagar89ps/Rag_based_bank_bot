"""
vector_store.py
Builds, saves, and loads the FAISS / Chroma vector store.
"""

import logging
import os
from typing import List

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
    VECTORSTORE_TYPE,
    TOP_K_DOCS,
)

logger = logging.getLogger(__name__)


def get_embeddings() -> OpenAIEmbeddings:
    """Return the configured embedding model."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )


# ── FAISS helpers ─────────────────────────────────────────────────────────────

def build_faiss_store(chunks: List[Document]) -> FAISS:
    """Create a new FAISS index from document chunks."""
    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    logger.info(f"FAISS index built with {len(chunks)} chunks.")
    return store


def save_faiss_store(store: FAISS, path: str = VECTORSTORE_PATH) -> None:
    os.makedirs(path, exist_ok=True)
    store.save_local(path)
    logger.info(f"FAISS index saved to: {path}")


def load_faiss_store(path: str = VECTORSTORE_PATH) -> FAISS:
    embeddings = get_embeddings()
    store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    logger.info(f"FAISS index loaded from: {path}")
    return store


# ── Chroma helpers ────────────────────────────────────────────────────────────

def build_chroma_store(chunks: List[Document], persist_dir: str = VECTORSTORE_PATH) -> Chroma:
    embeddings = get_embeddings()
    store = Chroma.from_documents(
        chunks, embeddings, persist_directory=persist_dir
    )
    store.persist()
    logger.info(f"Chroma store built and persisted at: {persist_dir}")
    return store


def load_chroma_store(persist_dir: str = VECTORSTORE_PATH) -> Chroma:
    embeddings = get_embeddings()
    store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    logger.info(f"Chroma store loaded from: {persist_dir}")
    return store


# ── Unified interface ─────────────────────────────────────────────────────────

def build_vector_store(chunks: List[Document]):
    """Build and persist the configured vector store type."""
    if VECTORSTORE_TYPE == "chroma":
        return build_chroma_store(chunks)
    return build_faiss_store(chunks)   # default: FAISS


def load_vector_store():
    """Load the persisted vector store."""
    if VECTORSTORE_TYPE == "chroma":
        return load_chroma_store()
    return load_faiss_store()


def get_retriever(store=None, k: int = TOP_K_DOCS):
    """Return a retriever from the vector store."""
    if store is None:
        store = load_vector_store()
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
