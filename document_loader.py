"""
document_loader.py
Loads and splits bank documents (TXT, PDF, DOCX) into chunks for embedding.
"""

import os
import logging
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    DirectoryLoader,
)

from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


# ── Loader map by extension ───────────────────────────────────────────────────
LOADER_MAP = {
    ".txt":  TextLoader,
    ".pdf":  PyMuPDFLoader,
    ".docx": Docx2txtLoader,
}


def load_documents(docs_dir: str = DOCS_DIR) -> List[Document]:
    """Load all supported documents from the given directory."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    all_docs: List[Document] = []

    for file_path in docs_path.rglob("*"):
        suffix = file_path.suffix.lower()
        if suffix not in LOADER_MAP:
            continue
        try:
            loader_cls = LOADER_MAP[suffix]
            loader = loader_cls(str(file_path))
            docs = loader.load()
            # Attach source metadata
            for doc in docs:
                doc.metadata["source"] = file_path.name
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} page(s) from {file_path.name}")
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}")

    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "Q:", "---", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def load_and_split(docs_dir: str = DOCS_DIR) -> List[Document]:
    """Convenience wrapper: load + split in one call."""
    raw_docs = load_documents(docs_dir)
    return split_documents(raw_docs)
