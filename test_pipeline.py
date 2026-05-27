"""
tests/test_pipeline.py
Unit tests for the RAG pipeline components (no LLM calls needed for most tests).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain.schema import Document
from src.document_loader import load_documents, split_documents


# ── Document Loader Tests ─────────────────────────────────────────────────────

def test_load_documents():
    docs = load_documents("data/sample_docs")
    assert len(docs) > 0, "Should load at least one document"

def test_document_has_metadata():
    docs = load_documents("data/sample_docs")
    for doc in docs:
        assert "source" in doc.metadata, "Each document should have a 'source' key"

def test_split_documents():
    sample_docs = [
        Document(page_content="This is a test document about SBI Bank loans and deposits. " * 30,
                 metadata={"source": "test.txt"})
    ]
    chunks = split_documents(sample_docs)
    assert len(chunks) > 1, "Long document should be split into multiple chunks"

def test_chunks_within_size_limit():
    from config import CHUNK_SIZE
    sample_docs = [
        Document(page_content="SBI home loan. " * 100, metadata={"source": "test.txt"})
    ]
    chunks = split_documents(sample_docs)
    for chunk in chunks:
        assert len(chunk.page_content) <= CHUNK_SIZE + 50, \
            f"Chunk size {len(chunk.page_content)} exceeds limit"

def test_chunks_preserve_metadata():
    sample_docs = [
        Document(page_content="Test content " * 50, metadata={"source": "loan_policy.txt"})
    ]
    chunks = split_documents(sample_docs)
    for chunk in chunks:
        assert chunk.metadata.get("source") == "loan_policy.txt"


# ── Prompt Template Tests ─────────────────────────────────────────────────────

def test_simple_qa_prompt_format():
    from src.prompts import SIMPLE_QA_PROMPT
    formatted = SIMPLE_QA_PROMPT.format(
        context="SBI home loan rate is 8.50% p.a.",
        question="What is the home loan rate?"
    )
    assert "8.50%" in formatted
    assert "home loan rate" in formatted

def test_condense_prompt_format():
    from src.prompts import CONDENSE_QUESTION_PROMPT
    formatted = CONDENSE_QUESTION_PROMPT.format(
        chat_history="User: What are FD rates?\nAssistant: SBI FD rates start at 3.50%.",
        question="What about for senior citizens?"
    )
    assert "senior citizens" in formatted


# ── Config Tests ──────────────────────────────────────────────────────────────

def test_config_values():
    from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_DOCS, BANK_NAME
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP < CHUNK_SIZE
    assert TOP_K_DOCS > 0
    assert BANK_NAME == "SBI Bank"


# ── FastAPI Tests ─────────────────────────────────────────────────────────────

def test_health_endpoint():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "bank" in data

def test_chat_endpoint_empty_question():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422   # Pydantic min_length validation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
