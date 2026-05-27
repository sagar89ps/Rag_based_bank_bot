"""
api/main.py
FastAPI backend for SBI Bank RAG Chatbot.

Endpoints:
  POST /chat          — Send a message, get a response
  POST /reset/{sid}   — Reset a session's memory
  GET  /health        — Health check
  GET  /sessions      — List active sessions
"""

import logging
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.rag_chain import BankChatSession
from config import BANK_NAME, BOT_NAME

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=f"{BANK_NAME} RAG Chatbot API",
    description=f"AI-powered Q&A chatbot for {BANK_NAME} using Retrieval-Augmented Generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────────────────
sessions: Dict[str, BankChatSession] = {}

MAX_SESSIONS = 100   # cap to avoid OOM in demo


def get_or_create_session(session_id: Optional[str]) -> tuple[str, BankChatSession]:
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    # Create new session
    new_id = session_id or str(uuid.uuid4())
    if len(sessions) >= MAX_SESSIONS:
        # Evict oldest session
        oldest = next(iter(sessions))
        del sessions[oldest]
        logger.warning(f"Evicted session {oldest} (max sessions reached).")
    sessions[new_id] = BankChatSession()
    logger.info(f"Created session: {new_id}")
    return new_id, sessions[new_id]


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, example="What is the SBI home loan interest rate?")
    session_id: Optional[str] = Field(None, example="abc-123")

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str]
    bot_name: str = BOT_NAME

class ResetResponse(BaseModel):
    session_id: str
    message: str

class HealthResponse(BaseModel):
    status: str
    bank: str
    bot: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(status="ok", bank=BANK_NAME, bot=BOT_NAME)


@app.get("/sessions", tags=["System"])
def list_sessions():
    return {"active_sessions": len(sessions), "session_ids": list(sessions.keys())}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Send a question to the SBI bank chatbot and receive an AI-generated answer
    grounded in official SBI documents.
    """
    try:
        session_id, session = get_or_create_session(request.session_id)
        result = session.chat(request.question)
        return ChatResponse(
            session_id=session_id,
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset/{session_id}", response_model=ResetResponse, tags=["Chat"])
def reset_session(session_id: str):
    """Reset the conversation memory for a given session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    sessions[session_id].reset()
    return ResetResponse(session_id=session_id, message="Session memory cleared.")


@app.delete("/session/{session_id}", tags=["Chat"])
def delete_session(session_id: str):
    """Delete a session entirely."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del sessions[session_id]
    return {"message": f"Session {session_id} deleted."}
