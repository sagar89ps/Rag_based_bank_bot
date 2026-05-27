"""
rag_chain.py
Builds the conversational RAG chain with memory for the SBI Bank chatbot.
"""

import logging
from typing import Dict, List, Any

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

from config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_TOKENS,
)
from src.vector_store import get_retriever
from src.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT, SIMPLE_QA_PROMPT

logger = logging.getLogger(__name__)


# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        openai_api_key=OPENAI_API_KEY,
    )


# ── Memory ────────────────────────────────────────────────────────────────────

def get_memory(window_size: int = 5) -> ConversationBufferWindowMemory:
    """Sliding-window memory — keeps last N conversation turns."""
    return ConversationBufferWindowMemory(
        k=window_size,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )


# ── RAG Chain ─────────────────────────────────────────────────────────────────

def build_rag_chain(retriever=None, memory=None) -> ConversationalRetrievalChain:
    """
    Build the full ConversationalRetrievalChain.
      1. Condenses follow-up questions using chat history.
      2. Retrieves relevant document chunks.
      3. Generates grounded answer via LLM.
    """
    if retriever is None:
        retriever = get_retriever()
    if memory is None:
        memory = get_memory()

    llm = get_llm()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    logger.info("RAG chain built successfully.")
    return chain


# ── Chat Session ──────────────────────────────────────────────────────────────

class BankChatSession:
    """
    Stateful chat session — wraps the RAG chain and exposes a clean chat() method.
    Each session has its own memory (per user / per session).
    """

    def __init__(self):
        self.retriever = get_retriever()
        self.memory    = get_memory()
        self.chain     = build_rag_chain(self.retriever, self.memory)
        self.history: List[Dict[str, str]] = []
        logger.info("New BankChatSession created.")

    def chat(self, question: str) -> Dict[str, Any]:
        """
        Send a question and get an answer with sources.

        Returns:
            {
                "answer": str,
                "sources": list[str],
                "chat_history": list[dict]
            }
        """
        if not question.strip():
            return {"answer": "Please type a valid question.", "sources": [], "chat_history": self.history}

        try:
            result = self.chain.invoke({"question": question})
        except Exception as e:
            logger.error(f"Chain invocation error: {e}")
            return {
                "answer": "Sorry, I encountered an error. Please try again or contact SBI at 1800-11-2211.",
                "sources": [],
                "chat_history": self.history,
            }

        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])
        sources = list({doc.metadata.get("source", "Unknown") for doc in source_docs})

        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": sources,
            "chat_history": self.history,
        }

    def reset(self) -> None:
        """Clear conversation memory and history."""
        self.memory.clear()
        self.history = []
        logger.info("Chat session reset.")
