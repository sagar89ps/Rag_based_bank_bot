"""
app.py  —  Streamlit frontend for SBI Bank RAG Chatbot
Run with:  streamlit run app.py
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SBI Bank Assistant",
    page_icon="🏦",
    layout="centered",
)

# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* SBI Blue theme */
    :root { --sbi-blue: #003087; --sbi-gold: #c8a951; }

    .stApp { background: #f5f7fa; }

    .chat-header {
        background: linear-gradient(135deg, #003087, #0057b8);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .chat-header h2 { margin: 0; font-size: 1.4rem; }
    .chat-header p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }

    .user-msg {
        background: #003087;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.4rem 0 0.4rem 15%;
        word-wrap: break-word;
    }
    .bot-msg {
        background: white;
        color: #1a1a2e;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.4rem 15% 0.4rem 0;
        border: 1px solid #e0e6f0;
        word-wrap: break-word;
    }
    .source-badge {
        font-size: 0.7rem;
        color: #666;
        margin-top: 0.3rem;
        padding-left: 0.5rem;
    }
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #003087;
        padding: 0.6rem 1.2rem;
    }
    .stButton > button {
        background: #003087;
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton > button:hover { background: #0057b8; }
    .suggestion-btn {
        background: #e8f0fe;
        border: 1px solid #003087;
        border-radius: 15px;
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
        color: #003087;
        cursor: pointer;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_chat_session():
    """Cache the RAG session so it isn't rebuilt on every interaction."""
    from src.rag_chain import BankChatSession
    return BankChatSession()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources"  not in st.session_state:
    st.session_state.sources  = []


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
  <span style="font-size:2rem">🏦</span>
  <div>
    <h2>SBI Assist</h2>
    <p>Your 24×7 AI-powered banking assistant</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/c/cc/SBI-logo.svg", width=100)
    st.markdown("### About SBI Assist")
    st.markdown("""
    SBI Assist uses **Retrieval-Augmented Generation (RAG)** to answer your queries
    based on official SBI documents.

    **Topics covered:**
    - 🏠 Home, Personal & Car Loans
    - 💳 Credit Cards
    - 💰 Savings & Fixed Deposits
    - 📱 Net Banking & YONO
    - 🔄 NEFT / RTGS / IMPS
    - 🔒 KYC & Security
    """)
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        try:
            session = load_chat_session()
            session.reset()
        except Exception:
            pass
        st.rerun()
    st.caption("For emergencies: **1800-11-2211** (24×7 Toll Free)")


# ── Suggested questions ───────────────────────────────────────────────────────
SUGGESTIONS = [
    "What is SBI home loan interest rate?",
    "How do I open a savings account?",
    "What are NEFT charges?",
    "What FD rates does SBI offer?",
    "How to register for net banking?",
    "What documents for personal loan?",
]

if not st.session_state.messages:
    st.markdown("**💡 Try asking:**")
    cols = st.columns(3)
    for i, suggestion in enumerate(SUGGESTIONS):
        if cols[i % 3].button(suggestion, key=f"sug_{i}", use_container_width=True):
            st.session_state["pending_question"] = suggestion
            st.rerun()


# ── Chat history display ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            sources_str = " | ".join(msg["sources"])
            st.markdown(f'<div class="source-badge">📄 Sources: {sources_str}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Input box ─────────────────────────────────────────────────────────────────
def handle_question(question: str):
    if not question.strip():
        return
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("SBI Assist is thinking..."):
        try:
            session = load_chat_session()
            result = session.chat(question)
            answer  = result["answer"]
            sources = result["sources"]
        except Exception as e:
            answer  = f"⚠️ Error: {e}. Please check your API key or try again."
            sources = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.rerun()


# Check pending question from suggestion buttons
if "pending_question" in st.session_state and st.session_state["pending_question"]:
    pq = st.session_state.pop("pending_question")
    handle_question(pq)

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    user_input = col1.text_input(
        "Ask a banking question...",
        placeholder="e.g. What is the minimum balance for SBI savings account?",
        label_visibility="collapsed",
    )
    submitted = col2.form_submit_button("Send")

if submitted and user_input:
    handle_question(user_input)

st.caption("⚠️ SBI Assist provides information based on official documents. Always verify critical details with SBI.")
