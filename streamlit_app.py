import streamlit as st
import os

from src.rag_pipeline import build_rag_chain

# ---------------- LOGIN ----------------
VALID_USERNAME = os.getenv("APP_USERNAME", "admin")
VALID_PASSWORD = os.getenv("APP_PASSWORD", "admin123")

def login():
    st.title("🔐 Medical RAG Chatbot Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- MAIN APP ----------------
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical RAG Chatbot")
st.caption("Answers are generated strictly from the provided medical document.")

@st.cache_resource
def load_chain():
    return build_rag_chain()

rag_chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a medical question…")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            q = user_input.lower().strip()
            if q in ["hi", "hello", "hey", "good morning", "good evening"]:
                answer = (
                    "Hello! I am a medical assistant trained on a specific medical document. "
                    "How can I help you today?"
                )
            elif q in ["who are you", "what are you"]:
                answer = (
                    "I am a retrieval-augmented medical chatbot. "
                    "I answer questions using only the provided medical document."
                )
            else:
                answer = rag_chain.invoke(user_input)

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
