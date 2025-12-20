import streamlit as st
from src.rag_pipeline import build_rag_chain

st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical RAG Chatbot")
st.write("Ask questions based on the medical book.")

@st.cache_resource
def load_chain():
    return build_rag_chain()

rag_chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            q = user_input.lower().strip()
            if q in ["hi", "hello", "hey", "good morning"]:
                answer = (
                    "Hello! I’m here to assist you with medical-related questions. "
                    "How can I help you today?"
                )
            else:
                answer = rag_chain.invoke(user_input)

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
