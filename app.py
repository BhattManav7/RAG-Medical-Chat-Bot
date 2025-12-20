from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_pipeline import build_rag_chain

app = FastAPI()
rag_chain = build_rag_chain()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    q = query.question.lower().strip()

    if q in ["hi", "hello", "hey", "good morning", "good evening"]:
        return {
            "answer": (
                "Hello! I am a medical RAG-based assistant trained on a medical textbook. "
                "You can ask me questions related to diseases, diagnostics, and medical procedures."
            )
        }

    if q in ["who are you", "what are you", "what is this"]:
        return {
            "answer": (
                "I am a medical chatbot that answers questions using a specific medical book. "
                "My responses are based only on that document."
            )
        }

    return {"answer": rag_chain.invoke(query.question)}
