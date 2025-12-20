import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.helper import load_pdf_files, filter_to_minimal_docs, text_split
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

def main():
    docs = load_pdf_files(DATA_PATH)
    minimal_docs = filter_to_minimal_docs(docs)
    texts_chunk = text_split(minimal_docs)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=texts_chunk,
        embedding=embedding,
        persist_directory=CHROMA_PATH
    )

    print("✅ Chroma vector store created successfully.")

if __name__ == "__main__":
    main()
