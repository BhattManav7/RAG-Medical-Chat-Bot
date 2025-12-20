from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

def build_rag_chain():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding
    )

    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": 5,
        }
    )
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=128
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a professional and friendly medical assistant.

Rules:
- If the user greets (hi, hello, hey), respond politely.
- If the question is non-medical, explain that you are a medical assistant.
- If the answer IS FOUND in the context, explain it clearly in simple language.
- If the answer is NOT FOUND in the context, say:
  "This information is not available in the provided medical document."
- Do NOT add information that is not present in the context.
- When possible, answer in 3–5 clear sentences.

Context:
{context}

Question:
{question}

Answer:
"""
)


    rag_chain = (
        {"context": retriever, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
