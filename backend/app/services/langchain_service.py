from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Tuple

from langchain_core.documents import Document
from langchain_classic.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain

from app.utils.xml_parser import XMLParser


class LangChainService:
    """Conversational RAG pipeline powered by Hugging Face-hosted Llama 2."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data"
        self._index_dir = Path(
            os.getenv("VECTORSTORE_DIR", self._data_dir / "faiss_index")
        ).expanduser()
        self._request_timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
        self._chain: ConversationalRetrievalChain | None = None
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    async def get_answer(self, question: str) -> Tuple[str, list[str]]:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        try:
            await asyncio.wait_for(self._ensure_chain(), timeout=self._request_timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("Request timed out while initializing the RAG pipeline.") from exc
        assert self._chain is not None

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self._chain.invoke, {"question": question}),
                timeout=self._request_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError("Request timed out while waiting for the model response.") from exc
        answer = response.get("answer") or response.get("result") or "No answer available."
        sources = [
            doc.metadata.get("title") or doc.metadata.get("url")
            for doc in response.get("source_documents", [])
            if isinstance(doc.metadata, dict)
        ]
        return answer, list(filter(None, sources))

    async def _ensure_chain(self) -> None:
        if self._chain is not None:
            return

        async with self._lock:
            if self._chain is not None:
                return

            self._logger.info("Initializing RAG pipeline")
            embeddings = HuggingFaceEmbeddings(
                model_name=os.getenv(
                    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                )
            )

            vectorstore = await asyncio.to_thread(self._load_or_build_vectorstore, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
            llm = self._build_llm()

            qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    "You are a professional fitness healthcare consultant. "
                    "Answer clearly and concisely using only the context below. "
                    "If the user greets (e.g., hi/hello/hey) or says thanks, "
                    "reply with a brief friendly response and ask how you can help, "
                    "without giving medical guidance. "
                    "If the context does not contain the answer, say you do not know and ask "
                    "for a more specific question. "
                    "Offer safe, general guidance only and suggest seeing a clinician for "
                    "urgent or severe symptoms. "
                    "Do not add unrelated topics. Do not switch languages. Do not include sources. "
                    "Keep the answer to 2-4 sentences.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}\n"
                    "Answer in 3-6 sentences, in the same language as the question."
                ),
            )

            self._chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                output_key="answer",
                combine_docs_chain_kwargs={"prompt": qa_prompt},
            )
            self._logger.info("RAG pipeline ready")

    async def warmup(self) -> None:
        try:
            await self._ensure_chain()
        except Exception:
            self._logger.exception("RAG warmup failed")

    def _load_documents(self) -> list[Document]:
        parser = XMLParser(self._data_dir)
        return parser.load_documents()

    def _load_or_build_vectorstore(self, embeddings: HuggingFaceEmbeddings) -> FAISS:
        if self._index_dir.exists():
            self._logger.info("Loading FAISS index from %s", self._index_dir)
            try:
                return FAISS.load_local(
                    str(self._index_dir),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            except TypeError:
                return FAISS.load_local(str(self._index_dir), embeddings)

        self._logger.info("Building FAISS index (first run)")
        documents = self._load_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self._index_dir))
        return vectorstore

    def _build_llm(self) -> HuggingFacePipeline:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required. Install with: pip install -U transformers"
            ) from exc

        model_id = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "128"))
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        device_map = os.getenv("LOCAL_LLM_DEVICE_MAP", "none")
        torch_dtype = os.getenv("LOCAL_LLM_TORCH_DTYPE", "auto")
        trust_remote_code = os.getenv("LOCAL_LLM_TRUST_REMOTE_CODE", "false").lower() in {"1", "true", "yes"}

        model_kwargs: dict[str, object] = {"dtype": torch_dtype}
        if device_map and device_map.lower() != "none":
            model_kwargs["device_map"] = device_map

        self._logger.info("Loading local model %s", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=trust_remote_code, **model_kwargs)

        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_full_text=False,
        )

        return HuggingFacePipeline(pipeline=text_gen)
