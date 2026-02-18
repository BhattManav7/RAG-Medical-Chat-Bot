# RAG-Medical-Chat-Bot

A local Retrieval-Augmented Generation (RAG) medical assistant built on MedlinePlus XML data. The backend parses and embeds MedlinePlus topics, stores them in a FAISS index, and serves answers through a FastAPI endpoint. The frontend is a simple React chat UI.

## What this project does

- Parses MedlinePlus XML into LangChain `Document` objects.
- Chunks, embeds, and indexes the corpus using FAISS (cached on disk).
- Uses a conversational RAG chain with memory for multi-turn chats.
- Runs a local Transformers model (default Qwen 1.5B) for answers.
- Returns answers plus the top sources that informed the response.

## Quick start

### Backend

1. Create and activate a Python virtual environment.
2. Install dependencies:
	- `pip install -r backend/requirements.txt`
3. Run the API from the backend folder:
	- `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`

The API should be available at `http://localhost:8000`.

### Frontend

1. From `frontend`, install dependencies:
	- `npm install`
2. Start the dev server:
	- `npm run dev`

If your backend runs on a different URL, set `VITE_API_BASE_URL` in `frontend/.env`.

## How it works

1. **Ingestion**: MedlinePlus XML is parsed and cleaned into page content.
2. **Indexing**: Documents are chunked, embedded, and stored in FAISS.
3. **Retrieval**: A retriever pulls top-k relevant chunks per question.
4. **Generation**: A local LLM answers using retrieved context.
5. **Memory**: ConversationBufferMemory stores prior turns.

Key files:
- Backend entrypoint: [backend/app/main.py](backend/app/main.py)
- RAG service: [backend/app/services/langchain_service.py](backend/app/services/langchain_service.py)
- XML parser: [backend/app/utils/xml_parser.py](backend/app/utils/xml_parser.py)
- Chat route: [backend/app/routes/chatbot.py](backend/app/routes/chatbot.py)

## Prompting

The chain uses a custom prompt template to enforce a professional fitness healthcare tone, concise answers, and context-only responses. You can tune the tone, length, and safety rules in [backend/app/services/langchain_service.py](backend/app/services/langchain_service.py).

## Memory

Conversation memory is enabled via `ConversationBufferMemory`. The chain stores only the `answer` output to avoid multiple-output-key errors.

## Configuration

Environment variables (all optional):

- `LOCAL_LLM_MODEL` (default `Qwen/Qwen2.5-1.5B-Instruct`)
- `LLM_MAX_NEW_TOKENS` (default `128`)
- `LLM_TEMPERATURE` (default `0.2`)
- `LOCAL_LLM_DEVICE_MAP` (default `none`)
- `LOCAL_LLM_TORCH_DTYPE` (default `auto`)
- `VECTORSTORE_DIR` (default `backend/app/data/faiss_index`)
- `REQUEST_TIMEOUT_SECONDS` (default `120`)

## Troubleshooting

- **Slow responses**: Local CPU inference can take 10-20s. Use a smaller model, lower `LLM_MAX_NEW_TOKENS`, or run on GPU.
- **First run is slow**: The FAISS index and model weights are built/downloaded once and then cached.
- **400 errors about multiple output keys**: Ensure `output_key="answer"` is set in both chain and memory.
- **Prompt echoing**: The pipeline uses `return_full_text=False` to avoid echoing the prompt.

## Future improvements (concise)

- Add a model switcher (local vs OpenAI API) with a config toggle.
- Persist chat history per user/session instead of in-memory only.
- Add citations inline in answers with score thresholds.
- Use a smaller, distilled embedding model for faster indexing.
- Add evaluation tests for answer quality and safety.
