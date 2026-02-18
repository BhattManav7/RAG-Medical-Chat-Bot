import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import chatbot
from app.routes.chatbot import service as chatbot_service


def _load_env_file(file_path: Path) -> None:
	if not file_path.exists():
		return

	for raw_line in file_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip("\"").strip("'")
		if key and key not in os.environ:
			os.environ[key] = value

def _load_env() -> None:
	app_dir = Path(__file__).resolve().parent
	backend_dir = app_dir.parent

	try:
		from dotenv import load_dotenv
		# Prefer explicit .env in backend root, then app/config/hf.env.
		load_dotenv(backend_dir / ".env", override=False)
		load_dotenv(app_dir / "config" / "hf.env", override=False)
		return
	except ImportError:
		pass

	_load_env_file(app_dir / "config" / "hf.env")


_load_env()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


app = FastAPI(title="RAG Medical Chatbot", version="0.1.0")

# Allow local development origins until stricter CORS rules are needed.
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(chatbot.router, prefix="/chat", tags=["chatbot"])


@app.on_event("startup")
async def warmup_rag_pipeline() -> None:
	asyncio.create_task(chatbot_service.warmup())


@app.get("/health")
def health_check():
	return {"status": "ok"}
