import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.langchain_service import LangChainService


router = APIRouter()
service = LangChainService()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
	question: str


class ChatResponse(BaseModel):
	answer: str
	sources: list[str] = []


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
	try:
		logger.info("Chat request received (length=%s)", len(request.question or ""))
		answer, sources = await service.get_answer(request.question)
		return ChatResponse(answer=answer, sources=sources)
	except TimeoutError as exc:
		raise HTTPException(status_code=504, detail=str(exc)) from exc
	except ValueError as exc:
		logger.warning("Chat request rejected: %s", exc)
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		logger.exception("Chat request failed")
		raise HTTPException(status_code=500, detail="Unexpected error") from exc
