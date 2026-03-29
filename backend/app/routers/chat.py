"""Chat API — SSE streaming with session management."""

import json
import uuid
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from config import Config
from app.dependencies import get_llm, get_retriever_tool
from app.store import create_session, get_session, update_session
from agent.graph import build_graph
from agent.checkpointer import create_memory_checkpointer
from rag.citation import CitationExtractor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

_checkpointer = create_memory_checkpointer()


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


def _build_graph():
    return build_graph(
        llm=get_llm(),
        retriever=get_retriever_tool(),
        citation_extractor=CitationExtractor,
        max_retries=Config.MAX_RETRIES,
        checkpointer=_checkpointer,
    )


async def _stream_response(graph, query: str, session_id: str) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": session_id}}

    yield json.dumps({"type": "session_id", "data": session_id})

    yield json.dumps({"type": "status", "data": "analyzing"})

    try:
        result = await graph.ainvoke(
            {
                "query": query,
                "messages": [],
                "summary": "",
                "documents": [],
                "sub_queries": [],
                "sub_answers": [],
                "answer": "",
                "citations": [],
            },
            config=config,
        )

        sub_queries = result.get("sub_queries", [])
        yield json.dumps({"type": "sub_queries", "data": sub_queries})

        answer = result.get("answer", "")
        yield json.dumps({"type": "answer", "data": answer})

        citations = result.get("citations", [])
        yield json.dumps({"type": "citations", "data": citations})

        title_hint = query[:50] + ("…" if len(query) > 50 else "")
        session = get_session(session_id)
        if session and not session.get("title"):
            update_session(session_id, title=title_hint)

        yield json.dumps({"type": "done", "data": None})

    except Exception as e:
        logger.exception("Chat error")
        yield json.dumps({"type": "error", "data": str(e)})


@router.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    if not get_session(session_id):
        create_session(session_id)

    graph = _build_graph()

    return EventSourceResponse(
        _stream_response(graph, req.query, session_id),
        media_type="text/event-stream",
    )
