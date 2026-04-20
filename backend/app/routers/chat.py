"""Chat API — SSE streaming with session management."""

import json
import uuid
import logging
from typing import AsyncGenerator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from config import Config
from app.dependencies import get_llm, get_retriever_tool, get_checkpointer
from app.store import create_session, get_session, update_session
from agent.graph import build_graph
from rag.citation import CitationExtractor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


def _build_graph():
    return build_graph(
        llm=get_llm(),
        retriever=get_retriever_tool(),
        citation_extractor=CitationExtractor,
        max_retries=Config.MAX_RETRIES,
        checkpointer=get_checkpointer(),
    )


async def _stream_response(graph, query: str, session_id: str) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": session_id}}
    graph_input = {"query": query}

    yield json.dumps({"type": "session_id", "data": session_id})
    yield json.dumps({"type": "status", "data": "analyzing"})

    try:
        final_citations = []
        final_answer = ""
        answer_buf = ""

        async for chunk in graph.astream(graph_input, config=config, stream_mode=["updates", "messages"]):
            stream_type, data = chunk

            if stream_type == "updates":
                for node_name, node_output in data.items():
                    if node_name == "analyze":
                        sq = node_output.get("sub_queries", [])
                        if sq:
                            yield json.dumps({"type": "sub_queries", "data": sq})
                            yield json.dumps({"type": "status", "data": "searching"})

                    if node_name == "prepare_synthesis":
                        final_citations = node_output.get("citations", [])
                        logger.info(f"prepare_synthesis: {len(final_citations)} citations")

            elif stream_type == "messages":
                msg, metadata = data
                if metadata.get("langgraph_node") == "synthesize" and hasattr(msg, "content") and msg.content:
                    answer_buf += msg.content
                    yield json.dumps({"type": "answer", "data": answer_buf})

        final_answer = answer_buf

        if not final_answer:
            yield json.dumps({"type": "answer", "data": ""})

        await graph.aupdate_state(config, {
            "messages": [
                HumanMessage(content=query),
                AIMessage(content=final_answer, additional_kwargs={"citations": final_citations}),
            ],
            "answer": final_answer,
        })

        yield json.dumps({"type": "citations", "data": final_citations})

        title_hint = query[:50] + ("…" if len(query) > 50 else "")
        session = await get_session(session_id)
        if session and not session.get("title"):
            await update_session(session_id, title=title_hint)

        yield json.dumps({"type": "done", "data": None})

    except Exception as e:
        logger.exception("Chat error")
        yield json.dumps({"type": "error", "data": str(e)})


@router.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    if not await get_session(session_id):
        await create_session(session_id)

    graph = _build_graph()

    return EventSourceResponse(
        _stream_response(graph, req.query, session_id),
        media_type="text/event-stream",
    )
