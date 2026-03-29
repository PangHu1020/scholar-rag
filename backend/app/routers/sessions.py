"""Session management APIs."""

import logging

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from app.store import list_sessions, get_session, delete_session
from app.routers.chat import _checkpointer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
async def get_sessions():
    return list_sessions()


@router.get("/{session_id}")
async def get_session_detail(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/{session_id}/history")
async def get_history(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    config = {"configurable": {"thread_id": session_id}}
    try:
        state = _checkpointer.get(config)
    except Exception:
        state = None

    if not state or not state.values:
        return {"session_id": session_id, "messages": []}

    messages = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        else:
            messages.append({"role": "assistant", "content": msg.content})

    return {"session_id": session_id, "messages": messages}


@router.delete("/{session_id}")
async def remove_session(session_id: str):
    ok = delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}
