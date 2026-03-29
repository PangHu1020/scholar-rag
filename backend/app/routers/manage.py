"""Collection management + health check APIs."""

import logging

from fastapi import APIRouter
from pymilvus import connections, utility

from config import Config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["management"])


@router.delete("/collection")
async def clear_collection():
    try:
        alias = "clear_conn"
        connections.connect(alias=alias, uri=Config.MILVUS_URI)
        dropped = []
        for suffix in ("_children", "_parents"):
            name = f"{Config.COLLECTION_NAME}{suffix}"
            if utility.has_collection(name, using=alias):
                utility.drop_collection(name, using=alias)
                dropped.append(name)
        connections.disconnect(alias)
        return {"ok": True, "dropped": dropped}
    except Exception as e:
        logger.exception("Failed to clear collection")
        return {"ok": False, "detail": str(e)}


@router.get("/health")
async def health():
    status = {"milvus": False, "llm": False}

    try:
        alias = "health_conn"
        connections.connect(alias=alias, uri=Config.MILVUS_URI)
        utility.list_collections(using=alias)
        connections.disconnect(alias)
        status["milvus"] = True
    except Exception:
        pass

    try:
        from app.dependencies import get_llm
        llm = get_llm()
        if llm:
            resp = llm.invoke("ping")
            status["llm"] = bool(resp.content)
    except Exception:
        pass

    ok = all(status.values())
    return {"ok": ok, **status}
