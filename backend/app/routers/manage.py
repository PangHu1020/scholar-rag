"""Collection management + health check APIs."""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter
from pymilvus import connections, utility

from config import Config
from app.store import clear_all_files
from app.dependencies import get_retriever

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

        # Invalidate retriever store caches
        retriever = get_retriever()
        for store in (retriever._child_store, retriever._parent_store):
            store._col_cache = None
            store._cache_key = None

        # Clear all uploaded PDFs
        upload_dir = Path(Config.UPLOAD_DIR)
        if upload_dir.exists():
            for f in upload_dir.iterdir():
                if f.suffix == ".pdf":
                    f.unlink(missing_ok=True)

        # Clear all extracted figures
        figures_dir = Path("data/figures")
        if figures_dir.exists():
            shutil.rmtree(figures_dir)

        # Clear file records from store
        await clear_all_files()

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
