"""Session & file metadata storage using PostgreSQL."""

import time
from typing import Optional

from psycopg_pool import AsyncConnectionPool

_pool: Optional[AsyncConnectionPool] = None

_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
)"""

_FILES_DDL = """
CREATE TABLE IF NOT EXISTS files (
    file_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    size_bytes BIGINT NOT NULL DEFAULT 0,
    page_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at DOUBLE PRECISION NOT NULL
)"""


async def init_store(pool: AsyncConnectionPool):
    global _pool
    _pool = pool
    async with pool.connection() as conn:
        await conn.execute(_SESSIONS_DDL)
        await conn.execute(_FILES_DDL)
        await conn.commit()


def _get_pool() -> AsyncConnectionPool:
    assert _pool is not None, "store not initialised — call init_store first"
    return _pool


# ── sessions ─────────────────────────────────────────────

async def create_session(session_id: str, title: str = "") -> dict:
    now = time.time()
    async with _get_pool().connection() as conn:
        await conn.execute(
            "INSERT INTO sessions (session_id, title, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (session_id) DO NOTHING",
            (session_id, title, now, now),
        )
        await conn.commit()
    return {"session_id": session_id, "title": title, "created_at": now, "updated_at": now}


async def update_session(session_id: str, title: Optional[str] = None) -> bool:
    parts, vals = ["updated_at = %s"], [time.time()]
    if title is not None:
        parts.append("title = %s")
        vals.append(title)
    vals.append(session_id)
    async with _get_pool().connection() as conn:
        cur = await conn.execute(
            f"UPDATE sessions SET {', '.join(parts)} WHERE session_id = %s", vals
        )
        await conn.commit()
        return cur.rowcount > 0


async def list_sessions() -> list[dict]:
    async with _get_pool().connection() as conn:
        cur = await conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in await cur.fetchall()]


async def get_session(session_id: str) -> Optional[dict]:
    async with _get_pool().connection() as conn:
        cur = await conn.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
        row = await cur.fetchone()
        if not row:
            return None
        cols = [d.name for d in cur.description]
        return dict(zip(cols, row))


async def delete_session(session_id: str) -> bool:
    async with _get_pool().connection() as conn:
        cur = await conn.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        await conn.commit()
        return cur.rowcount > 0


# ── files ────────────────────────────────────────────────

async def add_file(
    file_id: str, filename: str, paper_id: str,
    size_bytes: int = 0, page_count: int = 0, chunk_count: int = 0,
) -> dict:
    now = time.time()
    async with _get_pool().connection() as conn:
        await conn.execute(
            "INSERT INTO files (file_id, filename, paper_id, size_bytes, page_count, chunk_count, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (file_id) DO UPDATE SET "
            "filename=EXCLUDED.filename, paper_id=EXCLUDED.paper_id, "
            "size_bytes=EXCLUDED.size_bytes, page_count=EXCLUDED.page_count, "
            "chunk_count=EXCLUDED.chunk_count",
            (file_id, filename, paper_id, size_bytes, page_count, chunk_count, now),
        )
        await conn.commit()
    return {
        "file_id": file_id, "filename": filename, "paper_id": paper_id,
        "size_bytes": size_bytes, "page_count": page_count,
        "chunk_count": chunk_count, "created_at": now,
    }


async def list_files() -> list[dict]:
    async with _get_pool().connection() as conn:
        cur = await conn.execute("SELECT * FROM files ORDER BY created_at DESC")
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in await cur.fetchall()]


async def get_file(file_id: str) -> Optional[dict]:
    async with _get_pool().connection() as conn:
        cur = await conn.execute("SELECT * FROM files WHERE file_id = %s", (file_id,))
        row = await cur.fetchone()
        if not row:
            return None
        cols = [d.name for d in cur.description]
        return dict(zip(cols, row))


async def delete_file_record(file_id: str) -> Optional[dict]:
    async with _get_pool().connection() as conn:
        cur = await conn.execute("SELECT * FROM files WHERE file_id = %s", (file_id,))
        row = await cur.fetchone()
        if not row:
            return None
        cols = [d.name for d in cur.description]
        record = dict(zip(cols, row))
        await conn.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
        await conn.commit()
        return record


async def clear_all_files() -> int:
    async with _get_pool().connection() as conn:
        cur = await conn.execute("DELETE FROM files")
        await conn.commit()
        return cur.rowcount
