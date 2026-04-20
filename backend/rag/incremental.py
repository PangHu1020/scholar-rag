"""Incremental database update support."""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_milvus import Milvus


class IncrementalUpdater:
    """Handle incremental updates to Milvus collections."""

    def __init__(self, parent_store: Milvus, child_store: Milvus):
        self.parent_store = parent_store
        self.child_store = child_store

    def delete_paper(self, paper_id: str) -> bool:
        """Delete all chunks for a paper."""
        try:
            expr = f'paper_id == "{paper_id.replace(chr(34), "")}"'
            self.parent_store.delete(expr=expr)
            self.child_store.delete(expr=expr)
            return True
        except Exception:
            return False

    def has_content_hash(self, content_hash: str) -> Optional[str]:
        """Check if content_hash exists in Milvus. Returns paper_id if found."""
        try:
            col = self.child_store.col
            if col is None:
                return None
            safe_hash = content_hash.replace('"', "")
            results = col.query(
                expr=f'content_hash == "{safe_hash}"',
                output_fields=["paper_id"],
                limit=1,
            )
            if results:
                return results[0].get("paper_id")
            return None
        except Exception:
            return None

    def update_paper(
        self,
        paper_id: str,
        parents: List[Document],
        children: List[Document],
    ) -> bool:
        """Update paper by deleting old and inserting new."""
        self.delete_paper(paper_id)
        try:
            self.parent_store.add_documents(parents)
            self.child_store.add_documents(children)
            return True
        except Exception:
            return False
