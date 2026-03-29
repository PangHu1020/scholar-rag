"""Incremental database update support."""

from typing import List
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
            expr = f'paper_id == "{paper_id}"'
            self.parent_store.delete(expr=expr)
            self.child_store.delete(expr=expr)
            return True
        except Exception:
            return False
    
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
