"""Citation metadata extraction for retrieval results."""

from typing import Dict, List
from langchain_core.documents import Document


class CitationExtractor:
    """Extract citation metadata from retrieved documents."""
    
    @staticmethod
    def extract_citation(doc: Document) -> Dict:
        """Extract citation info from document metadata."""
        meta = doc.metadata
        return {
            "paper_id": meta.get("paper_id", ""),
            "section": meta.get("section_path", ""),
            "page": meta.get("page_num", ""),
            "chunk_id": meta.get("chunk_id", ""),
            "node_type": meta.get("node_type", ""),
        }
    
    @staticmethod
    def format_citation(citation: Dict) -> str:
        """Format citation as readable string."""
        parts = []
        if citation["paper_id"]:
            parts.append(f"Paper: {citation['paper_id']}")
        if citation["section"]:
            parts.append(f"Section: {citation['section']}")
        if citation["page"]:
            parts.append(f"Page: {citation['page']}")
        return " | ".join(parts) if parts else "Unknown source"
    
    @staticmethod
    def extract_all(docs: List[Document]) -> List[Dict]:
        """Extract citations from multiple documents."""
        return [CitationExtractor.extract_citation(doc) for doc in docs]
