"""Retrieval result caching."""

import hashlib
import json
from typing import List, Optional
from langchain_core.documents import Document


class RetrievalCache:
    """Simple in-memory cache for retrieval results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def _make_key(self, query: str, k: int, rerank: bool, expand_parent: bool) -> str:
        """Generate cache key from query parameters."""
        params = f"{query}|{k}|{rerank}|{expand_parent}"
        return hashlib.md5(params.encode()).hexdigest()
    
    def get(self, query: str, k: int, rerank: bool, expand_parent: bool) -> Optional[List[Document]]:
        """Get cached results."""
        key = self._make_key(query, k, rerank, expand_parent)
        return self.cache.get(key)
    
    def put(self, query: str, k: int, rerank: bool, expand_parent: bool, results: List[Document]):
        """Cache results with LRU eviction."""
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        key = self._make_key(query, k, rerank, expand_parent)
        self.cache[key] = results
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
