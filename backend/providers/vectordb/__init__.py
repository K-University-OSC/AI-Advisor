"""
VectorDB Provider 모듈

사용법:
    from providers.vectordb import get_vectordb_provider

    provider = get_vectordb_provider()
    await provider.upsert("collection", vectors)
    results = await provider.search("collection", query_vector, top_k=5)
"""

from providers.vectordb.base import BaseVectorDBProvider
from providers.vectordb.registry import get_vectordb_provider

__all__ = ["BaseVectorDBProvider", "get_vectordb_provider"]
