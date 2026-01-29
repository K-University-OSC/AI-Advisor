"""
Embedding Provider 모듈

사용법:
    from providers.embedding import get_embedding_provider

    provider = get_embedding_provider()
    vectors = await provider.embed(["텍스트1", "텍스트2"])
"""

from providers.embedding.base import BaseEmbeddingProvider
from providers.embedding.registry import get_embedding_provider

__all__ = ["BaseEmbeddingProvider", "get_embedding_provider"]
