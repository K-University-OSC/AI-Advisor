"""
Provider 패턴 구현

외부 서비스(LLM, DB, 검색엔진 등)를 추상화하여 교체 가능하게 합니다.

사용법:
    from providers import get_llm_provider, get_embedding_provider, get_vectordb_provider

    # 설정에 따라 자동으로 적절한 Provider 반환
    llm = get_llm_provider()
    response = await llm.chat([{"role": "user", "content": "안녕"}])
"""

from providers.llm import get_llm_provider, BaseLLMProvider
from providers.embedding import get_embedding_provider, BaseEmbeddingProvider
from providers.vectordb import get_vectordb_provider, BaseVectorDBProvider
from providers.reranker import get_reranker_provider, BaseRerankerProvider

__all__ = [
    # LLM
    "get_llm_provider",
    "BaseLLMProvider",
    # Embedding
    "get_embedding_provider",
    "BaseEmbeddingProvider",
    # VectorDB
    "get_vectordb_provider",
    "BaseVectorDBProvider",
    # Reranker
    "get_reranker_provider",
    "BaseRerankerProvider",
]
