"""
Embedding Provider Registry
"""

import logging
from typing import Dict, Type, Optional

from providers.embedding.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

_PROVIDER_REGISTRY: Dict[str, Type[BaseEmbeddingProvider]] = {}
_PROVIDER_INSTANCES: Dict[str, BaseEmbeddingProvider] = {}


def _ensure_providers_registered():
    """Provider 등록 확인"""
    if not _PROVIDER_REGISTRY:
        from providers.embedding.openai_provider import OpenAIEmbeddingProvider
        _PROVIDER_REGISTRY["openai"] = OpenAIEmbeddingProvider


def get_embedding_provider(
    provider_name: str = None,
    model: str = None,
    use_cache: bool = True
) -> BaseEmbeddingProvider:
    """
    설정에 따라 Embedding Provider 반환

    Args:
        provider_name: Provider 이름
        model: 모델명
        use_cache: 캐시 사용 여부

    Returns:
        BaseEmbeddingProvider: Embedding Provider 인스턴스
    """
    _ensure_providers_registered()

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from config import settings

    provider_name = provider_name or settings.providers.embedding_provider
    model = model or settings.providers.embedding_model

    cache_key = f"{provider_name}:{model}"

    if use_cache and cache_key in _PROVIDER_INSTANCES:
        return _PROVIDER_INSTANCES[cache_key]

    api_key = settings.api_keys.get_key_for_provider(provider_name)
    if not api_key:
        raise ValueError(f"API key not found for provider: {provider_name}")

    provider_class = _PROVIDER_REGISTRY.get(provider_name)
    if not provider_class:
        raise ValueError(f"Unknown embedding provider: {provider_name}")

    provider = provider_class(api_key=api_key, model=model)

    if use_cache:
        _PROVIDER_INSTANCES[cache_key] = provider

    logger.info(f"Embedding Provider 생성: {provider_name} / {model}")
    return provider
