"""
Reranker Provider Registry
"""

import logging
from typing import Dict, Type, Optional

from providers.reranker.base import BaseRerankerProvider

logger = logging.getLogger(__name__)

_PROVIDER_REGISTRY: Dict[str, Type[BaseRerankerProvider]] = {}
_PROVIDER_INSTANCES: Dict[str, BaseRerankerProvider] = {}


def _ensure_providers_registered():
    """Provider 등록 확인"""
    if not _PROVIDER_REGISTRY:
        from providers.reranker.bge_provider import BGERerankerProvider
        from providers.reranker.cohere_provider import CohereRerankerProvider
        _PROVIDER_REGISTRY["bge"] = BGERerankerProvider
        _PROVIDER_REGISTRY["cohere"] = CohereRerankerProvider


def get_reranker_provider(
    provider_name: str = None,
    use_cache: bool = True
) -> BaseRerankerProvider:
    """
    설정에 따라 Reranker Provider 반환

    Args:
        provider_name: Provider 이름
        use_cache: 캐시 사용 여부

    Returns:
        BaseRerankerProvider: Reranker Provider 인스턴스
    """
    _ensure_providers_registered()

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from config import settings

    provider_name = provider_name or settings.providers.reranker_provider

    if use_cache and provider_name in _PROVIDER_INSTANCES:
        return _PROVIDER_INSTANCES[provider_name]

    provider_class = _PROVIDER_REGISTRY.get(provider_name)
    if not provider_class:
        raise ValueError(f"Unknown reranker provider: {provider_name}")

    # Provider별 초기화
    if provider_name == "bge":
        provider = provider_class(model=settings.providers.bge_reranker_model)
    elif provider_name == "cohere":
        api_key = settings.api_keys.cohere
        if not api_key:
            raise ValueError("Cohere API key not found")
        provider = provider_class(api_key=api_key)
    else:
        provider = provider_class()

    if use_cache:
        _PROVIDER_INSTANCES[provider_name] = provider

    logger.info(f"Reranker Provider 생성: {provider_name}")
    return provider
