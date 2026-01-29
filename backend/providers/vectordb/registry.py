"""
VectorDB Provider Registry
"""

import logging
from typing import Dict, Type, Optional

from providers.vectordb.base import BaseVectorDBProvider

logger = logging.getLogger(__name__)

_PROVIDER_REGISTRY: Dict[str, Type[BaseVectorDBProvider]] = {}
_PROVIDER_INSTANCES: Dict[str, BaseVectorDBProvider] = {}


def _ensure_providers_registered():
    """Provider 등록 확인"""
    if not _PROVIDER_REGISTRY:
        from providers.vectordb.qdrant_provider import QdrantProvider
        _PROVIDER_REGISTRY["qdrant"] = QdrantProvider


def get_vectordb_provider(
    provider_name: str = None,
    use_cache: bool = True,
    **kwargs
) -> BaseVectorDBProvider:
    """
    설정에 따라 VectorDB Provider 반환

    Args:
        provider_name: Provider 이름
        use_cache: 캐시 사용 여부
        **kwargs: Provider 초기화 파라미터

    Returns:
        BaseVectorDBProvider: VectorDB Provider 인스턴스
    """
    _ensure_providers_registered()

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from config import settings

    provider_name = provider_name or settings.providers.vectordb_provider

    if use_cache and provider_name in _PROVIDER_INSTANCES:
        return _PROVIDER_INSTANCES[provider_name]

    provider_class = _PROVIDER_REGISTRY.get(provider_name)
    if not provider_class:
        raise ValueError(f"Unknown vectordb provider: {provider_name}")

    # Qdrant 설정
    if provider_name == "qdrant":
        kwargs.setdefault("host", settings.providers.qdrant_host)
        kwargs.setdefault("port", settings.providers.qdrant_port)

    provider = provider_class(**kwargs)

    if use_cache:
        _PROVIDER_INSTANCES[provider_name] = provider

    logger.info(f"VectorDB Provider 생성: {provider_name}")
    return provider
