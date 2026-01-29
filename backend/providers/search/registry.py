"""
Search Provider Registry
"""

import logging
from typing import Dict, Type, Optional

from providers.search.base import BaseSearchProvider

logger = logging.getLogger(__name__)

_PROVIDER_REGISTRY: Dict[str, Type[BaseSearchProvider]] = {}
_PROVIDER_INSTANCES: Dict[str, BaseSearchProvider] = {}


def _ensure_providers_registered():
    """Provider 등록 확인"""
    if not _PROVIDER_REGISTRY:
        from providers.search.tavily_provider import TavilyProvider
        _PROVIDER_REGISTRY["tavily"] = TavilyProvider


def get_search_provider(
    provider_name: str = None,
    use_cache: bool = True
) -> BaseSearchProvider:
    """
    설정에 따라 Search Provider 반환

    Args:
        provider_name: Provider 이름
        use_cache: 캐시 사용 여부

    Returns:
        BaseSearchProvider: Search Provider 인스턴스
    """
    _ensure_providers_registered()

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from config import settings

    provider_name = provider_name or settings.providers.search_provider

    if use_cache and provider_name in _PROVIDER_INSTANCES:
        return _PROVIDER_INSTANCES[provider_name]

    api_key = settings.api_keys.tavily  # 현재 Tavily만 지원
    if not api_key:
        raise ValueError(f"API key not found for search provider: {provider_name}")

    provider_class = _PROVIDER_REGISTRY.get(provider_name)
    if not provider_class:
        raise ValueError(f"Unknown search provider: {provider_name}")

    provider = provider_class(api_key=api_key)

    if use_cache:
        _PROVIDER_INSTANCES[provider_name] = provider

    logger.info(f"Search Provider 생성: {provider_name}")
    return provider
