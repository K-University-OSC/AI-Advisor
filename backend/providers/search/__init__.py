"""
Search Provider 모듈

사용법:
    from providers.search import get_search_provider

    provider = get_search_provider()
    results = await provider.search("검색어")
"""

from providers.search.base import BaseSearchProvider
from providers.search.registry import get_search_provider

__all__ = ["BaseSearchProvider", "get_search_provider"]
