"""
Tavily Search Provider
"""

import logging
from typing import List, Optional

from providers.search.base import BaseSearchProvider, SearchResult

logger = logging.getLogger(__name__)


class TavilyProvider(BaseSearchProvider):
    """
    Tavily Search Provider

    사용 예시:
        provider = TavilyProvider(api_key="tvly-...")
        results = await provider.search("오늘 날씨")
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self._client = None

    @property
    def provider_name(self) -> str:
        return "tavily"

    def _get_client(self):
        """Lazy 로딩으로 클라이언트 초기화"""
        if self._client is None:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                raise ImportError("tavily 패키지가 필요합니다: pip install tavily-python")
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
        **kwargs
    ) -> List[SearchResult]:
        """Tavily 웹 검색"""
        try:
            client = self._get_client()

            # Tavily는 동기 API이므로 asyncio.to_thread 사용 권장
            import asyncio
            response = await asyncio.to_thread(
                client.search,
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=include_answer
            )

            results = []
            for item in response.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score")
                ))

            logger.debug(f"Tavily 검색 완료: {query[:30]}... -> {len(results)}개 결과")
            return results

        except Exception as e:
            logger.error(f"Tavily 검색 오류: {e}")
            raise
