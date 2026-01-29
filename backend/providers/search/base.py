"""
Search Provider 베이스 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """검색 결과"""
    title: str
    url: str
    content: str
    score: Optional[float] = None


class BaseSearchProvider(ABC):
    """
    Search Provider 베이스 클래스

    모든 검색 제공자(Tavily, Google, Bing 등)는 이 클래스를 상속받아 구현합니다.
    """

    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.config = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider 이름"""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        웹 검색 실행

        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            **kwargs: 추가 파라미터

        Returns:
            List[SearchResult]: 검색 결과 리스트
        """
        pass

    def format_results_for_llm(self, results: List[SearchResult]) -> str:
        """
        검색 결과를 LLM 프롬프트용 텍스트로 포맷

        Args:
            results: 검색 결과 리스트

        Returns:
            str: 포맷된 텍스트
        """
        if not results:
            return "검색 결과가 없습니다."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"[{i}] {r.title}")
            formatted.append(f"    URL: {r.url}")
            formatted.append(f"    내용: {r.content[:300]}...")
            formatted.append("")

        return "\n".join(formatted)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
