"""
Reranker Provider 베이스 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class RerankResult:
    """Rerank 결과"""
    index: int
    score: float
    document: str


class BaseRerankerProvider(ABC):
    """
    Reranker Provider 베이스 클래스

    모든 Reranker(BGE, Cohere 등)는 이 클래스를 상속받아 구현합니다.
    """

    def __init__(self, **kwargs):
        self.config = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider 이름"""
        pass

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[RerankResult]:
        """
        문서 재정렬

        Args:
            query: 검색 쿼리
            documents: 문서 리스트
            top_k: 반환할 상위 문서 수

        Returns:
            List[RerankResult]: 재정렬된 결과
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
