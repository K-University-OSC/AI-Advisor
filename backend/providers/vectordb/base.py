"""
VectorDB Provider 베이스 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    """벡터 검색 결과"""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


class BaseVectorDBProvider(ABC):
    """
    VectorDB Provider 베이스 클래스

    모든 벡터 DB(Qdrant, Pinecone, Chroma 등)는 이 클래스를 상속받아 구현합니다.
    """

    def __init__(self, **kwargs):
        self.config = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider 이름"""
        pass

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        **kwargs
    ) -> bool:
        """
        컬렉션 생성

        Args:
            collection_name: 컬렉션 이름
            dimension: 벡터 차원
            **kwargs: 추가 설정

        Returns:
            bool: 성공 여부
        """
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """컬렉션 삭제"""
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """컬렉션 존재 여부 확인"""
        pass

    @abstractmethod
    async def upsert(
        self,
        collection_name: str,
        vectors: List[Dict[str, Any]]
    ) -> int:
        """
        벡터 삽입/업데이트

        Args:
            collection_name: 컬렉션 이름
            vectors: 벡터 데이터 리스트
                [{"id": "...", "vector": [...], "payload": {...}}]

        Returns:
            int: 처리된 벡터 수
        """
        pass

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        벡터 유사도 검색

        Args:
            collection_name: 컬렉션 이름
            query_vector: 검색 쿼리 벡터
            top_k: 반환할 결과 수
            filter: 필터 조건
            **kwargs: 추가 파라미터

        Returns:
            List[VectorSearchResult]: 검색 결과 리스트
        """
        pass

    @abstractmethod
    async def delete(
        self,
        collection_name: str,
        ids: List[str]
    ) -> int:
        """
        벡터 삭제

        Args:
            collection_name: 컬렉션 이름
            ids: 삭제할 ID 리스트

        Returns:
            int: 삭제된 벡터 수
        """
        pass

    @abstractmethod
    async def count(self, collection_name: str) -> int:
        """컬렉션의 벡터 수 조회"""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
