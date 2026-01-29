"""
Embedding Provider 베이스 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """
    Embedding Provider 베이스 클래스

    모든 임베딩 제공자(OpenAI, Cohere 등)는 이 클래스를 상속받아 구현합니다.
    """

    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.config = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider 이름"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """임베딩 차원"""
        pass

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트를 임베딩 벡터로 변환

        Args:
            texts: 텍스트 리스트

        Returns:
            List[List[float]]: 임베딩 벡터 리스트
        """
        pass

    async def embed_single(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        result = await self.embed([text])
        return result[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, dim={self.dimension})"
