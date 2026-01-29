"""
OpenAI Embedding Provider
"""

import logging
from typing import List

from openai import AsyncOpenAI

from providers.embedding.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI Embedding Provider

    지원 모델:
        - text-embedding-3-large (3072 차원)
        - text-embedding-3-small (1536 차원)
        - text-embedding-ada-002 (1536 차원)
    """

    MODEL_DIMENSIONS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: str, model: str = "text-embedding-3-large", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def dimension(self) -> int:
        return self.MODEL_DIMENSIONS.get(self.model, 3072)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트를 임베딩 벡터로 변환"""
        try:
            # 빈 텍스트 필터링
            texts = [t if t.strip() else " " for t in texts]

            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )

            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error(f"OpenAI Embedding 오류: {e}")
            raise
