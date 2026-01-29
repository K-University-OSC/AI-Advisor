"""
Cohere Reranker Provider (API 기반)
"""

import logging
import asyncio
from typing import List

from providers.reranker.base import BaseRerankerProvider, RerankResult

logger = logging.getLogger(__name__)


class CohereRerankerProvider(BaseRerankerProvider):
    """
    Cohere Reranker Provider

    API 기반 유료 Reranker
    """

    def __init__(self, api_key: str, model: str = "rerank-multilingual-v3.0", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def provider_name(self) -> str:
        return "cohere"

    def _get_client(self):
        """클라이언트 초기화 (Lazy Loading)"""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(self.api_key)
            except ImportError:
                raise ImportError("cohere 패키지가 필요합니다: pip install cohere")
        return self._client

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[RerankResult]:
        """Cohere API로 문서 재정렬"""
        try:
            client = self._get_client()

            # 동기 API를 비동기로 실행
            response = await asyncio.to_thread(
                client.rerank,
                query=query,
                documents=documents,
                model=self.model,
                top_n=top_k or len(documents)
            )

            return [
                RerankResult(
                    index=r.index,
                    score=r.relevance_score,
                    document=documents[r.index]
                )
                for r in response.results
            ]

        except Exception as e:
            logger.error(f"Cohere Rerank 오류: {e}")
            raise
