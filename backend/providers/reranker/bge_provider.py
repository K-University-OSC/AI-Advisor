"""
BGE Reranker Provider (로컬 모델)
"""

import logging
import asyncio
from typing import List

from providers.reranker.base import BaseRerankerProvider, RerankResult

logger = logging.getLogger(__name__)


class BGERerankerProvider(BaseRerankerProvider):
    """
    BGE Reranker Provider

    로컬에서 실행되는 무료 Reranker 모델

    지원 모델:
        - BAAI/bge-reranker-v2-m3 (기본, 다국어)
        - BAAI/bge-reranker-large
    """

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model
        self._model = None

    @property
    def provider_name(self) -> str:
        return "bge"

    def _load_model(self):
        """모델 로드 (Lazy Loading)"""
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(self.model_name, use_fp16=True)
                logger.info(f"BGE Reranker 로드 완료: {self.model_name}")
            except ImportError:
                raise ImportError("FlagEmbedding 패키지가 필요합니다: pip install FlagEmbedding")
        return self._model

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[RerankResult]:
        """BGE 모델로 문서 재정렬"""
        try:
            model = self._load_model()

            # 쿼리-문서 쌍 생성
            pairs = [[query, doc] for doc in documents]

            # 동기 함수를 비동기로 실행
            scores = await asyncio.to_thread(model.compute_score, pairs)

            # 점수와 인덱스 결합
            scored_docs = list(zip(range(len(documents)), scores, documents))

            # 점수 기준 정렬
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # top_k 적용
            if top_k:
                scored_docs = scored_docs[:top_k]

            return [
                RerankResult(index=idx, score=score, document=doc)
                for idx, score, doc in scored_docs
            ]

        except Exception as e:
            logger.error(f"BGE Rerank 오류: {e}")
            raise
