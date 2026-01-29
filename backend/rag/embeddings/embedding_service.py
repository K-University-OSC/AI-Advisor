"""
임베딩 서비스 모듈
Dense 임베딩과 Sparse 임베딩을 모두 지원
"""

import asyncio
from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional
import re
import httpx

from rag.chunkers import ParentChunk, ChildChunk


class EmbeddingService(ABC):
    """임베딩 서비스 추상 클래스"""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """단일 텍스트 임베딩"""
        pass

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """다중 텍스트 임베딩"""
        pass

    async def embed_chunks(
        self,
        parent_chunks: list[ParentChunk],
        child_chunks: list[ChildChunk],
    ) -> dict[str, list[float]]:
        """청크 리스트 임베딩"""
        all_chunks = []
        all_ids = []

        for chunk in parent_chunks:
            all_chunks.append(chunk.content)
            all_ids.append(chunk.chunk_id)

        for chunk in child_chunks:
            all_chunks.append(chunk.content)
            all_ids.append(chunk.chunk_id)

        embeddings = await self.embed_texts(all_chunks)

        return dict(zip(all_ids, embeddings))


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI 임베딩 서비스"""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",  # Baseline V1: 3072차원
        batch_size: int = 100,
    ):
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.api_url = "https://api.openai.com/v1/embeddings"

    async def embed_text(self, text: str) -> list[float]:
        """단일 텍스트 임베딩"""
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """다중 텍스트 임베딩 (배치 처리)"""
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """배치 임베딩"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        cleaned_texts = [self._clean_text(t) for t in texts]

        payload = {
            "model": self.model,
            "input": cleaned_texts,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )

        if response.status_code != 200:
            raise Exception(
                f"OpenAI 임베딩 API 오류: {response.status_code} - {response.text}"
            )

        result = response.json()

        embeddings = [None] * len(texts)
        for item in result["data"]:
            embeddings[item["index"]] = item["embedding"]

        return embeddings

    def _clean_text(self, text: str) -> str:
        """임베딩을 위한 텍스트 정제"""
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        if len(text) > 8000:
            text = text[:8000]
        return text


class SparseEmbeddingService:
    """Sparse 임베딩 서비스 (BM25 스타일)"""

    def __init__(
        self,
        vocab_size: int = 30000,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Args:
            vocab_size: 어휘 크기 (해시 버킷 수)
            k1: BM25 k1 파라미터
            b: BM25 b 파라미터
        """
        self.vocab_size = vocab_size
        self.k1 = k1
        self.b = b
        self._avg_doc_len = 100.0

    def embed_text(self, text: str) -> tuple[list[int], list[float]]:
        """단일 텍스트를 sparse 벡터로 변환"""
        tokens = self._tokenize(text)
        if not tokens:
            return [], []

        token_counts = Counter(tokens)
        doc_len = len(tokens)

        indices = []
        values = []

        # 해시 충돌 시 값을 누적하기 위한 딕셔너리
        index_value_map = {}

        for token, count in token_counts.items():
            idx = self._hash_token(token)

            tf = count / doc_len
            idf = 1.0
            norm = 1.0 - self.b + self.b * (doc_len / self._avg_doc_len)
            score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

            # 동일 인덱스에 대해 값을 누적
            if idx in index_value_map:
                index_value_map[idx] += score
            else:
                index_value_map[idx] = score

        # 정렬된 인덱스와 값 반환
        sorted_items = sorted(index_value_map.items())
        indices = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        return indices, values

    def embed_texts(
        self, texts: list[str]
    ) -> list[tuple[list[int], list[float]]]:
        """다중 텍스트를 sparse 벡터로 변환"""
        return [self.embed_text(text) for text in texts]

    def embed_chunks(
        self,
        parent_chunks: list[ParentChunk],
        child_chunks: list[ChildChunk],
    ) -> dict[str, tuple[list[int], list[float]]]:
        """청크 리스트를 sparse 벡터로 변환"""
        result = {}

        for chunk in parent_chunks:
            result[chunk.chunk_id] = self.embed_text(chunk.content)

        for chunk in child_chunks:
            result[chunk.chunk_id] = self.embed_text(chunk.content)

        return result

    def _tokenize(self, text: str) -> list[str]:
        """텍스트 토큰화 (한국어/영어 지원)"""
        text = text.lower()
        text = re.sub(r"[^\w\s가-힣]", " ", text)

        tokens = text.split()

        tokens = [t for t in tokens if len(t) >= 2]

        return tokens

    def _hash_token(self, token: str) -> int:
        """토큰을 어휘 인덱스로 해싱"""
        return hash(token) % self.vocab_size


class MultimodalEmbeddingService:
    """멀티모달 임베딩 서비스 (Dense + Sparse)"""

    def __init__(
        self,
        dense_service: EmbeddingService,
        sparse_service: Optional[SparseEmbeddingService] = None,
    ):
        self.dense_service = dense_service
        self.sparse_service = sparse_service or SparseEmbeddingService()

    async def embed_chunks(
        self,
        parent_chunks: list[ParentChunk],
        child_chunks: list[ChildChunk],
    ) -> tuple[dict[str, list[float]], dict[str, tuple[list[int], list[float]]]]:
        """
        청크들을 Dense와 Sparse 임베딩으로 변환

        Returns:
            (dense_embeddings, sparse_embeddings)
        """
        dense_embeddings = await self.dense_service.embed_chunks(
            parent_chunks, child_chunks
        )

        sparse_embeddings = self.sparse_service.embed_chunks(
            parent_chunks, child_chunks
        )

        return dense_embeddings, sparse_embeddings

    async def embed_query(
        self, query: str
    ) -> tuple[list[float], tuple[list[int], list[float]]]:
        """
        쿼리를 Dense와 Sparse 임베딩으로 변환

        Returns:
            (dense_embedding, sparse_embedding)
        """
        dense_embedding = await self.dense_service.embed_text(query)
        sparse_embedding = self.sparse_service.embed_text(query)

        return dense_embedding, sparse_embedding
