# -*- coding: utf-8 -*-
"""
HYBRID_V4 Semantic Chunker

의미 기반 텍스트 청킹 - 인접 문장 간 유사도를 계산하여 의미 단위로 분할

특징:
1. 문장 단위 임베딩 (all-MiniLM-L6-v2)
2. 코사인 유사도 기반 분할점 탐지
3. 최소/최대 청크 크기 제한
4. Fallback 청킹 (고정 크기)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ChunkType(str, Enum):
    """청크 타입"""
    TEXT = "text"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    TABLE = "table"
    IMAGE = "image"


@dataclass
class ChunkResult:
    """청크 결과"""
    content: str
    chunk_type: ChunkType
    file_name: str = ""
    page: int = 0
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "file_name": self.file_name,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


class SemanticChunker:
    """
    HYBRID_V4 Semantic Chunker

    의미 기반 청킹 - 문장 간 유사도가 threshold 미만인 지점에서 분할
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Args:
            threshold: 분할 임계값 (유사도가 이 값 미만이면 분할)
            min_chunk_size: 최소 청크 크기 (문자 수)
            max_chunk_size: 최대 청크 크기 (문자 수)
            model_name: Sentence Transformer 모델명
        """
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.model_name = model_name
        self.embeddings = None
        self._initialized = False

    def initialize(self) -> bool:
        """임베딩 모델 초기화"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Semantic Chunker 임베딩 모델 로딩: {self.model_name}")
            self.embeddings = SentenceTransformer(self.model_name)
            self._initialized = True
            logger.info("Semantic Chunker 초기화 완료")
            return True
        except ImportError:
            logger.warning("sentence-transformers 패키지가 설치되지 않았습니다. pip install sentence-transformers")
            return False
        except Exception as e:
            logger.error(f"Semantic Chunker 초기화 실패: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def chunk(self, text: str) -> List[str]:
        """
        의미 기반 청킹 수행

        Args:
            text: 청킹할 텍스트

        Returns:
            청크 리스트
        """
        if not self._initialized:
            logger.warning("Semantic Chunker가 초기화되지 않음, fallback 청킹 사용")
            return self._fallback_chunk(text)

        if not text or not text.strip():
            return []

        # 1. 문장 분리
        sentences = self._split_sentences(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 2:
            return [text] if text.strip() else []

        try:
            # 2. 문장별 임베딩 생성
            embeddings = self.embeddings.encode(sentences)

            # 3. 인접 문장 간 유사도 계산
            similarities = self._compute_similarities(embeddings)

            # 4. 분할점 탐지 (유사도가 threshold 미만인 지점)
            split_points = self._find_split_points(similarities)

            # 5. 청크 생성
            chunks = self._create_chunks(sentences, split_points)

            return chunks

        except Exception as e:
            logger.error(f"Semantic chunking 실패: {e}")
            return self._fallback_chunk(text)

    def _split_sentences(self, text: str) -> List[str]:
        """문장 분리"""
        # 한국어/영어 문장 끝 패턴
        pattern = r'(?<=[.!?。])\s+'
        sentences = re.split(pattern, text)
        return sentences

    def _compute_similarities(self, embeddings: np.ndarray) -> List[float]:
        """인접 문장 간 코사인 유사도 계산"""
        similarities = []
        for i in range(len(embeddings) - 1):
            # 코사인 유사도
            norm_i = np.linalg.norm(embeddings[i])
            norm_j = np.linalg.norm(embeddings[i + 1])
            if norm_i > 0 and norm_j > 0:
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (norm_i * norm_j)
            else:
                sim = 0.0
            similarities.append(sim)
        return similarities

    def _find_split_points(self, similarities: List[float]) -> List[int]:
        """분할점 탐지"""
        split_points = [0]
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                split_points.append(i + 1)
        split_points.append(len(similarities) + 1)
        return split_points

    def _create_chunks(self, sentences: List[str], split_points: List[int]) -> List[str]:
        """분할점 기준으로 청크 생성"""
        chunks = []

        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)

            # 최대 크기 초과 시 분할
            if len(chunk_text) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk_text)
                chunks.extend(sub_chunks)
            # 최소 크기 미만 시 이전 청크에 병합
            elif len(chunk_text) < self.min_chunk_size:
                if chunks:
                    chunks[-1] = chunks[-1] + ' ' + chunk_text
                else:
                    chunks.append(chunk_text)
            else:
                chunks.append(chunk_text)

        return [c.strip() for c in chunks if c.strip()]

    def _split_large_chunk(self, text: str) -> List[str]:
        """큰 청크 분할"""
        chunks = []
        overlap = 50
        for i in range(0, len(text), self.max_chunk_size - overlap):
            chunk = text[i:i + self.max_chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def _fallback_chunk(self, text: str) -> List[str]:
        """Fallback 청킹 (고정 크기)"""
        chunks = []
        chunk_size = 500
        overlap = 50
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def chunk_with_context(
        self,
        paragraphs: List[str],
        file_name: str = "",
        start_page: int = 0
    ) -> List[ChunkResult]:
        """
        문단 리스트를 청킹하고 컨텍스트 포함

        Args:
            paragraphs: 문단 리스트
            file_name: 파일명
            start_page: 시작 페이지

        Returns:
            ChunkResult 리스트
        """
        results = []
        chunk_index = 0

        for i, para in enumerate(paragraphs):
            if not para.strip() or len(para.strip()) < 50:
                continue

            # 500자 초과 문단은 Semantic Chunking
            if len(para) > 500 and self._initialized:
                semantic_chunks = self.chunk(para)
                for sc in semantic_chunks:
                    results.append(ChunkResult(
                        content=sc,
                        chunk_type=ChunkType.SEMANTIC,
                        file_name=file_name,
                        page=start_page,
                        chunk_index=chunk_index,
                        metadata={"original_paragraph_index": i}
                    ))
                    chunk_index += 1
            else:
                # 앞뒤 맥락 포함
                context_parts = []

                # 앞 맥락
                if i > 0 and paragraphs[i - 1].strip():
                    prev = paragraphs[i - 1][-200:] if len(paragraphs[i - 1]) > 200 else paragraphs[i - 1]
                    context_parts.append(f"[앞 맥락] {prev}")

                context_parts.append(para)

                # 뒤 맥락
                if i < len(paragraphs) - 1 and paragraphs[i + 1].strip():
                    next_p = paragraphs[i + 1][:200] if len(paragraphs[i + 1]) > 200 else paragraphs[i + 1]
                    context_parts.append(f"[뒤 맥락] {next_p}")

                results.append(ChunkResult(
                    content="\n\n".join(context_parts),
                    chunk_type=ChunkType.PARAGRAPH,
                    file_name=file_name,
                    page=start_page,
                    chunk_index=chunk_index,
                    metadata={"original_paragraph_index": i}
                ))
                chunk_index += 1

        return results
