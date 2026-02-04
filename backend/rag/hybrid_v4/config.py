# -*- coding: utf-8 -*-
"""
HYBRID_V4 RAG Configuration

환경변수 또는 기본값으로 설정 관리
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HybridV4Config:
    """HYBRID_V4 RAG 설정"""

    # ========== Qdrant 설정 ==========
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    collection_name: str = field(default_factory=lambda: os.getenv("HYBRID_V4_COLLECTION", "hybrid_v4"))

    # ========== 임베딩 설정 ==========
    embedding_model: str = field(default_factory=lambda: os.getenv("HYBRID_V4_EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_dimension: int = 1536  # text-embedding-3-small

    # ========== 청킹 설정 ==========
    chunk_size: int = 500
    chunk_overlap: int = 50
    semantic_threshold: float = 0.5  # Semantic chunking 유사도 임계값
    min_chunk_size: int = 100
    max_chunk_size: int = 1000

    # ========== 이미지 설정 ==========
    min_image_size: int = 5000  # 5KB 미만 이미지 제외
    image_caption_model: str = "gpt-4o"
    image_detail: str = "high"  # low, high, auto

    # ========== 검색 설정 ==========
    vector_top_k: int = 15  # 1단계 벡터 검색
    rerank_top_k: int = 5   # 2단계 리랭킹 후 최종
    similarity_threshold: float = 0.5

    # ========== Reranker 설정 ==========
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_max_length: int = 512

    # ========== LLM 설정 ==========
    llm_model: str = field(default_factory=lambda: os.getenv("HYBRID_V4_LLM_MODEL", "gpt-4o"))
    llm_temperature: float = 0.0
    llm_max_tokens: int = 500

    # ========== 테이블 추출 설정 ==========
    table_max_length: int = 2000  # 테이블 최대 길이

    # ========== API 키 (환경변수에서 로드) ==========
    @property
    def openai_api_key(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")

    def validate(self) -> bool:
        """설정 유효성 검증"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return True

    def to_dict(self) -> dict:
        """설정을 딕셔너리로 변환"""
        return {
            "qdrant_host": self.qdrant_host,
            "qdrant_port": self.qdrant_port,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "semantic_threshold": self.semantic_threshold,
            "vector_top_k": self.vector_top_k,
            "rerank_top_k": self.rerank_top_k,
            "reranker_model": self.reranker_model,
            "llm_model": self.llm_model,
        }
