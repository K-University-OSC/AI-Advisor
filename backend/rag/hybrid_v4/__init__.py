# -*- coding: utf-8 -*-
"""
HYBRID_V4 RAG Pipeline Module

90.9% 정확도 달성 (Azure/Upstage 미사용, OpenAI만 사용)

주요 구성요소:
1. SemanticChunker - 의미 기반 청킹 (all-MiniLM-L6-v2)
2. ImageCaptioner - GPT-4o Vision 구조화 캡셔닝
3. HybridV4RAGService - 전체 RAG 파이프라인

사용법:
    from rag.hybrid_v4 import HybridV4RAGService

    rag = HybridV4RAGService()
    await rag.initialize()

    # 문서 인덱싱
    await rag.index_document("document.pdf")

    # 검색 및 답변
    result = await rag.query("질문 내용")
"""

from .config import HybridV4Config
from .semantic_chunker import SemanticChunker, ChunkResult
from .image_captioner import ImageCaptioner, CaptionResult
from .rag_service import HybridV4RAGService

__all__ = [
    "HybridV4Config",
    "SemanticChunker",
    "ChunkResult",
    "ImageCaptioner",
    "CaptionResult",
    "HybridV4RAGService",
]

__version__ = "4.0.0"
