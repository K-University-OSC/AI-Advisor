# -*- coding: utf-8 -*-
"""
HYBRID_V4 RAG Service Wrapper

기존 advisor_osc 서비스와 HYBRID_V4 RAG 파이프라인 통합

사용법:
    from services.hybrid_v4_service import get_hybrid_v4_service

    # 서비스 인스턴스 가져오기
    service = await get_hybrid_v4_service()

    # 문서 인덱싱
    result = await service.index_document("/path/to/document.pdf")

    # 질의응답
    answer = await service.query("질문 내용")

    # RAG 채팅 (스트리밍)
    async for chunk in service.chat_stream("질문", session_id="..."):
        print(chunk)
"""

import logging
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

from rag.hybrid_v4 import HybridV4RAGService, HybridV4Config

logger = logging.getLogger(__name__)

# 싱글톤 인스턴스
_hybrid_v4_service: Optional[HybridV4RAGService] = None


async def get_hybrid_v4_service() -> HybridV4RAGService:
    """
    HYBRID_V4 RAG 서비스 싱글톤 인스턴스 반환

    Returns:
        초기화된 HybridV4RAGService 인스턴스
    """
    global _hybrid_v4_service

    if _hybrid_v4_service is None or not _hybrid_v4_service.is_initialized:
        logger.info("HYBRID_V4 RAG Service 초기화 중...")
        config = HybridV4Config()
        _hybrid_v4_service = HybridV4RAGService(config)
        success = await _hybrid_v4_service.initialize()

        if not success:
            raise RuntimeError("HYBRID_V4 RAG Service 초기화 실패")

        logger.info("HYBRID_V4 RAG Service 초기화 완료")

    return _hybrid_v4_service


async def reset_hybrid_v4_service():
    """서비스 재초기화"""
    global _hybrid_v4_service
    _hybrid_v4_service = None
    return await get_hybrid_v4_service()


class HybridV4ChatService:
    """
    HYBRID_V4 채팅 서비스

    기존 RAG 채팅 인터페이스와 호환되는 래퍼
    """

    def __init__(self):
        self.rag_service: Optional[HybridV4RAGService] = None
        self._initialized = False

    async def initialize(self):
        """서비스 초기화"""
        self.rag_service = await get_hybrid_v4_service()
        self._initialized = True

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        RAG 기반 채팅 (동기)

        Args:
            message: 사용자 메시지
            session_id: 세션 ID (옵션)
            user_id: 사용자 ID (옵션)

        Returns:
            응답 딕셔너리
        """
        if not self._initialized:
            await self.initialize()

        result = await self.rag_service.query(message)

        return {
            "answer": result.answer,
            "sources": [
                {
                    "content": s.content[:200] + "..." if len(s.content) > 200 else s.content,
                    "file_name": s.file_name,
                    "page": s.page,
                    "score": s.score,
                    "chunk_type": s.chunk_type
                }
                for s in result.sources
            ],
            "model": "hybrid_v4",
            "session_id": session_id,
            "metadata": result.metadata
        }

    async def chat_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        RAG 기반 채팅 (스트리밍)

        Args:
            message: 사용자 메시지
            session_id: 세션 ID
            user_id: 사용자 ID

        Yields:
            응답 청크
        """
        if not self._initialized:
            await self.initialize()

        # 검색
        search_results = await self.rag_service.search(message)

        if not search_results:
            yield "관련 정보를 찾을 수 없습니다."
            return

        # 컨텍스트 구성
        context = "\n\n".join([r.content for r in search_results])

        # 스트리밍 답변 생성
        from openai import AsyncOpenAI
        import os

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = await client.chat.completions.create(
            model=self.rag_service.config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "주어진 컨텍스트를 바탕으로 질문에 정확하고 간결하게 답변하세요. "
                               "이미지 관련 정보도 컨텍스트에 포함되어 있습니다. "
                               "컨텍스트에 없는 정보는 답변하지 마세요."
                },
                {
                    "role": "user",
                    "content": f"컨텍스트:\n{context}\n\n질문: {message}"
                }
            ],
            temperature=0,
            max_tokens=500,
            stream=True
        )

        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def index_document(self, file_path: str) -> Dict[str, Any]:
        """
        문서 인덱싱

        Args:
            file_path: 문서 파일 경로

        Returns:
            인덱싱 결과
        """
        if not self._initialized:
            await self.initialize()

        return await self.rag_service.index_document(file_path)

    async def search(self, query: str, top_k: int = 5) -> list:
        """
        검색만 수행

        Args:
            query: 검색 쿼리
            top_k: 결과 수

        Returns:
            검색 결과 리스트
        """
        if not self._initialized:
            await self.initialize()

        results = await self.rag_service.search(query, top_k)
        return [
            {
                "content": r.content,
                "score": r.score,
                "file_name": r.file_name,
                "page": r.page,
                "chunk_type": r.chunk_type
            }
            for r in results
        ]

    async def get_stats(self) -> Dict[str, Any]:
        """컬렉션 통계"""
        if not self._initialized:
            await self.initialize()

        return await self.rag_service.get_collection_info()


# 전역 채팅 서비스 인스턴스
_chat_service: Optional[HybridV4ChatService] = None


async def get_hybrid_v4_chat_service() -> HybridV4ChatService:
    """채팅 서비스 싱글톤 인스턴스"""
    global _chat_service

    if _chat_service is None:
        _chat_service = HybridV4ChatService()
        await _chat_service.initialize()

    return _chat_service
