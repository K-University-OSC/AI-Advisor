# -*- coding: utf-8 -*-
"""
MH_RAG 서비스 (Multimodal Hierarchical RAG Service)

일반 모드에서 MH_RAG 문서 검색을 제공하는 서비스
V7.1 컬렉션(mh_rag_finance_v7_1)과 연동
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# MH_RAG 활성화 여부 (환경변수로 제어)
MH_RAG_ENABLED = os.getenv("MH_RAG_ENABLED", "true").lower() == "true"
MH_RAG_COLLECTION = os.getenv("MH_RAG_COLLECTION", "mh_rag_finance_v7_1")

# 싱글톤 인스턴스
_mh_rag_service: Optional["MHRAGService"] = None


class MHRAGService:
    """MH_RAG 문서 검색 서비스"""

    def __init__(self, collection_name: str = None):
        """
        Args:
            collection_name: Qdrant 컬렉션 이름 (기본: mh_rag_finance_v7_1)
        """
        self.collection_name = collection_name or MH_RAG_COLLECTION
        self._rag_system = None
        self._initialized = False

    async def initialize(self) -> bool:
        """MH_RAG 시스템 초기화"""
        if self._initialized:
            return True

        try:
            from config import Settings
            from rag.mh_rag import MultimodalHierarchicalRAG

            settings = Settings()
            settings.qdrant_collection_name = self.collection_name
            # V7.1 설정
            settings.embedding_model = "text-embedding-3-large"
            settings.llm_model = "gpt-5.2"
            settings.vlm_model = "gpt-4o"

            self._rag_system = MultimodalHierarchicalRAG(settings=settings)
            await self._rag_system.initialize()

            self._initialized = True
            logger.info(f"MH_RAG 서비스 초기화 완료 (컬렉션: {self.collection_name})")
            return True

        except Exception as e:
            logger.error(f"MH_RAG 서비스 초기화 실패: {e}")
            return False

    def is_available(self) -> bool:
        """서비스 사용 가능 여부"""
        return self._initialized and self._rag_system is not None

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        문서 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            filter_source: 특정 소스만 검색

        Returns:
            검색 결과 리스트
        """
        if not self.is_available():
            logger.warning("MH_RAG 서비스가 초기화되지 않음")
            return []

        try:
            result = await self._rag_system.search(
                query=query,
                top_k=top_k,
                filter_source=filter_source,
            )

            # 결과를 딕셔너리 리스트로 변환
            documents = []
            if hasattr(result, 'documents') and result.documents:
                for doc in result.documents:
                    documents.append({
                        "content": doc.content if hasattr(doc, 'content') else str(doc),
                        "source": doc.metadata.get("source", "") if hasattr(doc, 'metadata') else "",
                        "page": doc.metadata.get("page", 0) if hasattr(doc, 'metadata') else 0,
                        "score": doc.score if hasattr(doc, 'score') else 0.0,
                    })

            logger.info(f"MH_RAG 검색 완료: {len(documents)}개 문서 발견")
            return documents

        except Exception as e:
            logger.error(f"MH_RAG 검색 오류: {e}")
            return []

    def format_context(self, documents: List[Dict[str, Any]], max_length: int = 3000) -> str:
        """
        검색 결과를 LLM 컨텍스트로 포맷팅

        Args:
            documents: 검색 결과 리스트
            max_length: 최대 컨텍스트 길이

        Returns:
            포맷팅된 컨텍스트 문자열
        """
        if not documents:
            return ""

        context_parts = []
        current_length = 0

        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            source = doc.get("source", "")
            page = doc.get("page", 0)

            # 출처 정보
            source_info = f"[출처: {source}"
            if page:
                source_info += f", 페이지 {page}"
            source_info += "]"

            doc_text = f"{source_info}\n{content}\n"

            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n---\n".join(context_parts)

    async def close(self):
        """리소스 정리"""
        if self._rag_system:
            await self._rag_system.close()
        self._initialized = False


async def get_mh_rag_service() -> Optional[MHRAGService]:
    """
    MH_RAG 서비스 싱글톤 인스턴스 반환

    Returns:
        MHRAGService 인스턴스 또는 None
    """
    global _mh_rag_service

    if not MH_RAG_ENABLED:
        return None

    if _mh_rag_service is None:
        _mh_rag_service = MHRAGService()
        await _mh_rag_service.initialize()

    return _mh_rag_service


# RAG 검색 필요 여부 판단
def needs_mh_rag_search(message: str) -> bool:
    """
    메시지가 MH_RAG 검색이 필요한지 판단

    Args:
        message: 사용자 메시지

    Returns:
        MH_RAG 검색 필요 여부
    """
    # 금융/문서 관련 키워드
    rag_keywords = [
        # 금융 용어
        "은행", "금리", "대출", "이자", "인가", "예비인가", "본인가",
        "연금", "퇴직", "기금", "투자", "펀드", "수익률",
        "녹색금융", "ESG", "기후", "탄소", "TCFD", "PCAF",
        "핀테크", "상생금융",
        # 문서 참조
        "문서", "보고서", "자료", "법률", "법령", "시행령",
        "규정", "지침", "가이드",
        # 질문 패턴
        "무엇인가", "설명해", "알려줘", "어떻게", "왜",
        "차이점", "비교", "요건", "절차", "조건",
    ]

    message_lower = message.lower()

    for keyword in rag_keywords:
        if keyword in message_lower:
            return True

    return False
