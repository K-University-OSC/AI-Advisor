"""
RAG 도구 (Retrieval-Augmented Generation Tool)
기존 memory_service를 활용한 문서 검색 도구
"""

from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)


class RAGTool:
    """
    기존 memory_service를 래핑한 RAG 도구
    업로드된 문서 및 이전 대화 기반 검색
    """

    def __init__(self, memory_service=None):
        """
        Args:
            memory_service: MemoryService 인스턴스 (None이면 lazy 초기화)
        """
        self._memory_service = memory_service

    async def _get_memory_service(self):
        """Memory service lazy 초기화 (async)"""
        if self._memory_service is None:
            try:
                from services.memory_service import MemoryService
                self._memory_service = MemoryService()
                await self._memory_service.initialize()  # 비동기 초기화 필수
                logger.info("MemoryService 초기화 완료")
            except Exception as e:
                logger.error(f"MemoryService 초기화 실패: {e}")
                raise
        return self._memory_service

    async def search(
        self,
        query: str,
        tenant_id: str = "default",
        user_id: str = "default",
        top_k: int = 5,
        pipeline_mode: str = "3-stage"
    ) -> Dict[str, Any]:
        """
        RAG 검색 수행 (async)

        Args:
            query: 검색 쿼리
            tenant_id: 테넌트 ID
            user_id: 사용자 ID (문자열 또는 정수)
            top_k: 반환할 결과 수
            pipeline_mode: 검색 파이프라인 모드

        Returns:
            검색 결과 딕셔너리
        """
        try:
            memory_service = await self._get_memory_service()

            # user_id를 정수로 변환 (MemoryService.search_memories는 int 필요)
            user_id_int = int(user_id) if isinstance(user_id, str) and user_id.isdigit() else 0
            logger.info(f"RAGTool.search - user_id 원본: {user_id}, 변환: {user_id_int}, tenant: {tenant_id}")

            # memory_service의 search_memories 메서드 사용 (async)
            results = await memory_service.search_memories(
                tenant_id=tenant_id,
                user_id=user_id_int,
                query=query,
                top_k=top_k,
                pipeline_mode=pipeline_mode
            )
            logger.info(f"RAGTool.search - 검색 결과: {len(results) if results else 0}개")

            return {
                "query": query,
                "results": results,
                "count": len(results) if results else 0,
                "pipeline_mode": pipeline_mode
            }

        except Exception as e:
            logger.error(f"RAG 검색 오류: {e}")
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "count": 0
            }

    def format_results_for_llm(self, search_result: Dict[str, Any]) -> str:
        """
        검색 결과를 LLM 프롬프트용으로 포맷팅

        Args:
            search_result: search() 메서드의 반환값

        Returns:
            포맷팅된 문자열
        """
        if "error" in search_result and search_result["error"]:
            return f"문서 검색 오류: {search_result['error']}"

        if not search_result.get("results"):
            return "관련 문서를 찾지 못했습니다."

        output = []
        output.append("이전 대화에서 추출한 사용자 정보:")

        for i, result in enumerate(search_result.get("results", []), 1):
            content = result.get("content", result.get("text", ""))
            score = result.get("vector_score", result.get("score", 0))

            # 점수가 float인 경우만 포맷팅
            score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
            output.append(f"\n[{i}] (관련도: {score_str})")
            output.append(f"    {content[:500]}")

        return "\n".join(output)


# 문서 검색 필요 여부 판단
def needs_rag_search(message: str, has_uploaded_docs: bool = False) -> bool:
    """
    메시지가 RAG 검색이 필요한지 판단

    Args:
        message: 사용자 메시지
        has_uploaded_docs: 업로드된 문서가 있는지 여부

    Returns:
        RAG 검색 필요 여부
    """
    # 문서 참조 키워드
    doc_keywords = [
        # 문서 참조
        "문서", "파일", "업로드", "첨부", "자료",
        "document", "file", "upload", "attachment",
        # 이전 대화 참조
        "아까", "전에", "이전에", "말했", "얘기",
        "earlier", "before", "previously", "mentioned", "said",
        # 요약/정리
        "요약", "정리", "알려줬", "설명했",
        "summarize", "summary", "told", "explained"
    ]

    message_lower = message.lower()

    # 업로드된 문서가 있으면 관련 질문일 가능성이 높음
    if has_uploaded_docs:
        return True

    return any(keyword in message_lower for keyword in doc_keywords)
