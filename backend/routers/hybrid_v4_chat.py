# -*- coding: utf-8 -*-
"""
HYBRID_V4 RAG 채팅 API 라우터

HYBRID_V4 파이프라인 전용 엔드포인트
- 90.9% 정확도 (Azure/Upstage 미사용, OpenAI만 사용)
- Semantic Chunking + GPT-4o Vision 이미지 캡셔닝
- BGE Reranker

엔드포인트:
- POST /api/v4/chat/send - RAG 채팅 (스트리밍)
- POST /api/v4/chat/query - RAG 질의 (비스트리밍)
- POST /api/v4/index - 문서 인덱싱
- GET /api/v4/search - 검색
- GET /api/v4/stats - 통계
"""

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
import json
import os
import aiofiles
from pathlib import Path

from routers.auth import get_current_user
from services.hybrid_v4_service import (
    get_hybrid_v4_chat_service,
    HybridV4ChatService
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v4", tags=["HYBRID_V4 RAG"])


# ============================================================================
# Request/Response 모델
# ============================================================================

class ChatRequest(BaseModel):
    """채팅 요청"""
    message: str
    session_id: Optional[str] = None


class QueryRequest(BaseModel):
    """질의 요청"""
    question: str
    top_k: Optional[int] = 5


class SearchRequest(BaseModel):
    """검색 요청"""
    query: str
    top_k: Optional[int] = 5


class ChatResponse(BaseModel):
    """채팅 응답"""
    answer: str
    sources: List[dict]
    model: str
    session_id: Optional[str] = None


# ============================================================================
# 엔드포인트
# ============================================================================

@router.post("/chat/send")
async def chat_send(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    HYBRID_V4 RAG 채팅 (스트리밍)

    SSE (Server-Sent Events) 형식으로 응답 스트리밍
    """
    try:
        service = await get_hybrid_v4_chat_service()

        async def generate():
            try:
                # 스트리밍 응답
                async for chunk in service.chat_stream(
                    message=request.message,
                    session_id=request.session_id,
                    user_id=current_user.get("user_id")
                ):
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

                # 완료 신호
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                logger.error(f"스트리밍 오류: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"채팅 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/query", response_model=ChatResponse)
async def chat_query(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    HYBRID_V4 RAG 질의 (비스트리밍)

    전체 응답을 한 번에 반환
    """
    try:
        service = await get_hybrid_v4_chat_service()

        result = await service.chat(
            message=request.question,
            user_id=current_user.get("user_id")
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            model=result["model"],
            session_id=result.get("session_id")
        )

    except Exception as e:
        logger.error(f"질의 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index")
async def index_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    문서 인덱싱

    PDF 파일을 HYBRID_V4 파이프라인으로 인덱싱
    """
    # 관리자 권한 확인 (선택적)
    # if current_user.get("role") != "admin":
    #     raise HTTPException(status_code=403, detail="관리자 권한 필요")

    try:
        # 파일 확장자 확인
        allowed_extensions = {".pdf", ".txt", ".md", ".docx"}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 파일 형식: {file_ext}. 지원: {allowed_extensions}"
            )

        # 임시 파일 저장
        upload_dir = Path("/tmp/hybrid_v4_uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # 인덱싱
        service = await get_hybrid_v4_chat_service()
        result = await service.index_document(str(file_path))

        # 임시 파일 삭제
        file_path.unlink(missing_ok=True)

        return {
            "status": "success",
            "message": f"문서 인덱싱 완료: {file.filename}",
            "result": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"인덱싱 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search(
    query: str,
    top_k: int = 5,
    current_user: dict = Depends(get_current_user)
):
    """
    검색만 수행 (답변 생성 없음)
    """
    try:
        service = await get_hybrid_v4_chat_service()
        results = await service.search(query, top_k)

        return {
            "query": query,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"검색 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats(
    current_user: dict = Depends(get_current_user)
):
    """
    HYBRID_V4 컬렉션 통계
    """
    try:
        service = await get_hybrid_v4_chat_service()
        stats = await service.get_stats()

        return {
            "status": "success",
            "stats": stats,
            "pipeline": {
                "name": "HYBRID_V4",
                "accuracy": "90.9%",
                "features": [
                    "Semantic Chunking (all-MiniLM-L6-v2)",
                    "GPT-4o Vision Image Captioning",
                    "BGE Reranker (bge-reranker-v2-m3)",
                    "text-embedding-3-small (1536 dims)"
                ]
            }
        }

    except Exception as e:
        logger.error(f"통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """HYBRID_V4 서비스 상태 확인"""
    try:
        service = await get_hybrid_v4_chat_service()
        stats = await service.get_stats()

        return {
            "status": "healthy",
            "service": "HYBRID_V4 RAG",
            "collection": stats.get("collection_name"),
            "points_count": stats.get("points_count", 0)
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
