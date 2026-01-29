"""
FastAPI 기반 API 서버
Multimodal Hierarchical RAG 시스템의 REST API 제공
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import aiofiles

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import get_settings
from rag.mh_rag import MultimodalHierarchicalRAG
from rag.chain import ChatMessage


rag_system: Optional[MultimodalHierarchicalRAG] = None


class ChatRequest(BaseModel):
    """채팅 요청"""
    query: str = Field(..., description="사용자 질문")
    conversation_history: Optional[list[dict]] = Field(
        default=None,
        description="이전 대화 기록 [{'role': 'user', 'content': '...'}, ...]"
    )
    stream: bool = Field(default=False, description="스트리밍 응답 여부")


class ChatResponse(BaseModel):
    """채팅 응답"""
    answer: str
    sources: list[dict]
    metadata: dict = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """검색 요청"""
    query: str = Field(..., description="검색 쿼리")
    top_k: int = Field(default=5, description="반환할 결과 수")
    filter_source: Optional[str] = Field(default=None, description="특정 소스만 검색")


class SearchResponse(BaseModel):
    """검색 응답"""
    query: str
    results: list[dict]
    context: str


class IngestResponse(BaseModel):
    """인덱싱 응답"""
    filename: str
    total_pages: int
    total_elements: int
    parent_chunks: int
    child_chunks: int
    captioned_images: int


class DeleteRequest(BaseModel):
    """삭제 요청"""
    source: str = Field(..., description="삭제할 문서명")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 생명주기 관리"""
    global rag_system
    rag_system = MultimodalHierarchicalRAG()
    await rag_system.initialize()
    print("✓ RAG 시스템 시작됨")

    yield

    if rag_system:
        await rag_system.close()
    print("✓ RAG 시스템 종료됨")


def create_app() -> FastAPI:
    """FastAPI 앱 생성"""
    app = FastAPI(
        title="Multimodal Hierarchical RAG API",
        description="복합 금융/학술 문서를 위한 멀티모달 계층적 RAG 시스템",
        version="1.0.0",
        lifespan=lifespan,
    )

    settings = get_settings()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "initialized": rag_system is not None and rag_system._initialized,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG 기반 채팅"""
    if not rag_system or not rag_system._initialized:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")

    try:
        history = None
        if request.conversation_history:
            history = [
                ChatMessage(role=m["role"], content=m["content"])
                for m in request.conversation_history
            ]

        if request.stream:
            async def generate():
                async for chunk in rag_system.chat_stream(
                    query=request.query,
                    conversation_history=history,
                ):
                    yield chunk

            return StreamingResponse(
                generate(),
                media_type="text/plain",
            )
        else:
            response = await rag_system.chat(
                query=request.query,
                conversation_history=history,
            )
            return ChatResponse(
                answer=response.answer,
                sources=response.sources,
                metadata=response.metadata,
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """스트리밍 RAG 채팅"""
    if not rag_system or not rag_system._initialized:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")

    try:
        history = None
        if request.conversation_history:
            history = [
                ChatMessage(role=m["role"], content=m["content"])
                for m in request.conversation_history
            ]

        async def generate():
            async for chunk in rag_system.chat_stream(
                query=request.query,
                conversation_history=history,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """문서 검색"""
    if not rag_system or not rag_system._initialized:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")

    try:
        result = await rag_system.search(
            query=request.query,
            top_k=request.top_k,
            filter_source=request.filter_source,
        )

        results = []
        for child in result.child_results:
            results.append({
                "chunk_id": child.chunk_id,
                "content": child.content,
                "score": child.score,
                "source": child.source,
                "page": child.page,
                "bbox": child.bbox,
                "heading": child.heading,
            })

        return SearchResponse(
            query=request.query,
            results=results,
            context=result.context,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    caption_images: bool = True,
):
    """문서 인덱싱"""
    if not rag_system or not rag_system._initialized:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")

    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".docx", ".pptx", ".xlsx"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {allowed_extensions}"
        )

    try:
        upload_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        result = await rag_system.ingest_document(
            file_path=file_path,
            caption_images=caption_images,
        )

        return IngestResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document")
async def delete_document(request: DeleteRequest):
    """문서 삭제"""
    if not rag_system or not rag_system._initialized:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")

    try:
        await rag_system.delete_document(request.source)
        return {"status": "success", "message": f"문서 '{request.source}' 삭제 완료"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources")
async def list_sources():
    """인덱싱된 소스 목록 조회"""
    if not rag_system or not rag_system._initialized:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")

    return {"message": "소스 목록 조회 기능은 추후 구현 예정입니다"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8630,
        reload=True,
    )
