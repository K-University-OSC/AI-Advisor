"""
데이터 관리 API 라우터
관리자가 RAG 챗봇의 데이터(파일, 동영상, 크롤링 사이트)를 관리하는 API
"""
import os
import uuid
import shutil
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import text

from database import get_db
from routers.auth import get_current_user
from routers.admin import verify_admin

router = APIRouter()

# ============================================================================
# 설정
# ============================================================================

# 데이터 저장 경로 (환경변수 또는 기본값)
DATA_BASE_PATH = os.getenv("DATA_BASE_PATH", "/home/aiedu/workspace/advisor/data")
DOCUMENTS_PATH = os.path.join(DATA_BASE_PATH, "documents")
VIDEOS_PATH = os.path.join(DATA_BASE_PATH, "videos")

# 허용 파일 확장자
ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".pptx", ".xlsx", ".csv", ".hwp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


# ============================================================================
# Request/Response 모델
# ============================================================================

class DocumentResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_size: int
    file_type: str
    status: str  # pending, processing, indexed, error
    uploaded_at: datetime
    indexed_at: Optional[datetime] = None
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None


class VideoResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_size: int
    duration: Optional[int] = None  # 초 단위
    status: str  # pending, processing, transcribed, indexed, error
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    transcript_length: Optional[int] = None
    error_message: Optional[str] = None


class CrawlSiteRequest(BaseModel):
    url: str
    name: str
    description: Optional[str] = None
    crawl_depth: int = Field(default=2, ge=1, le=5)
    crawl_frequency: str = Field(default="daily")  # daily, weekly, monthly, manual
    selectors: Optional[dict] = None  # CSS selectors for content extraction


class CrawlSiteResponse(BaseModel):
    id: str
    url: str
    name: str
    description: Optional[str] = None
    crawl_depth: int
    crawl_frequency: str
    status: str  # active, paused, error
    last_crawl_at: Optional[datetime] = None
    next_crawl_at: Optional[datetime] = None
    pages_crawled: int = 0
    error_message: Optional[str] = None
    created_at: datetime


class DataStatsResponse(BaseModel):
    total_documents: int
    indexed_documents: int
    total_videos: int
    processed_videos: int
    total_crawl_sites: int
    active_crawl_sites: int
    total_chunks: int
    storage_used_mb: float


# ============================================================================
# 유틸리티 함수
# ============================================================================

def ensure_directories():
    """데이터 디렉토리 생성"""
    Path(DOCUMENTS_PATH).mkdir(parents=True, exist_ok=True)
    Path(VIDEOS_PATH).mkdir(parents=True, exist_ok=True)


def get_file_extension(filename: str) -> str:
    """파일 확장자 추출"""
    return Path(filename).suffix.lower()


def generate_unique_filename(original_filename: str) -> str:
    """고유 파일명 생성"""
    ext = get_file_extension(original_filename)
    unique_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{unique_id}{ext}"


def get_directory_size(path: str) -> int:
    """디렉토리 크기 계산 (바이트)"""
    total = 0
    if os.path.exists(path):
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_directory_size(entry.path)
    return total


# ============================================================================
# 데이터 통계 API
# ============================================================================

@router.get("/stats", response_model=DataStatsResponse)
async def get_data_stats(
    request: Request,
    admin_info: dict = Depends(verify_admin)
):
    """데이터 관리 통계 조회"""
    # tenant removed

    async with get_db() as session:
        # 문서 통계
        doc_result = await session.execute(text("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'indexed' THEN 1 END) as indexed
            FROM rag_documents
        """))
        doc_stats = doc_result.mappings().first()

        # 동영상 통계
        video_result = await session.execute(text("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'indexed' THEN 1 END) as processed
            FROM rag_videos
        """))
        video_stats = video_result.mappings().first()

        # 크롤링 사이트 통계
        crawl_result = await session.execute(text("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active
            FROM crawl_sites
        """))
        crawl_stats = crawl_result.mappings().first()

        # 청크 수
        chunk_result = await session.execute(text("SELECT COUNT(*) FROM rag_chunks"))
        total_chunks = chunk_result.scalar() or 0

    # 스토리지 사용량
    doc_size = get_directory_size(DOCUMENTS_PATH)
    video_size = get_directory_size(VIDEOS_PATH)
    storage_mb = (doc_size + video_size) / (1024 * 1024)

    return DataStatsResponse(
        total_documents=doc_stats["total"] if doc_stats else 0,
        indexed_documents=doc_stats["indexed"] if doc_stats else 0,
        total_videos=video_stats["total"] if video_stats else 0,
        processed_videos=video_stats["processed"] if video_stats else 0,
        total_crawl_sites=crawl_stats["total"] if crawl_stats else 0,
        active_crawl_sites=crawl_stats["active"] if crawl_stats else 0,
        total_chunks=total_chunks,
        storage_used_mb=round(storage_mb, 2)
    )


# ============================================================================
# 문서 관리 API
# ============================================================================

@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    request: Request,
    page: int = 1,
    limit: int = 50,
    status: Optional[str] = None,
    admin_info: dict = Depends(verify_admin)
):
    """문서 목록 조회"""
    # tenant removed
    offset = (page - 1) * limit

    async with get_db() as session:
        query = """
            SELECT id, filename, original_filename, file_size, file_type,
                   status, uploaded_at, indexed_at, chunk_count, error_message
            FROM rag_documents
        """
        params = {"limit": limit, "offset": offset}

        if status:
            query += " WHERE status = :status"
            params["status"] = status

        query += " ORDER BY uploaded_at DESC LIMIT :limit OFFSET :offset"

        result = await session.execute(text(query), params)
        documents = result.mappings().all()

        return [DocumentResponse(**dict(doc)) for doc in documents]


@router.post("/documents/upload")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_index: bool = Form(default=True),
    admin_info: dict = Depends(verify_admin)
):
    """문서 업로드"""
    # tenant removed
    ensure_directories()

    # 파일 검증
    ext = get_file_extension(file.filename)
    if ext not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"허용되지 않는 파일 형식입니다. 허용: {', '.join(ALLOWED_DOCUMENT_EXTENSIONS)}"
        )

    # 파일 크기 확인 (스트리밍으로 읽으면서 체크)
    content = await file.read()
    file_size = len(content)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"파일 크기가 너무 큽니다. 최대: {MAX_FILE_SIZE // (1024*1024)}MB")

    # 고유 파일명 생성 및 저장
    unique_filename = generate_unique_filename(file.filename)
    doc_path = os.path.join(DOCUMENTS_PATH)
    Path(doc_path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(doc_path, unique_filename)

    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)

    # DB에 문서 정보 저장
    doc_id = str(uuid.uuid4())

    async with get_db() as session:
        await session.execute(text("""
            INSERT INTO rag_documents (id, filename, original_filename, file_size, file_type, status, uploaded_at)
            VALUES (:id, :filename, :original_filename, :file_size, :file_type, :status, :uploaded_at)
        """), {
            "id": doc_id,
            "filename": unique_filename,
            "original_filename": file.filename,
            "file_size": file_size,
            "file_type": ext[1:],  # .pdf -> pdf
            "status": "pending" if auto_index else "uploaded",
            "uploaded_at": datetime.now()
        })
        await session.commit()

    # 자동 인덱싱 요청 시 백그라운드 작업 추가
    if auto_index:
        background_tasks.add_task(index_document_task, doc_id, file_path)

    return {
        "id": doc_id,
        "filename": unique_filename,
        "original_filename": file.filename,
        "file_size": file_size,
        "status": "pending" if auto_index else "uploaded",
        "message": "문서가 업로드되었습니다." + (" 인덱싱이 시작됩니다." if auto_index else "")
    }


@router.delete("/documents/{doc_id}")
async def delete_document(
    request: Request,
    doc_id: str,
    admin_info: dict = Depends(verify_admin)
):
    """문서 삭제"""
    # tenant removed

    async with get_db() as session:
        # 문서 정보 조회
        result = await session.execute(
            text("SELECT filename FROM rag_documents WHERE id = :id"),
            {"id": doc_id}
        )
        doc = result.mappings().first()

        if not doc:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

        # 파일 삭제
        file_path = os.path.join(DOCUMENTS_PATH, "default", doc["filename"])
        if os.path.exists(file_path):
            os.remove(file_path)

        # 관련 청크 삭제
        await session.execute(
            text("DELETE FROM rag_chunks WHERE document_id = :doc_id"),
            {"doc_id": doc_id}
        )

        # 문서 레코드 삭제
        await session.execute(
            text("DELETE FROM rag_documents WHERE id = :id"),
            {"id": doc_id}
        )
        await session.commit()

    # TODO: 벡터 DB에서도 삭제

    return {"message": "문서가 삭제되었습니다.", "id": doc_id}


@router.post("/documents/{doc_id}/reindex")
async def reindex_document(
    request: Request,
    doc_id: str,
    background_tasks: BackgroundTasks,
    admin_info: dict = Depends(verify_admin)
):
    """문서 재인덱싱"""
    # tenant removed

    async with get_db() as session:
        result = await session.execute(
            text("SELECT filename FROM rag_documents WHERE id = :id"),
            {"id": doc_id}
        )
        doc = result.mappings().first()

        if not doc:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

        file_path = os.path.join(DOCUMENTS_PATH, "default", doc["filename"])
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

        # 상태 업데이트
        await session.execute(
            text("UPDATE rag_documents SET status = 'pending', error_message = NULL WHERE id = :id"),
            {"id": doc_id}
        )
        await session.commit()

    # 백그라운드 인덱싱 시작
    background_tasks.add_task(index_document_task, doc_id, file_path)

    return {"message": "재인덱싱이 시작됩니다.", "id": doc_id}


async def index_document_task(doc_id: str, file_path: str):
    """문서 인덱싱 백그라운드 작업 (V7.6.1 파이프라인)"""
    import sys
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    logger.info(f"[default] 문서 인덱싱 시작: {doc_id}, {file_path}")

    try:
        async with get_db() as session:
            await session.execute(
                text("UPDATE rag_documents SET status = 'processing' WHERE id = :id"),
                {"id": doc_id}
            )
            await session.commit()

        # 현재 프로젝트의 rag 모듈 로드
        from rag.parsers.parser_factory import get_document_parser
        from rag.embeddings.embedding_service import OpenAIEmbeddingService, GeminiEmbeddingService, SparseEmbeddingService, MultimodalEmbeddingService
        from rag.vectorstore.qdrant_store import QdrantVectorStore
        from rag.chunkers.hierarchical_chunker import HierarchicalChunker

        # 환경변수에서 설정 로드 (v1.6: Gemini 솔루션 사용)
        COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "advisor_osc_finance_gemini_embed")
        QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")  # Docker 컨테이너 이름
        QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
        EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google").lower()
        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

        # 서비스 초기화 (환경변수 DOCUMENT_PARSER로 선택, 기본값: gemini)
        parser = get_document_parser()
        chunker = HierarchicalChunker(
            parent_chunk_size=2000,
            child_chunk_size=500,
            chunk_overlap=50,
        )

        # 임베딩 서비스 선택 (EMBEDDING_PROVIDER 환경변수에 따라)
        if EMBEDDING_PROVIDER == "google":
            GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
            dense_embedding = GeminiEmbeddingService(
                api_key=GOOGLE_API_KEY,
                model=EMBEDDING_MODEL
            )
            logger.info(f"[index] Gemini 임베딩 사용: {EMBEDDING_MODEL}")
        else:
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            dense_embedding = OpenAIEmbeddingService(
                api_key=OPENAI_API_KEY,
                model=EMBEDDING_MODEL if EMBEDDING_MODEL else "text-embedding-3-large"
            )
            logger.info(f"[index] OpenAI 임베딩 사용: {EMBEDDING_MODEL}")

        sparse_embedding = SparseEmbeddingService()
        embedding_service = MultimodalEmbeddingService(
            dense_service=dense_embedding,
            sparse_service=sparse_embedding
        )

        # 컬렉션 확인 및 자동 생성
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams

        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = [c.name for c in qdrant_client.get_collections().collections]

        # 임베딩 모델에 따른 dimensions 설정
        # gemini-embedding-001: 3072, text-embedding-3-large: 3072
        VECTOR_SIZE = 3072  # Both Gemini and OpenAI use 3072 dimensions

        if COLLECTION_NAME not in collections:
            logger.info(f"[index] 새 컬렉션 생성: {COLLECTION_NAME} (dim={VECTOR_SIZE})")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
                },
                on_disk_payload=True,
            )
            logger.info(f"[index] 컬렉션 생성 완료: {COLLECTION_NAME}")

        vector_store = QdrantVectorStore(
            collection_name=COLLECTION_NAME,
            host=QDRANT_HOST,
            port=QDRANT_PORT,
        )

        file_ext = Path(file_path).suffix.lower()
        chunk_count = 0

        # PDF만 Azure 파서 사용, 나머지는 간단한 텍스트 처리
        if file_ext == '.pdf':
            # 문서 파서로 PDF 파싱 (기본: PyMuPDF, 환경변수로 변경 가능)
            parser_type = os.getenv("DOCUMENT_PARSER", "pymupdf")
            logger.info(f"[default] {parser_type} 파서로 PDF 파싱 중...")
            parsed_doc = await parser.parse(file_path)
            logger.info(f"[default] 파싱 완료: {len(parsed_doc.elements)}개 요소")

            # 청킹 (parent_chunks, child_chunks 튜플 반환)
            parent_chunks, child_chunks = chunker.chunk_document(parsed_doc)
            logger.info(f"[default] 청킹 완료: parent={len(parent_chunks)}, child={len(child_chunks)}")

            if parent_chunks or child_chunks:
                # 임베딩 생성 (dense, sparse 튜플 반환)
                dense_embeddings, sparse_embeddings = await embedding_service.embed_chunks(parent_chunks, child_chunks)

                # 벡터 DB에 저장
                await vector_store.add_chunks(
                    parent_chunks=parent_chunks,
                    child_chunks=child_chunks,
                    dense_embeddings=dense_embeddings,
                    sparse_embeddings=sparse_embeddings,
                )
                chunk_count = len(parent_chunks) + len(child_chunks)
                logger.info(f"[default] 벡터 DB 저장 완료: {chunk_count}개")
        else:
            # 텍스트 파일 간단 처리
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()

            if content.strip():
                # 간단한 청킹 (800자 단위)
                text_chunks = []
                for i in range(0, len(content), 700):
                    chunk_text = content[i:i+800]
                    if chunk_text.strip():
                        text_chunks.append(chunk_text)

                if text_chunks:
                    from qdrant_client.models import PointStruct
                    import hashlib

                    points = []
                    for idx, chunk_text in enumerate(text_chunks):
                        # 임베딩 생성
                        embedding = await dense_embedding.embed_text(chunk_text)
                        point_id = hashlib.md5(f"{doc_id}_{idx}".encode()).hexdigest()

                        points.append(PointStruct(
                            id=point_id,
                            vector={"dense": embedding},
                            payload={
                                "text": chunk_text,
                                "doc_id": doc_id,
                                "tenant_id": "default",
                                "source_file": Path(file_path).name,
                                "chunk_index": idx,
                            }
                        ))

                    # 벡터 DB에 저장
                    await vector_store.async_client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points
                    )
                    chunk_count = len(points)
                    logger.info(f"[default] 텍스트 파일 인덱싱 완료: {chunk_count}개 청크")

        # 성공 상태 업데이트
        async with get_db() as session:
            await session.execute(text("""
                UPDATE rag_documents
                SET status = 'indexed', indexed_at = :indexed_at, chunk_count = :chunk_count
                WHERE id = :id
            """), {"id": doc_id, "indexed_at": datetime.now(), "chunk_count": chunk_count})
            await session.commit()

        logger.info(f"[default] 문서 인덱싱 완료: {doc_id}, {chunk_count}개 청크")

    except Exception as e:
        import traceback
        logger.error(f"[default] 인덱싱 에러: {e}\n{traceback.format_exc()}")
        async with get_db() as session:
            await session.execute(text("""
                UPDATE rag_documents
                SET status = 'error', error_message = :error
                WHERE id = :id
            """), {"id": doc_id, "error": str(e)[:500]})
            await session.commit()


# ============================================================================
# 동영상 관리 API
# ============================================================================

@router.get("/videos", response_model=List[VideoResponse])
async def list_videos(
    request: Request,
    page: int = 1,
    limit: int = 50,
    status: Optional[str] = None,
    admin_info: dict = Depends(verify_admin)
):
    """동영상 목록 조회"""
    # tenant removed
    offset = (page - 1) * limit

    async with get_db() as session:
        query = """
            SELECT id, filename, original_filename, file_size, duration,
                   status, uploaded_at, processed_at, transcript_length, error_message
            FROM rag_videos
        """
        params = {"limit": limit, "offset": offset}

        if status:
            query += " WHERE status = :status"
            params["status"] = status

        query += " ORDER BY uploaded_at DESC LIMIT :limit OFFSET :offset"

        result = await session.execute(text(query), params)
        videos = result.mappings().all()

        return [VideoResponse(**dict(v)) for v in videos]


@router.post("/videos/upload")
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_transcribe: bool = Form(default=True),
    admin_info: dict = Depends(verify_admin)
):
    """동영상 업로드"""
    # tenant removed
    ensure_directories()

    # 파일 검증
    ext = get_file_extension(file.filename)
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"허용되지 않는 파일 형식입니다. 허용: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )

    # 파일 저장 (스트리밍)
    unique_filename = generate_unique_filename(file.filename)
    video_path = os.path.join(VIDEOS_PATH)
    Path(video_path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(video_path, unique_filename)

    file_size = 0
    async with aiofiles.open(file_path, 'wb') as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            await f.write(chunk)
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                os.remove(file_path)
                raise HTTPException(status_code=400, detail=f"파일 크기가 너무 큽니다. 최대: {MAX_FILE_SIZE // (1024*1024)}MB")

    # DB에 동영상 정보 저장
    video_id = str(uuid.uuid4())

    async with get_db() as session:
        await session.execute(text("""
            INSERT INTO rag_videos (id, filename, original_filename, file_size, status, uploaded_at)
            VALUES (:id, :filename, :original_filename, :file_size, :status, :uploaded_at)
        """), {
            "id": video_id,
            "filename": unique_filename,
            "original_filename": file.filename,
            "file_size": file_size,
            "status": "pending" if auto_transcribe else "uploaded",
            "uploaded_at": datetime.now()
        })
        await session.commit()

    # 자동 트랜스크립션 요청 시 백그라운드 작업 추가
    if auto_transcribe:
        background_tasks.add_task(transcribe_video_task, video_id, file_path)

    return {
        "id": video_id,
        "filename": unique_filename,
        "original_filename": file.filename,
        "file_size": file_size,
        "status": "pending" if auto_transcribe else "uploaded",
        "message": "동영상이 업로드되었습니다." + (" 트랜스크립션이 시작됩니다." if auto_transcribe else "")
    }


@router.delete("/videos/{video_id}")
async def delete_video(
    request: Request,
    video_id: str,
    admin_info: dict = Depends(verify_admin)
):
    """동영상 삭제"""
    # tenant removed

    async with get_db() as session:
        # 동영상 정보 조회
        result = await session.execute(
            text("SELECT filename FROM rag_videos WHERE id = :id"),
            {"id": video_id}
        )
        video = result.mappings().first()

        if not video:
            raise HTTPException(status_code=404, detail="동영상을 찾을 수 없습니다.")

        # 파일 삭제
        file_path = os.path.join(VIDEOS_PATH, "default", video["filename"])
        if os.path.exists(file_path):
            os.remove(file_path)

        # 관련 청크 삭제
        await session.execute(
            text("DELETE FROM rag_chunks WHERE video_id = :video_id"),
            {"video_id": video_id}
        )

        # 동영상 레코드 삭제
        await session.execute(
            text("DELETE FROM rag_videos WHERE id = :id"),
            {"id": video_id}
        )
        await session.commit()

    return {"message": "동영상이 삭제되었습니다.", "id": video_id}


async def transcribe_video_task(video_id: str, file_path: str):
    """동영상 트랜스크립션 백그라운드 작업"""
    try:
        async with get_db() as session:
            await session.execute(
                text("UPDATE rag_videos SET status = 'processing' WHERE id = :id"),
                {"id": video_id}
            )
            await session.commit()

        # TODO: 실제 트랜스크립션 로직 구현
        # 1. Whisper API 또는 로컬 모델로 음성 인식
        # 2. 텍스트 청크 분할
        # 3. 임베딩 생성
        # 4. 벡터 DB 저장

        # 임시: 성공으로 표시
        async with get_db() as session:
            await session.execute(text("""
                UPDATE rag_videos
                SET status = 'indexed', processed_at = :processed_at, transcript_length = 0
                WHERE id = :id
            """), {"id": video_id, "processed_at": datetime.now()})
            await session.commit()

    except Exception as e:
        async with get_db() as session:
            await session.execute(text("""
                UPDATE rag_videos
                SET status = 'error', error_message = :error
                WHERE id = :id
            """), {"id": video_id, "error": str(e)})
            await session.commit()


# ============================================================================
# 크롤링 사이트 관리 API
# ============================================================================

@router.get("/crawl-sites", response_model=List[CrawlSiteResponse])
async def list_crawl_sites(
    request: Request,
    page: int = 1,
    limit: int = 50,
    status: Optional[str] = None,
    admin_info: dict = Depends(verify_admin)
):
    """크롤링 사이트 목록 조회"""
    # tenant removed
    offset = (page - 1) * limit

    async with get_db() as session:
        query = """
            SELECT id, url, name, description, crawl_depth, crawl_frequency,
                   status, last_crawl_at, next_crawl_at, pages_crawled, error_message, created_at
            FROM crawl_sites
        """
        params = {"limit": limit, "offset": offset}

        if status:
            query += " WHERE status = :status"
            params["status"] = status

        query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"

        result = await session.execute(text(query), params)
        sites = result.mappings().all()

        return [CrawlSiteResponse(**dict(s)) for s in sites]


@router.post("/crawl-sites", response_model=CrawlSiteResponse)
async def create_crawl_site(
    request: Request,
    site: CrawlSiteRequest,
    admin_info: dict = Depends(verify_admin)
):
    """크롤링 사이트 등록"""
    # tenant removed
    site_id = str(uuid.uuid4())
    now = datetime.now()

    async with get_db() as session:
        # URL 중복 체크
        result = await session.execute(
            text("SELECT id FROM crawl_sites WHERE url = :url"),
            {"url": site.url}
        )
        if result.first():
            raise HTTPException(status_code=400, detail="이미 등록된 URL입니다.")

        await session.execute(text("""
            INSERT INTO crawl_sites (id, url, name, description, crawl_depth, crawl_frequency, status, created_at)
            VALUES (:id, :url, :name, :description, :crawl_depth, :crawl_frequency, :status, :created_at)
        """), {
            "id": site_id,
            "url": site.url,
            "name": site.name,
            "description": site.description,
            "crawl_depth": site.crawl_depth,
            "crawl_frequency": site.crawl_frequency,
            "status": "active",
            "created_at": now
        })
        await session.commit()

    return CrawlSiteResponse(
        id=site_id,
        url=site.url,
        name=site.name,
        description=site.description,
        crawl_depth=site.crawl_depth,
        crawl_frequency=site.crawl_frequency,
        status="active",
        pages_crawled=0,
        created_at=now
    )


@router.put("/crawl-sites/{site_id}")
async def update_crawl_site(
    request: Request,
    site_id: str,
    site: CrawlSiteRequest,
    admin_info: dict = Depends(verify_admin)
):
    """크롤링 사이트 수정"""
    # tenant removed

    async with get_db() as session:
        result = await session.execute(
            text("SELECT id FROM crawl_sites WHERE id = :id"),
            {"id": site_id}
        )
        if not result.first():
            raise HTTPException(status_code=404, detail="사이트를 찾을 수 없습니다.")

        await session.execute(text("""
            UPDATE crawl_sites
            SET url = :url, name = :name, description = :description,
                crawl_depth = :crawl_depth, crawl_frequency = :crawl_frequency
            WHERE id = :id
        """), {
            "id": site_id,
            "url": site.url,
            "name": site.name,
            "description": site.description,
            "crawl_depth": site.crawl_depth,
            "crawl_frequency": site.crawl_frequency
        })
        await session.commit()

    return {"message": "사이트가 수정되었습니다.", "id": site_id}


@router.delete("/crawl-sites/{site_id}")
async def delete_crawl_site(
    request: Request,
    site_id: str,
    admin_info: dict = Depends(verify_admin)
):
    """크롤링 사이트 삭제"""
    # tenant removed

    async with get_db() as session:
        result = await session.execute(
            text("SELECT id FROM crawl_sites WHERE id = :id"),
            {"id": site_id}
        )
        if not result.first():
            raise HTTPException(status_code=404, detail="사이트를 찾을 수 없습니다.")

        # 관련 청크 삭제
        await session.execute(
            text("DELETE FROM rag_chunks WHERE crawl_site_id = :site_id"),
            {"site_id": site_id}
        )

        # 사이트 삭제
        await session.execute(
            text("DELETE FROM crawl_sites WHERE id = :id"),
            {"id": site_id}
        )
        await session.commit()

    return {"message": "사이트가 삭제되었습니다.", "id": site_id}


@router.post("/crawl-sites/{site_id}/pause")
async def pause_crawl_site(
    request: Request,
    site_id: str,
    admin_info: dict = Depends(verify_admin)
):
    """크롤링 일시정지"""
    # tenant removed

    async with get_db() as session:
        await session.execute(
            text("UPDATE crawl_sites SET status = 'paused' WHERE id = :id"),
            {"id": site_id}
        )
        await session.commit()

    return {"message": "크롤링이 일시정지되었습니다.", "id": site_id}


@router.post("/crawl-sites/{site_id}/resume")
async def resume_crawl_site(
    request: Request,
    site_id: str,
    admin_info: dict = Depends(verify_admin)
):
    """크롤링 재개"""
    # tenant removed

    async with get_db() as session:
        await session.execute(
            text("UPDATE crawl_sites SET status = 'active', error_message = NULL WHERE id = :id"),
            {"id": site_id}
        )
        await session.commit()

    return {"message": "크롤링이 재개되었습니다.", "id": site_id}


@router.post("/crawl-sites/{site_id}/crawl-now")
async def trigger_crawl(
    request: Request,
    site_id: str,
    background_tasks: BackgroundTasks,
    admin_info: dict = Depends(verify_admin)
):
    """즉시 크롤링 실행"""
    # tenant removed

    async with get_db() as session:
        result = await session.execute(
            text("SELECT url, crawl_depth FROM crawl_sites WHERE id = :id"),
            {"id": site_id}
        )
        site = result.mappings().first()

        if not site:
            raise HTTPException(status_code=404, detail="사이트를 찾을 수 없습니다.")

        await session.execute(
            text("UPDATE crawl_sites SET status = 'crawling' WHERE id = :id"),
            {"id": site_id}
        )
        await session.commit()

    background_tasks.add_task(crawl_site_task, site_id, site["url"], site["crawl_depth"])

    return {"message": "크롤링이 시작됩니다.", "id": site_id}


async def crawl_site_task(site_id: str, url: str, depth: int):
    """사이트 크롤링 백그라운드 작업"""
    try:
        # TODO: 실제 크롤링 로직 구현
        # 1. 웹 페이지 크롤링
        # 2. 콘텐츠 추출
        # 3. 텍스트 청크 분할
        # 4. 임베딩 생성
        # 5. 벡터 DB 저장

        # 임시: 성공으로 표시
        async with get_db() as session:
            await session.execute(text("""
                UPDATE crawl_sites
                SET status = 'active', last_crawl_at = :now, pages_crawled = pages_crawled + 1
                WHERE id = :id
            """), {"id": site_id, "now": datetime.now()})
            await session.commit()

    except Exception as e:
        async with get_db() as session:
            await session.execute(text("""
                UPDATE crawl_sites
                SET status = 'error', error_message = :error
                WHERE id = :id
            """), {"id": site_id, "error": str(e)})
            await session.commit()
