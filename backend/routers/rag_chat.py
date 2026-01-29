"""
Advisor RAG 채팅 API 라우터 (OSC 공개 버전)
MH_rag baseline_v1 기반 RAG 전용 채팅
- LLM 직접 호출 없음 (모든 질문은 RAG 파이프라인 통과)
- 모델 선택 UI 없음 (백엔드에서 고정 모델 사용)
"""
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse, Response
from pydantic import BaseModel
from typing import Optional, List
import uuid
import json
import logging
import asyncio
import os
import urllib.parse
import aiofiles
import base64
import httpx

from routers.auth import get_current_user
from database import get_db
from sqlalchemy import text

# RAG 시스템
from rag.mh_rag import MultimodalHierarchicalRAG
from rag.chain import ChatMessage
from config.settings import Settings

# PDF 하이라이트 모듈
from rag.utils import PDFPageExtractor, get_keywords_string

logger = logging.getLogger(__name__)

# 개인화 메모리 서비스
try:
    from services.memory_service import get_memory_service
except ImportError:
    get_memory_service = None
    logger.warning("Memory service not available")

router = APIRouter()

# RAG 시스템 싱글톤
_rag_system = None  # 단일 RAG 시스템

# 기본 컬렉션 (레거시 호환)
MH_RAG_COLLECTION = "mh_rag_finance_v7_6_azure"

# ============================================================================
# 이미지 분석 (Vision API)
# ============================================================================
async def analyze_image_with_vision(file_path: str, filename: str) -> str:
    """OpenAI GPT-4 Vision API를 사용하여 이미지 내용 분석"""
    try:
        # 이미지 파일 읽기 및 base64 인코딩
        async with aiofiles.open(file_path, 'rb') as f:
            image_data = await f.read()

        base64_image = base64.b64encode(image_data).decode('utf-8')

        # 파일 확장자로 MIME 타입 결정
        ext = os.path.splitext(filename)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }
        mime_type = mime_types.get(ext, 'image/png')

        # OpenAI API 호출
        settings = Settings()
        api_key = settings.OPENAI_API_KEY

        if not api_key:
            logger.warning("OpenAI API key not found for vision analysis")
            return "(이미지 분석 불가 - API 키 없음)"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "이 이미지의 내용을 상세하게 설명해주세요. 텍스트가 있다면 모두 추출해주세요. 표가 있다면 표 내용도 정리해주세요."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 2000
                }
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                logger.info(f"Vision analysis completed: {len(content)} chars")
                return content
            else:
                logger.warning(f"Vision API error: {response.status_code} - {response.text}")
                return f"(이미지 분석 실패: {response.status_code})"

    except Exception as e:
        logger.error(f"Vision analysis error: {e}")
        return f"(이미지 분석 오류: {str(e)})"

# ============================================================================
# 파일 업로드 설정
# ============================================================================
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

ALLOWED_EXTENSIONS = {
    # 이미지
    '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp',
    # 문서
    '.pdf', '.doc', '.docx', '.txt', '.rtf',
    # 스프레드시트
    '.xls', '.xlsx', '.csv',
}

def is_allowed_file(filename: str) -> bool:
    """허용된 파일 확장자인지 확인"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def get_file_type(filename: str) -> str:
    """파일 유형 반환"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}:
        return 'image'
    elif ext == '.pdf':
        return 'pdf'
    elif ext in {'.doc', '.docx', '.txt', '.rtf'}:
        return 'document'
    elif ext in {'.xls', '.xlsx', '.csv'}:
        return 'spreadsheet'
    return 'other'


async def get_rag_system() -> MultimodalHierarchicalRAG:
    """RAG 시스템 반환 (싱글톤)"""
    global _rag_system

    if _rag_system is None:
        logger.info(f"Initializing MH_rag system (collection: {MH_RAG_COLLECTION})...")
        settings = Settings()
        settings.qdrant_collection_name = MH_RAG_COLLECTION
        _rag_system = MultimodalHierarchicalRAG(settings=settings)
        await _rag_system.initialize()
        logger.info(f"MH_rag system initialized for collection: {MH_RAG_COLLECTION}")

    return _rag_system


# ============================================================================
# 데이터베이스 함수들
# ============================================================================

async def create_session(session_id: str, user_id: int, title: str = None) -> str:
    """새 대화 세션 생성 (RAG 전용 - 모델 고정)"""
    async with get_db() as session:
        await session.execute(
            text("""
                INSERT INTO sessions (id, user_id, title, model)
                VALUES (:id, :user_id, :title, 'rag')
            """),
            {"id": session_id, "user_id": user_id, "title": title}
        )
        return session_id


async def get_session(session_id: str) -> dict:
    """세션 조회"""
    async with get_db() as session:
        result = await session.execute(
            text("SELECT * FROM sessions WHERE id = :id"),
            {"id": session_id}
        )
        row = result.mappings().first()
        return dict(row) if row else None


async def get_user_sessions(user_id: int) -> list:
    """사용자의 모든 세션 조회"""
    async with get_db() as session:
        result = await session.execute(
            text("""
                SELECT id, title, created_at, updated_at
                FROM sessions
                WHERE user_id = :user_id
                ORDER BY updated_at DESC
            """),
            {"user_id": user_id}
        )
        return [dict(row) for row in result.mappings().all()]


async def update_session_title(session_id: str, title: str):
    """세션 제목 업데이트"""
    async with get_db() as session:
        await session.execute(
            text("UPDATE sessions SET title = :title, updated_at = NOW() WHERE id = :id"),
            {"title": title, "id": session_id}
        )


async def delete_session(session_id: str):
    """세션 삭제"""
    async with get_db() as session:
        await session.execute(
            text("DELETE FROM messages WHERE session_id = :id"),
            {"id": session_id}
        )
        await session.execute(
            text("DELETE FROM sessions WHERE id = :id"),
            {"id": session_id}
        )


async def add_message(session_id: str, role: str, content: str) -> int:
    """메시지 추가"""
    async with get_db() as session:
        result = await session.execute(
            text("""
                INSERT INTO messages (session_id, role, content)
                VALUES (:session_id, :role, :content)
                RETURNING id
            """),
            {"session_id": session_id, "role": role, "content": content}
        )
        await session.execute(
            text("UPDATE sessions SET updated_at = NOW() WHERE id = :id"),
            {"id": session_id}
        )
        return result.scalar()


async def get_session_messages(session_id: str) -> list:
    """세션의 모든 메시지 조회"""
    async with get_db() as session:
        result = await session.execute(
            text("""
                SELECT id, role, content, created_at
                FROM messages
                WHERE session_id = :session_id
                ORDER BY created_at ASC
            """),
            {"session_id": session_id}
        )
        return [dict(row) for row in result.mappings().all()]


# ============================================================================
# Request/Response 모델
# ============================================================================

class RAGChatRequest(BaseModel):
    """RAG 채팅 요청 (모델 선택 없음)"""
    message: str
    session_id: Optional[str] = None
    attachment_ids: Optional[List[int]] = None  # 첨부파일 ID 목록


class SessionRenameRequest(BaseModel):
    title: str


# ============================================================================
# RAG 채팅 API
# ============================================================================

@router.post("/send")
async def send_rag_message(request: RAGChatRequest, current_user: dict = Depends(get_current_user)):
    """
    RAG 기반 채팅 메시지 전송 (스트리밍 응답)

    모든 질문은 MH_rag 파이프라인을 통해 처리됩니다.
    - 벡터 검색으로 관련 문서 검색
    - 컨텍스트 기반 답변 생성
    """
    user_id = current_user["id"]
    

    # 세션 생성 또는 확인
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        await create_session(session_id, user_id)
    else:
        session = await get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        if session["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="접근 권한이 없습니다")

    # 첨부파일 내용 로드
    attachment_context = ""
    logger.info(f"[Chat] User {user_id}: attachment_ids={request.attachment_ids}")
    if request.attachment_ids:
        try:
            async with get_db() as db_session:
                for att_id in request.attachment_ids:
                    result = await db_session.execute(
                        text("SELECT filename, original_filename, file_type, file_path, user_id FROM attachments WHERE id = :id"),
                        {"id": att_id}
                    )
                    row = result.fetchone()
                    logger.info(f"[Chat] Attachment {att_id}: row={row}, row.user_id={row.user_id if row else None}, user_id={user_id}")
                    if row:
                        file_path = row.file_path
                        logger.info(f"[Chat] Attachment file_path={file_path}, exists={os.path.exists(file_path)}, file_type={row.file_type}")
                        if os.path.exists(file_path):
                            # 텍스트 파일인 경우 내용 읽기
                            if row.file_type in ['document', 'code'] or row.original_filename.endswith(('.txt', '.md', '.json', '.csv')):
                                try:
                                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                                        content = await f.read()
                                        if len(content) > 10000:
                                            content = content[:10000] + "\n... (이하 생략)"
                                        attachment_context += f"\n\n[첨부파일: {row.original_filename}]\n{content}"
                                        logger.info(f"[Chat] Attachment loaded: {row.original_filename}, {len(content)} chars")
                                except Exception as read_err:
                                    logger.warning(f"[Chat] Attachment read error: {read_err}")
                                    attachment_context += f"\n\n[첨부파일: {row.original_filename}] (읽기 불가)"
                            else:
                                # 이미지 파일은 Vision API로 분석
                                if row.file_type == 'image' or row.original_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                                    try:
                                        logger.info(f"[Chat] Analyzing image with Vision API: {row.original_filename}")
                                        image_analysis = await analyze_image_with_vision(file_path, row.original_filename)
                                        attachment_context += f"\n\n[첨부파일: {row.original_filename}] (이미지 분석 결과)\n{image_analysis}"
                                        logger.info(f"[Chat] Image analysis completed: {row.original_filename}, {len(image_analysis)} chars")
                                    except Exception as vision_err:
                                        logger.warning(f"[Chat] Vision API error: {vision_err}")
                                        attachment_context += f"\n\n[첨부파일: {row.original_filename}] (이미지 파일 - 분석 실패: {str(vision_err)[:100]})"
                                else:
                                    # 기타 파일
                                    attachment_context += f"\n\n[첨부파일: {row.original_filename}] (지원되지 않는 파일 형식)"
                                    logger.info(f"[Chat] Unsupported attachment: {row.original_filename}")
        except Exception as e:
            logger.warning(f"Failed to load attachments: {e}")

    # 사용자 메시지 저장 (첨부파일 정보 포함)
    message_to_save = request.message
    if attachment_context:
        message_to_save += attachment_context
    await add_message(session_id, "user", message_to_save)

    # 첫 메시지인 경우 세션 제목 설정
    messages = await get_session_messages(session_id)
    if len(messages) == 1:
        title = request.message[:30] + ("..." if len(request.message) > 30 else "")
        await update_session_title(session_id, title)

    # 변수 캡처
    
    _session_id = session_id
    _user_id = current_user.get("username", str(user_id))  # username 사용 (메모리 서비스 호환)
    _user_query = request.message
    _attachment_context = attachment_context  # 첨부파일 내용
    _messages = messages

    async def generate():
        """RAG 스트리밍 응답 생성 (실시간 LLM 스트리밍 + 개인화 메모리)"""
        try:
            # RAG 시스템 가져오기 (테넌트별 컬렉션)
            rag = await get_rag_system()

            # ============ 개인화 메모리 컨텍스트 조회 ============
            personalized_context = ""
            memory_svc = None
            if get_memory_service:
                try:
                    memory_svc = await get_memory_service()
                    if memory_svc and memory_svc.is_available():
                        personalized_context = await memory_svc.get_personalized_context(
                            
                            user_id=_user_id,
                            current_query=_user_query
                        )
                        if personalized_context:
                            logger.info(f"[Chat] User {_user_id}: Personalized context loaded ({len(personalized_context)} chars)")
                except Exception as e:
                    logger.warning(f"Personalization context retrieval failed: {e}")

            # 대화 히스토리 구성 (개인화 컨텍스트를 시스템 메시지로 추가)
            conversation_history = []

            # 개인화 컨텍스트가 있으면 시스템 메시지로 추가
            if personalized_context:
                conversation_history.append(
                    ChatMessage(role="system", content=f"다음은 이 사용자에 대해 이전에 알게 된 정보입니다. 답변 시 참고하세요:\n{personalized_context}")
                )

            # 첨부파일 컨텍스트가 있으면 시스템 메시지로 추가
            if _attachment_context:
                conversation_history.append(
                    ChatMessage(role="system", content=f"사용자가 첨부한 파일의 내용입니다. 질문에 답변할 때 이 내용을 참고하세요:{_attachment_context}")
                )
                logger.info(f"[Chat] Attachment context added to conversation: {len(_attachment_context)} chars")

            # 기존 대화 히스토리 추가
            for msg in _messages[:-1]:  # 현재 메시지 제외
                conversation_history.append(
                    ChatMessage(role=msg["role"], content=msg["content"])
                )

            # 먼저 검색 수행하여 소스 정보 획득
            search_result = await rag.search(
                query=_user_query,
                top_k=5
            )

            # 소스 정보 추출 (키워드 포함 - 하이라이팅용)
            # PDF 하이라이트 모듈의 키워드 추출기 사용
            keywords_str = get_keywords_string(_user_query)

            sources = []
            if search_result.sources:
                # 점수 기반 필터링: 관련도 낮은 문서 제외
                top_score = None
                for idx, s in enumerate(search_result.sources[:5]):  # 최대 5개 검토
                    if isinstance(s, dict):
                        score = s.get("score", 0.0)
                        source_info = {
                            "source": s.get("source", ""),
                            "page": s.get("page", 0),
                            "score": score,
                            "keywords": keywords_str
                        }
                    else:
                        score = s.score
                        source_info = {
                            "source": s.source,
                            "page": s.page,
                            "score": score,
                            "keywords": keywords_str
                        }

                    # 첫 번째 결과의 점수 저장
                    if top_score is None:
                        top_score = score

                    # 필터링 조건:
                    # 1. 절대 점수 0.5 이상
                    # 2. 1위 점수 대비 80% 이상
                    if score >= 0.5 and (top_score == 0 or score >= top_score * 0.8):
                        sources.append(source_info)
                        if len(sources) >= 3:  # 최대 3개
                            break

            # 실시간 LLM 스트리밍 응답 생성 (개인화 컨텍스트 포함)
            full_response = ""
            async for chunk in rag.chat_stream(
                query=_user_query,
                conversation_history=conversation_history,
            ):
                full_response += chunk
                yield f"data: {json.dumps({'content': chunk, 'done': False, 'session_id': _session_id}, ensure_ascii=False)}\n\n"

            # 어시스턴트 응답 저장
            message_id = await add_message(_session_id, "assistant", full_response)

            # ============ 개인화 메모리 저장 (백그라운드) ============
            if memory_svc and memory_svc.is_available():
                try:
                    conversation_text = f"User: {_user_query}\nAssistant: {full_response[:500]}"
                    asyncio.create_task(
                        memory_svc.store_memory(
                            
                            user_id=_user_id,
                            text=conversation_text,
                            category="history"
                        )
                    )
                    logger.debug(f"[Chat] User {_user_id}: Memory storage scheduled")
                except Exception as e:
                    logger.warning(f"Memory storage failed: {e}")

            yield f"data: {json.dumps({'content': '', 'done': True, 'session_id': _session_id, 'message_id': message_id, 'sources': sources}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"RAG chat error: {e}")
            yield f"data: {json.dumps({'error': str(e), 'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/sessions")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    """사용자의 대화 세션 목록 조회"""
    
    sessions = await get_user_sessions(current_user["id"])
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session_detail(session_id: str, current_user: dict = Depends(get_current_user)):
    """세션 상세 조회 (메시지 포함)"""
    
    session = await get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    if session["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다")

    messages = await get_session_messages(session_id)

    return {
        "session": session,
        "messages": messages
    }


@router.put("/sessions/{session_id}")
async def rename_session(
    session_id: str,
    request: SessionRenameRequest,
    current_user: dict = Depends(get_current_user)
):
    """세션 이름 변경"""
    
    session = await get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    if session["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다")

    await update_session_title(session_id, request.title)
    return {"status": "updated"}


@router.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str, current_user: dict = Depends(get_current_user)):
    """세션 삭제"""
    
    session = await get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    if session["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다")

    await delete_session(session_id)
    return {"status": "deleted"}


# ============================================================================
# RAG 검색 API (LLM 응답 없이 검색만)
# ============================================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/search")
async def search_documents(request: SearchRequest, current_user: dict = Depends(get_current_user)):
    """
    문서 검색만 수행 (LLM 응답 없음)

    RAG 파이프라인의 검색 단계만 실행하여
    관련 문서를 반환합니다.
    """
    
    try:
        rag = await get_rag_system()
        result = await rag.search(
            query=request.query,
            top_k=request.top_k
        )

        return {
            "query": request.query,
            "results": [
                {
                    "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                    "source": r.metadata.get("source", "unknown"),
                    "page": r.metadata.get("page", 0),
                    "score": r.score
                }
                for r in result.child_results
            ]
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.api_route("/pdf/{domain}/{filename:path}", methods=["GET", "HEAD"])
async def get_pdf_file(domain: str, filename: str):
    """
    PDF 파일 제공 (공개 API - 인증 불필요)

    RAG 소스의 PDF 파일을 제공합니다.
    예: /api/chat/pdf/finance/은행법.pdf

    설정 파일(config/settings.py)에서 PDF_BASE_PATH, PDF_VALID_DOMAINS 설정 가능

    Note: 새 탭에서 PDF를 열기 위해 인증 없이 접근 가능하도록 설정.
    도메인/파일명 검증으로 보안 유지.
    """
    settings = Settings()

    # 설정에서 PDF 기본 경로 및 허용 도메인 가져오기
    pdf_base_path = settings.pdf_base_path
    valid_domains = [d.strip() for d in settings.pdf_valid_domains.split(",")]

    # 도메인 검증
    if domain not in valid_domains:
        raise HTTPException(status_code=400, detail=f"Invalid domain: {domain}. Allowed: {valid_domains}")

    # 파일명 URL 디코딩
    decoded_filename = urllib.parse.unquote(filename)

    # 파일 경로 구성
    file_path = os.path.join(pdf_base_path, domain, decoded_filename)

    # 경로 탐색 공격 방지
    real_path = os.path.realpath(file_path)
    if not real_path.startswith(os.path.realpath(pdf_base_path)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(file_path):
        logger.warning(f"PDF not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"PDF not found: {decoded_filename}")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename=\"{decoded_filename}\"",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges"
        }
    )


@router.api_route("/pdf-pages/{domain}/{filename:path}", methods=["GET", "HEAD"])
async def get_pdf_pages(
    domain: str,
    filename: str,
    page: int = 1,
    range_pages: int = 3,  # 앞뒤 포함 총 페이지 수 (기본 3페이지)
    highlight: str = None  # 하이라이트할 텍스트
):
    """
    PDF 특정 페이지 범위만 추출하여 제공 (빠른 로딩용 + 하이라이팅)

    예: /api/chat/pdf-pages/finance/은행법.pdf?page=38&range_pages=3&highlight=검색어
    → 37, 38, 39 페이지만 추출하고 "검색어"를 하이라이트하여 전송

    Args:
        domain: PDF 도메인 (finance, medical 등)
        filename: PDF 파일명
        page: 중심 페이지 번호 (1-based)
        range_pages: 추출할 총 페이지 수 (기본 3)
        highlight: 하이라이트할 텍스트 (URL 인코딩됨, 파이프 구분)
    """
    settings = Settings()
    pdf_base_path = settings.pdf_base_path
    valid_domains = [d.strip() for d in settings.pdf_valid_domains.split(",")]

    if domain not in valid_domains:
        raise HTTPException(status_code=400, detail=f"Invalid domain: {domain}")

    decoded_filename = urllib.parse.unquote(filename)
    file_path = os.path.join(pdf_base_path, domain, decoded_filename)

    real_path = os.path.realpath(file_path)
    if not real_path.startswith(os.path.realpath(pdf_base_path)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"PDF not found: {decoded_filename}")

    try:
        # PDF 페이지 추출 및 하이라이트 모듈 사용
        extractor = PDFPageExtractor()
        output, metadata = extractor.extract_pages(
            pdf_path=file_path,
            center_page=page,
            range_pages=range_pages,
            highlight_text=highlight
        )

        # 한글 파일명 인코딩 (RFC 5987)
        encoded_filename = urllib.parse.quote(decoded_filename, safe='')
        return Response(
            content=output.read(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename*=UTF-8''{encoded_filename}",
                "Access-Control-Allow-Origin": "*",
                "X-Total-Pages": str(metadata["total_pages"]),
                "X-Start-Page": str(metadata["start_page"]),
                "X-End-Page": str(metadata["end_page"]),
                "X-Requested-Page": str(metadata["requested_page"])
            }
        )
    except Exception as e:
        logger.error(f"PDF page extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 파일 업로드 API
# ============================================================================

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """
    파일 업로드 (이미지, 문서 등)

    클립보드 붙여넣기, 드래그앤드롭, 파일 선택 모두 지원
    """
    user_id = current_user["id"]
    

    # 파일 검증
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다")

    if not is_allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="허용되지 않는 파일 형식입니다")

    # 세션이 없으면 임시 세션 ID 생성
    if not session_id:
        session_id = f"temp_{str(uuid.uuid4())}"

    # 사용자별 디렉토리 생성
    user_upload_dir = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(user_upload_dir, exist_ok=True)

    # 고유 파일명 생성
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(user_upload_dir, unique_filename)

    # 스트리밍 방식으로 파일 저장
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB 청크

    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                file_size += len(chunk)
                # 파일 크기 체크
                if file_size > MAX_FILE_SIZE:
                    await out_file.close()
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=400,
                        detail=f"파일 크기가 {MAX_FILE_SIZE // (1024*1024)}MB를 초과합니다"
                    )
                await out_file.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {str(e)}")

    # DB에 기록 (attachments 테이블)
    file_type = get_file_type(file.filename)
    try:
        async with get_db() as session:
            result = await session.execute(
                text("""
                    INSERT INTO attachments (session_id, user_id, filename, original_filename, file_type, file_size, file_path)
                    VALUES (:session_id, :user_id, :filename, :original_filename, :file_type, :file_size, :file_path)
                    RETURNING id
                """),
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "filename": unique_filename,
                    "original_filename": file.filename,
                    "file_type": file_type,
                    "file_size": file_size,
                    "file_path": file_path
                }
            )
            attachment_id = result.scalar()
    except Exception as e:
        logger.warning(f"DB 기록 실패 (테이블 없을 수 있음): {e}")
        attachment_id = str(uuid.uuid4())  # DB 없으면 임시 ID

    logger.info(f"파일 업로드 완료: {file.filename} -> {unique_filename}")

    return {
        "id": attachment_id,
        "filename": unique_filename,
        "original_filename": file.filename,
        "file_type": file_type,
        "file_size": file_size,
        "session_id": session_id
    }
