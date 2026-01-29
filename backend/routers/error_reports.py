"""
오류 리포트 API 라우터 (OSC 공개 버전)
사용자가 오류를 신고하고, 관리자가 확인할 수 있는 API
"""
from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from sqlalchemy import text
import uuid
import os
import base64
import logging

from database import get_db
from routers.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

# 첨부파일 저장 경로
UPLOAD_DIR = "/app/uploads/error_reports"


# ============================================================================
# Request/Response 모델
# ============================================================================

class ErrorReportCreate(BaseModel):
    """오류 리포트 생성 요청"""
    title: str = Field(..., min_length=1, max_length=200, description="오류 제목")
    description: str = Field(..., min_length=1, max_length=5000, description="오류 설명")
    error_type: str = Field(default="general", description="오류 유형: general, ui, api, data, other")
    browser_info: Optional[str] = Field(None, description="브라우저 정보")
    page_url: Optional[str] = Field(None, description="오류 발생 페이지 URL")


class ErrorReportAttachment(BaseModel):
    """첨부파일 정보"""
    filename: str
    file_type: str
    file_size: int
    file_url: str


class ErrorReportResponse(BaseModel):
    """오류 리포트 응답"""
    id: str
    title: str
    description: str
    error_type: str
    status: str
    user_id: int
    username: str
    browser_info: Optional[str]
    page_url: Optional[str]
    attachments: List[dict]
    created_at: datetime
    updated_at: datetime
    admin_response: Optional[str]


class ErrorReportUpdate(BaseModel):
    """오류 리포트 상태 업데이트 (관리자용)"""
    status: Optional[str] = Field(None, description="상태: pending, in_progress, resolved, closed")
    admin_response: Optional[str] = Field(None, description="관리자 응답")


# ============================================================================
# 데이터베이스 초기화
# ============================================================================

async def ensure_error_reports_table():
    """오류 리포트 테이블이 없으면 생성"""
    async with get_db() as session:
        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS error_reports (
                id VARCHAR(100) PRIMARY KEY,
                title VARCHAR(200) NOT NULL,
                description TEXT NOT NULL,
                error_type VARCHAR(50) DEFAULT 'general',
                status VARCHAR(50) DEFAULT 'pending',
                user_id INTEGER NOT NULL,
                username VARCHAR(100) NOT NULL,
                browser_info TEXT,
                page_url TEXT,
                attachments JSONB DEFAULT '[]',
                admin_response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # 인덱스 생성
        await session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_error_reports_status ON error_reports(status)
        """))
        await session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_error_reports_user_id ON error_reports(user_id)
        """))
        await session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_error_reports_created_at ON error_reports(created_at DESC)
        """))


# ============================================================================
# 사용자 API - 오류 신고
# ============================================================================

@router.post("/reports", response_model=dict)
async def create_error_report(
    report: ErrorReportCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    오류 리포트 생성 (사용자)
    """
    user_id = current_user["id"]
    username = current_user.get("username", "unknown")

    # 테이블 확인
    await ensure_error_reports_table()

    report_id = str(uuid.uuid4())

    async with get_db() as session:
        await session.execute(text("""
            INSERT INTO error_reports (id, title, description, error_type, user_id, username, browser_info, page_url)
            VALUES (:id, :title, :description, :error_type, :user_id, :username, :browser_info, :page_url)
        """), {
            "id": report_id,
            "title": report.title,
            "description": report.description,
            "error_type": report.error_type,
            "user_id": user_id,
            "username": username,
            "browser_info": report.browser_info,
            "page_url": report.page_url
        })

    logger.info(f"Error report created: {report_id} by user {username}")

    return {
        "message": "오류 리포트가 등록되었습니다.",
        "report_id": report_id
    }


@router.post("/reports/{report_id}/attachments")
async def upload_attachment(
    report_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    오류 리포트에 첨부파일 업로드
    """
    user_id = current_user["id"]

    # 파일 크기 제한 (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="파일 크기는 10MB를 초과할 수 없습니다.")

    # 리포트 존재 및 권한 확인
    async with get_db() as session:
        result = await session.execute(
            text("SELECT * FROM error_reports WHERE id = :id AND user_id = :user_id"),
            {"id": report_id, "user_id": user_id}
        )
        report = result.mappings().first()
        if not report:
            raise HTTPException(status_code=404, detail="리포트를 찾을 수 없습니다.")

    # 파일 저장
    upload_dir = os.path.join(UPLOAD_DIR, report_id)
    os.makedirs(upload_dir, exist_ok=True)

    file_ext = os.path.splitext(file.filename)[1] if file.filename else ".bin"
    file_id = str(uuid.uuid4())[:8]
    saved_filename = f"{file_id}{file_ext}"
    file_path = os.path.join(upload_dir, saved_filename)

    with open(file_path, "wb") as f:
        f.write(contents)

    # DB 업데이트
    attachment_info = {
        "id": file_id,
        "filename": file.filename,
        "saved_filename": saved_filename,
        "file_type": file.content_type,
        "file_size": len(contents),
        "uploaded_at": datetime.now().isoformat()
    }

    async with get_db() as session:
        await session.execute(text("""
            UPDATE error_reports
            SET attachments = attachments || :attachment::jsonb,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :id
        """), {
            "id": report_id,
            "attachment": f'[{{"id": "{file_id}", "filename": "{file.filename}", "saved_filename": "{saved_filename}", "file_type": "{file.content_type}", "file_size": {len(contents)}}}]'
        })

    return {
        "message": "파일이 업로드되었습니다.",
        "attachment": attachment_info
    }


@router.post("/reports/{report_id}/attachments/base64")
async def upload_attachment_base64(
    report_id: str,
    filename: str = Form(...),
    file_type: str = Form(...),
    data: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Base64 인코딩된 첨부파일 업로드 (클립보드 이미지 등)
    """
    user_id = current_user["id"]

    # Base64 디코딩
    try:
        # data:image/png;base64, 접두사 제거
        if "," in data:
            data = data.split(",")[1]
        contents = base64.b64decode(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail="잘못된 Base64 데이터입니다.")

    # 파일 크기 제한 (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="파일 크기는 10MB를 초과할 수 없습니다.")

    # 리포트 존재 및 권한 확인
    async with get_db() as session:
        result = await session.execute(
            text("SELECT * FROM error_reports WHERE id = :id AND user_id = :user_id"),
            {"id": report_id, "user_id": user_id}
        )
        report = result.mappings().first()
        if not report:
            raise HTTPException(status_code=404, detail="리포트를 찾을 수 없습니다.")

    # 파일 저장
    upload_dir = os.path.join(UPLOAD_DIR, report_id)
    os.makedirs(upload_dir, exist_ok=True)

    file_ext = os.path.splitext(filename)[1] if filename else ".png"
    file_id = str(uuid.uuid4())[:8]
    saved_filename = f"{file_id}{file_ext}"
    file_path = os.path.join(upload_dir, saved_filename)

    with open(file_path, "wb") as f:
        f.write(contents)

    # DB 업데이트
    async with get_db() as session:
        await session.execute(text("""
            UPDATE error_reports
            SET attachments = attachments || :attachment::jsonb,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :id
        """), {
            "id": report_id,
            "attachment": f'[{{"id": "{file_id}", "filename": "{filename}", "saved_filename": "{saved_filename}", "file_type": "{file_type}", "file_size": {len(contents)}}}]'
        })

    return {
        "message": "파일이 업로드되었습니다.",
        "attachment": {
            "id": file_id,
            "filename": filename,
            "saved_filename": saved_filename,
            "file_type": file_type,
            "file_size": len(contents)
        }
    }


@router.get("/reports/my")
async def get_my_reports(
    current_user: dict = Depends(get_current_user)
):
    """
    내가 작성한 오류 리포트 목록
    """
    user_id = current_user["id"]

    await ensure_error_reports_table()

    async with get_db() as session:
        result = await session.execute(text("""
            SELECT * FROM error_reports
            WHERE user_id = :user_id
            ORDER BY created_at DESC
        """), {"user_id": user_id})
        reports = [dict(row) for row in result.mappings().all()]

    return {"reports": reports}


# ============================================================================
# 관리자 API - 리포트 조회/관리
# ============================================================================

@router.get("/admin/reports")
async def get_all_reports(
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    모든 오류 리포트 조회 (관리자용)
    """
    # 관리자 권한 확인
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

    await ensure_error_reports_table()

    async with get_db() as session:
        if status:
            result = await session.execute(text("""
                SELECT * FROM error_reports
                WHERE status = :status
                ORDER BY created_at DESC
            """), {"status": status})
        else:
            result = await session.execute(text("""
                SELECT * FROM error_reports
                ORDER BY created_at DESC
            """))
        reports = [dict(row) for row in result.mappings().all()]

    return {"reports": reports, "total": len(reports)}


@router.get("/admin/reports/{report_id}")
async def get_report_detail(
    report_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    오류 리포트 상세 조회 (관리자용)
    """
    # 관리자 권한 확인
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

    async with get_db() as session:
        result = await session.execute(
            text("SELECT * FROM error_reports WHERE id = :id"),
            {"id": report_id}
        )
        report = result.mappings().first()

        if not report:
            raise HTTPException(status_code=404, detail="리포트를 찾을 수 없습니다.")

    return {"report": dict(report)}


@router.patch("/admin/reports/{report_id}")
async def update_report(
    report_id: str,
    update: ErrorReportUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    오류 리포트 상태 업데이트 (관리자용)
    """
    # 관리자 권한 확인
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

    updates = []
    params = {"id": report_id}

    if update.status:
        updates.append("status = :status")
        params["status"] = update.status

    if update.admin_response is not None:
        updates.append("admin_response = :admin_response")
        params["admin_response"] = update.admin_response

    if not updates:
        raise HTTPException(status_code=400, detail="업데이트할 내용이 없습니다.")

    updates.append("updated_at = CURRENT_TIMESTAMP")

    async with get_db() as session:
        result = await session.execute(
            text(f"UPDATE error_reports SET {', '.join(updates)} WHERE id = :id RETURNING *"),
            params
        )
        updated = result.mappings().first()

        if not updated:
            raise HTTPException(status_code=404, detail="리포트를 찾을 수 없습니다.")

    logger.info(f"Error report updated: {report_id} by admin {current_user.get('username')}")

    return {"message": "리포트가 업데이트되었습니다.", "report": dict(updated)}


@router.get("/admin/reports/{report_id}/attachment/{attachment_id}")
async def get_attachment(
    report_id: str,
    attachment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    첨부파일 다운로드 URL 반환
    """
    # 관리자 권한 확인
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

    async with get_db() as session:
        result = await session.execute(
            text("SELECT attachments FROM error_reports WHERE id = :id"),
            {"id": report_id}
        )
        row = result.mappings().first()

        if not row:
            raise HTTPException(status_code=404, detail="리포트를 찾을 수 없습니다.")

        attachments = row.get("attachments", [])
        attachment = next((a for a in attachments if a.get("id") == attachment_id), None)

        if not attachment:
            raise HTTPException(status_code=404, detail="첨부파일을 찾을 수 없습니다.")

    file_path = os.path.join(UPLOAD_DIR, report_id, attachment.get("saved_filename", ""))

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일이 존재하지 않습니다.")

    return {
        "file_path": file_path,
        "filename": attachment.get("filename"),
        "file_type": attachment.get("file_type")
    }
