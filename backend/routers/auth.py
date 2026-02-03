"""
인증 API 라우터 (OSC 공개 버전)
회원가입, 로그인, JWT 토큰 관리
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional
import bcrypt
import jwt
import os
import logging

from sqlalchemy import text
from database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

# JWT 설정
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24


# Request/Response 모델
class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=4)
    display_name: Optional[str] = None
    email: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    id: int
    username: str
    display_name: str
    role: str
    created_at: str


# 비밀번호 해싱 (bcrypt - 보안 강화)
def hash_password(password: str) -> str:
    """비밀번호를 bcrypt로 해싱 (보안 강화)"""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """비밀번호 검증 (bcrypt)"""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except Exception:
        # SHA256 해시 호환성 (기존 사용자 마이그레이션용)
        import hashlib
        if len(password_hash) == 64:  # SHA256 해시 길이
            return hashlib.sha256(password.encode()).hexdigest() == password_hash
        return False


# JWT 토큰 생성/검증
def create_access_token(user_id: int, username: str, role: str = "user") -> str:
    """JWT 액세스 토큰 생성"""
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    """JWT 토큰 검증"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="토큰이 만료되었습니다")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")


# 사용자 조회 함수
async def get_user_by_username(username: str) -> Optional[dict]:
    """username으로 사용자 조회"""
    try:
        async with get_db() as session:
            result = await session.execute(
                text("SELECT * FROM users WHERE username = :username"),
                {"username": username}
            )
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        return None


async def get_user_by_id(user_id: int) -> Optional[dict]:
    """ID로 사용자 조회"""
    try:
        async with get_db() as session:
            result = await session.execute(
                text("SELECT * FROM users WHERE id = :id"),
                {"id": user_id}
            )
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"Error getting user by id: {e}")
        return None


async def create_user(username: str, password_hash: str, display_name: str = None, email: str = None) -> int:
    """새 사용자 생성"""
    async with get_db() as session:
        result = await session.execute(
            text("""
                INSERT INTO users (username, password_hash, display_name, email, role)
                VALUES (:username, :password_hash, :display_name, :email, 'user')
                RETURNING id
            """),
            {
                "username": username,
                "password_hash": password_hash,
                "display_name": display_name or username,
                "email": email
            }
        )
        user_id = result.scalar()

        # 기본 설정 생성
        await session.execute(
            text("INSERT INTO user_preferences (user_id) VALUES (:user_id)"),
            {"user_id": user_id}
        )
        return user_id


async def get_user_preferences(user_id: int) -> Optional[dict]:
    """사용자 설정 조회"""
    try:
        async with get_db() as session:
            result = await session.execute(
                text("SELECT * FROM user_preferences WHERE user_id = :user_id"),
                {"user_id": user_id}
            )
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        return None


async def update_last_login(user_id: int):
    """마지막 로그인 시간 업데이트"""
    try:
        async with get_db() as session:
            await session.execute(
                text("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = :id"),
                {"id": user_id}
            )
    except Exception as e:
        logger.error(f"Error updating last login: {e}")


# 의존성: 현재 사용자 가져오기
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """현재 로그인된 사용자 정보 반환"""
    token = credentials.credentials
    payload = verify_token(token)

    user_id = int(payload.get("sub"))
    user = await get_user_by_id(user_id)

    if not user:
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없습니다")

    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="비활성화된 계정입니다")

    return user


# API 엔드포인트
@router.post("/signup", response_model=TokenResponse)
async def signup(request: SignupRequest):
    """회원가입"""
    # 입력값 trim 처리
    username = request.username.strip() if request.username else ""
    password = request.password.strip() if request.password else ""
    display_name = request.display_name.strip() if request.display_name else None
    email = request.email.strip() if request.email else None

    # 중복 확인
    existing_user = await get_user_by_username(username)
    if existing_user:
        raise HTTPException(status_code=400, detail="이미 존재하는 사용자명입니다")

    # 사용자 생성
    password_hash = hash_password(password)
    try:
        user_id = await create_user(
            username=username,
            password_hash=password_hash,
            display_name=display_name,
            email=email
        )
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=f"사용자 생성 실패: {str(e)}")

    # 토큰 생성
    access_token = create_access_token(user_id, username, "user")

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "username": username,
            "display_name": display_name or username,
            "role": "user"
        }
    }


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """로그인"""
    # 입력값 trim 처리
    username = request.username.strip() if request.username else ""
    password = request.password.strip() if request.password else ""

    # 사용자 조회
    user = await get_user_by_username(username)

    if not user:
        raise HTTPException(status_code=401, detail="사용자명 또는 비밀번호가 올바르지 않습니다")

    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="비활성화된 계정입니다")

    if not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="사용자명 또는 비밀번호가 올바르지 않습니다")

    # 마지막 로그인 시간 업데이트
    await update_last_login(user["id"])

    # 토큰 생성
    role = user.get("role", "user")
    access_token = create_access_token(user["id"], user["username"], role)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user.get("display_name", user["username"]),
            "role": role
        }
    }


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """현재 로그인된 사용자 정보"""
    preferences = await get_user_preferences(current_user["id"])

    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "display_name": current_user.get("display_name", current_user["username"]),
        "email": current_user.get("email"),
        "role": current_user.get("role", "user"),
        "created_at": str(current_user.get("created_at", "")),
        "preferences": preferences
    }


@router.post("/verify")
async def verify_token_endpoint(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """토큰 유효성 검증"""
    token = credentials.credentials
    payload = verify_token(token)
    return {
        "valid": True,
        "user_id": payload.get("sub"),
        "role": payload.get("role", "user")
    }
