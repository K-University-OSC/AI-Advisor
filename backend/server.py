"""
Advisor RAG Chatbot - FastAPI 메인 서버 (OSC 공개 버전)
- 단일 테넌트 구조 (일반 사용자 + 관리자)
- RAG 전용 서비스
"""
import logging
import sys
import time
import os
from collections import defaultdict
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

# 라우터
from routers import auth
from routers import rag_chat
from routers import admin
from routers import data_management
from routers import error_reports
import app_config
from database import init_db, close_db

# ============ 로깅 설정 ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ============ Rate Limiter (Redis 기반) ============
import redis.asyncio as aioredis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class RedisRateLimiter:
    """
    Redis 기반 Rate Limiting
    - 멀티 워커 환경에서 공유 가능
    - 슬라이딩 윈도우 알고리즘
    """

    def __init__(self, requests_per_minute: int = 1000, requests_per_hour: int = 30000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.redis: aioredis.Redis = None
        self._fallback = defaultdict(list)  # Redis 연결 실패 시 폴백

    async def init(self):
        """Redis 연결 초기화"""
        try:
            self.redis = await aioredis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            await self.redis.ping()
            logger.info("Redis Rate Limiter initialized")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory fallback: {e}")
            self.redis = None

    async def close(self):
        """Redis 연결 종료"""
        if self.redis:
            await self.redis.close()

    async def is_allowed(self, ip: str) -> tuple[bool, str]:
        """요청 허용 여부 확인 (비동기)"""
        if not self.redis:
            return self._fallback_check(ip)

        try:
            now = int(time.time())
            minute_key = f"rate_limit:{ip}:minute:{now // 60}"
            hour_key = f"rate_limit:{ip}:hour:{now // 3600}"

            pipe = self.redis.pipeline()
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)
            pipe.incr(hour_key)
            pipe.expire(hour_key, 3600)
            results = await pipe.execute()

            minute_count = results[0]
            hour_count = results[2]

            if minute_count > self.requests_per_minute:
                return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"

            if hour_count > self.requests_per_hour:
                return False, f"Rate limit exceeded: {self.requests_per_hour} requests per hour"

            return True, ""

        except Exception as e:
            logger.warning(f"Redis rate limit check failed: {e}")
            return self._fallback_check(ip)

    def _fallback_check(self, ip: str) -> tuple[bool, str]:
        """Redis 실패 시 인메모리 폴백"""
        now = time.time()
        self._fallback[ip] = [t for t in self._fallback[ip] if now - t < 60]
        if len(self._fallback[ip]) >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"
        self._fallback[ip].append(now)
        return True, ""


rate_limiter = RedisRateLimiter(requests_per_minute=1000, requests_per_hour=30000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    # 시작 시
    logger.info("Starting Advisor RAG Chatbot Server...")

    # Redis Rate Limiter 초기화
    logger.info("Initializing Redis Rate Limiter...")
    await rate_limiter.init()

    # 데이터베이스 초기화
    logger.info("Initializing Database...")
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    logger.info("Server started successfully")
    yield

    # 종료 시
    logger.info("Shutting down server...")
    await rate_limiter.close()
    await close_db()
    logger.info("Server shutdown complete")


app = FastAPI(
    title="Advisor RAG Chatbot API",
    description="대학교 학사정보 RAG 챗봇 API (OSC 공개 버전)",
    version="1.0.0",
    lifespan=lifespan
)


# ============ 미들웨어 ============

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate Limiting 미들웨어"""
    # 헬스체크 및 문서 API 제외
    if request.url.path in ["/", "/health", "/api/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)

    # CORS preflight 요청(OPTIONS)은 rate limit 제외
    if request.method == "OPTIONS":
        return await call_next(request)

    client_ip = request.client.host
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

    allowed, message = await rate_limiter.is_allowed(client_ip)
    if not allowed:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": message}
        )

    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청 로깅 미들웨어"""
    start_time = time.time()

    client_ip = request.client.host
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

    response = await call_next(request)

    process_time = time.time() - start_time

    if request.url.path not in ["/", "/health", "/api/health"]:
        logger.info(
            f"{client_ip} - {request.method} {request.url.path} - "
            f"{response.status_code} - {process_time:.3f}s"
        )

    response.headers["X-Process-Time"] = str(process_time)
    return response


# ============ 전역 예외 핸들러 ============

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)} - {request.url.path}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )


# ============ 라우터 등록 ============

# 인증 API
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])

# RAG 채팅 API
app.include_router(rag_chat.router, prefix="/api/chat", tags=["RAG Chat"])

# 관리자 API
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])

# 데이터 관리 API (관리자 전용)
app.include_router(data_management.router, prefix="/api/admin/data", tags=["Data Management"])

# 오류 리포트 API
app.include_router(error_reports.router, prefix="/api/errors", tags=["Error Reports"])


# ============ 헬스체크 ============

@app.get("/")
async def root():
    return {
        "message": "Advisor RAG Chatbot API",
        "version": "1.0.0",
        "status": "healthy",
        "rag_based": True
    }


@app.get("/health")
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "rag_based": True
    }


# ============ 서버 실행 ============
if __name__ == "__main__":
    host = app_config.SERVER_HOST
    port = app_config.SERVER_PORT
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        workers=1,
        log_level="info",
        access_log=True
    )
