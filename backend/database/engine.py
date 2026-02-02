"""
데이터베이스 엔진 설정
PostgreSQL + asyncpg + SQLAlchemy
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# PostgreSQL 연결 설정
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://advisor:***REMOVED***@localhost:10312/advisor_osc_db"
)

# 커넥션 풀 설정
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "30"))
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))


# SQLAlchemy Base
class Base(DeclarativeBase):
    pass


# 비동기 엔진 생성
engine = create_async_engine(
    DATABASE_URL,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_timeout=POOL_TIMEOUT,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False
)

# 세션 팩토리
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """비동기 DB 세션 컨텍스트 매니저"""
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        await session.close()


async def close_db():
    """DB 연결 종료"""
    await engine.dispose()
    logger.info("Database connections closed")
