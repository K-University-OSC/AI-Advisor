"""
PostgreSQL 데이터베이스 설정 및 모델
단일 테넌트 버전 (OSC 공개용)
- 기존 auth.py, chat.py와의 호환성을 위해 유지
- engine.py의 엔진과 세션 팩토리를 재사용
"""
import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from sqlalchemy import text, Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv

# engine.py의 것들을 재사용
from .engine import engine, async_session_factory

load_dotenv(override=True)

logger = logging.getLogger(__name__)


# SQLAlchemy Base
class Base(DeclarativeBase):
    pass


@asynccontextmanager
async def get_db():
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


async def init_db():
    """데이터베이스 초기화 - 테이블 생성"""
    async with engine.begin() as conn:
        # 사용자 테이블 (role 필드 추가: admin, user)
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                display_name VARCHAR(255),
                email VARCHAR(255),
                role VARCHAR(20) DEFAULT 'user',
                is_active BOOLEAN DEFAULT true,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """))

        # role 컬럼이 없으면 추가 (기존 DB 마이그레이션용)
        await conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='users' AND column_name='role') THEN
                    ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'user';
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='users' AND column_name='email') THEN
                    ALTER TABLE users ADD COLUMN email VARCHAR(255);
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='users' AND column_name='is_active') THEN
                    ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT true;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='users' AND column_name='last_login') THEN
                    ALTER TABLE users ADD COLUMN last_login TIMESTAMP;
                END IF;
            END $$;
        """))

        # 관리자 확장 정보 테이블
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS admins (
                id SERIAL PRIMARY KEY,
                user_id INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                permissions JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # 사용자 설정 테이블
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id SERIAL PRIMARY KEY,
                user_id INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                preferred_model VARCHAR(50) DEFAULT 'gpt4o',
                temperature REAL DEFAULT 0.7,
                system_prompt TEXT,
                usage_limits JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # usage_limits 컬럼이 없으면 추가
        await conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='user_preferences' AND column_name='usage_limits') THEN
                    ALTER TABLE user_preferences ADD COLUMN usage_limits JSONB DEFAULT '{}';
                END IF;
            END $$;
        """))

        # 세션 테이블
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sessions (
                id VARCHAR(255) PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title VARCHAR(500),
                model VARCHAR(50) DEFAULT 'gpt4o',
                total_tokens INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # 메시지 테이블
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                tokens_used INTEGER DEFAULT 0,
                model VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # tokens_used, model 컬럼이 없으면 추가
        await conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='messages' AND column_name='tokens_used') THEN
                    ALTER TABLE messages ADD COLUMN tokens_used INTEGER DEFAULT 0;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                               WHERE table_name='messages' AND column_name='model') THEN
                    ALTER TABLE messages ADD COLUMN model VARCHAR(50);
                END IF;
            END $$;
        """))

        # 첨부파일 테이블
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS attachments (
                id SERIAL PRIMARY KEY,
                message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
                session_id VARCHAR(255) NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filename VARCHAR(500) NOT NULL,
                original_filename VARCHAR(500) NOT NULL,
                file_type VARCHAR(50) NOT NULL,
                file_size INTEGER,
                file_path VARCHAR(1000) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # 사용자 프로파일 테이블 (개인화)
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id SERIAL PRIMARY KEY,
                user_id INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                location VARCHAR(255),
                occupation VARCHAR(255),
                food_preferences JSONB DEFAULT '{}',
                hobbies JSONB DEFAULT '[]',
                communication_style JSONB DEFAULT '{}',
                profile_summary TEXT,
                extraction_count INTEGER DEFAULT 0,
                last_extracted_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # 사용자 팩트 테이블
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                category VARCHAR(50) NOT NULL,
                fact_key VARCHAR(100),
                fact_value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source_session_id VARCHAR(255),
                source_message_id INTEGER,
                is_active BOOLEAN DEFAULT true,
                invalidated_at TIMESTAMP,
                invalidated_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # 선호도 변화 이력 테이블
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS preference_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                category VARCHAR(50) NOT NULL,
                old_value TEXT,
                new_value TEXT NOT NULL,
                change_type VARCHAR(20) NOT NULL,
                source_session_id VARCHAR(255),
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # 사용량 로그 테이블 (일별 통계)
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id SERIAL PRIMARY KEY,
                log_date DATE NOT NULL UNIQUE,
                active_users INTEGER DEFAULT 0,
                total_messages INTEGER DEFAULT 0,
                llm_tokens_used BIGINT DEFAULT 0,
                storage_used_mb REAL DEFAULT 0,
                api_calls INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # 전역 설정 테이블
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS global_settings (
                key VARCHAR(100) PRIMARY KEY,
                value JSONB,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # 인덱스 생성
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_attachments_session_id ON attachments(session_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_attachments_user_id ON attachments(user_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_facts_user_id ON user_facts(user_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_usage_logs_date ON usage_logs(log_date)"))

    logger.info("✅ PostgreSQL database initialized successfully")
    print("✅ PostgreSQL database initialized successfully")


async def close_db():
    """DB 연결 종료"""
    await engine.dispose()
    logger.info("Database connections closed")


# ============ 사용자 관련 함수 ============

async def create_user(username: str, password_hash: str, display_name: str = None) -> int:
    """새 사용자 생성"""
    async with get_db() as session:
        result = await session.execute(
            text("""
                INSERT INTO users (username, password_hash, display_name)
                VALUES (:username, :password_hash, :display_name)
                RETURNING id
            """),
            {"username": username, "password_hash": password_hash, "display_name": display_name or username}
        )
        user_id = result.scalar()

        # 기본 설정 생성
        await session.execute(
            text("INSERT INTO user_preferences (user_id) VALUES (:user_id)"),
            {"user_id": user_id}
        )
        return user_id


async def get_user_by_username(username: str) -> Optional[Dict]:
    """username으로 사용자 조회"""
    async with get_db() as session:
        result = await session.execute(
            text("SELECT * FROM users WHERE username = :username"),
            {"username": username}
        )
        row = result.mappings().first()
        return dict(row) if row else None


async def get_user_by_id(user_id: int) -> Optional[Dict]:
    """ID로 사용자 조회"""
    async with get_db() as session:
        result = await session.execute(
            text("SELECT * FROM users WHERE id = :id"),
            {"id": user_id}
        )
        row = result.mappings().first()
        return dict(row) if row else None


# ============ 세션 관련 함수 ============

async def create_session(session_id: str, user_id: int, title: str = None, model: str = "gpt4o") -> str:
    """새 대화 세션 생성"""
    async with get_db() as session:
        await session.execute(
            text("""
                INSERT INTO sessions (id, user_id, title, model)
                VALUES (:id, :user_id, :title, :model)
            """),
            {"id": session_id, "user_id": user_id, "title": title, "model": model}
        )
        return session_id


async def get_session(session_id: str) -> Optional[Dict]:
    """세션 조회"""
    async with get_db() as session:
        result = await session.execute(
            text("SELECT * FROM sessions WHERE id = :id"),
            {"id": session_id}
        )
        row = result.mappings().first()
        return dict(row) if row else None


async def get_user_sessions(user_id: int, limit: int = 50) -> List[Dict]:
    """사용자의 세션 목록 조회"""
    async with get_db() as session:
        result = await session.execute(
            text("""
                SELECT s.*,
                    (SELECT content FROM messages WHERE session_id = s.id ORDER BY created_at DESC LIMIT 1) as last_message,
                    (SELECT COUNT(*) FROM messages WHERE session_id = s.id) as message_count
                FROM sessions s
                WHERE user_id = :user_id
                ORDER BY updated_at DESC
                LIMIT :limit
            """),
            {"user_id": user_id, "limit": limit}
        )
        return [dict(row) for row in result.mappings()]


async def update_session_title(session_id: str, title: str):
    """세션 제목 업데이트"""
    async with get_db() as session:
        await session.execute(
            text("""
                UPDATE sessions
                SET title = :title, updated_at = CURRENT_TIMESTAMP
                WHERE id = :id
            """),
            {"id": session_id, "title": title}
        )


async def delete_session(session_id: str):
    """세션 삭제"""
    async with get_db() as session:
        await session.execute(
            text("DELETE FROM sessions WHERE id = :id"),
            {"id": session_id}
        )


# ============ 메시지 관련 함수 ============

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
        message_id = result.scalar()

        # 세션 updated_at 갱신
        await session.execute(
            text("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = :id"),
            {"id": session_id}
        )
        return message_id


async def get_session_messages(session_id: str, limit: int = 100) -> List[Dict]:
    """세션의 메시지 조회"""
    async with get_db() as session:
        result = await session.execute(
            text("""
                SELECT * FROM (
                    SELECT * FROM messages WHERE session_id = :session_id
                    ORDER BY created_at DESC LIMIT :limit
                ) sub ORDER BY created_at ASC
            """),
            {"session_id": session_id, "limit": limit}
        )
        return [dict(row) for row in result.mappings()]


# ============ 사용자 설정 관련 함수 ============

async def get_user_preferences(user_id: int) -> Optional[Dict]:
    """사용자 설정 조회"""
    async with get_db() as session:
        result = await session.execute(
            text("SELECT * FROM user_preferences WHERE user_id = :user_id"),
            {"user_id": user_id}
        )
        row = result.mappings().first()
        return dict(row) if row else None


async def update_user_preferences(user_id: int, **kwargs):
    """사용자 설정 업데이트"""
    allowed_fields = ['preferred_model', 'temperature', 'system_prompt']
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

    if not updates:
        return

    set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys()])
    updates["user_id"] = user_id

    async with get_db() as session:
        await session.execute(
            text(f"UPDATE user_preferences SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE user_id = :user_id"),
            updates
        )


# ============ 첨부파일 관련 함수 ============

async def add_attachment(
    session_id: str,
    user_id: int,
    filename: str,
    original_filename: str,
    file_type: str,
    file_size: int,
    file_path: str,
    message_id: int = None
) -> int:
    """첨부파일 추가"""
    async with get_db() as session:
        result = await session.execute(
            text("""
                INSERT INTO attachments
                (message_id, session_id, user_id, filename, original_filename, file_type, file_size, file_path)
                VALUES (:message_id, :session_id, :user_id, :filename, :original_filename, :file_type, :file_size, :file_path)
                RETURNING id
            """),
            {
                "message_id": message_id,
                "session_id": session_id,
                "user_id": user_id,
                "filename": filename,
                "original_filename": original_filename,
                "file_type": file_type,
                "file_size": file_size,
                "file_path": file_path
            }
        )
        return result.scalar()


async def get_attachment(attachment_id: int) -> Optional[Dict]:
    """첨부파일 조회"""
    async with get_db() as session:
        result = await session.execute(
            text("SELECT * FROM attachments WHERE id = :id"),
            {"id": attachment_id}
        )
        row = result.mappings().first()
        return dict(row) if row else None


async def get_session_attachments(session_id: str) -> List[Dict]:
    """세션의 모든 첨부파일 조회"""
    async with get_db() as session:
        result = await session.execute(
            text("SELECT * FROM attachments WHERE session_id = :session_id ORDER BY created_at ASC"),
            {"session_id": session_id}
        )
        return [dict(row) for row in result.mappings()]


async def get_message_attachments(message_id: int) -> List[Dict]:
    """메시지의 첨부파일 조회"""
    async with get_db() as session:
        result = await session.execute(
            text("SELECT * FROM attachments WHERE message_id = :message_id ORDER BY created_at ASC"),
            {"message_id": message_id}
        )
        return [dict(row) for row in result.mappings()]


async def update_attachment_message_id(attachment_id: int, message_id: int):
    """첨부파일에 메시지 ID 연결"""
    async with get_db() as session:
        await session.execute(
            text("UPDATE attachments SET message_id = :message_id WHERE id = :id"),
            {"id": attachment_id, "message_id": message_id}
        )


async def delete_attachment(attachment_id: int):
    """첨부파일 삭제"""
    async with get_db() as session:
        await session.execute(
            text("DELETE FROM attachments WHERE id = :id"),
            {"id": attachment_id}
        )


# ============ 동기 래퍼 (하위 호환성) ============
# 기존 동기 코드와의 호환을 위한 래퍼
import asyncio

def _run_async(coro):
    """비동기 함수를 동기적으로 실행"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# 동기 래퍼 함수들 (필요시 사용)
def sync_create_user(username: str, password_hash: str, display_name: str = None) -> int:
    return _run_async(create_user(username, password_hash, display_name))

def sync_get_user_by_username(username: str) -> Optional[Dict]:
    return _run_async(get_user_by_username(username))

def sync_get_user_by_id(user_id: int) -> Optional[Dict]:
    return _run_async(get_user_by_id(user_id))


if __name__ == "__main__":
    asyncio.run(init_db())
