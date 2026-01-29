"""
사용량 제한 관리 모듈 (OSC 공개 버전)
- 관리자: 전역 기본값 설정
- 사용자별 제한 설정
"""

from typing import Dict, Optional
from datetime import datetime, date, timedelta
import json
import logging
from sqlalchemy import text

from database import get_db

logger = logging.getLogger(__name__)

# ============================================================================
# 기본 사용량 제한 설정
# ============================================================================

DEFAULT_LIMITS = {
    # 일일 제한
    "daily_messages": 100,          # 일일 메시지 수
    "daily_tokens": 50000,          # 일일 토큰 수
    "daily_cost_usd": 1.0,          # 일일 비용 한도 (USD)

    # 월간 제한
    "monthly_messages": 2000,       # 월간 메시지 수
    "monthly_tokens": 1000000,      # 월간 토큰 수
    "monthly_cost_usd": 20.0,       # 월간 비용 한도 (USD)

    # 요청당 제한
    "max_tokens_per_request": 4000, # 요청당 최대 토큰
    "max_context_messages": 20,     # 컨텍스트 메시지 최대 수

    # 동시성 제한
    "max_concurrent_requests": 3,   # 동시 요청 수

    # 특수 제한
    "allowed_models": ["gpt-4o-mini", "gpt-3.5-turbo", "claude-3-haiku", "gemini-1.5-flash", "llama3"],
    "premium_models": ["gpt-4o", "gpt-4-turbo", "claude-3-sonnet", "claude-3-opus", "gemini-1.5-pro"]
}


# ============================================================================
# 전역 제한 관리 (관리자용)
# ============================================================================

async def get_global_limits() -> Dict:
    """전역 기본 제한값 조회"""
    try:
        async with get_db() as session:
            result = await session.execute(
                text("SELECT value FROM global_settings WHERE key = 'usage_limits'")
            )
            row = result.fetchone()

            if row and row[0]:
                return row[0] if isinstance(row[0], dict) else json.loads(row[0])
    except Exception as e:
        logger.warning(f"Failed to get global limits: {e}")

    return DEFAULT_LIMITS.copy()


async def set_global_limits(limits: Dict) -> Dict:
    """전역 기본 제한값 설정"""
    # 기본값과 병합
    merged = DEFAULT_LIMITS.copy()
    merged.update(limits)

    async with get_db() as session:
        await session.execute(
            text("""
                INSERT INTO global_settings (key, value, description, updated_at)
                VALUES ('usage_limits', :value, '전역 사용량 제한 설정', CURRENT_TIMESTAMP)
                ON CONFLICT (key) DO UPDATE
                SET value = :value, updated_at = CURRENT_TIMESTAMP
            """),
            {"value": json.dumps(merged)}
        )

    logger.info("Global usage limits updated")
    return merged


# ============================================================================
# 사용자별 제한 관리
# ============================================================================

async def get_user_limits(user_id: int) -> Dict:
    """
    사용자별 제한값 조회
    사용자 설정이 없으면 전역 기본값 반환
    """
    # 전역 기본값 로드
    global_limits = await get_global_limits()

    try:
        async with get_db() as session:
            result = await session.execute(
                text("""
                    SELECT usage_limits FROM user_preferences
                    WHERE user_id = :user_id
                """),
                {"user_id": user_id}
            )
            row = result.fetchone()

            if row and row[0]:
                # 사용자 설정으로 오버라이드
                limits = global_limits.copy()
                user_limits = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                limits.update(user_limits)
                return limits
    except Exception as e:
        logger.warning(f"Failed to get user limits: {e}")

    return global_limits


async def set_user_limits(user_id: int, limits: Dict) -> Dict:
    """사용자별 제한값 설정 (관리자용)"""
    try:
        async with get_db() as session:
            # user_preferences 테이블에 usage_limits 컬럼이 있는지 확인하고 없으면 추가
            await session.execute(text("""
                ALTER TABLE user_preferences
                ADD COLUMN IF NOT EXISTS usage_limits JSONB DEFAULT '{}'
            """))

            # upsert
            await session.execute(
                text("""
                    INSERT INTO user_preferences (user_id, usage_limits, updated_at)
                    VALUES (:user_id, :limits, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id) DO UPDATE
                    SET usage_limits = :limits, updated_at = CURRENT_TIMESTAMP
                """),
                {"user_id": user_id, "limits": json.dumps(limits)}
            )

        logger.info(f"User {user_id} usage limits updated")
        return limits
    except Exception as e:
        logger.error(f"Failed to set user limits: {e}")
        raise


# ============================================================================
# 사용량 조회
# ============================================================================

async def get_user_usage(user_id: int, period: str = "daily") -> Dict:
    """
    사용자 사용량 조회

    Args:
        user_id: 사용자 ID
        period: "daily" | "monthly"
    """
    if period == "daily":
        date_filter = "DATE(created_at) = CURRENT_DATE"
    else:  # monthly
        date_filter = "DATE(created_at) >= DATE_TRUNC('month', CURRENT_DATE)"

    usage = {
        "period": period,
        "messages": 0,
        "tokens": 0,
        "estimated_cost_usd": 0
    }

    try:
        async with get_db() as session:
            result = await session.execute(
                text(f"""
                    SELECT
                        COUNT(*) as message_count,
                        COALESCE(SUM(m.tokens_used), 0) as tokens
                    FROM messages m
                    JOIN sessions s ON m.session_id = s.id
                    WHERE s.user_id = :user_id AND {date_filter}
                """),
                {"user_id": user_id}
            )
            row = result.fetchone()

            if row:
                usage["messages"] = int(row[0])
                usage["tokens"] = int(row[1])
                # 간단한 비용 추정 (평균 가격 기준)
                usage["estimated_cost_usd"] = round(usage["tokens"] / 1000 * 0.002, 4)

    except Exception as e:
        logger.warning(f"Failed to get user usage: {e}")

    return usage


# ============================================================================
# 제한 검사
# ============================================================================

async def check_user_limits(user_id: int, requested_tokens: int = 0) -> Dict:
    """
    사용자 제한 검사

    Returns:
        {
            "allowed": True/False,
            "reason": "제한 사유 (제한 시)",
            "usage": {...},
            "limits": {...},
            "remaining": {...}
        }
    """
    limits = await get_user_limits(user_id)
    daily_usage = await get_user_usage(user_id, "daily")
    monthly_usage = await get_user_usage(user_id, "monthly")

    result = {
        "allowed": True,
        "reason": None,
        "usage": {
            "daily": daily_usage,
            "monthly": monthly_usage
        },
        "limits": limits,
        "remaining": {
            "daily_messages": limits["daily_messages"] - daily_usage["messages"],
            "daily_tokens": limits["daily_tokens"] - daily_usage["tokens"],
            "monthly_messages": limits["monthly_messages"] - monthly_usage["messages"],
            "monthly_tokens": limits["monthly_tokens"] - monthly_usage["tokens"]
        }
    }

    # 일일 메시지 제한
    if daily_usage["messages"] >= limits["daily_messages"]:
        result["allowed"] = False
        result["reason"] = f"일일 메시지 한도({limits['daily_messages']}건)에 도달했습니다. 내일 다시 시도해주세요."
        return result

    # 일일 토큰 제한
    if daily_usage["tokens"] + requested_tokens > limits["daily_tokens"]:
        result["allowed"] = False
        result["reason"] = f"일일 토큰 한도({limits['daily_tokens']:,})에 도달했습니다. 내일 다시 시도해주세요."
        return result

    # 월간 메시지 제한
    if monthly_usage["messages"] >= limits["monthly_messages"]:
        result["allowed"] = False
        result["reason"] = f"월간 메시지 한도({limits['monthly_messages']}건)에 도달했습니다. 다음 달에 다시 시도해주세요."
        return result

    # 월간 토큰 제한
    if monthly_usage["tokens"] + requested_tokens > limits["monthly_tokens"]:
        result["allowed"] = False
        result["reason"] = f"월간 토큰 한도({limits['monthly_tokens']:,})에 도달했습니다. 다음 달에 다시 시도해주세요."
        return result

    # 요청당 토큰 제한
    if requested_tokens > limits["max_tokens_per_request"]:
        result["allowed"] = False
        result["reason"] = f"요청당 최대 토큰({limits['max_tokens_per_request']:,})을 초과했습니다."
        return result

    return result


async def check_model_access(user_id: int, model: str) -> Dict:
    """
    모델 접근 권한 검사

    Returns:
        {"allowed": True/False, "reason": "사유"}
    """
    limits = await get_user_limits(user_id)

    allowed_models = limits.get("allowed_models", [])
    premium_models = limits.get("premium_models", [])

    # 허용된 모델 확인
    if model in allowed_models:
        return {"allowed": True, "reason": None}

    # 프리미엄 모델 확인
    if model in premium_models:
        # 프리미엄 모델은 특별 권한 필요 (추후 구현 가능)
        return {
            "allowed": False,
            "reason": f"'{model}'은(는) 프리미엄 모델입니다. 관리자에게 문의하세요."
        }

    return {
        "allowed": True,  # 알 수 없는 모델은 일단 허용 (로컬 모델 등)
        "reason": None
    }


# ============================================================================
# 제한 요약 정보
# ============================================================================

async def get_limits_summary(user_id: int) -> Dict:
    """
    사용자의 제한 및 사용량 요약 정보
    """
    limits = await get_user_limits(user_id)
    daily_usage = await get_user_usage(user_id, "daily")
    monthly_usage = await get_user_usage(user_id, "monthly")

    return {
        "limits": {
            "daily_messages": limits["daily_messages"],
            "daily_tokens": limits["daily_tokens"],
            "monthly_messages": limits["monthly_messages"],
            "monthly_tokens": limits["monthly_tokens"],
            "allowed_models": limits.get("allowed_models", []),
            "premium_models": limits.get("premium_models", [])
        },
        "usage": {
            "daily": daily_usage,
            "monthly": monthly_usage
        },
        "remaining": {
            "daily_messages": max(0, limits["daily_messages"] - daily_usage["messages"]),
            "daily_tokens": max(0, limits["daily_tokens"] - daily_usage["tokens"]),
            "monthly_messages": max(0, limits["monthly_messages"] - monthly_usage["messages"]),
            "monthly_tokens": max(0, limits["monthly_tokens"] - monthly_usage["tokens"])
        },
        "percentages": {
            "daily_messages": min(100, round(daily_usage["messages"] / limits["daily_messages"] * 100, 1)) if limits["daily_messages"] > 0 else 0,
            "daily_tokens": min(100, round(daily_usage["tokens"] / limits["daily_tokens"] * 100, 1)) if limits["daily_tokens"] > 0 else 0,
            "monthly_messages": min(100, round(monthly_usage["messages"] / limits["monthly_messages"] * 100, 1)) if limits["monthly_messages"] > 0 else 0,
            "monthly_tokens": min(100, round(monthly_usage["tokens"] / limits["monthly_tokens"] * 100, 1)) if limits["monthly_tokens"] > 0 else 0
        }
    }
