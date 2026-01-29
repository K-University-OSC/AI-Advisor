"""
관리자 API 라우터 (OSC 공개 버전)
- 사용자 관리, 통계, 비용 분석
- 단일 테넌트 구조
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from sqlalchemy import text

from database import get_db
from routers.auth import get_current_user
from utils.cost_calculator import calculate_cost, get_model_pricing, get_pricing_summary

router = APIRouter()


# ============================================================================
# Request/Response 모델
# ============================================================================

class UserUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class UsageLimitsRequest(BaseModel):
    """사용량 제한 설정 요청"""
    daily_messages: Optional[int] = None
    daily_tokens: Optional[int] = None
    monthly_messages: Optional[int] = None
    monthly_tokens: Optional[int] = None
    max_tokens_per_request: Optional[int] = None
    allowed_models: Optional[List[str]] = None


# ============================================================================
# 권한 검증 함수
# ============================================================================

async def verify_admin(current_user: dict = Depends(get_current_user)):
    """
    관리자 권한 검증
    users 테이블에서 role이 'admin'인 경우에만 허용
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return current_user


# ============================================================================
# 사용자 관리 API
# ============================================================================

@router.get("/users")
async def list_users(
    page: int = 1,
    limit: int = 50,
    search: Optional[str] = None,
    admin_user: dict = Depends(verify_admin)
):
    """사용자 목록 조회"""
    offset = (page - 1) * limit

    async with get_db() as session:
        # 전체 수 조회
        if search:
            count_result = await session.execute(
                text("""
                    SELECT COUNT(*) FROM users
                    WHERE username ILIKE :search OR display_name ILIKE :search
                """),
                {"search": f"%{search}%"}
            )
        else:
            count_result = await session.execute(text("SELECT COUNT(*) FROM users"))

        total = count_result.scalar()

        # 사용자 목록 조회
        if search:
            result = await session.execute(
                text("""
                    SELECT id, username, display_name, email, role, is_active, created_at, last_login
                    FROM users
                    WHERE username ILIKE :search OR display_name ILIKE :search
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """),
                {"search": f"%{search}%", "limit": limit, "offset": offset}
            )
        else:
            result = await session.execute(
                text("""
                    SELECT id, username, display_name, email, role, is_active, created_at, last_login
                    FROM users
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """),
                {"limit": limit, "offset": offset}
            )

        users = [dict(row) for row in result.mappings()]

    return {
        "users": users,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit
    }


@router.get("/users/{user_id}")
async def get_user_detail(
    user_id: int,
    admin_user: dict = Depends(verify_admin)
):
    """사용자 상세 정보 조회"""
    async with get_db() as session:
        # 사용자 정보
        result = await session.execute(
            text("""
                SELECT u.*, up.preferred_model, up.temperature
                FROM users u
                LEFT JOIN user_preferences up ON u.id = up.user_id
                WHERE u.id = :user_id
            """),
            {"user_id": user_id}
        )
        user = result.mappings().first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # 최근 세션 수
        session_count = await session.execute(
            text("SELECT COUNT(*) FROM sessions WHERE user_id = :user_id"),
            {"user_id": user_id}
        )

        # 최근 메시지 수 (30일)
        message_count = await session.execute(
            text("""
                SELECT COUNT(*) FROM messages m
                JOIN sessions s ON m.session_id = s.id
                WHERE s.user_id = :user_id
                AND m.created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
            """),
            {"user_id": user_id}
        )

    return {
        "user": dict(user),
        "stats": {
            "total_sessions": session_count.scalar(),
            "messages_last_30_days": message_count.scalar()
        }
    }


@router.put("/users/{user_id}")
async def update_user(
    user_id: int,
    update_request: UserUpdateRequest,
    admin_user: dict = Depends(verify_admin)
):
    """사용자 정보 수정"""
    updates = update_request.dict(exclude_none=True)

    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    # role 변경은 admin만 가능
    if "role" in updates and updates["role"] not in ["user", "admin"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys()])
    updates["user_id"] = user_id

    async with get_db() as session:
        result = await session.execute(
            text(f"UPDATE users SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = :user_id"),
            updates
        )

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")

    return {"message": "User updated", "user_id": user_id}


@router.post("/users/{user_id}/suspend")
async def suspend_user(
    user_id: int,
    admin_user: dict = Depends(verify_admin)
):
    """사용자 계정 정지"""
    # 자기 자신 정지 불가
    if user_id == admin_user["id"]:
        raise HTTPException(status_code=400, detail="Cannot suspend yourself")

    async with get_db() as session:
        result = await session.execute(
            text("UPDATE users SET is_active = false, updated_at = CURRENT_TIMESTAMP WHERE id = :user_id"),
            {"user_id": user_id}
        )

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")

    return {"message": "User suspended", "user_id": user_id}


@router.post("/users/{user_id}/activate")
async def activate_user(
    user_id: int,
    admin_user: dict = Depends(verify_admin)
):
    """사용자 계정 활성화"""
    async with get_db() as session:
        result = await session.execute(
            text("UPDATE users SET is_active = true, updated_at = CURRENT_TIMESTAMP WHERE id = :user_id"),
            {"user_id": user_id}
        )

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")

    return {"message": "User activated", "user_id": user_id}


# ============================================================================
# 통계 API
# ============================================================================

@router.get("/stats")
async def get_stats(admin_user: dict = Depends(verify_admin)):
    """전체 통계"""
    async with get_db() as session:
        # 전체 사용자 수
        total_users = await session.execute(text("SELECT COUNT(*) FROM users"))

        # 활성 사용자 수
        active_users = await session.execute(
            text("SELECT COUNT(*) FROM users WHERE is_active = true")
        )

        # 최근 7일 활성 사용자
        recent_active = await session.execute(
            text("""
                SELECT COUNT(DISTINCT user_id) FROM sessions
                WHERE updated_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
            """)
        )

        # 전체 세션 수
        total_sessions = await session.execute(text("SELECT COUNT(*) FROM sessions"))

        # 전체 메시지 수
        total_messages = await session.execute(text("SELECT COUNT(*) FROM messages"))

        # 오늘 메시지 수
        today_messages = await session.execute(
            text("""
                SELECT COUNT(*) FROM messages
                WHERE created_at >= CURRENT_DATE
            """)
        )

    return {
        "users": {
            "total": total_users.scalar(),
            "active": active_users.scalar(),
            "active_last_7_days": recent_active.scalar()
        },
        "sessions": {
            "total": total_sessions.scalar()
        },
        "messages": {
            "total": total_messages.scalar(),
            "today": today_messages.scalar()
        }
    }


@router.get("/stats/daily")
async def get_daily_stats(
    days: int = 30,
    admin_user: dict = Depends(verify_admin)
):
    """일별 통계"""
    async with get_db() as session:
        result = await session.execute(
            text("""
                SELECT
                    DATE(m.created_at) as date,
                    COUNT(*) as message_count,
                    COUNT(DISTINCT s.user_id) as active_users
                FROM messages m
                JOIN sessions s ON m.session_id = s.id
                WHERE m.created_at >= CURRENT_DATE - CAST(:days AS INTEGER) * INTERVAL '1 day'
                GROUP BY DATE(m.created_at)
                ORDER BY date DESC
            """),
            {"days": days}
        )

        stats = [dict(row) for row in result.mappings()]

    return {
        "period": f"last {days} days",
        "daily_stats": stats
    }


# ============================================================================
# 관리자 대시보드 API
# ============================================================================

@router.get("/dashboard")
async def get_admin_dashboard(admin_user: dict = Depends(verify_admin)):
    """
    관리자 대시보드

    반환 데이터:
    - 사용자 현황
    - 메시지/세션 통계
    - 최근 활동
    - 모델 사용량
    """
    dashboard = {
        "timestamp": datetime.now().isoformat(),
        "users": {},
        "usage": {},
        "recent_activity": [],
        "model_usage": {}
    }

    async with get_db() as session:
        # 1. 사용자 현황
        user_stats = await session.execute(text("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE is_active = true) as active,
                COUNT(*) FILTER (WHERE created_at >= CURRENT_DATE - INTERVAL '7 days') as new_this_week,
                COUNT(*) FILTER (WHERE last_login >= CURRENT_DATE - INTERVAL '7 days') as active_7d,
                COUNT(*) FILTER (WHERE last_login >= CURRENT_DATE - INTERVAL '30 days') as active_30d,
                COUNT(*) FILTER (WHERE role = 'admin') as admin_count
            FROM users
        """))
        user_row = user_stats.fetchone()
        dashboard["users"] = {
            "total": user_row[0],
            "active": user_row[1],
            "new_this_week": user_row[2],
            "active_7_days": user_row[3],
            "active_30_days": user_row[4],
            "admin_count": user_row[5]
        }

        # 2. 사용량 통계
        usage_stats = await session.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM sessions) as total_sessions,
                (SELECT COUNT(*) FROM sessions WHERE created_at >= CURRENT_DATE) as sessions_today,
                (SELECT COUNT(*) FROM messages) as total_messages,
                (SELECT COUNT(*) FROM messages WHERE created_at >= CURRENT_DATE) as messages_today,
                (SELECT COALESCE(SUM(tokens_used), 0) FROM messages) as total_tokens,
                (SELECT COALESCE(SUM(tokens_used), 0) FROM messages WHERE created_at >= CURRENT_DATE) as tokens_today
        """))
        usage_row = usage_stats.fetchone()
        dashboard["usage"] = {
            "total_sessions": usage_row[0],
            "sessions_today": usage_row[1],
            "total_messages": usage_row[2],
            "messages_today": usage_row[3],
            "total_tokens": usage_row[4],
            "tokens_today": usage_row[5]
        }

        # 3. 일별 추이 (최근 14일)
        daily_trend = await session.execute(text("""
            SELECT
                DATE(m.created_at) as date,
                COUNT(*) as messages,
                COUNT(DISTINCT s.user_id) as users,
                COALESCE(SUM(m.tokens_used), 0) as tokens
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE m.created_at >= CURRENT_DATE - INTERVAL '14 days'
            GROUP BY DATE(m.created_at)
            ORDER BY date
        """))
        dashboard["daily_trend"] = [
            {
                "date": str(row[0]),
                "messages": row[1],
                "users": row[2],
                "tokens": row[3]
            }
            for row in daily_trend.fetchall()
        ]

        # 4. 최근 활동 사용자 (최근 7일 기준)
        recent_users = await session.execute(text("""
            SELECT
                u.id, u.username, u.display_name, u.last_login,
                COUNT(DISTINCT s.id) as session_count,
                COUNT(m.id) as message_count
            FROM users u
            LEFT JOIN sessions s ON u.id = s.user_id
                AND s.updated_at >= CURRENT_DATE - INTERVAL '7 days'
            LEFT JOIN messages m ON s.id = m.session_id
                AND m.created_at >= CURRENT_DATE - INTERVAL '7 days'
            WHERE u.last_login >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY u.id, u.username, u.display_name, u.last_login
            ORDER BY message_count DESC
            LIMIT 10
        """))
        dashboard["recent_activity"] = [
            {
                "user_id": row[0],
                "username": row[1],
                "display_name": row[2],
                "last_login": row[3].isoformat() if row[3] else None,
                "session_count": row[4],
                "message_count": row[5]
            }
            for row in recent_users.fetchall()
        ]

        # 5. 모델별 사용량
        model_usage = await session.execute(text("""
            SELECT
                COALESCE(model, 'unknown') as model,
                COUNT(*) as count,
                COALESCE(SUM(tokens_used), 0) as tokens
            FROM messages
            WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY model
            ORDER BY count DESC
        """))
        dashboard["model_usage"] = [
            {
                "model": row[0],
                "count": row[1],
                "tokens": row[2]
            }
            for row in model_usage.fetchall()
        ]

    return dashboard


@router.get("/dashboard/usage-patterns")
async def get_usage_patterns(
    days: int = 30,
    admin_user: dict = Depends(verify_admin)
):
    """
    사용 패턴 분석

    - 시간대별 사용량
    - 요일별 사용량
    - 사용자 성향 분석
    """
    patterns = {
        "period_days": days,
        "hourly_distribution": [0] * 24,
        "weekday_distribution": [0] * 7,  # 0=월요일
        "user_patterns": [],
        "model_preferences": {},
        "avg_session_length": 0
    }

    async with get_db() as session:
        # 시간대별 분포
        hourly = await session.execute(text("""
            SELECT EXTRACT(HOUR FROM created_at)::int as hour, COUNT(*)
            FROM messages
            WHERE created_at >= CURRENT_DATE - CAST(:days AS INTEGER) * INTERVAL '1 day'
            GROUP BY hour
            ORDER BY hour
        """), {"days": days})
        for row in hourly.fetchall():
            patterns["hourly_distribution"][row[0]] = row[1]

        # 요일별 분포
        weekday = await session.execute(text("""
            SELECT EXTRACT(DOW FROM created_at)::int as dow, COUNT(*)
            FROM messages
            WHERE created_at >= CURRENT_DATE - CAST(:days AS INTEGER) * INTERVAL '1 day'
            GROUP BY dow
            ORDER BY dow
        """), {"days": days})
        for row in weekday.fetchall():
            # PostgreSQL DOW: 0=일요일, 변환: 월=0, 일=6
            idx = (row[0] - 1) % 7
            patterns["weekday_distribution"][idx] = row[1]

        # 사용자별 성향 분석
        user_patterns = await session.execute(text("""
            SELECT
                u.id, u.username, u.display_name,
                COUNT(DISTINCT s.id) as session_count,
                COUNT(m.id) as message_count,
                COALESCE(SUM(m.tokens_used), 0) as total_tokens,
                MODE() WITHIN GROUP (ORDER BY m.model) as preferred_model,
                AVG(EXTRACT(HOUR FROM m.created_at)) as avg_hour,
                MIN(m.created_at) as first_message,
                MAX(m.created_at) as last_message
            FROM users u
            JOIN sessions s ON u.id = s.user_id
            JOIN messages m ON s.id = m.session_id
            WHERE m.created_at >= CURRENT_DATE - CAST(:days AS INTEGER) * INTERVAL '1 day'
            GROUP BY u.id, u.username, u.display_name
            ORDER BY message_count DESC
            LIMIT 20
        """), {"days": days})

        patterns["user_patterns"] = [
            {
                "user_id": row[0],
                "username": row[1],
                "display_name": row[2],
                "session_count": row[3],
                "message_count": row[4],
                "total_tokens": row[5],
                "preferred_model": row[6],
                "avg_usage_hour": round(float(row[7] or 0), 1),
                "first_message": row[8].isoformat() if row[8] else None,
                "last_message": row[9].isoformat() if row[9] else None,
                "engagement": "high" if row[4] > 50 else "medium" if row[4] > 10 else "low"
            }
            for row in user_patterns.fetchall()
        ]

        # 모델 선호도
        model_prefs = await session.execute(text("""
            SELECT
                COALESCE(m.model, 'unknown') as model,
                COUNT(DISTINCT s.user_id) as unique_users,
                COUNT(*) as usage_count,
                COALESCE(SUM(m.tokens_used), 0) as total_tokens
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE m.created_at >= CURRENT_DATE - CAST(:days AS INTEGER) * INTERVAL '1 day'
            GROUP BY m.model
            ORDER BY usage_count DESC
        """), {"days": days})

        patterns["model_preferences"] = [
            {
                "model": row[0],
                "unique_users": row[1],
                "usage_count": row[2],
                "total_tokens": row[3]
            }
            for row in model_prefs.fetchall()
        ]

        # 평균 세션 길이
        avg_session = await session.execute(text("""
            SELECT AVG(msg_count) FROM (
                SELECT COUNT(*) as msg_count
                FROM messages m
                WHERE m.created_at >= CURRENT_DATE - CAST(:days AS INTEGER) * INTERVAL '1 day'
                GROUP BY m.session_id
            ) sub
        """), {"days": days})
        patterns["avg_session_length"] = round(float(avg_session.scalar() or 0), 2)

    return patterns


@router.get("/dashboard/top-users")
async def get_top_users(
    days: int = 30,
    limit: int = 20,
    admin_user: dict = Depends(verify_admin)
):
    """상위 활성 사용자 조회"""
    async with get_db() as session:
        result = await session.execute(text("""
            SELECT
                u.id, u.username, u.display_name, u.email,
                u.created_at as registered_at, u.last_login,
                COUNT(DISTINCT s.id) as session_count,
                COUNT(m.id) as message_count,
                COALESCE(SUM(m.tokens_used), 0) as total_tokens,
                MAX(m.created_at) as last_activity
            FROM users u
            LEFT JOIN sessions s ON u.id = s.user_id
            LEFT JOIN messages m ON s.id = m.session_id
                AND m.created_at >= CURRENT_DATE - CAST(:days AS INTEGER) * INTERVAL '1 day'
            WHERE u.is_active = true
            GROUP BY u.id, u.username, u.display_name, u.email, u.created_at, u.last_login
            ORDER BY message_count DESC
            LIMIT :limit
        """), {"days": days, "limit": limit})

        users = [
            {
                "user_id": row[0],
                "username": row[1],
                "display_name": row[2],
                "email": row[3],
                "registered_at": row[4].isoformat() if row[4] else None,
                "last_login": row[5].isoformat() if row[5] else None,
                "session_count": row[6],
                "message_count": row[7],
                "total_tokens": row[8],
                "last_activity": row[9].isoformat() if row[9] else None
            }
            for row in result.fetchall()
        ]

    return {
        "period_days": days,
        "users": users
    }


@router.get("/dashboard/costs")
async def get_costs(
    days: int = 30,
    admin_user: dict = Depends(verify_admin)
):
    """
    비용 분석 (토큰 기반)

    모델별 공식 가격을 기준으로 비용을 추정합니다.
    """
    costs = {
        "period_days": days,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0,
        "by_model": [],
        "by_user": [],
        "daily_costs": [],
        "pricing_info": get_pricing_summary()
    }

    async with get_db() as session:
        # 모델별 토큰 사용량 (input/output 구분)
        model_usage = await session.execute(text("""
            SELECT
                COALESCE(model, 'unknown') as model,
                COUNT(*) as message_count,
                COALESCE(SUM(CASE WHEN role = 'user' THEN tokens_used ELSE 0 END), 0) as input_tokens,
                COALESCE(SUM(CASE WHEN role = 'assistant' THEN tokens_used ELSE 0 END), 0) as output_tokens,
                COALESCE(SUM(tokens_used), 0) as total_tokens
            FROM messages
            WHERE created_at >= CURRENT_DATE - CAST(:days AS INTEGER) * INTERVAL '1 day'
            GROUP BY model
            ORDER BY total_tokens DESC
        """), {"days": days})

        for row in model_usage.fetchall():
            model = row[0]
            input_tokens = int(row[2])
            output_tokens = int(row[3])
            total_tokens = int(row[4])

            # 중앙 비용 계산 모듈 사용
            cost = calculate_cost(model, input_tokens, output_tokens, total_tokens)
            pricing = get_model_pricing(model)

            costs["by_model"].append({
                "model": model,
                "display_name": pricing["display_name"],
                "message_count": row[1],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": round(cost, 4),
                "is_free": pricing["input"] == 0 and pricing["output"] == 0
            })
            costs["total_input_tokens"] += input_tokens
            costs["total_output_tokens"] += output_tokens
            costs["total_tokens"] += total_tokens
            costs["estimated_cost_usd"] += cost

        costs["estimated_cost_usd"] = round(costs["estimated_cost_usd"], 4)

        # 사용자별 비용 (상위 10명)
        user_costs = await session.execute(text("""
            SELECT
                u.id, u.username, u.display_name,
                COALESCE(SUM(CASE WHEN m.role = 'user' THEN m.tokens_used ELSE 0 END), 0) as input_tokens,
                COALESCE(SUM(CASE WHEN m.role = 'assistant' THEN m.tokens_used ELSE 0 END), 0) as output_tokens,
                COALESCE(SUM(m.tokens_used), 0) as total_tokens,
                COUNT(m.id) as message_count,
                array_agg(DISTINCT COALESCE(m.model, 'unknown')) as models_used
            FROM users u
            JOIN sessions s ON u.id = s.user_id
            JOIN messages m ON s.id = m.session_id
            WHERE m.created_at >= CURRENT_DATE - CAST(:days AS INTEGER) * INTERVAL '1 day'
            GROUP BY u.id, u.username, u.display_name
            ORDER BY total_tokens DESC
            LIMIT 10
        """), {"days": days})

        for row in user_costs.fetchall():
            input_tokens = int(row[3])
            output_tokens = int(row[4])
            total_tokens = int(row[5])
            cost = calculate_cost("unknown", input_tokens, output_tokens, total_tokens)

            costs["by_user"].append({
                "user_id": row[0],
                "username": row[1],
                "display_name": row[2],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "message_count": row[6],
                "models_used": row[7] if row[7] else [],
                "estimated_cost_usd": round(cost, 4)
            })

        # 일별 비용
        daily_usage = await session.execute(text("""
            SELECT
                DATE(created_at) as date,
                COALESCE(model, 'unknown') as model,
                COALESCE(SUM(CASE WHEN role = 'user' THEN tokens_used ELSE 0 END), 0) as input_tokens,
                COALESCE(SUM(CASE WHEN role = 'assistant' THEN tokens_used ELSE 0 END), 0) as output_tokens,
                COUNT(*) as messages
            FROM messages
            WHERE created_at >= CURRENT_DATE - CAST(:days AS INTEGER) * INTERVAL '1 day'
            GROUP BY DATE(created_at), model
            ORDER BY date
        """), {"days": days})

        # 일별로 집계
        daily_data = {}
        for row in daily_usage.fetchall():
            date_str = str(row[0])
            model = row[1]
            input_tokens = int(row[2])
            output_tokens = int(row[3])

            if date_str not in daily_data:
                daily_data[date_str] = {
                    "date": date_str,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "messages": 0,
                    "cost_usd": 0
                }

            cost = calculate_cost(model, input_tokens, output_tokens)
            daily_data[date_str]["input_tokens"] += input_tokens
            daily_data[date_str]["output_tokens"] += output_tokens
            daily_data[date_str]["total_tokens"] += input_tokens + output_tokens
            daily_data[date_str]["messages"] += row[4]
            daily_data[date_str]["cost_usd"] += cost

        for date_str in sorted(daily_data.keys()):
            day = daily_data[date_str]
            day["cost_usd"] = round(day["cost_usd"], 4)
            costs["daily_costs"].append(day)

    return costs


# ============================================================================
# 사용량 제한 관리
# ============================================================================

# 기본 제한값
DEFAULT_LIMITS = {
    "daily_messages": 100,
    "daily_tokens": 100000,
    "monthly_messages": 2000,
    "monthly_tokens": 2000000,
    "max_tokens_per_request": 4000
}


@router.get("/settings/usage-limits")
async def get_usage_limits(admin_user: dict = Depends(verify_admin)):
    """현재 사용량 제한 조회"""
    async with get_db() as session:
        result = await session.execute(
            text("SELECT value FROM global_settings WHERE key = 'usage_limits'")
        )
        row = result.fetchone()
        limits = row[0] if row else DEFAULT_LIMITS

    return {
        "limits": limits,
        "defaults": DEFAULT_LIMITS
    }


@router.put("/settings/usage-limits")
async def update_usage_limits(
    request_body: UsageLimitsRequest,
    admin_user: dict = Depends(verify_admin)
):
    """사용량 제한 설정 변경"""
    updates = {k: v for k, v in request_body.dict().items() if v is not None}

    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    async with get_db() as session:
        # 현재 설정 조회
        result = await session.execute(
            text("SELECT value FROM global_settings WHERE key = 'usage_limits'")
        )
        row = result.fetchone()
        current = row[0] if row else DEFAULT_LIMITS.copy()

        # 업데이트
        current.update(updates)

        # 저장
        import json
        await session.execute(
            text("""
                INSERT INTO global_settings (key, value, description, updated_at)
                VALUES ('usage_limits', :value::jsonb, 'Global usage limits', CURRENT_TIMESTAMP)
                ON CONFLICT (key) DO UPDATE SET value = :value::jsonb, updated_at = CURRENT_TIMESTAMP
            """),
            {"value": json.dumps(current)}
        )

    return {
        "message": "Usage limits updated",
        "limits": current
    }


@router.get("/users/{user_id}/usage-limits")
async def get_user_usage_limits(
    user_id: int,
    admin_user: dict = Depends(verify_admin)
):
    """특정 사용자의 사용량 제한 및 현재 사용량 조회"""
    async with get_db() as session:
        # 사용자 정보
        result = await session.execute(
            text("SELECT id, username, display_name FROM users WHERE id = :user_id"),
            {"user_id": user_id}
        )
        user = result.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # 사용자 설정에서 제한 조회
        pref_result = await session.execute(
            text("SELECT usage_limits FROM user_preferences WHERE user_id = :user_id"),
            {"user_id": user_id}
        )
        pref_row = pref_result.fetchone()
        user_limits = pref_row[0] if pref_row and pref_row[0] else {}

        # 현재 사용량 (일별)
        daily_usage = await session.execute(
            text("""
                SELECT COUNT(*) as messages, COALESCE(SUM(m.tokens_used), 0) as tokens
                FROM messages m
                JOIN sessions s ON m.session_id = s.id
                WHERE s.user_id = :user_id AND m.created_at >= CURRENT_DATE
            """),
            {"user_id": user_id}
        )
        daily_row = daily_usage.fetchone()

        # 현재 사용량 (월별)
        monthly_usage = await session.execute(
            text("""
                SELECT COUNT(*) as messages, COALESCE(SUM(m.tokens_used), 0) as tokens
                FROM messages m
                JOIN sessions s ON m.session_id = s.id
                WHERE s.user_id = :user_id
                AND m.created_at >= DATE_TRUNC('month', CURRENT_DATE)
            """),
            {"user_id": user_id}
        )
        monthly_row = monthly_usage.fetchone()

    return {
        "user": {
            "id": user[0],
            "username": user[1],
            "display_name": user[2]
        },
        "limits": user_limits if user_limits else "Using global defaults",
        "defaults": DEFAULT_LIMITS,
        "current_usage": {
            "daily": {
                "messages": daily_row[0],
                "tokens": daily_row[1]
            },
            "monthly": {
                "messages": monthly_row[0],
                "tokens": monthly_row[1]
            }
        }
    }


@router.put("/users/{user_id}/usage-limits")
async def update_user_usage_limits(
    user_id: int,
    request_body: UsageLimitsRequest,
    admin_user: dict = Depends(verify_admin)
):
    """특정 사용자의 사용량 제한 설정"""
    updates = {k: v for k, v in request_body.dict().items() if v is not None}

    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    async with get_db() as session:
        # 사용자 존재 확인
        result = await session.execute(
            text("SELECT id FROM users WHERE id = :user_id"),
            {"user_id": user_id}
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="User not found")

        # 현재 설정 조회
        pref_result = await session.execute(
            text("SELECT usage_limits FROM user_preferences WHERE user_id = :user_id"),
            {"user_id": user_id}
        )
        pref_row = pref_result.fetchone()
        current = pref_row[0] if pref_row and pref_row[0] else {}

        # 업데이트
        current.update(updates)

        # 저장
        import json
        await session.execute(
            text("""
                UPDATE user_preferences
                SET usage_limits = :value::jsonb, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = :user_id
            """),
            {"user_id": user_id, "value": json.dumps(current)}
        )

    return {
        "message": "User usage limits updated",
        "user_id": user_id,
        "limits": current
    }


@router.delete("/users/{user_id}/usage-limits")
async def reset_user_usage_limits(
    user_id: int,
    admin_user: dict = Depends(verify_admin)
):
    """사용자의 사용량 제한을 기본값으로 리셋"""
    async with get_db() as session:
        await session.execute(
            text("""
                UPDATE user_preferences
                SET usage_limits = '{}', updated_at = CURRENT_TIMESTAMP
                WHERE user_id = :user_id
            """),
            {"user_id": user_id}
        )

    return {
        "message": "User usage limits reset to defaults",
        "user_id": user_id
    }


# ============================================================================
# 관리자 관리
# ============================================================================

@router.get("/admins")
async def list_admins(admin_user: dict = Depends(verify_admin)):
    """관리자 목록"""
    async with get_db() as session:
        result = await session.execute(
            text("""
                SELECT id, username, display_name, email, created_at, last_login
                FROM users
                WHERE role = 'admin'
                ORDER BY created_at
            """)
        )
        admins = [dict(row) for row in result.mappings()]

    return {"admins": admins}


@router.post("/admins/{user_id}")
async def add_admin(
    user_id: int,
    admin_user: dict = Depends(verify_admin)
):
    """관리자 추가"""
    async with get_db() as session:
        # 사용자 존재 확인
        user = await session.execute(
            text("SELECT id, role FROM users WHERE id = :user_id"),
            {"user_id": user_id}
        )
        row = user.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        if row[1] == "admin":
            raise HTTPException(status_code=400, detail="User is already an admin")

        # 관리자로 변경
        await session.execute(
            text("UPDATE users SET role = 'admin', updated_at = CURRENT_TIMESTAMP WHERE id = :user_id"),
            {"user_id": user_id}
        )

    return {"message": "Admin added", "user_id": user_id}


@router.delete("/admins/{user_id}")
async def remove_admin(
    user_id: int,
    admin_user: dict = Depends(verify_admin)
):
    """관리자 제거"""
    # 자기 자신 제거 불가
    if user_id == admin_user["id"]:
        raise HTTPException(status_code=400, detail="Cannot remove yourself")

    async with get_db() as session:
        result = await session.execute(
            text("UPDATE users SET role = 'user', updated_at = CURRENT_TIMESTAMP WHERE id = :user_id AND role = 'admin'"),
            {"user_id": user_id}
        )

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Admin not found")

    return {"message": "Admin removed", "user_id": user_id}
