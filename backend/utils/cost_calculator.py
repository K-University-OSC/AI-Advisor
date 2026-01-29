"""
LLM 비용 계산 모듈
중앙에서 관리되는 API 키를 사용하므로, 테넌트별 비용은 토큰 사용량으로 계산
"""

from typing import Dict, Optional
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# LLM 모델별 가격 정책 (USD per 1K tokens)
# 2024년 기준 공식 가격 (변경 시 업데이트 필요)
# ============================================================================

MODEL_PRICING = {
    # OpenAI GPT 모델
    "gpt-4o": {
        "input": 0.005,      # $5 / 1M tokens
        "output": 0.015,     # $15 / 1M tokens
        "display_name": "GPT-4o"
    },
    "gpt-4o-mini": {
        "input": 0.00015,    # $0.15 / 1M tokens
        "output": 0.0006,    # $0.60 / 1M tokens
        "display_name": "GPT-4o Mini"
    },
    "gpt-4-turbo": {
        "input": 0.01,       # $10 / 1M tokens
        "output": 0.03,      # $30 / 1M tokens
        "display_name": "GPT-4 Turbo"
    },
    "gpt-4": {
        "input": 0.03,       # $30 / 1M tokens
        "output": 0.06,      # $60 / 1M tokens
        "display_name": "GPT-4"
    },
    "gpt-3.5-turbo": {
        "input": 0.0005,     # $0.50 / 1M tokens
        "output": 0.0015,    # $1.50 / 1M tokens
        "display_name": "GPT-3.5 Turbo"
    },

    # Anthropic Claude 모델
    "claude-3-5-sonnet": {
        "input": 0.003,      # $3 / 1M tokens
        "output": 0.015,     # $15 / 1M tokens
        "display_name": "Claude 3.5 Sonnet"
    },
    "claude-3-sonnet": {
        "input": 0.003,      # $3 / 1M tokens
        "output": 0.015,     # $15 / 1M tokens
        "display_name": "Claude 3 Sonnet"
    },
    "claude-3-opus": {
        "input": 0.015,      # $15 / 1M tokens
        "output": 0.075,     # $75 / 1M tokens
        "display_name": "Claude 3 Opus"
    },
    "claude-3-haiku": {
        "input": 0.00025,    # $0.25 / 1M tokens
        "output": 0.00125,   # $1.25 / 1M tokens
        "display_name": "Claude 3 Haiku"
    },

    # Google Gemini 모델
    "gemini-1.5-pro": {
        "input": 0.00125,    # $1.25 / 1M tokens (<=128k)
        "output": 0.005,     # $5 / 1M tokens
        "display_name": "Gemini 1.5 Pro"
    },
    "gemini-1.5-flash": {
        "input": 0.000075,   # $0.075 / 1M tokens
        "output": 0.0003,    # $0.30 / 1M tokens
        "display_name": "Gemini 1.5 Flash"
    },
    "gemini-pro": {
        "input": 0.0005,     # $0.50 / 1M tokens
        "output": 0.0015,    # $1.50 / 1M tokens
        "display_name": "Gemini Pro"
    },

    # Ollama 로컬 모델 (무료)
    "llama3": {
        "input": 0,
        "output": 0,
        "display_name": "Llama 3 (Local)"
    },
    "llama3.1": {
        "input": 0,
        "output": 0,
        "display_name": "Llama 3.1 (Local)"
    },
    "mistral": {
        "input": 0,
        "output": 0,
        "display_name": "Mistral (Local)"
    },
    "codellama": {
        "input": 0,
        "output": 0,
        "display_name": "Code Llama (Local)"
    },

    # 기본값 (알 수 없는 모델)
    "unknown": {
        "input": 0.001,
        "output": 0.002,
        "display_name": "Unknown Model"
    }
}

# 모델명 매핑 (다양한 표기 지원)
MODEL_ALIASES = {
    # GPT
    "gpt4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
    "gpt4-turbo": "gpt-4-turbo",
    "gpt4": "gpt-4",
    "gpt35-turbo": "gpt-3.5-turbo",
    "gpt-35-turbo": "gpt-3.5-turbo",

    # Claude
    "claude-3.5-sonnet": "claude-3-5-sonnet",
    "claude-sonnet": "claude-3-sonnet",
    "claude-opus": "claude-3-opus",
    "claude-haiku": "claude-3-haiku",

    # Gemini
    "gemini-pro-1.5": "gemini-1.5-pro",
    "gemini-flash": "gemini-1.5-flash",

    # Ollama
    "llama-3": "llama3",
    "llama-3.1": "llama3.1",
}


def normalize_model_name(model: str) -> str:
    """모델명 정규화"""
    if not model:
        return "unknown"

    model_lower = model.lower().strip()

    # 별칭 확인
    if model_lower in MODEL_ALIASES:
        return MODEL_ALIASES[model_lower]

    # 직접 매핑
    if model_lower in MODEL_PRICING:
        return model_lower

    # 부분 매칭 시도
    for key in MODEL_PRICING:
        if key in model_lower or model_lower in key:
            return key

    return "unknown"


def get_model_pricing(model: str) -> Dict:
    """모델의 가격 정보 조회"""
    normalized = normalize_model_name(model)
    return MODEL_PRICING.get(normalized, MODEL_PRICING["unknown"])


def calculate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0
) -> float:
    """
    토큰 사용량으로 비용 계산

    Args:
        model: 모델명
        input_tokens: 입력 토큰 수
        output_tokens: 출력 토큰 수
        total_tokens: 총 토큰 수 (input/output 구분 없을 때)

    Returns:
        예상 비용 (USD)
    """
    pricing = get_model_pricing(model)

    if input_tokens > 0 or output_tokens > 0:
        # input/output 구분된 경우
        cost = (input_tokens / 1000) * pricing["input"]
        cost += (output_tokens / 1000) * pricing["output"]
    elif total_tokens > 0:
        # 구분 없는 경우 평균 비율 적용 (input:output = 3:1 가정)
        avg_rate = (pricing["input"] * 0.75) + (pricing["output"] * 0.25)
        cost = (total_tokens / 1000) * avg_rate
    else:
        cost = 0

    return round(cost, 6)


def calculate_batch_cost(usage_data: list) -> Dict:
    """
    여러 사용 기록의 비용 일괄 계산

    Args:
        usage_data: [{"model": str, "input_tokens": int, "output_tokens": int}, ...]

    Returns:
        {"total_cost": float, "by_model": {...}}
    """
    result = {
        "total_cost_usd": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "by_model": {}
    }

    for item in usage_data:
        model = item.get("model", "unknown")
        input_tokens = item.get("input_tokens", 0)
        output_tokens = item.get("output_tokens", 0)
        total_tokens = item.get("total_tokens", 0)

        if total_tokens > 0 and input_tokens == 0 and output_tokens == 0:
            # 구분 없는 경우 추정 분배
            input_tokens = int(total_tokens * 0.75)
            output_tokens = total_tokens - input_tokens

        cost = calculate_cost(model, input_tokens, output_tokens)
        normalized = normalize_model_name(model)

        if normalized not in result["by_model"]:
            pricing = get_model_pricing(model)
            result["by_model"][normalized] = {
                "display_name": pricing["display_name"],
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0,
                "call_count": 0
            }

        result["by_model"][normalized]["input_tokens"] += input_tokens
        result["by_model"][normalized]["output_tokens"] += output_tokens
        result["by_model"][normalized]["cost_usd"] += cost
        result["by_model"][normalized]["call_count"] += 1

        result["total_cost_usd"] += cost
        result["total_input_tokens"] += input_tokens
        result["total_output_tokens"] += output_tokens

    # 반올림
    result["total_cost_usd"] = round(result["total_cost_usd"], 4)
    for model in result["by_model"]:
        result["by_model"][model]["cost_usd"] = round(
            result["by_model"][model]["cost_usd"], 4
        )

    return result


def get_pricing_summary() -> Dict:
    """모든 모델의 가격 정보 요약"""
    summary = {
        "updated_at": datetime.now().isoformat(),
        "currency": "USD",
        "unit": "per 1K tokens",
        "models": {}
    }

    for model_id, pricing in MODEL_PRICING.items():
        if model_id == "unknown":
            continue

        summary["models"][model_id] = {
            "display_name": pricing["display_name"],
            "input_cost": pricing["input"],
            "output_cost": pricing["output"],
            "is_free": pricing["input"] == 0 and pricing["output"] == 0
        }

    return summary


def estimate_monthly_cost(
    daily_messages: int,
    avg_tokens_per_message: int = 500,
    model: str = "gpt-4o-mini"
) -> Dict:
    """
    월간 비용 추정

    Args:
        daily_messages: 일일 평균 메시지 수
        avg_tokens_per_message: 메시지당 평균 토큰 수
        model: 사용 모델
    """
    daily_tokens = daily_messages * avg_tokens_per_message
    monthly_tokens = daily_tokens * 30

    # input:output = 3:1 가정
    input_tokens = int(monthly_tokens * 0.75)
    output_tokens = monthly_tokens - input_tokens

    cost = calculate_cost(model, input_tokens, output_tokens)
    pricing = get_model_pricing(model)

    return {
        "model": model,
        "display_name": pricing["display_name"],
        "daily_messages": daily_messages,
        "avg_tokens_per_message": avg_tokens_per_message,
        "monthly_tokens": monthly_tokens,
        "estimated_monthly_cost_usd": round(cost, 2),
        "breakdown": {
            "input_tokens": input_tokens,
            "input_cost_usd": round((input_tokens / 1000) * pricing["input"], 4),
            "output_tokens": output_tokens,
            "output_cost_usd": round((output_tokens / 1000) * pricing["output"], 4)
        }
    }
