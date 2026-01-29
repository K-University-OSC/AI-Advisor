"""
LLM Provider Registry

Provider를 등록하고 설정에 따라 적절한 Provider를 반환합니다.
"""

import logging
from typing import Dict, Type, Optional

from providers.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

# Provider 클래스 레지스트리
_PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {}

# 싱글톤 인스턴스 캐시
_PROVIDER_INSTANCES: Dict[str, BaseLLMProvider] = {}


def register_provider(name: str):
    """
    Provider 등록 데코레이터

    사용 예시:
        @register_provider("openai")
        class OpenAIProvider(BaseLLMProvider):
            ...
    """
    def decorator(cls: Type[BaseLLMProvider]):
        _PROVIDER_REGISTRY[name] = cls
        logger.debug(f"LLM Provider 등록: {name} -> {cls.__name__}")
        return cls
    return decorator


class LLMProviderRegistry:
    """
    LLM Provider Registry

    Provider를 관리하고 설정에 따라 반환합니다.
    """

    @staticmethod
    def register(name: str, provider_class: Type[BaseLLMProvider]):
        """Provider 등록"""
        _PROVIDER_REGISTRY[name] = provider_class

    @staticmethod
    def get_provider_class(name: str) -> Type[BaseLLMProvider]:
        """Provider 클래스 조회"""
        if name not in _PROVIDER_REGISTRY:
            raise ValueError(f"Unknown LLM provider: {name}. Available: {list(_PROVIDER_REGISTRY.keys())}")
        return _PROVIDER_REGISTRY[name]

    @staticmethod
    def list_providers() -> list:
        """등록된 Provider 목록"""
        return list(_PROVIDER_REGISTRY.keys())

    @staticmethod
    def create(
        provider_name: str,
        api_key: str,
        model: str,
        **kwargs
    ) -> BaseLLMProvider:
        """Provider 인스턴스 생성"""
        provider_class = LLMProviderRegistry.get_provider_class(provider_name)
        return provider_class(api_key=api_key, model=model, **kwargs)


def _ensure_providers_registered():
    """Provider 클래스들이 등록되었는지 확인하고, 없으면 등록"""
    if not _PROVIDER_REGISTRY:
        # Lazy import로 순환 참조 방지
        from providers.llm.openai_provider import OpenAIProvider
        from providers.llm.claude_provider import ClaudeProvider
        from providers.llm.google_provider import GoogleProvider

        _PROVIDER_REGISTRY["openai"] = OpenAIProvider
        _PROVIDER_REGISTRY["claude"] = ClaudeProvider
        _PROVIDER_REGISTRY["google"] = GoogleProvider


def get_llm_provider(
    provider_name: str = None,
    model: str = None,
    use_cache: bool = True
) -> BaseLLMProvider:
    """
    설정에 따라 LLM Provider 인스턴스 반환

    Args:
        provider_name: Provider 이름 (None이면 설정에서 로드)
        model: 모델명 (None이면 설정에서 로드)
        use_cache: 캐시된 인스턴스 사용 여부

    Returns:
        BaseLLMProvider: LLM Provider 인스턴스

    사용 예시:
        # 기본 설정 사용
        provider = get_llm_provider()

        # 특정 Provider 지정
        provider = get_llm_provider("claude", "claude-sonnet-4-20250514")
    """
    _ensure_providers_registered()

    # 설정에서 기본값 로드
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from config import settings

    provider_name = provider_name or settings.providers.llm_provider
    model = model or settings.providers.llm_model

    # 캐시 키
    cache_key = f"{provider_name}:{model}"

    # 캐시된 인스턴스 반환
    if use_cache and cache_key in _PROVIDER_INSTANCES:
        return _PROVIDER_INSTANCES[cache_key]

    # API 키 조회
    api_key = settings.api_keys.get_key_for_provider(provider_name)
    if not api_key:
        raise ValueError(f"API key not found for provider: {provider_name}")

    # 모델 ID 변환 (config의 키 -> 실제 API 모델 ID)
    try:
        actual_model_id = settings.get_model_id(model)
    except ValueError:
        actual_model_id = model  # 이미 실제 모델 ID인 경우

    # Provider 생성
    provider = LLMProviderRegistry.create(
        provider_name=provider_name,
        api_key=api_key,
        model=actual_model_id
    )

    # 캐시에 저장
    if use_cache:
        _PROVIDER_INSTANCES[cache_key] = provider

    logger.info(f"LLM Provider 생성: {provider_name} / {actual_model_id}")
    return provider


def clear_provider_cache():
    """Provider 캐시 초기화 (테스트용)"""
    global _PROVIDER_INSTANCES
    _PROVIDER_INSTANCES = {}
