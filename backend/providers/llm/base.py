"""
LLM Provider 베이스 인터페이스

모든 LLM Provider는 이 인터페이스를 구현해야 합니다.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM 응답 데이터"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


class BaseLLMProvider(ABC):
    """
    LLM Provider 베이스 클래스

    모든 LLM 제공자(OpenAI, Claude, Google 등)는 이 클래스를 상속받아 구현합니다.

    사용 예시:
        provider = OpenAIProvider(api_key="...", model="gpt-5")
        response = await provider.chat([{"role": "user", "content": "안녕"}])
    """

    def __init__(self, api_key: str, model: str, **kwargs):
        """
        Provider 초기화

        Args:
            api_key: API 키
            model: 사용할 모델명
            **kwargs: 추가 설정
        """
        self.api_key = api_key
        self.model = model
        self.config = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider 이름 (예: 'openai', 'claude', 'google')"""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        채팅 완성 API 호출

        Args:
            messages: 대화 메시지 리스트 [{"role": "user", "content": "..."}]
            temperature: 샘플링 온도 (0.0 ~ 2.0)
            max_tokens: 최대 생성 토큰 수
            **kwargs: 추가 파라미터

        Returns:
            LLMResponse: 응답 데이터
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 채팅 API 호출

        Args:
            messages: 대화 메시지 리스트
            temperature: 샘플링 온도
            max_tokens: 최대 생성 토큰 수
            **kwargs: 추가 파라미터

        Yields:
            str: 응답 토큰
        """
        pass

    async def chat_with_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        JSON 응답 요청 (지원하는 모델만)

        Args:
            messages: 대화 메시지 리스트
            temperature: 샘플링 온도
            **kwargs: 추가 파라미터

        Returns:
            Dict: 파싱된 JSON 응답
        """
        import json
        response = await self.chat(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            **kwargs
        )
        return json.loads(response.content)

    def get_available_models(self) -> List[str]:
        """
        사용 가능한 모델 목록 반환

        Returns:
            List[str]: 모델 ID 리스트
        """
        return [self.model]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
