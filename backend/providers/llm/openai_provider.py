"""
OpenAI LLM Provider

GPT-4o, GPT-5, O1, O3 등 OpenAI 모델 지원
"""

import logging
from typing import AsyncGenerator, List, Dict, Any, Optional

from openai import AsyncOpenAI

from providers.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM Provider

    지원 모델:
        - gpt-4o, gpt-4o-mini
        - gpt-5, gpt-5-mini
        - o1, o1-mini, o3-mini

    사용 예시:
        provider = OpenAIProvider(api_key="sk-...", model="gpt-5")
        response = await provider.chat([{"role": "user", "content": "안녕"}])
    """

    # 지원하는 모델 목록
    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5",
        "gpt-5-mini",
        "o1",
        "o1-mini",
        "o3-mini",
    ]

    # temperature 변경이 지원되지 않는 모델 (기본값 1만 사용)
    FIXED_TEMPERATURE_MODELS = [
        "gpt-5-mini",
        "o1",
        "o1-mini",
        "o3-mini",
    ]

    def __init__(self, api_key: str, model: str = "gpt-5", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """OpenAI 채팅 완성 API 호출"""
        try:
            params = {
                "model": self.model,
                "messages": messages,
            }

            # temperature 지원 여부 확인
            # gpt-5-mini, o1, o1-mini, o3-mini는 temperature=1만 지원
            if self.model not in self.FIXED_TEMPERATURE_MODELS:
                params["temperature"] = temperature

            if max_tokens:
                params["max_tokens"] = max_tokens

            # JSON 응답 형식 지원
            if kwargs.get("response_format"):
                params["response_format"] = kwargs["response_format"]

            response = await self.client.chat.completions.create(**params)

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
            )

        except Exception as e:
            logger.error(f"OpenAI API 오류: {e}")
            raise

    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """OpenAI 스트리밍 API 호출"""
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "stream": True,
            }

            # temperature 지원 여부 확인
            if self.model not in self.FIXED_TEMPERATURE_MODELS:
                params["temperature"] = temperature

            if max_tokens:
                params["max_tokens"] = max_tokens

            response = await self.client.chat.completions.create(**params)

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI 스트리밍 오류: {e}")
            raise

    def get_available_models(self) -> List[str]:
        return self.SUPPORTED_MODELS
