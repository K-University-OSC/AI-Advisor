"""
Claude (Anthropic) LLM Provider

Claude Sonnet, Haiku 등 Anthropic 모델 지원
"""

import logging
from typing import AsyncGenerator, List, Dict, Any, Optional

from anthropic import AsyncAnthropic

from providers.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class ClaudeProvider(BaseLLMProvider):
    """
    Claude (Anthropic) LLM Provider

    지원 모델:
        - claude-sonnet-4-20250514
        - claude-3-5-haiku-20241022
        - claude-opus-4-5-20251101

    사용 예시:
        provider = ClaudeProvider(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
        response = await provider.chat([{"role": "user", "content": "안녕"}])
    """

    SUPPORTED_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
        "claude-opus-4-5-20251101",
    ]

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = AsyncAnthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "claude"

    def _convert_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """OpenAI 형식 메시지를 Claude 형식으로 변환"""
        system_prompt = None
        claude_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            else:
                # Claude는 assistant/user만 지원
                claude_role = "assistant" if role == "assistant" else "user"
                claude_messages.append({"role": claude_role, "content": content})

        return system_prompt, claude_messages

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Claude 채팅 API 호출"""
        try:
            system_prompt, claude_messages = self._convert_messages(messages)

            params = {
                "model": self.model,
                "messages": claude_messages,
                "max_tokens": max_tokens or 4096,
            }

            # Claude는 temperature가 1.0 초과 불가
            if temperature <= 1.0:
                params["temperature"] = temperature

            if system_prompt:
                params["system"] = system_prompt

            response = await self.client.messages.create(**params)

            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                finish_reason=response.stop_reason,
            )

        except Exception as e:
            logger.error(f"Claude API 오류: {e}")
            raise

    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Claude 스트리밍 API 호출"""
        try:
            system_prompt, claude_messages = self._convert_messages(messages)

            params = {
                "model": self.model,
                "messages": claude_messages,
                "max_tokens": max_tokens or 4096,
            }

            if temperature <= 1.0:
                params["temperature"] = temperature

            if system_prompt:
                params["system"] = system_prompt

            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Claude 스트리밍 오류: {e}")
            raise

    def get_available_models(self) -> List[str]:
        return self.SUPPORTED_MODELS
