"""
Google (Gemini) LLM Provider

Gemini Pro, Flash 등 Google 모델 지원
"""

import logging
from typing import AsyncGenerator, List, Dict, Any, Optional

import google.generativeai as genai

from providers.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class GoogleProvider(BaseLLMProvider):
    """
    Google Gemini LLM Provider

    지원 모델:
        - gemini-1.5-pro
        - gemini-1.5-flash
        - gemini-2.0-flash-exp

    사용 예시:
        provider = GoogleProvider(api_key="...", model="gemini-1.5-pro")
        response = await provider.chat([{"role": "user", "content": "안녕"}])
    """

    SUPPORTED_MODELS = [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash-exp",
    ]

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro", **kwargs):
        super().__init__(api_key, model, **kwargs)
        genai.configure(api_key=api_key)
        self.genai_model = genai.GenerativeModel(model)

    @property
    def provider_name(self) -> str:
        return "google"

    def _convert_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """OpenAI 형식 메시지를 Gemini 형식으로 변환"""
        system_instruction = None
        gemini_history = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            else:
                # Gemini는 user/model 역할 사용
                gemini_role = "model" if role == "assistant" else "user"
                gemini_history.append({"role": gemini_role, "parts": [content]})

        return system_instruction, gemini_history

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Gemini 채팅 API 호출"""
        try:
            system_instruction, gemini_history = self._convert_messages(messages)

            # 시스템 프롬프트가 있으면 새 모델 생성
            if system_instruction:
                model = genai.GenerativeModel(
                    self.model,
                    system_instruction=system_instruction
                )
            else:
                model = self.genai_model

            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
            )

            if max_tokens:
                generation_config.max_output_tokens = max_tokens

            # 마지막 메시지 추출
            if gemini_history:
                last_message = gemini_history[-1]["parts"][0]
                history = gemini_history[:-1] if len(gemini_history) > 1 else []
            else:
                last_message = ""
                history = []

            chat = model.start_chat(history=history)
            response = await chat.send_message_async(
                last_message,
                generation_config=generation_config
            )

            return LLMResponse(
                content=response.text,
                model=self.model,
                usage=None,  # Gemini는 사용량 정보가 다름
                finish_reason="stop",
            )

        except Exception as e:
            logger.error(f"Gemini API 오류: {e}")
            raise

    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Gemini 스트리밍 API 호출"""
        try:
            system_instruction, gemini_history = self._convert_messages(messages)

            if system_instruction:
                model = genai.GenerativeModel(
                    self.model,
                    system_instruction=system_instruction
                )
            else:
                model = self.genai_model

            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
            )

            if max_tokens:
                generation_config.max_output_tokens = max_tokens

            if gemini_history:
                last_message = gemini_history[-1]["parts"][0]
                history = gemini_history[:-1] if len(gemini_history) > 1 else []
            else:
                last_message = ""
                history = []

            chat = model.start_chat(history=history)
            response = await chat.send_message_async(
                last_message,
                generation_config=generation_config,
                stream=True
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Gemini 스트리밍 오류: {e}")
            raise

    def get_available_models(self) -> List[str]:
        return self.SUPPORTED_MODELS
