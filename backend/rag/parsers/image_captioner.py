"""
이미지/차트 캡셔닝 모듈
VLM(Vision Language Model)을 사용하여 차트, 그래프, 표 이미지를 텍스트로 변환
"""

import asyncio
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import httpx

from .document_parser import ParsedElement, ElementType


@dataclass
class CaptionResult:
    """캡셔닝 결과"""
    element_id: str
    original_content: str
    caption: str
    summary: str
    key_values: dict
    metadata: dict


class ImageCaptioner(ABC):
    """이미지 캡셔너 추상 클래스"""

    @abstractmethod
    async def caption(
        self,
        image_data: bytes,
        context: Optional[str] = None,
    ) -> str:
        """이미지를 설명하는 텍스트 생성"""
        pass

    @abstractmethod
    async def caption_element(
        self,
        element: ParsedElement,
        context: Optional[str] = None,
    ) -> CaptionResult:
        """ParsedElement의 이미지를 캡셔닝"""
        pass


class OpenAIImageCaptioner(ImageCaptioner):
    """OpenAI GPT-4 Vision 기반 캡셔너"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",  # Baseline V1: gpt-4o Vision
    ):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"

    async def caption(
        self,
        image_data: bytes,
        context: Optional[str] = None,
    ) -> str:
        """이미지를 설명하는 텍스트 생성"""
        base64_image = base64.b64encode(image_data).decode("utf-8")

        # V7.8: 향상된 Vision 프롬프트 - 검색 최적화
        system_prompt = """당신은 문서에서 추출한 이미지와 차트를 분석하는 전문가입니다.
주어진 이미지를 분석하고 다음 형식으로 상세히 설명해주세요:

1. **유형**: 차트/그래프/표/다이어그램/플로우차트/인포그래픽 중 무엇인지
2. **제목**: 이미지에 표시된 제목 (정확히 기재)
3. **주요 내용**:
   - 모든 텍스트를 정확히 추출 (법률명, 기관명, 용어 등)
   - 구조화된 정보가 있다면 번호를 매겨 나열 (예: 4가지 요인, 3단계 등)
   - 핵심 데이터와 수치를 구체적으로 나열
   - 항목별 값과 단위를 명확히 기재
   - 화살표/연결선이 있다면 관계와 흐름 설명
4. **상세 설명**:
   - 각 항목의 세부 내용 (박스 안 텍스트 전체 추출)
   - 범례, 주석, 출처 등 부가 정보
5. **키워드**: 검색에 도움될 핵심 키워드 10-15개 (고유명사, 전문용어 포함)

중요: 이미지 내 모든 텍스트를 빠짐없이 추출하세요. 특히 법률명, 기관명, 조직명, 전문용어는 정확히 기재해야 합니다."""

        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high",
                },
            },
        ]

        if context:
            user_content.insert(0, {
                "type": "text",
                "text": f"문서 컨텍스트: {context}\n\n위 컨텍스트를 참고하여 아래 이미지를 분석해주세요.",
            })
        else:
            user_content.insert(0, {
                "type": "text",
                "text": "아래 이미지를 분석해주세요.",
            })

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_completion_tokens": 2500,  # V7.8: 상세 캡션을 위해 증가
            "temperature": 0.2,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )

        if response.status_code != 200:
            raise Exception(
                f"OpenAI API 오류: {response.status_code} - {response.text}"
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def caption_element(
        self,
        element: ParsedElement,
        context: Optional[str] = None,
    ) -> CaptionResult:
        """ParsedElement의 이미지를 캡셔닝"""
        if not element.image_data:
            raise ValueError(f"요소 {element.element_id}에 이미지 데이터가 없습니다.")

        caption = await self.caption(element.image_data, context)

        summary = await self._generate_summary(caption)
        key_values = await self._extract_key_values(caption)

        return CaptionResult(
            element_id=element.element_id,
            original_content=element.content,
            caption=caption,
            summary=summary,
            key_values=key_values,
            metadata={
                "model": self.model,
                "element_type": element.element_type.value,
                "page": element.page,
            },
        )

    async def _generate_summary(self, caption: str) -> str:
        """캡션에서 핵심 요약 생성"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "차트/그래프 분석 내용을 1-2문장으로 핵심만 요약해주세요.",
                },
                {"role": "user", "content": caption},
            ],
            "max_completion_tokens": 200,
            "temperature": 0.1,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )

        if response.status_code != 200:
            return caption[:200]

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def _extract_key_values(self, caption: str) -> dict:
        """캡션에서 주요 수치 추출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": """차트/그래프 분석 내용에서 주요 수치와 데이터를 JSON 형식으로 추출해주세요.
예시: {"최고값": "15.2%", "최저값": "3.1%", "평균": "8.5%", "기간": "2019-2023"}
수치가 없으면 빈 객체 {}를 반환하세요.""",
                },
                {"role": "user", "content": caption},
            ],
            "max_completion_tokens": 300,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )

        if response.status_code != 200:
            return {}

        import json
        try:
            result = response.json()
            return json.loads(result["choices"][0]["message"]["content"])
        except (json.JSONDecodeError, KeyError):
            return {}


class GeminiImageCaptioner(ImageCaptioner):
    """Gemini Vision 기반 캡셔너 (GPT-4o 대안)"""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3-flash-preview",
    ):
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    async def caption(
        self,
        image_data: bytes,
        context: Optional[str] = None,
    ) -> str:
        """이미지를 설명하는 텍스트 생성"""
        base64_image = base64.b64encode(image_data).decode("utf-8")

        system_prompt = """당신은 문서에서 추출한 이미지와 차트를 분석하는 전문가입니다.
주어진 이미지를 분석하고 다음 형식으로 상세히 설명해주세요:

1. **유형**: 차트/그래프/표/다이어그램/플로우차트/인포그래픽 중 무엇인지
2. **제목**: 이미지에 표시된 제목 (정확히 기재)
3. **주요 내용**:
   - 모든 텍스트를 정확히 추출 (법률명, 기관명, 용어 등)
   - 구조화된 정보가 있다면 번호를 매겨 나열 (예: 4가지 요인, 3단계 등)
   - 핵심 데이터와 수치를 구체적으로 나열
   - 항목별 값과 단위를 명확히 기재
   - 화살표/연결선이 있다면 관계와 흐름 설명
4. **상세 설명**:
   - 각 항목의 세부 내용 (박스 안 텍스트 전체 추출)
   - 범례, 주석, 출처 등 부가 정보
5. **키워드**: 검색에 도움될 핵심 키워드 10-15개 (고유명사, 전문용어 포함)

중요: 이미지 내 모든 텍스트를 빠짐없이 추출하세요. 특히 법률명, 기관명, 조직명, 전문용어는 정확히 기재해야 합니다."""

        user_text = "아래 이미지를 분석해주세요."
        if context:
            user_text = f"문서 컨텍스트: {context}\n\n위 컨텍스트를 참고하여 아래 이미지를 분석해주세요."

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt + "\n\n" + user_text},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2500,
            }
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.api_url}?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
            )

        if response.status_code != 200:
            raise Exception(
                f"Gemini API 오류: {response.status_code} - {response.text}"
            )

        result = response.json()
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise Exception(f"Gemini 응답 파싱 오류: {result}")

    async def caption_element(
        self,
        element: ParsedElement,
        context: Optional[str] = None,
    ) -> CaptionResult:
        """ParsedElement의 이미지를 캡셔닝"""
        if not element.image_data:
            raise ValueError(f"요소 {element.element_id}에 이미지 데이터가 없습니다.")

        caption = await self.caption(element.image_data, context)

        # 간단한 요약 생성 (Gemini로)
        summary = await self._generate_summary(caption)

        return CaptionResult(
            element_id=element.element_id,
            original_content=element.content,
            caption=caption,
            summary=summary,
            key_values={},  # Gemini는 JSON 모드 제한적
            metadata={
                "model": self.model,
                "element_type": element.element_type.value,
                "page": element.page,
            },
        )

    async def _generate_summary(self, caption: str) -> str:
        """캡션에서 핵심 요약 생성"""
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"다음 차트/그래프 분석 내용을 1-2문장으로 핵심만 요약해주세요:\n\n{caption}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 200,
            }
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_url}?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )

            if response.status_code != 200:
                return caption[:200]

            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return caption[:200]


class BatchImageCaptioner:
    """여러 이미지를 배치로 캡셔닝"""

    def __init__(
        self,
        captioner: ImageCaptioner,
        max_concurrent: int = 3,
    ):
        self.captioner = captioner
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def caption_elements(
        self,
        elements: list[ParsedElement],
        context_map: Optional[dict[str, str]] = None,
    ) -> list[CaptionResult]:
        """여러 요소를 동시에 캡셔닝"""
        image_elements = [
            e for e in elements
            if e.element_type in (ElementType.IMAGE, ElementType.CHART)
            and e.image_data
        ]

        if not image_elements:
            return []

        context_map = context_map or {}

        async def caption_with_limit(element: ParsedElement) -> CaptionResult:
            async with self._semaphore:
                context = context_map.get(element.element_id)
                return await self.captioner.caption_element(element, context)

        tasks = [caption_with_limit(e) for e in image_elements]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_results = []
        for result in results:
            if isinstance(result, CaptionResult):
                successful_results.append(result)
            elif isinstance(result, Exception):
                print(f"캡셔닝 오류: {result}")

        return successful_results
