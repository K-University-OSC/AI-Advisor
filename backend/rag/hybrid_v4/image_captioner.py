# -*- coding: utf-8 -*-
"""
HYBRID_V4 Image Captioner

GPT-4o Vision을 사용한 구조화된 이미지 캡셔닝

특징:
1. 이미지 유형 분류 (차트/그래프/표/다이어그램/사진/로고)
2. 주요 내용, 세부 사항, 문맥 연결 정보 추출
3. 주변 텍스트 컨텍스트 활용
4. 검색에 유용한 키워드 포함
"""

import base64
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# 구조화된 캡셔닝 프롬프트
CAPTION_PROMPT = """이 이미지를 분석하여 다음 형식으로 설명해주세요:

[이미지 유형]: (차트/그래프/표/다이어그램/사진/로고/기타 중 하나)
[주요 내용]: (이미지가 전달하는 핵심 정보를 2-3문장으로)
[세부 사항]: (숫자, 레이블, 범례 등 구체적인 데이터나 텍스트)
[문맥 연결]: (이 이미지가 문서에서 어떤 맥락에서 사용될 수 있는지)

한국어로 작성하고, 검색에 유용한 키워드를 포함해주세요."""


@dataclass
class CaptionResult:
    """이미지 캡션 결과"""
    image_type: str = "unknown"
    main_content: str = ""
    details: str = ""
    context: str = ""
    full_caption: str = ""
    page: int = 0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return bool(self.full_caption) and self.error is None

    def to_chunk_content(self, surrounding_text: str = "") -> str:
        """검색용 청크 콘텐츠 생성"""
        # 이미지 유형에 따른 접두사
        type_prefix = {
            '차트': '[차트/그래프]',
            '그래프': '[차트/그래프]',
            '표': '[표 이미지]',
            '다이어그램': '[다이어그램]',
            '사진': '[사진]',
            '로고': '[로고]',
        }.get(self.image_type, '[이미지]')

        content_parts = [type_prefix]

        if self.main_content:
            content_parts.append(f"내용: {self.main_content}")

        if self.details:
            content_parts.append(f"세부사항: {self.details}")

        if self.context:
            content_parts.append(f"문맥: {self.context}")

        # 주변 텍스트 일부 포함 (검색 연결성 강화)
        if surrounding_text:
            content_parts.append(f"[관련 텍스트] {surrounding_text[:300]}")

        return '\n'.join(content_parts)

    def to_dict(self) -> dict:
        return {
            "image_type": self.image_type,
            "main_content": self.main_content,
            "details": self.details,
            "context": self.context,
            "full_caption": self.full_caption,
            "page": self.page,
            "error": self.error,
            "metadata": self.metadata,
        }


class ImageCaptioner:
    """
    HYBRID_V4 Image Captioner

    GPT-4o Vision을 사용한 구조화된 이미지 캡셔닝
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        detail: str = "high",
        max_tokens: int = 500,
        temperature: float = 0.0
    ):
        """
        Args:
            model: OpenAI Vision 모델명
            detail: 이미지 디테일 수준 (low, high, auto)
            max_tokens: 최대 출력 토큰
            temperature: 생성 온도
        """
        self.model = model
        self.detail = detail
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.openai_client = None
        self._initialized = False

    def initialize(self, openai_client=None) -> bool:
        """
        초기화

        Args:
            openai_client: OpenAI 클라이언트 (없으면 새로 생성)
        """
        try:
            if openai_client:
                self.openai_client = openai_client
            else:
                import os
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
                self.openai_client = OpenAI(api_key=api_key)

            self._initialized = True
            logger.info(f"Image Captioner 초기화 완료 (model: {self.model})")
            return True

        except Exception as e:
            logger.error(f"Image Captioner 초기화 실패: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def caption_image(
        self,
        image_bytes: bytes,
        page_num: int = 0,
        surrounding_text: str = ""
    ) -> CaptionResult:
        """
        이미지 캡셔닝 수행

        Args:
            image_bytes: 이미지 바이트 데이터
            page_num: 페이지 번호
            surrounding_text: 주변 텍스트 (컨텍스트)

        Returns:
            CaptionResult
        """
        if not self._initialized:
            return CaptionResult(error="Image Captioner가 초기화되지 않았습니다.", page=page_num)

        try:
            # 이미지를 base64로 인코딩
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            # 컨텍스트 추가 프롬프트
            prompt = CAPTION_PROMPT
            if surrounding_text:
                prompt += f"\n\n[참고] 이 이미지 주변의 텍스트:\n{surrounding_text[:500]}"

            # GPT-4o Vision API 호출
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": self.detail
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            caption = response.choices[0].message.content.strip()

            # 캡션 파싱
            return self._parse_caption(caption, page_num)

        except Exception as e:
            logger.error(f"이미지 캡셔닝 오류: {e}")
            return CaptionResult(
                error=str(e),
                page=page_num
            )

    async def caption_image_async(
        self,
        image_bytes: bytes,
        page_num: int = 0,
        surrounding_text: str = ""
    ) -> CaptionResult:
        """
        비동기 이미지 캡셔닝

        Args:
            image_bytes: 이미지 바이트 데이터
            page_num: 페이지 번호
            surrounding_text: 주변 텍스트

        Returns:
            CaptionResult
        """
        if not self._initialized:
            return CaptionResult(error="Image Captioner가 초기화되지 않았습니다.", page=page_num)

        try:
            from openai import AsyncOpenAI
            import os

            async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            prompt = CAPTION_PROMPT
            if surrounding_text:
                prompt += f"\n\n[참고] 이 이미지 주변의 텍스트:\n{surrounding_text[:500]}"

            response = await async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": self.detail
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            caption = response.choices[0].message.content.strip()
            return self._parse_caption(caption, page_num)

        except Exception as e:
            logger.error(f"비동기 이미지 캡셔닝 오류: {e}")
            return CaptionResult(error=str(e), page=page_num)

    def _parse_caption(self, caption: str, page_num: int) -> CaptionResult:
        """캡션 파싱"""
        result = CaptionResult(
            full_caption=caption,
            page=page_num
        )

        lines = caption.split('\n')
        current_field = None

        for line in lines:
            line = line.strip()

            if line.startswith('[이미지 유형]'):
                value = line.replace('[이미지 유형]:', '').replace('[이미지 유형]', '').strip()
                result.image_type = value
            elif line.startswith('[주요 내용]'):
                value = line.replace('[주요 내용]:', '').replace('[주요 내용]', '').strip()
                result.main_content = value
                current_field = 'main_content'
            elif line.startswith('[세부 사항]'):
                value = line.replace('[세부 사항]:', '').replace('[세부 사항]', '').strip()
                result.details = value
                current_field = 'details'
            elif line.startswith('[문맥 연결]'):
                value = line.replace('[문맥 연결]:', '').replace('[문맥 연결]', '').strip()
                result.context = value
                current_field = 'context'
            elif current_field and line:
                # 현재 필드에 내용 추가
                if current_field == 'main_content':
                    result.main_content += ' ' + line
                elif current_field == 'details':
                    result.details += ' ' + line
                elif current_field == 'context':
                    result.context += ' ' + line

        return result
