"""
Upstage Document Parse API를 사용한 이미지 OCR
Azure OCR의 대안으로 사용
"""

import asyncio
import base64
import httpx
from dataclasses import dataclass
from typing import Optional


@dataclass
class UpstageOCRResult:
    """Upstage OCR 결과"""
    text: str
    confidence: float
    tables: list
    word_count: int
    error: Optional[str] = None


class UpstageOCR:
    """Upstage Document Parse API 기반 이미지 OCR"""

    API_URL = "https://api.upstage.ai/v1/document-ai/document-parse"

    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: Upstage API 키 (없으면 환경변수에서 로드)
        """
        import os
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        self._initialized = False

    def initialize(self) -> bool:
        """초기화 및 API 키 검증"""
        if not self.api_key:
            print("  [Upstage OCR] API 키가 설정되지 않았습니다.")
            return False
        self._initialized = True
        return True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def analyze_image(self, image_bytes: bytes) -> UpstageOCRResult:
        """
        이미지에서 텍스트/테이블 추출

        Args:
            image_bytes: 이미지 바이트 데이터

        Returns:
            UpstageOCRResult
        """
        if not self._initialized:
            return UpstageOCRResult(
                text="", confidence=0.0, tables=[], word_count=0,
                error="OCR이 초기화되지 않았습니다."
            )

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }

            # 이미지를 파일로 전송
            files = {
                "document": ("image.png", image_bytes, "image/png"),
            }

            data = {
                "ocr": "true",
                "coordinates": "false",
                "output_formats": "text,html",
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.API_URL,
                    headers=headers,
                    files=files,
                    data=data,
                )

            if response.status_code != 200:
                return UpstageOCRResult(
                    text="", confidence=0.0, tables=[], word_count=0,
                    error=f"API 오류: {response.status_code}"
                )

            result = response.json()
            return self._process_response(result)

        except Exception as e:
            return UpstageOCRResult(
                text="", confidence=0.0, tables=[], word_count=0,
                error=str(e)
            )

    def analyze_image_sync(self, image_bytes: bytes) -> UpstageOCRResult:
        """동기 버전의 이미지 분석"""
        return asyncio.run(self.analyze_image(image_bytes))

    def _process_response(self, response: dict) -> UpstageOCRResult:
        """API 응답 처리"""
        elements = response.get("elements", [])

        text_parts = []
        tables = []
        total_confidence = 0.0
        confidence_count = 0

        for elem in elements:
            category = elem.get("category", "").lower()
            content = elem.get("content", {})

            # 텍스트 추출
            text = content.get("text", "")
            if text:
                text_parts.append(text)

            # 테이블 추출
            if category == "table":
                html = content.get("html", "")
                markdown = content.get("markdown", "")
                tables.append({
                    "html": html,
                    "markdown": markdown,
                    "text": text,
                })

            # 신뢰도 수집
            confidence = elem.get("confidence")
            if confidence is not None:
                total_confidence += confidence
                confidence_count += 1

        combined_text = " ".join(text_parts)
        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.8

        return UpstageOCRResult(
            text=combined_text,
            confidence=avg_confidence,
            tables=tables,
            word_count=len(combined_text.split()),
        )


class UpstageHybridImageProcessor:
    """Upstage OCR + VLM 하이브리드 이미지 프로세서"""

    def __init__(
        self,
        upstage_api_key: str,
        openai_api_key: str,
        vlm_model: str = "gpt-4o",
        ocr_confidence_threshold: float = 0.7,
    ):
        from .image_captioner import OpenAIImageCaptioner

        self.ocr = UpstageOCR(api_key=upstage_api_key)
        self.vlm = OpenAIImageCaptioner(api_key=openai_api_key, model=vlm_model)
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self._initialized = False

    def initialize(self) -> bool:
        """초기화"""
        self._initialized = self.ocr.initialize()
        return self._initialized

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def process_image(
        self,
        image_data: bytes,
        context: Optional[str] = None,
        element_id: str = "",
    ) -> dict:
        """
        이미지를 하이브리드 방식으로 처리

        Returns:
            dict with keys: element_id, ocr_text, ocr_confidence, ocr_tables,
                           vlm_caption, vlm_summary, combined_content, metadata
        """
        # 1. Upstage OCR로 텍스트/테이블 추출
        ocr_result = await self.ocr.analyze_image(image_data)

        ocr_text = ocr_result.text
        ocr_confidence = ocr_result.confidence
        ocr_tables = ocr_result.tables

        # 2. VLM으로 차트 해석
        vlm_caption = ""
        vlm_summary = ""
        try:
            # OCR 결과를 VLM 컨텍스트에 추가
            enhanced_context = context or ""
            if ocr_text and ocr_confidence >= self.ocr_confidence_threshold:
                enhanced_context += f"\n\n[OCR 추출 텍스트]\n{ocr_text[:1000]}"

            vlm_caption = await self.vlm.caption(image_data, enhanced_context)
            sentences = vlm_caption.split('.')
            vlm_summary = '.'.join(sentences[:2]) + '.' if sentences else vlm_caption[:200]
        except Exception as e:
            print(f"VLM 캡셔닝 실패: {e}")
            vlm_caption = ocr_text

        # 3. 결과 통합
        combined_content = self._combine_results(
            ocr_text, ocr_confidence, ocr_tables, vlm_caption
        )

        return {
            "element_id": element_id,
            "ocr_text": ocr_text,
            "ocr_confidence": ocr_confidence,
            "ocr_tables": ocr_tables,
            "vlm_caption": vlm_caption,
            "vlm_summary": vlm_summary,
            "combined_content": combined_content,
            "metadata": {
                "ocr_word_count": ocr_result.word_count,
                "table_count": len(ocr_tables),
                "has_vlm": bool(vlm_caption),
                "ocr_provider": "upstage",
            }
        }

    def _combine_results(
        self,
        ocr_text: str,
        ocr_confidence: float,
        ocr_tables: list,
        vlm_caption: str,
    ) -> str:
        """OCR과 VLM 결과를 통합"""
        parts = []

        # VLM 캡션
        if vlm_caption:
            parts.append("[차트 분석]\n" + vlm_caption)

        # OCR 테이블
        for i, table in enumerate(ocr_tables):
            if table.get('markdown'):
                parts.append(f"[테이블 {i+1}]\n{table['markdown']}")

        # OCR 텍스트
        if ocr_text and ocr_confidence >= self.ocr_confidence_threshold:
            parts.append(f"[OCR 추출 텍스트 (신뢰도: {ocr_confidence:.2f})]\n{ocr_text[:500]}")

        return "\n\n".join(parts) if parts else vlm_caption or ocr_text
