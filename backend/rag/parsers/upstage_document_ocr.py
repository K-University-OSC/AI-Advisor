"""
Upstage Document OCR API
순수 OCR 기능 - 이미지에서 텍스트 추출에 특화

Document Parse와 달리 구조화 없이 순수 텍스트 추출만 수행
더 빠르고 저렴할 것으로 예상
"""

import asyncio
import base64
import httpx
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class UpstageDocOCRResult:
    """Upstage Document OCR 결과"""
    text: str
    confidence: float
    words: list  # 단어별 정보 (text, confidence, bbox)
    word_count: int
    error: Optional[str] = None


class UpstageDocumentOCR:
    """Upstage Document OCR API 클라이언트

    Document Parse와 달리 순수 OCR만 수행:
    - 더 빠른 응답
    - 더 저렴한 비용
    - 단어별 신뢰도 점수 제공
    """

    API_URL = "https://api.upstage.ai/v1/document-ai/ocr"

    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: Upstage API 키 (없으면 환경변수에서 로드)
        """
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        self._initialized = False

    def initialize(self) -> bool:
        """초기화 및 API 키 검증"""
        if not self.api_key:
            print("  [Upstage Document OCR] API 키가 설정되지 않았습니다.")
            return False
        self._initialized = True
        print("  [Upstage Document OCR] 초기화 완료")
        return True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def analyze_image(self, image_bytes: bytes) -> UpstageDocOCRResult:
        """
        이미지에서 텍스트 추출 (Document OCR API)

        Args:
            image_bytes: 이미지 바이트 데이터

        Returns:
            UpstageDocOCRResult
        """
        if not self._initialized:
            return UpstageDocOCRResult(
                text="", confidence=0.0, words=[], word_count=0,
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

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.API_URL,
                    headers=headers,
                    files=files,
                )

            if response.status_code != 200:
                return UpstageDocOCRResult(
                    text="", confidence=0.0, words=[], word_count=0,
                    error=f"API 오류: {response.status_code} - {response.text}"
                )

            result = response.json()
            return self._process_response(result)

        except Exception as e:
            return UpstageDocOCRResult(
                text="", confidence=0.0, words=[], word_count=0,
                error=str(e)
            )

    def analyze_image_sync(self, image_bytes: bytes) -> UpstageDocOCRResult:
        """동기 버전의 이미지 분석"""
        return asyncio.run(self.analyze_image(image_bytes))

    def _process_response(self, response: dict) -> UpstageDocOCRResult:
        """API 응답 처리"""
        # Document OCR API 응답 구조 파싱
        pages = response.get("pages", [])

        all_text = []
        all_words = []
        total_confidence = 0.0
        confidence_count = 0

        for page in pages:
            # 페이지별 텍스트
            page_text = page.get("text", "")
            if page_text:
                all_text.append(page_text)

            # 단어별 정보
            words = page.get("words", [])
            for word in words:
                word_info = {
                    "text": word.get("text", ""),
                    "confidence": word.get("confidence", 0.0),
                    "bbox": word.get("boundingBox", {}),
                }
                all_words.append(word_info)

                # 신뢰도 집계
                conf = word.get("confidence", 0.0)
                if conf > 0:
                    total_confidence += conf
                    confidence_count += 1

        combined_text = " ".join(all_text)
        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0

        return UpstageDocOCRResult(
            text=combined_text,
            confidence=avg_confidence,
            words=all_words,
            word_count=len(all_words),
        )


class UpstageDocOCRHybridProcessor:
    """Upstage Document OCR + VLM 하이브리드 이미지 프로세서

    Document Parse 대신 Document OCR 사용:
    - 더 빠른 OCR 처리
    - 단어별 신뢰도 활용 가능
    """

    def __init__(
        self,
        upstage_api_key: str,
        openai_api_key: str,
        vlm_model: str = "gpt-4o",
        ocr_confidence_threshold: float = 0.7,
    ):
        from .image_captioner import OpenAIImageCaptioner

        self.ocr = UpstageDocumentOCR(api_key=upstage_api_key)
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
            dict with keys: element_id, ocr_text, ocr_confidence,
                           vlm_caption, vlm_summary, combined_content, metadata
        """
        # 1. Upstage Document OCR로 텍스트 추출
        ocr_result = await self.ocr.analyze_image(image_data)

        ocr_text = ocr_result.text
        ocr_confidence = ocr_result.confidence
        word_count = ocr_result.word_count

        # 2. VLM으로 이미지 분석
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
            ocr_text, ocr_confidence, vlm_caption
        )

        return {
            "element_id": element_id,
            "ocr_text": ocr_text,
            "ocr_confidence": ocr_confidence,
            "ocr_words": ocr_result.words[:20],  # 상위 20개 단어만
            "vlm_caption": vlm_caption,
            "vlm_summary": vlm_summary,
            "combined_content": combined_content,
            "metadata": {
                "ocr_word_count": word_count,
                "has_vlm": bool(vlm_caption),
                "ocr_provider": "upstage_document_ocr",
            }
        }

    def _combine_results(
        self,
        ocr_text: str,
        ocr_confidence: float,
        vlm_caption: str,
    ) -> str:
        """OCR과 VLM 결과를 통합"""
        parts = []

        # VLM 캡션
        if vlm_caption:
            parts.append("[이미지 분석]\n" + vlm_caption)

        # OCR 텍스트 (신뢰도가 높은 경우)
        if ocr_text and ocr_confidence >= self.ocr_confidence_threshold:
            parts.append(f"[OCR 추출 텍스트 (신뢰도: {ocr_confidence:.2f})]\n{ocr_text[:500]}")

        return "\n\n".join(parts) if parts else vlm_caption or ocr_text
