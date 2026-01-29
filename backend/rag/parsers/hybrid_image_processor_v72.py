"""
V7.2 하이브리드 이미지 프로세서

Upstage Document OCR + VLM 결합:
- Upstage Document OCR: 빠른 텍스트 추출 (Azure 대비 9배 빠름)
- VLM: 차트 해석 및 인사이트 생성

V7.1 대비 변경점:
- Azure OCR → Upstage Document OCR
- 속도 향상 (4524ms → 486ms)
- 유사한 신뢰도 (0.95)
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from .upstage_document_ocr import UpstageDocumentOCR, UpstageDocOCRResult
from .image_captioner import OpenAIImageCaptioner, CaptionResult
from .document_parser import ParsedElement, ElementType


@dataclass
class HybridImageResultV72:
    """V7.2 하이브리드 이미지 처리 결과"""
    element_id: str
    ocr_text: str
    ocr_confidence: float
    ocr_word_count: int
    vlm_caption: str
    vlm_summary: str
    combined_content: str
    metadata: dict


class HybridImageProcessorV72:
    """V7.2: Upstage Document OCR + VLM 하이브리드 이미지 프로세서

    V7.1 (Azure OCR) 대비:
    - 속도: 9배 빠름 (4524ms → 486ms)
    - 신뢰도: 동등 (0.95)
    - 비용: Upstage API 사용
    """

    def __init__(
        self,
        openai_api_key: str,
        upstage_api_key: str,
        vlm_model: str = "gpt-4o",
        ocr_confidence_threshold: float = 0.7,
    ):
        """
        Args:
            openai_api_key: OpenAI API 키 (VLM용)
            upstage_api_key: Upstage API 키 (OCR용)
            vlm_model: VLM 모델 이름
            ocr_confidence_threshold: OCR 신뢰도 임계값
        """
        self.ocr = UpstageDocumentOCR(api_key=upstage_api_key)
        self.vlm = OpenAIImageCaptioner(api_key=openai_api_key, model=vlm_model)
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self._ocr_initialized = False

    def initialize(self) -> bool:
        """초기화"""
        self._ocr_initialized = self.ocr.initialize()
        return self._ocr_initialized

    @property
    def is_initialized(self) -> bool:
        return self._ocr_initialized

    async def process_image(
        self,
        image_data: bytes,
        context: Optional[str] = None,
        element_id: str = "",
    ) -> HybridImageResultV72:
        """
        이미지를 하이브리드 방식으로 처리

        Args:
            image_data: 이미지 바이트 데이터
            context: 문서 컨텍스트
            element_id: 요소 ID

        Returns:
            HybridImageResultV72
        """
        # 1. Upstage Document OCR로 텍스트 추출
        ocr_result = UpstageDocOCRResult(text='', confidence=0.0, words=[], word_count=0)
        if self._ocr_initialized:
            ocr_result = await self.ocr.analyze_image(image_data)

        ocr_text = ocr_result.text
        ocr_confidence = ocr_result.confidence
        ocr_word_count = ocr_result.word_count

        # 2. VLM으로 차트 해석
        vlm_caption = ""
        vlm_summary = ""
        try:
            # OCR 결과를 VLM 컨텍스트에 추가
            enhanced_context = context or ""
            if ocr_text and ocr_confidence >= self.ocr_confidence_threshold:
                enhanced_context += f"\n\n[OCR 추출 텍스트]\n{ocr_text[:1000]}"

            vlm_caption = await self.vlm.caption(image_data, enhanced_context)
            # 요약은 캡션에서 첫 2문장 추출
            sentences = vlm_caption.split('.')
            vlm_summary = '.'.join(sentences[:2]) + '.' if sentences else vlm_caption[:200]
        except Exception as e:
            print(f"VLM 캡셔닝 실패: {e}")
            vlm_caption = ocr_text  # OCR 텍스트로 대체

        # 3. 결과 통합
        combined_content = self._combine_results(
            ocr_text, ocr_confidence, vlm_caption
        )

        return HybridImageResultV72(
            element_id=element_id,
            ocr_text=ocr_text,
            ocr_confidence=ocr_confidence,
            ocr_word_count=ocr_word_count,
            vlm_caption=vlm_caption,
            vlm_summary=vlm_summary,
            combined_content=combined_content,
            metadata={
                'ocr_word_count': ocr_word_count,
                'ocr_provider': 'upstage_document_ocr',
                'has_vlm': bool(vlm_caption),
                'version': 'v7.2',
            }
        )

    async def process_element(
        self,
        element: ParsedElement,
        context: Optional[str] = None,
    ) -> HybridImageResultV72:
        """ParsedElement 처리"""
        if not element.image_data:
            raise ValueError(f"요소 {element.element_id}에 이미지 데이터가 없습니다.")

        return await self.process_image(
            image_data=element.image_data,
            context=context,
            element_id=element.element_id,
        )

    def _combine_results(
        self,
        ocr_text: str,
        ocr_confidence: float,
        vlm_caption: str,
    ) -> str:
        """OCR과 VLM 결과를 통합"""
        parts = []

        # VLM 캡션 (차트 해석)
        if vlm_caption:
            parts.append("[차트 분석]\n" + vlm_caption)

        # OCR 텍스트 (신뢰도가 높은 경우)
        if ocr_text and ocr_confidence >= self.ocr_confidence_threshold:
            parts.append(f"[OCR 추출 텍스트 (신뢰도: {ocr_confidence:.2f})]\n{ocr_text[:500]}")

        return "\n\n".join(parts) if parts else vlm_caption or ocr_text


class BatchHybridProcessorV72:
    """여러 이미지를 배치로 처리 (V7.2)"""

    def __init__(
        self,
        processor: HybridImageProcessorV72,
        max_concurrent: int = 5,  # Upstage OCR이 빠르므로 동시성 증가
    ):
        self.processor = processor
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def process_elements(
        self,
        elements: list[ParsedElement],
        context_map: Optional[dict[str, str]] = None,
    ) -> list[HybridImageResultV72]:
        """여러 요소를 동시에 처리"""
        image_elements = [
            e for e in elements
            if e.element_type in (ElementType.IMAGE, ElementType.CHART)
            and e.image_data
        ]

        if not image_elements:
            return []

        context_map = context_map or {}

        async def process_with_limit(element: ParsedElement) -> HybridImageResultV72:
            async with self._semaphore:
                context = context_map.get(element.element_id)
                return await self.processor.process_element(element, context)

        tasks = [process_with_limit(e) for e in image_elements]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_results = []
        for result in results:
            if isinstance(result, HybridImageResultV72):
                successful_results.append(result)
            elif isinstance(result, Exception):
                print(f"V7.2 하이브리드 처리 오류: {result}")

        return successful_results
