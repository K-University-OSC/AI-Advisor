"""
하이브리드 이미지 프로세서

Azure OCR + VLM을 결합하여 이미지/차트 처리 정확도 향상:
- Azure OCR: 텍스트/테이블 추출 (높은 OCR 정확도)
- VLM: 차트 해석 및 인사이트 생성
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from .azure_ocr import AzureOCR
from .image_captioner import OpenAIImageCaptioner, CaptionResult
from .document_parser import ParsedElement, ElementType


@dataclass
class HybridImageResult:
    """하이브리드 이미지 처리 결과"""
    element_id: str
    ocr_text: str
    ocr_confidence: float
    ocr_tables: list
    vlm_caption: str
    vlm_summary: str
    combined_content: str
    metadata: dict


class HybridImageProcessor:
    """Azure OCR + VLM 하이브리드 이미지 프로세서"""

    def __init__(
        self,
        openai_api_key: str,
        vlm_model: str = "gpt-4o",
        azure_endpoint: str = None,
        azure_key: str = None,
        ocr_confidence_threshold: float = 0.7,
    ):
        """
        Args:
            openai_api_key: OpenAI API 키
            vlm_model: VLM 모델 이름
            azure_endpoint: Azure 엔드포인트
            azure_key: Azure API 키
            ocr_confidence_threshold: OCR 신뢰도 임계값
        """
        self.ocr = AzureOCR(endpoint=azure_endpoint, key=azure_key)
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
    ) -> HybridImageResult:
        """
        이미지를 하이브리드 방식으로 처리

        Args:
            image_data: 이미지 바이트 데이터
            context: 문서 컨텍스트
            element_id: 요소 ID

        Returns:
            HybridImageResult
        """
        # 1. Azure OCR로 텍스트/테이블 추출
        ocr_result = {'text': '', 'confidence': 0.0, 'tables': []}
        if self._ocr_initialized:
            ocr_result = self.ocr.analyze_image(image_data)

        ocr_text = ocr_result.get('text', '')
        ocr_confidence = ocr_result.get('confidence', 0.0)
        ocr_tables = ocr_result.get('tables', [])

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
            ocr_text, ocr_confidence, ocr_tables, vlm_caption
        )

        return HybridImageResult(
            element_id=element_id,
            ocr_text=ocr_text,
            ocr_confidence=ocr_confidence,
            ocr_tables=ocr_tables,
            vlm_caption=vlm_caption,
            vlm_summary=vlm_summary,
            combined_content=combined_content,
            metadata={
                'ocr_word_count': ocr_result.get('word_count', 0),
                'table_count': len(ocr_tables),
                'has_vlm': bool(vlm_caption),
            }
        )

    async def process_element(
        self,
        element: ParsedElement,
        context: Optional[str] = None,
    ) -> HybridImageResult:
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
        ocr_tables: list,
        vlm_caption: str,
    ) -> str:
        """OCR과 VLM 결과를 통합"""
        parts = []

        # VLM 캡션 (차트 해석)
        if vlm_caption:
            parts.append("[차트 분석]\n" + vlm_caption)

        # OCR 테이블 (높은 정확도의 테이블 데이터)
        for i, table in enumerate(ocr_tables):
            if table.get('markdown'):
                parts.append(f"[테이블 {i+1}]\n{table['markdown']}")

        # OCR 텍스트 (신뢰도가 높은 경우)
        if ocr_text and ocr_confidence >= self.ocr_confidence_threshold:
            # VLM 캡션에 이미 OCR 내용이 반영되었을 수 있으므로 간략히 추가
            parts.append(f"[OCR 추출 텍스트 (신뢰도: {ocr_confidence:.2f})]\n{ocr_text[:500]}")

        return "\n\n".join(parts) if parts else vlm_caption or ocr_text


class BatchHybridProcessor:
    """여러 이미지를 배치로 처리"""

    def __init__(
        self,
        processor: HybridImageProcessor,
        max_concurrent: int = 3,
    ):
        self.processor = processor
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def process_elements(
        self,
        elements: list[ParsedElement],
        context_map: Optional[dict[str, str]] = None,
    ) -> list[HybridImageResult]:
        """여러 요소를 동시에 처리"""
        image_elements = [
            e for e in elements
            if e.element_type in (ElementType.IMAGE, ElementType.CHART)
            and e.image_data
        ]

        if not image_elements:
            return []

        context_map = context_map or {}

        async def process_with_limit(element: ParsedElement) -> HybridImageResult:
            async with self._semaphore:
                context = context_map.get(element.element_id)
                return await self.processor.process_element(element, context)

        tasks = [process_with_limit(e) for e in image_elements]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_results = []
        for result in results:
            if isinstance(result, HybridImageResult):
                successful_results.append(result)
            elif isinstance(result, Exception):
                print(f"하이브리드 처리 오류: {result}")

        return successful_results
