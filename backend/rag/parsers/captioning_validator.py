# -*- coding: utf-8 -*-
"""
캡셔닝 검증 및 재시도 모듈

인덱싱 시 이미지 캡셔닝 품질을 검증하고 실패 시 재시도
"""

import asyncio
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CaptioningStats:
    """캡셔닝 통계"""
    total: int = 0
    success: int = 0
    failed: int = 0
    retried: int = 0
    failed_items: List[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return (self.success / self.total * 100) if self.total > 0 else 0


class CaptioningValidator:
    """캡셔닝 검증기"""

    # 캡셔닝 실패 패턴
    FAILURE_PATTERNS = [
        "죄송하지만",
        "이미지를 분석할 수 없습니다",
        "정보를 제공할 수 없습니다",
        "분석을 제공할 수 없습니다",
        "설명할 수 없습니다",
        "사람을 식별할 수 없습니다",
        "사람이 누구인지 알 수 없습니다",
        "I cannot",
        "I'm unable to",
        "Sorry, I cannot",
    ]

    # 최소 유효 콘텐츠 길이
    MIN_CONTENT_LENGTH = 50

    # 최대 재시도 횟수
    MAX_RETRIES = 3

    # 재시도 간 대기 시간 (초)
    RETRY_DELAY = 2.0

    def __init__(self):
        self.stats = CaptioningStats()

    def is_caption_failed(self, caption: str) -> Tuple[bool, str]:
        """
        캡셔닝 실패 여부 확인

        Returns:
            (실패 여부, 실패 이유)
        """
        if not caption:
            return True, "빈 캡션"

        caption_lower = caption.lower()

        # 실패 패턴 확인
        for pattern in self.FAILURE_PATTERNS:
            if pattern.lower() in caption_lower:
                return True, f"실패 패턴 감지: {pattern}"

        # 최소 길이 확인 (실패 메시지만 있는 경우 방지)
        # [IMAGE] 태그와 메타데이터 제외한 실제 콘텐츠 길이
        content_only = caption
        for tag in ["[IMAGE]", "[차트 분석]", "## 유형", "## 제목", "## 주요 내용"]:
            content_only = content_only.replace(tag, "")

        if len(content_only.strip()) < self.MIN_CONTENT_LENGTH:
            return True, f"콘텐츠 부족 (길이: {len(content_only.strip())})"

        return False, ""

    def extract_useful_content(self, caption: str) -> str:
        """실패한 캡션에서 유용한 부분만 추출"""
        lines = caption.split('\n')
        useful_lines = []

        for line in lines:
            # 실패 메시지 라인 제외
            is_failure_line = any(
                pattern.lower() in line.lower()
                for pattern in self.FAILURE_PATTERNS
            )
            if not is_failure_line and line.strip():
                useful_lines.append(line)

        return '\n'.join(useful_lines)

    async def caption_with_retry(
        self,
        captioner,
        image_data: bytes,
        context: Optional[str] = None,
        element_id: str = "",
    ) -> Tuple[str, bool]:
        """
        재시도 로직이 포함된 캡셔닝

        Returns:
            (캡션 결과, 성공 여부)
        """
        self.stats.total += 1
        last_error = ""

        for attempt in range(self.MAX_RETRIES):
            try:
                caption = await captioner.caption(image_data, context)

                failed, reason = self.is_caption_failed(caption)

                if not failed:
                    self.stats.success += 1
                    return caption, True

                last_error = reason
                logger.warning(
                    f"[{element_id}] 캡셔닝 실패 (시도 {attempt + 1}/{self.MAX_RETRIES}): {reason}"
                )

                if attempt < self.MAX_RETRIES - 1:
                    self.stats.retried += 1
                    await asyncio.sleep(self.RETRY_DELAY)

            except Exception as e:
                last_error = str(e)
                logger.error(f"[{element_id}] 캡셔닝 예외: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY)

        # 모든 재시도 실패
        self.stats.failed += 1
        self.stats.failed_items.append({
            "element_id": element_id,
            "reason": last_error,
            "timestamp": datetime.now().isoformat()
        })

        # 유용한 부분이라도 추출
        if caption:
            useful = self.extract_useful_content(caption)
            if useful:
                return useful, False

        return "", False

    async def caption_with_fallback(
        self,
        vlm_captioner,
        ocr_processor,
        image_data: bytes,
        context: Optional[str] = None,
        element_id: str = "",
    ) -> Tuple[str, str]:
        """
        VLM + OCR 폴백이 포함된 캡셔닝

        Returns:
            (최종 캡션, 사용된 방법)
        """
        # 1. VLM 캡셔닝 시도 (재시도 포함)
        caption, success = await self.caption_with_retry(
            vlm_captioner, image_data, context, element_id
        )

        if success:
            return caption, "vlm"

        # 2. VLM 실패 시 OCR로 폴백
        if ocr_processor and ocr_processor.is_initialized:
            try:
                ocr_result = ocr_processor.analyze_image(image_data)
                ocr_text = ocr_result.get('text', '')
                ocr_tables = ocr_result.get('tables', [])

                if ocr_text or ocr_tables:
                    fallback_content = f"[IMAGE]\n[OCR 추출]\n{ocr_text}"
                    if ocr_tables:
                        fallback_content += f"\n\n[테이블]\n{ocr_tables}"

                    # VLM에서 추출한 유용한 부분 추가
                    if caption:
                        fallback_content = f"{caption}\n\n{fallback_content}"

                    logger.info(f"[{element_id}] OCR 폴백 사용")
                    return fallback_content, "ocr_fallback"

            except Exception as e:
                logger.error(f"[{element_id}] OCR 폴백 실패: {e}")

        # 3. 모두 실패 시 VLM 결과라도 반환 (경고 포함)
        if caption:
            return f"[WARNING: 캡셔닝 품질 낮음]\n{caption}", "partial"

        return "[ERROR: 이미지 처리 실패]", "failed"

    def get_report(self) -> str:
        """캡셔닝 결과 리포트"""
        report = f"""
========================================
캡셔닝 품질 리포트
========================================
전체: {self.stats.total}개
성공: {self.stats.success}개 ({self.stats.success_rate:.1f}%)
실패: {self.stats.failed}개
재시도: {self.stats.retried}회

실패 목록:
"""
        for item in self.stats.failed_items[:20]:  # 상위 20개만
            report += f"  - [{item['element_id']}] {item['reason']}\n"

        if len(self.stats.failed_items) > 20:
            report += f"  ... 외 {len(self.stats.failed_items) - 20}개\n"

        return report

    def save_failed_items(self, filepath: str):
        """실패 항목을 파일로 저장"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "stats": {
                    "total": self.stats.total,
                    "success": self.stats.success,
                    "failed": self.stats.failed,
                    "success_rate": self.stats.success_rate
                },
                "failed_items": self.stats.failed_items
            }, f, ensure_ascii=False, indent=2)
