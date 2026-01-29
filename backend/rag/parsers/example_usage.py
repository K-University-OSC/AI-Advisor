# -*- coding: utf-8 -*-
"""
CaptioningValidator 사용 예시

인덱싱 시 이 방식으로 캡셔닝 검증을 적용하면
품질 문제를 방지할 수 있습니다.
"""

import asyncio
from pathlib import Path


async def index_with_validation():
    """검증이 포함된 인덱싱 예시"""

    from rag.parsers import OpenAIImageCaptioner
    from rag.parsers.azure_ocr import AzureOCR
    from rag.parsers.captioning_validator import CaptioningValidator

    # 1. 캡셔너 및 검증기 초기화
    vlm = OpenAIImageCaptioner(api_key="...", model="gpt-4o")
    ocr = AzureOCR()
    ocr.initialize()

    validator = CaptioningValidator()

    # 2. 이미지 처리 (기존 방식)
    # caption = await vlm.caption(image_data)  # ❌ 실패해도 그대로 저장

    # 3. 이미지 처리 (검증 포함 - 권장)
    image_data = b"..."  # 이미지 바이트
    element_id = "page_5_image_2"

    caption, method = await validator.caption_with_fallback(
        vlm_captioner=vlm,
        ocr_processor=ocr,
        image_data=image_data,
        context="금융 보고서",
        element_id=element_id
    )

    # method: "vlm" | "ocr_fallback" | "partial" | "failed"
    print(f"사용된 방법: {method}")
    print(f"캡션 길이: {len(caption)}")

    # 4. 인덱싱 완료 후 리포트 출력
    print(validator.get_report())

    # 5. 실패 항목 저장 (재인덱싱 대상)
    validator.save_failed_items("failed_captions.json")


# ================================================================
# 기존 인덱싱 코드 수정 방법
# ================================================================

"""
[BEFORE] 기존 코드 (문제 발생 가능)
------------------------------------
async def process_image(self, image_data, context, element_id):
    caption = await self.vlm.caption(image_data, context)
    return caption  # 실패해도 그대로 반환


[AFTER] 개선된 코드 (검증 포함)
------------------------------------
from rag.parsers.captioning_validator import CaptioningValidator

class ImprovedImageProcessor:
    def __init__(self):
        self.validator = CaptioningValidator()

    async def process_image(self, image_data, context, element_id):
        caption, method = await self.validator.caption_with_fallback(
            vlm_captioner=self.vlm,
            ocr_processor=self.ocr,
            image_data=image_data,
            context=context,
            element_id=element_id
        )

        if method == "failed":
            # 완전 실패 시 로그에 기록하고 빈 값 대신 OCR 결과라도 반환
            logger.error(f"[{element_id}] 이미지 처리 완전 실패")

        return caption
"""


# ================================================================
# 인덱싱 후 품질 검사 스크립트
# ================================================================

def check_index_quality(collection_name: str):
    """인덱싱된 데이터 품질 검사"""
    from qdrant_client import QdrantClient

    client = QdrantClient(host='localhost', port=6333)

    # 실패 패턴이 포함된 데이터 검색
    failure_patterns = [
        "죄송하지만",
        "분석할 수 없습니다",
        "제공할 수 없습니다"
    ]

    offset = None
    failed_points = []

    while True:
        results = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True
        )

        points, offset = results
        if not points:
            break

        for point in points:
            content = str(point.payload.get('content', ''))
            for pattern in failure_patterns:
                if pattern in content:
                    failed_points.append({
                        'id': point.id,
                        'pattern': pattern,
                        'content_preview': content[:200]
                    })
                    break

        if offset is None:
            break

    print(f"품질 문제가 있는 데이터: {len(failed_points)}개")
    for fp in failed_points[:10]:
        print(f"  - ID: {fp['id']}")
        print(f"    패턴: {fp['pattern']}")
        print(f"    미리보기: {fp['content_preview'][:100]}...")
        print()

    return failed_points


if __name__ == "__main__":
    # 품질 검사 실행
    check_index_quality("mh_rag_finance_v7_1")
