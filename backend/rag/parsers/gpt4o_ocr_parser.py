# -*- coding: utf-8 -*-
"""
GPT-4o Vision OCR 기반 문서 파서

GPT-4o Vision을 사용한 고품질 OCR
- 페이지별 병렬 OCR 처리
- 마크다운 출력 지원
- [IMAGE: 설명] 형식으로 이미지/차트 설명
- 테이블 구조 보존

v1.6.8 기준 최고 성능: 73.3% (finance dataset)
"""

import asyncio
import base64
import os
import re
import uuid
from pathlib import Path
from typing import Optional, List

from openai import AsyncOpenAI

from .document_parser import (
    DocumentParser,
    ParsedDocument,
    ParsedElement,
    ElementType,
)


# OCR 시스템 프롬프트
OCR_SYSTEM_PROMPT = """당신은 문서 OCR 전문가입니다. 이미지에서 텍스트를 정확하게 추출하세요.

## 출력 규칙
1. 마크다운 형식으로 출력
2. 제목은 # ## ### 사용
3. 테이블은 마크다운 테이블 형식 (| col1 | col2 |)
   - 테이블의 모든 행과 열을 정확히 추출
   - 숫자, 날짜, 금액 등 데이터 정확히 기재
4. 목록은 - 또는 1. 2. 3. 사용
5. 이미지/차트/그래프는 [IMAGE: 상세 설명] 형식으로 내용을 설명
   - 차트의 경우 데이터, 트렌드, 수치를 포함
   - 다이어그램의 경우 구조와 관계를 설명
   - 모든 텍스트/레이블/범례를 정확히 추출
6. 원본 레이아웃 최대한 유지
7. 모든 수치, 날짜, 기관명, 법률명 등 정확히 기재

## 출력 형식
텍스트 내용만 출력 (추가 설명이나 코멘트 없이)"""


class GPT4oOCRParser(DocumentParser):
    """GPT-4o Vision 기반 문서 파서

    GPT-4o Vision을 사용한 고품질 OCR/문서 분석
    - 페이지별 병렬 처리
    - 이미지 설명 자동 생성
    - 테이블 구조 보존
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o",
        dpi: int = 150,
        ocr_concurrency: int = 5,
    ):
        """
        Args:
            api_key: OpenAI API 키 (없으면 환경변수에서 로드)
            model: GPT 모델명 (기본: gpt-4o)
            dpi: PDF 렌더링 DPI (기본: 150)
            ocr_concurrency: 페이지 OCR 병렬 수 (기본: 5)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.dpi = dpi
        self.ocr_concurrency = ocr_concurrency
        self.client = None

    def _ensure_client(self):
        """AsyncOpenAI 클라이언트 초기화"""
        if self.client is None:
            self.client = AsyncOpenAI(api_key=self.api_key)

    async def parse(self, file_path: str | Path) -> ParsedDocument:
        """파일 경로로부터 문서 파싱"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        with open(file_path, "rb") as f:
            content = f.read()

        return await self.parse_bytes(content, file_path.name)

    async def parse_bytes(
        self, content: bytes, filename: str
    ) -> ParsedDocument:
        """바이트 데이터로부터 문서 파싱"""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다")

        self._ensure_client()

        # PDF를 이미지로 변환
        images = await self._pdf_to_images(content)

        if not images:
            raise RuntimeError("PDF 이미지 변환 실패")

        total_pages = len(images)

        # 페이지별 병렬 OCR
        full_text, image_count = await self._ocr_all_pages(images, filename)

        # ParsedElement 생성 (전체 텍스트를 하나의 요소로)
        elements = [ParsedElement(
            element_id=str(uuid.uuid4()),
            element_type=ElementType.PARAGRAPH,
            content=full_text,
            page=1,
            markdown_content=full_text,
            metadata={
                "parser": "gpt4o-vision-ocr",
                "source": filename,
                "images": image_count,
                "total_pages": total_pages,
            }
        )]

        return ParsedDocument(
            source=filename,
            filename=filename,
            total_pages=total_pages,
            elements=elements,
            metadata={
                "parser": "gpt4o-vision-ocr",
                "model": self.model,
                "images": image_count,
            }
        )

    async def _pdf_to_images(self, content: bytes) -> List[bytes]:
        """PDF를 이미지 리스트로 변환"""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=content, filetype="pdf")
            images = []
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)

            for page in doc:
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                images.append(img_data)

            doc.close()
            return images

        except Exception as e:
            print(f"PDF 이미지 변환 실패: {e}")
            return []

    async def _ocr_all_pages(
        self, images: List[bytes], filename: str
    ) -> tuple[str, int]:
        """모든 페이지 OCR (병렬 처리)"""
        semaphore = asyncio.Semaphore(self.ocr_concurrency)

        async def ocr_with_limit(img_data: bytes, page_num: int) -> tuple:
            async with semaphore:
                text = await self._ocr_page(img_data, page_num)
                return page_num, text

        tasks = [
            ocr_with_limit(img_data, page_num)
            for page_num, img_data in enumerate(images, 1)
        ]
        results = await asyncio.gather(*tasks)

        # 페이지 순서대로 정렬 및 병합
        results.sort(key=lambda x: x[0])
        full_text = ""
        for page_num, text in results:
            if text.strip():
                full_text += f"\n\n<!-- Page {page_num} -->\n{text}"

        # [IMAGE: ...] 개수 카운트
        image_count = len(re.findall(r'\[IMAGE:', full_text))

        return full_text.strip(), image_count

    async def _ocr_page(self, image_data: bytes, page_num: int) -> str:
        """단일 페이지 OCR (GPT-4o Vision)"""
        img_base64 = base64.b64encode(image_data).decode()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": OCR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"이 문서 페이지({page_num}페이지)의 내용을 추출해주세요."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=8192,
                temperature=0
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"GPT-4o OCR 실패 (page {page_num}): {e}")
            return ""

    def _generate_element_id(
        self, filename: str, page_num: int, idx: int
    ) -> str:
        """요소 ID 생성"""
        return f"{filename}_p{page_num}_{idx}_{uuid.uuid4().hex[:8]}"
