# -*- coding: utf-8 -*-
"""
Gemini Vision OCR 기반 문서 파서

Google Gemini 3 Flash를 사용한 고품질 OCR
- OCR Arena 1위 성능
- 마크다운 출력 지원
- 테이블/이미지 분석
"""

import asyncio
import base64
import json
import os
import re
import uuid
from pathlib import Path
from typing import Optional
import httpx

from .document_parser import (
    DocumentParser,
    ParsedDocument,
    ParsedElement,
    ElementType,
    BoundingBox,
)


class GeminiOCRParser(DocumentParser):
    """Gemini Vision 기반 문서 파서

    Gemini 3 Flash를 사용한 고품질 OCR/문서 분석
    - OCR Arena 리더보드 1위
    - 인쇄물 95%+ 정확도
    - 손글씨 85%+ 정확도
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-3-flash-preview",
        extract_tables: bool = True,
        extract_images: bool = True,
        output_format: str = "markdown",
    ):
        """
        Args:
            api_key: Google API 키 (없으면 환경변수에서 로드)
            model: Gemini 모델명 (기본: gemini-3-flash-preview)
            extract_tables: 테이블 추출 여부
            extract_images: 이미지 설명 추출 여부
            output_format: 출력 형식 (markdown, text, structured)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.model = model
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.output_format = output_format
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

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
            raise RuntimeError("GOOGLE_API_KEY가 설정되지 않았습니다")

        # PDF를 이미지로 변환
        images = await self._pdf_to_images(content)

        if not images:
            raise RuntimeError("PDF 이미지 변환 실패")

        elements: list[ParsedElement] = []
        total_pages = len(images)

        # 각 페이지를 Gemini로 OCR
        for page_num, image_data in enumerate(images, 1):
            page_elements = await self._ocr_page(
                image_data, page_num, filename
            )
            elements.extend(page_elements)

        return ParsedDocument(
            source=filename,
            filename=filename,
            total_pages=total_pages,
            elements=elements,
            metadata={
                "parser": "gemini-ocr",
                "model": self.model,
            }
        )

    async def _pdf_to_images(self, content: bytes) -> list[bytes]:
        """PDF를 이미지 리스트로 변환"""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=content, filetype="pdf")
            images = []

            for page in doc:
                # 고해상도 렌더링 (DPI 150)
                mat = fitz.Matrix(150 / 72, 150 / 72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                images.append(img_data)

            doc.close()
            return images

        except Exception as e:
            print(f"PDF 이미지 변환 실패: {e}")
            return []

    async def _ocr_page(
        self, image_data: bytes, page_num: int, filename: str
    ) -> list[ParsedElement]:
        """단일 페이지 OCR"""

        # 시스템 프롬프트
        system_prompt = """당신은 문서 OCR 전문가입니다. 이미지에서 텍스트를 정확하게 추출하세요.

## 출력 규칙
1. 마크다운 형식으로 출력
2. 제목은 # ## ### 사용
3. 테이블은 마크다운 테이블 형식
4. 목록은 - 또는 1. 2. 3. 사용
5. 이미지/차트는 [IMAGE: 설명] 형식
6. 원본 레이아웃 최대한 유지

## 출력 형식
텍스트 내용만 출력 (추가 설명 없이)"""

        # Base64 인코딩
        image_base64 = base64.b64encode(image_data).decode()

        # API 요청
        payload = {
            "contents": [{
                "parts": [
                    {"text": system_prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 8192,
            }
        }

        try:
            url = f"{self.api_url}?key={self.api_key}"

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload
                )

            if response.status_code != 200:
                print(f"Gemini API 오류: {response.status_code} - {response.text}")
                return []

            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]

            # 텍스트를 요소로 파싱
            return self._parse_markdown_to_elements(text, page_num, filename)

        except Exception as e:
            print(f"Gemini OCR 실패 (page {page_num}): {e}")
            return []

    def _parse_markdown_to_elements(
        self, text: str, page_num: int, filename: str
    ) -> list[ParsedElement]:
        """마크다운 텍스트를 ParsedElement 리스트로 변환"""
        elements = []
        lines = text.split('\n')
        idx = 0
        current_heading = None

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            element_type = ElementType.PARAGRAPH
            heading_level = None
            content = line

            # 헤딩 감지
            if line.startswith('###'):
                element_type = ElementType.HEADING
                heading_level = 3
                content = line.lstrip('#').strip()
                current_heading = content
            elif line.startswith('##'):
                element_type = ElementType.HEADING
                heading_level = 2
                content = line.lstrip('#').strip()
                current_heading = content
            elif line.startswith('#'):
                element_type = ElementType.HEADING
                heading_level = 1
                content = line.lstrip('#').strip()
                current_heading = content

            # 테이블 감지
            elif line.startswith('|'):
                table_lines = [line]
                i += 1
                while i < len(lines) and lines[i].strip().startswith('|'):
                    table_lines.append(lines[i].strip())
                    i += 1
                i -= 1  # 다음 반복에서 i += 1 되므로

                element_type = ElementType.TABLE
                content = '\n'.join(table_lines)

            # 이미지/차트 감지
            elif line.startswith('[IMAGE:') or line.startswith('[CHART:'):
                element_type = ElementType.IMAGE if '[IMAGE:' in line else ElementType.CHART
                content = line

            # 리스트 감지
            elif line.startswith('- ') or re.match(r'^\d+\.', line):
                element_type = ElementType.LIST
                list_lines = [line]
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith('- ') or re.match(r'^\d+\.', next_line):
                        list_lines.append(next_line)
                        i += 1
                    else:
                        break
                i -= 1
                content = '\n'.join(list_lines)

            element = ParsedElement(
                element_id=self._generate_element_id(filename, page_num, idx),
                element_type=element_type,
                content=content,
                page=page_num,
                markdown_content=content,
                heading_level=heading_level,
                parent_heading=current_heading if element_type != ElementType.HEADING else None,
            )
            elements.append(element)
            idx += 1
            i += 1

        return elements
