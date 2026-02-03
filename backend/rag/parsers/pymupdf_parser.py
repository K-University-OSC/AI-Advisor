# -*- coding: utf-8 -*-
"""
PyMuPDF4LLM 기반 문서 파서

무료 오픈소스 PDF 파서 - RAG 시스템에 최적화
- 마크다운 출력 지원
- 테이블/이미지 추출
- 빠른 속도
"""

import asyncio
import base64
import io
import re
import uuid
from pathlib import Path
from typing import Optional

from .document_parser import (
    DocumentParser,
    ParsedDocument,
    ParsedElement,
    ElementType,
    BoundingBox,
)


class PyMuPDFParser(DocumentParser):
    """PyMuPDF4LLM 기반 문서 파서

    Azure Document Intelligence와 동일한 인터페이스 제공
    완전 무료, 로컬 처리
    """

    def __init__(
        self,
        extract_images: bool = True,
        extract_tables: bool = True,
        page_chunks: bool = False,
        write_images: bool = False,
        image_path: Optional[str] = None,
    ):
        """
        Args:
            extract_images: 이미지 추출 여부
            extract_tables: 테이블 추출 여부
            page_chunks: 페이지별 청킹 여부
            write_images: 이미지 파일 저장 여부
            image_path: 이미지 저장 경로
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.page_chunks = page_chunks
        self.write_images = write_images
        self.image_path = image_path
        self._initialized = False

    def _initialize(self) -> bool:
        """라이브러리 초기화 확인"""
        if self._initialized:
            return True

        try:
            import pymupdf4llm
            import fitz  # PyMuPDF
            self._initialized = True
            return True
        except ImportError as e:
            print(f"PyMuPDF 패키지 설치 필요: {e}")
            return False

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
        if not self._initialize():
            raise RuntimeError("PyMuPDF 초기화 실패")

        # 동기 API를 비동기로 실행
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._parse_document, content, filename
        )

        return result

    def _parse_document(self, content: bytes, filename: str) -> ParsedDocument:
        """문서 파싱 (동기)"""
        import pymupdf4llm
        import fitz

        # PDF 열기
        doc = fitz.open(stream=content, filetype="pdf")
        total_pages = len(doc)

        elements: list[ParsedElement] = []
        element_idx = 0

        # pymupdf4llm으로 마크다운 추출
        md_text = pymupdf4llm.to_markdown(
            doc,
            page_chunks=self.page_chunks,
            write_images=self.write_images,
            image_path=self.image_path,
        )

        # 페이지별로 처리
        for page_num in range(total_pages):
            page = doc[page_num]
            page_elements = self._extract_page_elements(
                page, page_num + 1, filename, element_idx
            )
            elements.extend(page_elements)
            element_idx += len(page_elements)

        doc.close()

        return ParsedDocument(
            source=filename,
            filename=filename,
            total_pages=total_pages,
            elements=elements,
            metadata={
                "parser": "pymupdf4llm",
                "markdown_full": md_text if isinstance(md_text, str) else "\n\n".join(md_text),
            }
        )

    def _extract_page_elements(
        self, page, page_num: int, filename: str, start_idx: int
    ) -> list[ParsedElement]:
        """페이지에서 요소 추출"""
        import fitz

        elements = []
        idx = start_idx

        # 텍스트 블록 추출
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        current_heading = None

        for block in blocks:
            if block["type"] == 0:  # 텍스트 블록
                for line in block.get("lines", []):
                    text = ""
                    max_size = 0
                    for span in line.get("spans", []):
                        text += span.get("text", "")
                        max_size = max(max_size, span.get("size", 12))

                    text = text.strip()
                    if not text:
                        continue

                    # 요소 타입 결정
                    element_type = ElementType.PARAGRAPH
                    heading_level = None

                    # 폰트 크기로 헤딩 판단
                    if max_size >= 16:
                        element_type = ElementType.HEADING
                        if max_size >= 24:
                            heading_level = 1
                        elif max_size >= 20:
                            heading_level = 2
                        else:
                            heading_level = 3
                        current_heading = text

                    # BoundingBox 생성
                    bbox = BoundingBox(
                        x1=block["bbox"][0],
                        y1=block["bbox"][1],
                        x2=block["bbox"][2],
                        y2=block["bbox"][3],
                    )

                    element = ParsedElement(
                        element_id=self._generate_element_id(filename, page_num, idx),
                        element_type=element_type,
                        content=text,
                        page=page_num,
                        bbox=bbox,
                        markdown_content=text,
                        heading_level=heading_level,
                        parent_heading=current_heading if element_type != ElementType.HEADING else None,
                    )
                    elements.append(element)
                    idx += 1

            elif block["type"] == 1 and self.extract_images:  # 이미지 블록
                # 이미지 추출
                try:
                    bbox = BoundingBox(
                        x1=block["bbox"][0],
                        y1=block["bbox"][1],
                        x2=block["bbox"][2],
                        y2=block["bbox"][3],
                    )

                    # 이미지 데이터 추출
                    image_data = block.get("image", None)

                    element = ParsedElement(
                        element_id=self._generate_element_id(filename, page_num, idx),
                        element_type=ElementType.IMAGE,
                        content=f"[Image on page {page_num}]",
                        page=page_num,
                        bbox=bbox,
                        image_data=image_data,
                        parent_heading=current_heading,
                    )
                    elements.append(element)
                    idx += 1
                except Exception as e:
                    print(f"이미지 추출 실패: {e}")

        # 테이블 추출
        if self.extract_tables:
            tables = page.find_tables()
            for table in tables:
                try:
                    # 테이블을 마크다운으로 변환
                    table_data = table.extract()
                    md_table = self._table_to_markdown(table_data)

                    bbox = BoundingBox(
                        x1=table.bbox[0],
                        y1=table.bbox[1],
                        x2=table.bbox[2],
                        y2=table.bbox[3],
                    )

                    element = ParsedElement(
                        element_id=self._generate_element_id(filename, page_num, idx),
                        element_type=ElementType.TABLE,
                        content=md_table,
                        page=page_num,
                        bbox=bbox,
                        markdown_content=md_table,
                        parent_heading=current_heading,
                    )
                    elements.append(element)
                    idx += 1
                except Exception as e:
                    print(f"테이블 추출 실패: {e}")

        return elements

    def _table_to_markdown(self, table_data: list[list]) -> str:
        """테이블 데이터를 마크다운으로 변환"""
        if not table_data:
            return ""

        lines = []
        for i, row in enumerate(table_data):
            # None 값을 빈 문자열로 변환
            row = [str(cell) if cell is not None else "" for cell in row]
            line = "| " + " | ".join(row) + " |"
            lines.append(line)

            # 헤더 구분선
            if i == 0:
                separator = "| " + " | ".join(["---"] * len(row)) + " |"
                lines.append(separator)

        return "\n".join(lines)
