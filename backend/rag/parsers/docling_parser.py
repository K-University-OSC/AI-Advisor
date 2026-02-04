# -*- coding: utf-8 -*-
"""
Docling 기반 문서 파서

IBM의 오픈소스 문서 분석 도구
- DocLayNet 기반 고품질 레이아웃 분석
- 테이블/이미지/차트 추출
- 마크다운 출력 지원
- 완전 무료 (MIT 라이선스)
"""

import asyncio
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


class DoclingParser(DocumentParser):
    """Docling 기반 문서 파서

    IBM의 오픈소스 DocLayNet 모델을 사용한 고품질 레이아웃 분석
    - GitHub 30K+ Stars
    - MIT 라이선스 (완전 무료)
    - PDF, DOCX, PPTX, XLSX, HTML, 이미지 지원
    """

    def __init__(
        self,
        extract_tables: bool = True,
        extract_images: bool = True,
        ocr_enabled: bool = True,
        output_format: str = "markdown",
    ):
        """
        Args:
            extract_tables: 테이블 추출 여부
            extract_images: 이미지 추출 여부
            ocr_enabled: OCR 사용 여부
            output_format: 출력 형식 (markdown, text)
        """
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled
        self.output_format = output_format
        self._initialized = False
        self._converter = None

    def _initialize(self) -> bool:
        """Docling 초기화"""
        if self._initialized:
            return True

        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import PdfFormatOption

            # 파이프라인 옵션 설정
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.ocr_enabled
            pipeline_options.do_table_structure = self.extract_tables
            pipeline_options.images_scale = 2.0  # 고해상도 이미지

            # DocumentConverter 초기화
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            self._initialized = True
            return True

        except ImportError as e:
            print(f"Docling 패키지 설치 필요: {e}")
            print("설치: pip install docling")
            return False
        except Exception as e:
            print(f"Docling 초기화 실패: {e}")
            return False

    async def parse(self, file_path: str | Path) -> ParsedDocument:
        """파일 경로로부터 문서 파싱"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        # 동기 API를 비동기로 실행
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._parse_file, file_path
        )
        return result

    async def parse_bytes(
        self, content: bytes, filename: str
    ) -> ParsedDocument:
        """바이트 데이터로부터 문서 파싱"""
        import tempfile
        import os

        # 임시 파일로 저장 후 파싱
        suffix = Path(filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = await self.parse(tmp_path)
            result.source = filename
            result.filename = filename
            return result
        finally:
            os.unlink(tmp_path)

    def _parse_file(self, file_path: Path) -> ParsedDocument:
        """파일 파싱 (동기)"""
        if not self._initialize():
            raise RuntimeError("Docling 초기화 실패")

        # Docling으로 변환
        result = self._converter.convert(str(file_path))
        doc = result.document

        elements: list[ParsedElement] = []
        element_idx = 0

        # 마크다운 출력
        markdown_content = doc.export_to_markdown()

        # 문서 요소 추출
        for item in doc.iterate_items():
            element = self._convert_item_to_element(
                item, file_path.name, element_idx
            )
            if element:
                elements.append(element)
                element_idx += 1

        # 요소가 없으면 마크다운에서 추출
        if not elements and markdown_content:
            elements = self._parse_markdown_to_elements(
                markdown_content, file_path.name
            )

        return ParsedDocument(
            source=str(file_path),
            filename=file_path.name,
            total_pages=doc.num_pages() if hasattr(doc, 'num_pages') else len(set(e.page for e in elements if e.page)),
            elements=elements,
            metadata={
                "parser": "docling",
                "markdown_full": markdown_content,
            }
        )

    def _convert_item_to_element(
        self, item, filename: str, idx: int
    ) -> Optional[ParsedElement]:
        """Docling 아이템을 ParsedElement로 변환"""
        try:
            from docling.datamodel.document import (
                TextItem, TableItem, PictureItem, SectionHeaderItem
            )

            content = ""
            element_type = ElementType.PARAGRAPH
            page_num = 1
            heading_level = None
            bbox = None

            # 페이지 번호 추출
            if hasattr(item, 'prov') and item.prov:
                prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                if hasattr(prov, 'page_no'):
                    page_num = prov.page_no
                # BoundingBox 추출
                if hasattr(prov, 'bbox'):
                    b = prov.bbox
                    bbox = BoundingBox(
                        x1=b.l, y1=b.t, x2=b.r, y2=b.b
                    )

            if isinstance(item, SectionHeaderItem):
                element_type = ElementType.HEADING
                content = item.text if hasattr(item, 'text') else str(item)
                heading_level = item.level if hasattr(item, 'level') else 1

            elif isinstance(item, TextItem):
                element_type = ElementType.PARAGRAPH
                content = item.text if hasattr(item, 'text') else str(item)

            elif isinstance(item, TableItem):
                element_type = ElementType.TABLE
                # 테이블을 마크다운으로 변환
                content = item.export_to_markdown() if hasattr(item, 'export_to_markdown') else str(item)

            elif isinstance(item, PictureItem):
                element_type = ElementType.IMAGE
                content = f"[Image on page {page_num}]"
                if hasattr(item, 'caption') and item.caption:
                    content = f"[Image: {item.caption}]"

            else:
                # 기타 요소
                content = str(item) if item else ""
                if not content.strip():
                    return None

            if not content.strip():
                return None

            return ParsedElement(
                element_id=self._generate_element_id(filename, page_num, idx),
                element_type=element_type,
                content=content,
                page=page_num,
                bbox=bbox,
                markdown_content=content,
                heading_level=heading_level,
            )

        except Exception as e:
            print(f"요소 변환 실패: {e}")
            return None

    def _parse_markdown_to_elements(
        self, markdown: str, filename: str
    ) -> list[ParsedElement]:
        """마크다운 텍스트를 ParsedElement 리스트로 변환"""
        import re

        elements = []
        lines = markdown.split('\n')
        idx = 0
        page_num = 1
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

            # 페이지 마커 감지
            if line.startswith('<!-- page'):
                page_match = re.search(r'page\s*(\d+)', line)
                if page_match:
                    page_num = int(page_match.group(1))
                i += 1
                continue

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
                i -= 1
                element_type = ElementType.TABLE
                content = '\n'.join(table_lines)

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

            # 이미지 감지
            elif line.startswith('![') or '[IMAGE:' in line.upper():
                element_type = ElementType.IMAGE

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
