"""
Azure Document Intelligence 기반 문서 파서
Upstage Document Parse API와 동일한 인터페이스 제공
"""

import asyncio
import base64
import io
import os
import subprocess
from pathlib import Path
from typing import Optional
import uuid

from .document_parser import (
    DocumentParser,
    ParsedDocument,
    ParsedElement,
    ElementType,
    BoundingBox,
)


def _load_azure_key() -> str:
    """Azure Document Intelligence API 키 로드"""
    # 환경변수에서 먼저 시도
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
    if key:
        return key

    # 오타 버전도 확인 (기존 호환성)
    key = os.getenv("AZURE_DOCUMENT_INTELLEGENCE_KEY", "")
    if key:
        return key

    # bashrc에서 시도
    try:
        result = subprocess.run(
            ['bash', '-c', 'source ~/.bashrc && echo $AZURE_DOCUMENT_INTELLIGENCE_KEY'],
            capture_output=True, text=True
        )
        key = result.stdout.strip()
        if key:
            return key
    except:
        pass

    return ""


class AzureDocumentParser(DocumentParser):
    """Azure Document Intelligence 기반 문서 파서

    Upstage Document Parser와 동일한 인터페이스 제공
    """

    DEFAULT_ENDPOINT = "https://rag-di.cognitiveservices.azure.com/"

    # Azure의 paragraph role을 ElementType으로 매핑
    ELEMENT_TYPE_MAP = {
        "title": ElementType.HEADING,
        "sectionHeading": ElementType.HEADING,
        "pageHeader": ElementType.HEADER,
        "pageFooter": ElementType.FOOTER,
        "pageNumber": ElementType.FOOTER,
        "footnote": ElementType.FOOTER,
    }

    def __init__(
        self,
        api_key: str = None,
        endpoint: str = None,
        output_formats: Optional[list[str]] = None,
        coordinates: bool = True,
        ocr: bool = True,
        base64_encoding: Optional[list[str]] = None,
    ):
        """
        Args:
            api_key: Azure API 키 (없으면 환경변수에서 로드)
            endpoint: Azure 엔드포인트
            output_formats: 출력 형식 (호환성용, Azure에서는 무시)
            coordinates: 좌표 정보 포함 여부
            ocr: OCR 활성화 여부 (Azure에서는 항상 활성화)
            base64_encoding: base64로 인코딩할 요소 (figure 등)
        """
        self.api_key = api_key or _load_azure_key()
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self.output_formats = output_formats or ["html", "markdown", "text"]
        self.coordinates = coordinates
        self.ocr = ocr
        self.base64_encoding = base64_encoding or ["figure", "chart"]

        self.client = None
        self._initialized = False

    def _initialize(self) -> bool:
        """클라이언트 초기화"""
        if self._initialized:
            return True

        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential

            if not self.api_key:
                print("Azure Document Intelligence API 키가 설정되지 않았습니다.")
                return False

            self.client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key)
            )
            self._initialized = True
            return True

        except ImportError:
            print("azure-ai-documentintelligence 패키지를 설치하세요")
            return False
        except Exception as e:
            print(f"Azure Document Intelligence 초기화 실패: {e}")
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
            raise RuntimeError("Azure Document Intelligence 초기화 실패")

        # 동기 API를 비동기로 실행
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._analyze_document, content, filename
        )

        return result

    def _analyze_document(self, content: bytes, filename: str) -> ParsedDocument:
        """문서 분석 (동기)"""
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

        # base64 인코딩
        content_base64 = base64.b64encode(content).decode()

        # prebuilt-layout 모델 사용 (테이블, 이미지, 텍스트 모두 추출)
        poller = self.client.begin_analyze_document(
            "prebuilt-layout",
            body={"base64Source": content_base64},
            output_content_format="markdown",  # Markdown 형식으로 출력
        )
        result = poller.result()

        return self._process_response(result, filename, content)

    def _process_response(self, result, filename: str, original_content: bytes) -> ParsedDocument:
        """API 응답을 ParsedDocument로 변환"""
        elements = []
        current_heading = None
        element_counter = {}

        total_pages = len(result.pages) if result.pages else 1

        # 1. 테이블 처리
        if result.tables:
            for table_idx, table in enumerate(result.tables):
                page = table.bounding_regions[0].page_number if table.bounding_regions else 1

                if page not in element_counter:
                    element_counter[page] = 0
                element_counter[page] += 1

                element_id = self._generate_element_id(filename, page, element_counter[page])

                # 테이블을 Markdown으로 변환
                markdown_table = self._table_to_markdown(table)
                html_table = self._table_to_html(table)

                # bbox 추출
                bbox = None
                if self.coordinates and table.bounding_regions:
                    br = table.bounding_regions[0]
                    if br.polygon and len(br.polygon) >= 4:
                        # polygon은 [x1,y1,x2,y2,x3,y3,x4,y4] 형식
                        xs = [br.polygon[i] for i in range(0, len(br.polygon), 2)]
                        ys = [br.polygon[i] for i in range(1, len(br.polygon), 2)]
                        bbox = BoundingBox(
                            x1=min(xs), y1=min(ys),
                            x2=max(xs), y2=max(ys)
                        )

                parsed_element = ParsedElement(
                    element_id=element_id,
                    element_type=ElementType.TABLE,
                    content=markdown_table,  # 텍스트 콘텐츠로 Markdown 사용
                    page=page,
                    bbox=bbox,
                    html_content=html_table,
                    markdown_content=markdown_table,
                    parent_heading=current_heading,
                    metadata={
                        "category": "table",
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                    },
                )
                elements.append(parsed_element)

        # 2. 이미지(Figure) 처리
        if result.figures:
            for fig_idx, figure in enumerate(result.figures):
                page = figure.bounding_regions[0].page_number if figure.bounding_regions else 1

                if page not in element_counter:
                    element_counter[page] = 0
                element_counter[page] += 1

                element_id = self._generate_element_id(filename, page, element_counter[page])

                # bbox 추출
                bbox = None
                if self.coordinates and figure.bounding_regions:
                    br = figure.bounding_regions[0]
                    if br.polygon and len(br.polygon) >= 4:
                        xs = [br.polygon[i] for i in range(0, len(br.polygon), 2)]
                        ys = [br.polygon[i] for i in range(1, len(br.polygon), 2)]
                        bbox = BoundingBox(
                            x1=min(xs), y1=min(ys),
                            x2=max(xs), y2=max(ys)
                        )

                # 이미지 데이터 추출 (PDF에서 crop)
                image_data = None
                if bbox and "figure" in self.base64_encoding:
                    image_data = self._extract_image_from_pdf(
                        original_content, page, bbox
                    )

                # Figure caption
                caption = ""
                if hasattr(figure, 'caption') and figure.caption:
                    caption = figure.caption.content if hasattr(figure.caption, 'content') else str(figure.caption)

                parsed_element = ParsedElement(
                    element_id=element_id,
                    element_type=ElementType.IMAGE,
                    content=caption or f"[Figure on page {page}]",
                    page=page,
                    bbox=bbox,
                    image_data=image_data,
                    image_path=f"images/{element_id}.png" if image_data else None,
                    parent_heading=current_heading,
                    metadata={
                        "category": "figure",
                    },
                )
                elements.append(parsed_element)

        # 3. 단락(Paragraph) 처리
        if result.paragraphs:
            for para_idx, paragraph in enumerate(result.paragraphs):
                page = paragraph.bounding_regions[0].page_number if paragraph.bounding_regions else 1

                if page not in element_counter:
                    element_counter[page] = 0
                element_counter[page] += 1

                element_id = self._generate_element_id(filename, page, element_counter[page])

                # role에 따른 타입 결정
                role = getattr(paragraph, 'role', None)
                element_type = self.ELEMENT_TYPE_MAP.get(role, ElementType.PARAGRAPH)

                # 헤딩 레벨 결정
                heading_level = None
                if role == "title":
                    heading_level = 1
                    current_heading = paragraph.content
                elif role == "sectionHeading":
                    heading_level = 2
                    current_heading = paragraph.content

                # bbox 추출
                bbox = None
                if self.coordinates and paragraph.bounding_regions:
                    br = paragraph.bounding_regions[0]
                    if br.polygon and len(br.polygon) >= 4:
                        xs = [br.polygon[i] for i in range(0, len(br.polygon), 2)]
                        ys = [br.polygon[i] for i in range(1, len(br.polygon), 2)]
                        bbox = BoundingBox(
                            x1=min(xs), y1=min(ys),
                            x2=max(xs), y2=max(ys)
                        )

                parsed_element = ParsedElement(
                    element_id=element_id,
                    element_type=element_type,
                    content=paragraph.content,
                    page=page,
                    bbox=bbox,
                    heading_level=heading_level,
                    parent_heading=current_heading if element_type != ElementType.HEADING else None,
                    metadata={
                        "category": role or "paragraph",
                    },
                )
                elements.append(parsed_element)

        # 요소들을 페이지 순서로 정렬
        elements.sort(key=lambda e: (e.page, e.bbox.y1 if e.bbox else 0))

        return ParsedDocument(
            source=filename,
            filename=filename,
            total_pages=total_pages,
            elements=elements,
            metadata={
                "parser": "azure_document_intelligence",
                "model": "prebuilt-layout",
            },
        )

    def _table_to_markdown(self, table) -> str:
        """테이블을 Markdown으로 변환"""
        if not table.cells:
            return ""

        rows = table.row_count
        cols = table.column_count

        # 그리드 생성
        grid = [["" for _ in range(cols)] for _ in range(rows)]

        # 셀 데이터 채우기
        for cell in table.cells:
            r = cell.row_index
            c = cell.column_index
            if 0 <= r < rows and 0 <= c < cols:
                content = cell.content.replace('\n', ' ').replace('|', '\\|')
                grid[r][c] = content

        # Markdown 생성
        lines = []
        if grid:
            # 헤더
            lines.append("| " + " | ".join(grid[0]) + " |")
            # 구분선
            lines.append("| " + " | ".join(["---"] * cols) + " |")
            # 데이터 행
            for row in grid[1:]:
                lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _table_to_html(self, table) -> str:
        """테이블을 HTML로 변환"""
        if not table.cells:
            return ""

        rows = table.row_count
        cols = table.column_count

        # 그리드 생성
        grid = [["" for _ in range(cols)] for _ in range(rows)]

        for cell in table.cells:
            r = cell.row_index
            c = cell.column_index
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = cell.content

        # HTML 생성
        html_parts = ["<table>"]
        for i, row in enumerate(grid):
            html_parts.append("<tr>")
            tag = "th" if i == 0 else "td"
            for cell in row:
                html_parts.append(f"<{tag}>{cell}</{tag}>")
            html_parts.append("</tr>")
        html_parts.append("</table>")

        return "".join(html_parts)

    def _extract_image_from_pdf(
        self, pdf_content: bytes, page: int, bbox: BoundingBox
    ) -> Optional[bytes]:
        """PDF에서 이미지 영역 추출"""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=pdf_content, filetype="pdf")
            if page <= len(doc):
                pdf_page = doc[page - 1]

                # bbox를 PDF 좌표로 변환
                # Azure는 인치 단위, PDF는 포인트 (1인치 = 72포인트)
                rect = fitz.Rect(
                    bbox.x1 * 72,
                    bbox.y1 * 72,
                    bbox.x2 * 72,
                    bbox.y2 * 72,
                )

                # 이미지로 렌더링
                mat = fitz.Matrix(2, 2)  # 2배 해상도
                pix = pdf_page.get_pixmap(matrix=mat, clip=rect)

                doc.close()
                return pix.tobytes("png")
        except Exception as e:
            print(f"이미지 추출 실패: {e}")

        return None

    def _get_mime_type(self, filename: str) -> str:
        """파일 확장자로부터 MIME 타입 추론"""
        ext = Path(filename).suffix.lower()
        mime_types = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".bmp": "image/bmp",
        }
        return mime_types.get(ext, "application/octet-stream")
