"""
Upstage Document Parse API를 사용한 문서 파서
한국어 문서, 복잡한 표, 다단 레이아웃 처리에 최적화
"""

import asyncio
import base64
import httpx
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


class UpstageDocumentParser(DocumentParser):
    """Upstage Document Parse API 기반 파서"""

    API_URL = "https://api.upstage.ai/v1/document-ai/document-parse"

    ELEMENT_TYPE_MAP = {
        "paragraph": ElementType.PARAGRAPH,
        "heading1": ElementType.HEADING,
        "heading2": ElementType.HEADING,
        "heading3": ElementType.HEADING,
        "table": ElementType.TABLE,
        "figure": ElementType.IMAGE,
        "chart": ElementType.CHART,
        "image": ElementType.IMAGE,
        "list": ElementType.LIST,
        "footer": ElementType.FOOTER,
        "header": ElementType.HEADER,
        "caption": ElementType.CAPTION,
        "equation": ElementType.EQUATION,
    }

    def __init__(
        self,
        api_key: str,
        output_formats: Optional[list[str]] = None,
        coordinates: bool = True,
        ocr: bool = True,
        base64_encoding: Optional[list[str]] = None,
    ):
        """
        Args:
            api_key: Upstage API 키
            output_formats: 출력 형식 (html, markdown, text)
            coordinates: 좌표 정보 포함 여부 (하이라이팅용)
            ocr: OCR 활성화 여부
            base64_encoding: base64로 인코딩할 요소 타입 (figure, table 등)
        """
        self.api_key = api_key
        self.output_formats = output_formats or ["html", "markdown", "text"]
        self.coordinates = coordinates
        self.ocr = ocr
        self.base64_encoding = base64_encoding or ["figure", "chart"]

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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        files = {
            "document": (filename, content, self._get_mime_type(filename)),
        }

        data = {
            "ocr": str(self.ocr).lower(),
            "coordinates": str(self.coordinates).lower(),
            "output_formats": ",".join(self.output_formats),
        }

        if self.base64_encoding:
            data["base64_encoding"] = ",".join(self.base64_encoding)

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                self.API_URL,
                headers=headers,
                files=files,
                data=data,
            )

        if response.status_code != 200:
            raise Exception(
                f"Upstage API 오류: {response.status_code} - {response.text}"
            )

        result = response.json()
        return self._process_response(result, filename)

    def _process_response(self, response: dict, filename: str) -> ParsedDocument:
        """API 응답을 ParsedDocument로 변환"""
        elements = []
        current_heading = None
        element_counter = {}

        api_elements = response.get("elements", [])
        total_pages = response.get("num_pages", 1)

        for elem in api_elements:
            page = elem.get("page", 1)
            category = elem.get("category", "paragraph").lower()

            if page not in element_counter:
                element_counter[page] = 0
            element_counter[page] += 1

            element_type = self.ELEMENT_TYPE_MAP.get(
                category, ElementType.PARAGRAPH
            )

            element_id = self._generate_element_id(
                filename, page, element_counter[page]
            )

            bbox = None
            if self.coordinates and "bounding_box" in elem:
                bb = elem["bounding_box"]
                bbox = BoundingBox(
                    x1=bb.get("x1", bb.get("x", 0)),
                    y1=bb.get("y1", bb.get("y", 0)),
                    x2=bb.get("x2", bb.get("x", 0) + bb.get("width", 0)),
                    y2=bb.get("y2", bb.get("y", 0) + bb.get("height", 0)),
                )

            heading_level = None
            if category.startswith("heading"):
                try:
                    heading_level = int(category[-1])
                except (ValueError, IndexError):
                    heading_level = 1
                current_heading = elem.get("content", {}).get("text", "")

            content = elem.get("content", {})
            text_content = content.get("text", "")
            html_content = content.get("html")
            markdown_content = content.get("markdown")

            image_data = None
            image_path = None
            if element_type in (ElementType.IMAGE, ElementType.CHART):
                base64_data = elem.get("base64_encoding")
                if base64_data:
                    image_data = base64.b64decode(base64_data)
                    image_path = f"images/{element_id}.png"

            parsed_element = ParsedElement(
                element_id=element_id,
                element_type=element_type,
                content=text_content,
                page=page,
                bbox=bbox,
                html_content=html_content,
                markdown_content=markdown_content,
                image_data=image_data,
                image_path=image_path,
                heading_level=heading_level,
                parent_heading=current_heading if element_type != ElementType.HEADING else None,
                metadata={
                    "category": category,
                    "confidence": elem.get("confidence"),
                },
            )
            elements.append(parsed_element)

        return ParsedDocument(
            source=filename,
            filename=filename,
            total_pages=total_pages,
            elements=elements,
            metadata={
                "parser": "upstage",
                "model": response.get("model"),
                "api_version": response.get("api", "v1"),
            },
        )

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
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        return mime_types.get(ext, "application/octet-stream")
