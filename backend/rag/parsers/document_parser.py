"""
문서 파서 기본 인터페이스 및 데이터 모델
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ElementType(str, Enum):
    """문서 요소 타입"""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    LIST = "list"
    FOOTER = "footer"
    HEADER = "header"
    CAPTION = "caption"
    EQUATION = "equation"


@dataclass
class BoundingBox:
    """요소의 위치 좌표 (PDF 하이라이팅용)"""
    x1: float
    y1: float
    x2: float
    y2: float
    width: float = 0.0
    height: float = 0.0

    def __post_init__(self):
        if self.width == 0.0:
            self.width = self.x2 - self.x1
        if self.height == 0.0:
            self.height = self.y2 - self.y1

    def to_dict(self) -> dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class ParsedElement:
    """파싱된 문서 요소"""
    element_id: str
    element_type: ElementType
    content: str
    page: int
    bbox: Optional[BoundingBox] = None
    html_content: Optional[str] = None
    markdown_content: Optional[str] = None
    image_data: Optional[bytes] = None
    image_path: Optional[str] = None
    heading_level: Optional[int] = None
    parent_heading: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "content": self.content,
            "page": self.page,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "html_content": self.html_content,
            "markdown_content": self.markdown_content,
            "image_path": self.image_path,
            "heading_level": self.heading_level,
            "parent_heading": self.parent_heading,
            "metadata": self.metadata,
        }


@dataclass
class ParsedDocument:
    """파싱된 문서 전체"""
    source: str
    filename: str
    total_pages: int
    elements: list[ParsedElement] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def get_elements_by_type(self, element_type: ElementType) -> list[ParsedElement]:
        """특정 타입의 요소만 반환"""
        return [e for e in self.elements if e.element_type == element_type]

    def get_elements_by_page(self, page: int) -> list[ParsedElement]:
        """특정 페이지의 요소만 반환"""
        return [e for e in self.elements if e.page == page]

    def get_images_and_charts(self) -> list[ParsedElement]:
        """이미지와 차트 요소만 반환"""
        return [
            e for e in self.elements
            if e.element_type in (ElementType.IMAGE, ElementType.CHART)
        ]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "filename": self.filename,
            "total_pages": self.total_pages,
            "elements": [e.to_dict() for e in self.elements],
            "metadata": self.metadata,
        }


class DocumentParser(ABC):
    """문서 파서 추상 클래스"""

    @abstractmethod
    async def parse(self, file_path: str | Path) -> ParsedDocument:
        """문서를 파싱하여 구조화된 결과 반환"""
        pass

    @abstractmethod
    async def parse_bytes(
        self, content: bytes, filename: str
    ) -> ParsedDocument:
        """바이트 데이터를 파싱"""
        pass

    def _generate_element_id(self, source: str, page: int, index: int) -> str:
        """요소 ID 생성"""
        return f"{Path(source).stem}_p{page}_e{index}"
