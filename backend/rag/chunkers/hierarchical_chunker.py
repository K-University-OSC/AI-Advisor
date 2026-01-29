"""
계층적 청킹 모듈 (Parent-Child 전략)
문맥 유지를 위해 부모 청크와 자식 청크를 분리하여 관리
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid

from rag.parsers import ParsedDocument, ParsedElement, ElementType, CaptionResult


class ChunkType(str, Enum):
    """청크 타입"""
    PARENT = "parent"
    CHILD = "child"


@dataclass
class ChunkRelation:
    """청크 간 관계"""
    parent_id: str
    child_ids: list[str] = field(default_factory=list)


@dataclass
class BaseChunk:
    """청크 기본 클래스"""
    chunk_id: str
    content: str
    chunk_type: ChunkType
    source: str
    page: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ChildChunk(BaseChunk):
    """자식 청크 - 실제 벡터 검색용"""
    parent_id: str = ""
    bbox: Optional[dict] = None
    element_type: str = "paragraph"
    heading: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "source": self.source,
            "page": self.page,
            "parent_id": self.parent_id,
            "bbox": self.bbox,
            "element_type": self.element_type,
            "heading": self.heading,
            "metadata": self.metadata,
        }


@dataclass
class ParentChunk(BaseChunk):
    """부모 청크 - 문맥 파악용"""
    heading: str = ""
    child_ids: list[str] = field(default_factory=list)
    start_page: int = 0
    end_page: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "source": self.source,
            "page": self.page,
            "heading": self.heading,
            "child_ids": self.child_ids,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "metadata": self.metadata,
        }


class HierarchicalChunker:
    """계층적 청킹 처리기"""

    def __init__(
        self,
        parent_chunk_size: int = 2000,  # V7.7.2: 2000자 유지 (하이브리드 fallback으로 보완)
        child_chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Args:
            parent_chunk_size: 부모 청크 최대 크기 (2000자)
            child_chunk_size: 자식 청크 최대 크기 (약 500자)
            chunk_overlap: 청크 간 오버랩 크기
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        document: ParsedDocument,
        caption_results: Optional[list[CaptionResult]] = None,
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """
        문서를 계층적으로 청킹

        Args:
            document: 파싱된 문서
            caption_results: 이미지/차트 캡셔닝 결과

        Returns:
            (부모 청크 리스트, 자식 청크 리스트)
        """
        caption_map = {}
        if caption_results:
            caption_map = {r.element_id: r for r in caption_results}

        sections = self._group_by_sections(document)

        parent_chunks = []
        child_chunks = []

        for section_heading, elements in sections:
            parent_chunk, children = self._create_hierarchical_chunks(
                elements=elements,
                section_heading=section_heading,
                source=document.source,
                caption_map=caption_map,
            )

            if parent_chunk:
                parent_chunks.append(parent_chunk)
                child_chunks.extend(children)

        return parent_chunks, child_chunks

    def _group_by_sections(
        self, document: ParsedDocument
    ) -> list[tuple[str, list[ParsedElement]]]:
        """문서를 섹션(헤딩) 기준으로 그룹화"""
        sections = []
        current_heading = "서론"
        current_elements = []

        for element in document.elements:
            if element.element_type == ElementType.HEADING:
                if current_elements:
                    sections.append((current_heading, current_elements))
                current_heading = element.content or f"섹션 {len(sections) + 1}"
                current_elements = [element]
            else:
                current_elements.append(element)

        if current_elements:
            sections.append((current_heading, current_elements))

        return sections

    def _create_hierarchical_chunks(
        self,
        elements: list[ParsedElement],
        section_heading: str,
        source: str,
        caption_map: dict[str, CaptionResult],
    ) -> tuple[Optional[ParentChunk], list[ChildChunk]]:
        """섹션을 부모-자식 청크로 변환"""
        if not elements:
            return None, []

        full_content_parts = []
        child_chunks = []
        pages = set()

        parent_id = str(uuid.uuid4())

        for element in elements:
            pages.add(element.page)

            content = self._get_element_content(element, caption_map)
            if not content.strip():
                continue

            full_content_parts.append(content)

            element_children = self._create_child_chunks(
                content=content,
                element=element,
                parent_id=parent_id,
                source=source,
                section_heading=section_heading,
            )
            child_chunks.extend(element_children)

        full_content = "\n\n".join(full_content_parts)

        if len(full_content) > self.parent_chunk_size:
            full_content = self._truncate_with_summary(full_content)

        if not full_content.strip():
            return None, []

        parent_chunk = ParentChunk(
            chunk_id=parent_id,
            content=full_content,
            chunk_type=ChunkType.PARENT,
            source=source,
            page=min(pages) if pages else 1,
            heading=section_heading,
            child_ids=[c.chunk_id for c in child_chunks],
            start_page=min(pages) if pages else 1,
            end_page=max(pages) if pages else 1,
            metadata={
                "element_count": len(elements),
                "child_count": len(child_chunks),
            },
        )

        return parent_chunk, child_chunks

    def _get_element_content(
        self,
        element: ParsedElement,
        caption_map: dict[str, CaptionResult],
    ) -> str:
        """요소의 콘텐츠 추출 (이미지/차트는 캡션 포함)"""
        if element.element_type in (ElementType.IMAGE, ElementType.CHART):
            caption_result = caption_map.get(element.element_id)
            if caption_result:
                return f"[{element.element_type.value.upper()}]\n{caption_result.caption}"
            elif element.content:
                return f"[{element.element_type.value.upper()}] {element.content}"
            return ""

        if element.element_type == ElementType.TABLE:
            if element.markdown_content:
                return f"[TABLE]\n{element.markdown_content}"
            elif element.html_content:
                return f"[TABLE]\n{element.html_content}"
            return f"[TABLE] {element.content}"

        return element.content

    def _create_child_chunks(
        self,
        content: str,
        element: ParsedElement,
        parent_id: str,
        source: str,
        section_heading: str,
    ) -> list[ChildChunk]:
        """콘텐츠를 자식 청크로 분할"""
        chunks = []

        if len(content) <= self.child_chunk_size:
            chunk = ChildChunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                chunk_type=ChunkType.CHILD,
                source=source,
                page=element.page,
                parent_id=parent_id,
                bbox=element.bbox.to_dict() if element.bbox else None,
                element_type=element.element_type.value,
                heading=section_heading,
                metadata={
                    "original_element_id": element.element_id,
                },
            )
            chunks.append(chunk)
        else:
            sub_chunks = self._split_text(content, self.child_chunk_size, self.chunk_overlap)
            for i, sub_content in enumerate(sub_chunks):
                chunk = ChildChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=sub_content,
                    chunk_type=ChunkType.CHILD,
                    source=source,
                    page=element.page,
                    parent_id=parent_id,
                    bbox=element.bbox.to_dict() if element.bbox else None,
                    element_type=element.element_type.value,
                    heading=section_heading,
                    metadata={
                        "original_element_id": element.element_id,
                        "sub_chunk_index": i,
                        "total_sub_chunks": len(sub_chunks),
                    },
                )
                chunks.append(chunk)

        return chunks

    def _split_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """텍스트를 지정된 크기로 분할"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        separators = ["\n\n", "\n", ". ", " "]

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))

            if end < len(text):
                best_split = end
                for sep in separators:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + chunk_size // 2:
                        best_split = last_sep + len(sep)
                        break
                end = best_split

            chunks.append(text[start:end].strip())
            start = max(start + 1, end - overlap)

        return [c for c in chunks if c]

    def _truncate_with_summary(self, content: str) -> str:
        """콘텐츠가 너무 길면 자르기"""
        if len(content) <= self.parent_chunk_size:
            return content

        truncated = content[:self.parent_chunk_size]
        last_period = truncated.rfind(".")
        if last_period > self.parent_chunk_size // 2:
            truncated = truncated[:last_period + 1]

        return truncated + "\n...(이하 생략)"
