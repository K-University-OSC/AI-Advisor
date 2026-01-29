# -*- coding: utf-8 -*-
"""
V7.4 Semantic Chunking 모듈

개선사항:
1. 의미 단위 기반 청킹 (문장/문단 경계 존중)
2. Context-Aware 청킹 (헤딩/섹션 컨텍스트 유지)
3. 테이블/이미지 전용 청킹 (구조 보존)
4. 메타데이터 강화 (검색 품질 향상)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid
import re

from rag.parsers import ParsedDocument, ParsedElement, ElementType, CaptionResult


class ChunkType(str, Enum):
    """청크 타입"""
    PARENT = "parent"
    CHILD = "child"


@dataclass
class ChildChunk:
    """자식 청크 - 실제 벡터 검색용"""
    chunk_id: str
    content: str
    chunk_type: ChunkType
    source: str
    page: int
    parent_id: str = ""
    bbox: Optional[dict] = None
    element_type: str = "paragraph"
    heading: Optional[str] = None
    metadata: dict = field(default_factory=dict)

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
class ParentChunk:
    """부모 청크 - 문맥 파악용"""
    chunk_id: str
    content: str
    chunk_type: ChunkType
    source: str
    page: int
    heading: str = ""
    child_ids: list[str] = field(default_factory=list)
    start_page: int = 0
    end_page: int = 0
    metadata: dict = field(default_factory=dict)

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


class SemanticChunker:
    """
    V7.4 Semantic Chunker

    개선사항:
    1. 문장 경계 존중 청킹
    2. 테이블 구조 보존
    3. 이미지/차트 메타데이터 강화
    4. 검색 키워드 추출
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 500,
        chunk_overlap: int = 100,  # V7.4: 오버랩 증가 (50→100)
        min_chunk_size: int = 100,  # V7.4: 최소 청크 크기
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(
        self,
        document: ParsedDocument,
        caption_results: Optional[list[CaptionResult]] = None,
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """
        문서를 의미 단위로 청킹

        V7.4 개선:
        - 테이블은 분할하지 않고 전체 유지
        - 이미지/차트는 캡션과 함께 단일 청크
        - 문단은 문장 경계 존중
        """
        caption_map = {}
        if caption_results:
            caption_map = {r.element_id: r for r in caption_results}

        sections = self._group_by_sections(document)

        parent_chunks = []
        child_chunks = []

        for section_heading, elements in sections:
            parent_chunk, children = self._create_semantic_chunks(
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

    def _create_semantic_chunks(
        self,
        elements: list[ParsedElement],
        section_heading: str,
        source: str,
        caption_map: dict[str, CaptionResult],
    ) -> tuple[Optional[ParentChunk], list[ChildChunk]]:
        """V7.4: 요소 타입별 최적화된 청킹"""
        if not elements:
            return None, []

        full_content_parts = []
        child_chunks = []
        pages = set()

        parent_id = str(uuid.uuid4())

        for element in elements:
            pages.add(element.page)

            # V7.4: 요소 타입별 처리
            if element.element_type == ElementType.TABLE:
                # 테이블: 구조 보존, 분할 금지
                content, children = self._process_table_element(
                    element, parent_id, source, section_heading
                )
            elif element.element_type in (ElementType.IMAGE, ElementType.CHART):
                # 이미지/차트: 캡션 + 메타데이터 강화
                content, children = self._process_image_element(
                    element, parent_id, source, section_heading, caption_map
                )
            else:
                # 문단: 의미 단위 청킹
                content, children = self._process_paragraph_element(
                    element, parent_id, source, section_heading
                )

            if content.strip():
                full_content_parts.append(content)
                child_chunks.extend(children)

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
                "section_keywords": self._extract_keywords(full_content),
            },
        )

        return parent_chunk, child_chunks

    def _process_table_element(
        self,
        element: ParsedElement,
        parent_id: str,
        source: str,
        section_heading: str,
    ) -> tuple[str, list[ChildChunk]]:
        """
        V7.4: 테이블 처리 - 구조 보존

        - 마크다운/HTML 형식 유지
        - 테이블 분할 금지 (검색 품질 저하 방지)
        - 메타데이터에 테이블 구조 정보 추가
        """
        # 테이블 콘텐츠 추출
        if element.markdown_content:
            table_content = element.markdown_content
        elif element.html_content:
            table_content = element.html_content
        else:
            table_content = element.content

        # V7.4: 테이블 헤더 추가 (검색 키워드 강화)
        table_header = self._extract_table_header(table_content)
        enhanced_content = f"[TABLE: {section_heading}]\n"
        if table_header:
            enhanced_content += f"헤더: {table_header}\n"
        enhanced_content += table_content

        # V7.4: 테이블은 분할하지 않고 단일 청크로 유지
        chunk = ChildChunk(
            chunk_id=str(uuid.uuid4()),
            content=enhanced_content,
            chunk_type=ChunkType.CHILD,
            source=source,
            page=element.page,
            parent_id=parent_id,
            bbox=element.bbox.to_dict() if element.bbox else None,
            element_type="table",
            heading=section_heading,
            metadata={
                "original_element_id": element.element_id,
                "content_type": "table",
                "table_header": table_header,
                "preserves_structure": True,
            },
        )

        return enhanced_content, [chunk]

    def _process_image_element(
        self,
        element: ParsedElement,
        parent_id: str,
        source: str,
        section_heading: str,
        caption_map: dict[str, CaptionResult],
    ) -> tuple[str, list[ChildChunk]]:
        """
        V7.4: 이미지/차트 처리 - 캡션 + 메타데이터 강화

        - VLM 캡션 활용
        - 차트 유형, 데이터 포인트 추출
        - 검색 키워드 강화
        """
        caption_result = caption_map.get(element.element_id)

        if caption_result:
            caption = caption_result.caption
            # V7.4: 캡션에서 키워드 추출
            keywords = self._extract_keywords(caption)
        else:
            caption = element.content or "이미지"
            keywords = []

        # V7.4: 강화된 이미지 콘텐츠
        element_type_upper = element.element_type.value.upper()
        enhanced_content = f"[{element_type_upper}: {section_heading}]\n"
        enhanced_content += caption

        # V7.4: 수치 데이터 추출 (차트/그래프용)
        numbers = self._extract_numbers(caption)
        if numbers:
            enhanced_content += f"\n주요 수치: {', '.join(numbers)}"

        chunk = ChildChunk(
            chunk_id=str(uuid.uuid4()),
            content=enhanced_content,
            chunk_type=ChunkType.CHILD,
            source=source,
            page=element.page,
            parent_id=parent_id,
            bbox=element.bbox.to_dict() if element.bbox else None,
            element_type=element.element_type.value,
            heading=section_heading,
            metadata={
                "original_element_id": element.element_id,
                "content_type": "image",
                "has_caption": caption_result is not None,
                "keywords": keywords,
                "numbers": numbers,
            },
        )

        return enhanced_content, [chunk]

    def _process_paragraph_element(
        self,
        element: ParsedElement,
        parent_id: str,
        source: str,
        section_heading: str,
    ) -> tuple[str, list[ChildChunk]]:
        """
        V7.4: 문단 처리 - 의미 단위 청킹

        - 문장 경계 존중
        - 너무 짧은 청크 병합
        - 컨텍스트 오버랩 증가
        """
        content = element.content.strip()
        if not content:
            return "", []

        chunks = []

        if len(content) <= self.child_chunk_size:
            # 짧은 문단: 그대로 유지
            chunk = ChildChunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                chunk_type=ChunkType.CHILD,
                source=source,
                page=element.page,
                parent_id=parent_id,
                bbox=element.bbox.to_dict() if element.bbox else None,
                element_type="paragraph",
                heading=section_heading,
                metadata={
                    "original_element_id": element.element_id,
                    "content_type": "paragraph",
                },
            )
            chunks.append(chunk)
        else:
            # V7.4: 문장 기반 분할
            sub_chunks = self._split_by_sentences(content)

            for i, sub_content in enumerate(sub_chunks):
                # V7.4: 컨텍스트 프리픽스 추가
                if i > 0 and section_heading:
                    contextualized = f"[{section_heading}] {sub_content}"
                else:
                    contextualized = sub_content

                chunk = ChildChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=contextualized,
                    chunk_type=ChunkType.CHILD,
                    source=source,
                    page=element.page,
                    parent_id=parent_id,
                    bbox=element.bbox.to_dict() if element.bbox else None,
                    element_type="paragraph",
                    heading=section_heading,
                    metadata={
                        "original_element_id": element.element_id,
                        "content_type": "paragraph",
                        "sub_chunk_index": i,
                        "total_sub_chunks": len(sub_chunks),
                    },
                )
                chunks.append(chunk)

        return content, chunks

    def _split_by_sentences(self, text: str) -> list[str]:
        """
        V7.4: 문장 경계 기반 분할

        - 한국어/영어 문장 끝 패턴 인식
        - 최소 청크 크기 보장
        - 오버랩 증가
        """
        # 문장 끝 패턴
        sentence_endings = re.compile(r'(?<=[.!?。])\s+')
        sentences = sentence_endings.split(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            # 현재 청크에 추가해도 크기 내인 경우
            if current_length + sentence_length <= self.child_chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # 공백 포함
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    elif chunks:
                        # 너무 짧으면 이전 청크에 병합
                        chunks[-1] += " " + chunk_text

                # V7.4: 오버랩을 위해 마지막 문장 유지
                if current_chunk and self.chunk_overlap > 0:
                    overlap_text = current_chunk[-1] if current_chunk else ""
                    current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                    current_length = len(overlap_text) + sentence_length + 2
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length

        # 마지막 청크 저장
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            elif chunks:
                chunks[-1] += " " + chunk_text

        return chunks if chunks else [text]

    def _extract_table_header(self, table_content: str) -> str:
        """테이블에서 헤더 행 추출"""
        lines = table_content.strip().split("\n")
        if lines:
            # 첫 번째 행이 헤더라고 가정
            header = lines[0].replace("|", " ").replace("-", "").strip()
            return header[:100] if len(header) > 100 else header
        return ""

    def _extract_keywords(self, text: str) -> list[str]:
        """텍스트에서 검색 키워드 추출"""
        # 불용어
        stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로',
                     '와', '과', '도', '만', '까지', '부터', '에게', '한테', '께',
                     '무엇', '어떤', '어떻게', '왜', '언제', '어디', '누가', '무슨',
                     '있는', '하는', '되는', '되어', '하여', '있다', '한다', '된다'}

        # 단어 추출
        words = re.findall(r'[가-힣]{2,}|[A-Za-z]{3,}|\d+(?:\.\d+)?%?', text)
        keywords = [w for w in words if w.lower() not in stopwords]

        # 빈도 기반 상위 키워드
        from collections import Counter
        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(10)]

    def _extract_numbers(self, text: str) -> list[str]:
        """텍스트에서 수치 데이터 추출"""
        # 숫자 + 단위 패턴
        patterns = [
            r'\d+(?:\.\d+)?%',  # 퍼센트
            r'\d+(?:,\d{3})*(?:\.\d+)?(?:원|달러|엔|유로)?',  # 금액
            r'\d{4}년',  # 연도
            r'\d+(?:\.\d+)?(?:조|억|만)?(?:원)?',  # 큰 단위 금액
        ]

        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            numbers.extend(matches)

        return list(set(numbers))[:10]

    def _truncate_with_summary(self, content: str) -> str:
        """콘텐츠가 너무 길면 자르기"""
        if len(content) <= self.parent_chunk_size:
            return content

        truncated = content[:self.parent_chunk_size]
        last_period = truncated.rfind(".")
        if last_period > self.parent_chunk_size // 2:
            truncated = truncated[:last_period + 1]

        return truncated + "\n...(이하 생략)"
