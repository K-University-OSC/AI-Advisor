# -*- coding: utf-8 -*-
"""
V7.5 Contextual Retrieval 모듈

Anthropic의 Contextual Retrieval 방식 구현:
- 각 청크에 문맥 정보(출처, 섹션, 요약)를 추가
- 검색 실패율 49% 감소 효과 (Anthropic 벤치마크)

Reference: https://www.anthropic.com/news/contextual-retrieval
"""

import os
import asyncio
from dataclasses import dataclass, field
from typing import Optional
import httpx

from rag.parsers import ParsedDocument, ParsedElement, ElementType, CaptionResult
from rag.chunkers.hierarchical_chunker import (
    HierarchicalChunker,
    ParentChunk,
    ChildChunk,
    ChunkRelation,
)


@dataclass
class ContextualChunk:
    """문맥 정보가 추가된 청크"""
    original_content: str
    context_header: str
    contextualized_content: str
    chunk_id: str
    source: str
    page: int
    section: str = ""
    element_type: str = "paragraph"


class ContextGenerator:
    """
    LLM을 사용하여 청크별 문맥 정보 생성

    각 청크에 50-100 토큰의 설명적 문맥을 추가하여
    검색 시 해당 청크가 어떤 맥락에서 나온 것인지 알 수 있게 함
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.2",  # 쿼리 분석 품질 향상
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"

    async def generate_context(
        self,
        chunk_content: str,
        document_name: str,
        document_summary: str = "",
        section_title: str = "",
        page: int = 0,
    ) -> str:
        """
        청크에 대한 문맥 설명 생성

        Args:
            chunk_content: 청크 내용
            document_name: 문서 이름
            document_summary: 문서 전체 요약 (선택)
            section_title: 섹션 제목 (선택)
            page: 페이지 번호

        Returns:
            문맥 설명 (50-100 토큰)
        """
        prompt = f"""<document>
문서명: {document_name}
{f'문서 요약: {document_summary}' if document_summary else ''}
{f'섹션: {section_title}' if section_title else ''}
페이지: {page}
</document>

아래 청크의 내용을 문서 전체 맥락에서 이해할 수 있도록 간결한 문맥 설명을 작성하세요.
문맥 설명은 청크 앞에 추가되어 검색 시 해당 청크가 무엇에 관한 것인지 명확히 알 수 있게 합니다.

<chunk>
{chunk_content[:1500]}
</chunk>

다음 형식으로 50-100자의 문맥 설명만 작성하세요:
"이 청크는 [문서명]의 [섹션/주제]에서 [핵심 내용]에 대해 설명합니다."

문맥 설명:"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_completion_tokens": 150,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.api_url, headers=headers, json=payload)

            if response.status_code != 200:
                return self._create_fallback_context(document_name, section_title, page)

            result = response.json()
            context = result["choices"][0]["message"]["content"].strip()
            return context

        except Exception as e:
            print(f"문맥 생성 실패: {e}")
            return self._create_fallback_context(document_name, section_title, page)

    def _create_fallback_context(
        self,
        document_name: str,
        section_title: str,
        page: int,
    ) -> str:
        """LLM 실패 시 기본 문맥 생성"""
        parts = [f"[출처: {document_name}"]
        if section_title:
            parts.append(f", 섹션: {section_title}")
        if page > 0:
            parts.append(f", 페이지: {page}")
        parts.append("]")
        return "".join(parts)

    async def generate_contexts_batch(
        self,
        chunks: list[dict],
        document_name: str,
        document_summary: str = "",
        batch_size: int = 5,
    ) -> list[str]:
        """
        여러 청크에 대해 배치로 문맥 생성

        Args:
            chunks: 청크 정보 리스트 [{content, section, page}, ...]
            document_name: 문서 이름
            document_summary: 문서 요약
            batch_size: 동시 처리 수

        Returns:
            문맥 설명 리스트
        """
        contexts = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            tasks = [
                self.generate_context(
                    chunk_content=c.get("content", ""),
                    document_name=document_name,
                    document_summary=document_summary,
                    section_title=c.get("section", ""),
                    page=c.get("page", 0),
                )
                for c in batch
            ]
            batch_contexts = await asyncio.gather(*tasks)
            contexts.extend(batch_contexts)

            # Rate limiting
            if i + batch_size < len(chunks):
                await asyncio.sleep(0.5)

        return contexts


class ContextualHierarchicalChunker(HierarchicalChunker):
    """
    V7.5 Contextual Hierarchical Chunker

    기존 HierarchicalChunker를 확장하여 각 청크에 문맥 정보 추가
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 500,
        chunk_overlap: int = 50,
        api_key: Optional[str] = None,
        use_llm_context: bool = True,  # LLM 문맥 생성 사용 여부
    ):
        super().__init__(
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.context_generator = ContextGenerator(api_key=api_key)
        self.use_llm_context = use_llm_context

    async def chunk_document_with_context(
        self,
        document: ParsedDocument,
        caption_results: Optional[list[CaptionResult]] = None,
        document_summary: str = "",
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """
        문서를 청킹하고 각 청크에 문맥 정보 추가

        Args:
            document: 파싱된 문서
            caption_results: 이미지/차트 캡션 결과
            document_summary: 문서 전체 요약 (선택)

        Returns:
            (부모 청크 리스트, 자식 청크 리스트) - 문맥 정보 포함
        """
        # 1. 기본 청킹 수행
        parent_chunks, child_chunks = self.chunk_document(
            document=document,
            caption_results=caption_results,
        )

        # 2. 문맥 정보 추가
        if self.use_llm_context:
            child_chunks = await self._add_llm_context(
                child_chunks=child_chunks,
                document_name=document.source,
                document_summary=document_summary,
            )
        else:
            child_chunks = self._add_simple_context(
                child_chunks=child_chunks,
                document_name=document.source,
            )

        # 3. 부모 청크도 문맥 추가
        parent_chunks = self._add_parent_context(
            parent_chunks=parent_chunks,
            document_name=document.source,
        )

        return parent_chunks, child_chunks

    async def _add_llm_context(
        self,
        child_chunks: list[ChildChunk],
        document_name: str,
        document_summary: str = "",
    ) -> list[ChildChunk]:
        """LLM을 사용하여 각 청크에 문맥 추가"""
        print(f"  📝 LLM 문맥 생성 중: {len(child_chunks)}개 청크...")

        # 청크 정보 준비
        chunk_infos = [
            {
                "content": chunk.content,
                "section": chunk.heading or "",
                "page": chunk.page,
            }
            for chunk in child_chunks
        ]

        # 배치로 문맥 생성
        contexts = await self.context_generator.generate_contexts_batch(
            chunks=chunk_infos,
            document_name=document_name,
            document_summary=document_summary,
            batch_size=10,
        )

        # 문맥 추가
        for chunk, context in zip(child_chunks, contexts):
            chunk.content = f"{context}\n\n{chunk.content}"

        print(f"  ✓ 문맥 생성 완료")
        return child_chunks

    def _add_simple_context(
        self,
        child_chunks: list[ChildChunk],
        document_name: str,
    ) -> list[ChildChunk]:
        """간단한 규칙 기반 문맥 추가 (LLM 없이)"""
        for chunk in child_chunks:
            context_parts = [f"[출처: {document_name}"]

            if chunk.heading:
                context_parts.append(f", 섹션: {chunk.heading}")

            if chunk.page > 0:
                context_parts.append(f", 페이지: {chunk.page}")

            # 요소 타입 추가
            if hasattr(chunk, 'element_type') and chunk.element_type:
                type_map = {
                    "table": "테이블",
                    "image": "이미지/차트",
                    "chart": "차트",
                    "paragraph": "본문",
                }
                element_type_kr = type_map.get(chunk.element_type, chunk.element_type)
                context_parts.append(f", 유형: {element_type_kr}")

            context_parts.append("]")
            context_header = "".join(context_parts)

            chunk.content = f"{context_header}\n\n{chunk.content}"

        return child_chunks

    def _add_parent_context(
        self,
        parent_chunks: list[ParentChunk],
        document_name: str,
    ) -> list[ParentChunk]:
        """부모 청크에 문맥 추가"""
        for chunk in parent_chunks:
            context_parts = [f"[문서: {document_name}"]

            if chunk.heading:
                context_parts.append(f", 섹션: {chunk.heading}")

            if chunk.start_page > 0:
                if chunk.start_page == chunk.end_page:
                    context_parts.append(f", 페이지: {chunk.start_page}")
                else:
                    context_parts.append(f", 페이지: {chunk.start_page}-{chunk.end_page}")

            context_parts.append("]")
            context_header = "".join(context_parts)

            chunk.content = f"{context_header}\n\n{chunk.content}"

        return parent_chunks


async def generate_document_summary(
    document: ParsedDocument,
    api_key: Optional[str] = None,
    model: str = "gpt-5.2",
) -> str:
    """
    문서 전체 요약 생성 (선택적 사용)

    Args:
        document: 파싱된 문서
        api_key: OpenAI API 키
        model: 사용할 모델

    Returns:
        문서 요약 (200-300자)
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    # 문서의 처음 부분에서 텍스트 추출
    text_parts = []
    for element in document.elements[:20]:  # 처음 20개 요소
        if element.element_type in (ElementType.PARAGRAPH, ElementType.HEADING):
            text_parts.append(element.content)

    sample_text = "\n".join(text_parts)[:3000]

    prompt = f"""다음 문서의 내용을 200-300자로 요약하세요.
문서의 주제, 목적, 주요 내용을 포함해주세요.

문서명: {document.source}

내용:
{sample_text}

요약:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": 500,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"문서 요약 생성 실패: {e}")

    return ""
