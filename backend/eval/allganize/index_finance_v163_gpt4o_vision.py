# -*- coding: utf-8 -*-
"""
Advisor OSC Finance v1.6.3 인덱싱 스크립트

파이프라인:
- Document Parser: GPT-4o Vision (PDF → 이미지 → GPT-4o)
- Image Captioning: GPT-4o
- Embedding: OpenAI text-embedding-3-large (3072 dim)
- Collection: advisor_osc_finance_v163_gpt4o_vision
"""

import os
import sys
import asyncio
import base64
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional
import fitz  # PyMuPDF

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
local_env = Path(__file__).parent / ".env"
if local_env.exists():
    load_dotenv(local_env, override=True)
else:
    load_dotenv(PROJECT_ROOT / ".env.docker", override=True)

from openai import AsyncOpenAI
from config import get_settings
from rag.chunkers import HierarchicalChunker
from rag.embeddings import OpenAIEmbeddingService, SparseEmbeddingService, MultimodalEmbeddingService
from rag.vectorstore import QdrantVectorStore
from rag.parsers.document_parser import ParsedDocument, ParsedElement, ElementType
import uuid

# v1.6.3 설정
VERSION = "v1.6.3"
COLLECTION_NAME = "advisor_osc_finance_v163_gpt4o_vision"
PDF_DIR = Path(__file__).parent / "files" / "finance"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@dataclass
class GPT4oVisionParser:
    """GPT-4o Vision 기반 PDF 파서"""
    api_key: str
    model: str = "gpt-4o"
    dpi: int = 150
    max_concurrent: int = 3

    async def parse(self, pdf_path: Path) -> ParsedDocument:
        """PDF를 GPT-4o Vision으로 파싱"""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        elements = []

        client = AsyncOpenAI(api_key=self.api_key)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def parse_page(page_num: int) -> List[ParsedElement]:
            async with semaphore:
                page = doc[page_num]
                pix = page.get_pixmap(dpi=self.dpi)
                img_bytes = pix.tobytes("png")
                img_base64 = base64.b64encode(img_bytes).decode()

                prompt = """이 문서 페이지의 모든 내용을 Markdown 형식으로 추출해주세요.

규칙:
1. 텍스트는 그대로 추출
2. 테이블은 Markdown 테이블 형식으로 변환
3. 차트/그래프가 있으면 [CHART: 설명] 형식으로 설명
4. 이미지가 있으면 [IMAGE: 설명] 형식으로 설명
5. 제목/헤더는 # 마크다운 형식 사용
6. 순서는 문서에 나타난 순서대로

모든 텍스트, 수치, 데이터를 누락 없이 추출해주세요."""

                try:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}", "detail": "high"}}
                            ]
                        }],
                        max_tokens=4096,
                        temperature=0
                    )

                    content = response.choices[0].message.content

                    return [ParsedElement(
                        element_id=str(uuid.uuid4()),
                        element_type=ElementType.PARAGRAPH,
                        content=content,
                        page=page_num + 1,
                        markdown_content=content,
                        metadata={"parser": "gpt-4o-vision", "source": pdf_path.name}
                    )]

                except Exception as e:
                    print(f"    페이지 {page_num + 1} 파싱 오류: {e}")
                    return []

        # 병렬로 페이지 파싱
        tasks = [parse_page(i) for i in range(total_pages)]
        results = await asyncio.gather(*tasks)

        for page_elements in results:
            elements.extend(page_elements)

        doc.close()

        return ParsedDocument(
            source=str(pdf_path),
            filename=pdf_path.name,
            total_pages=total_pages,
            elements=elements,
            metadata={"parser": "gpt-4o-vision", "version": VERSION}
        )


async def main():
    """Finance v1.6.3 인덱싱 실행"""
    print("=" * 60)
    print(f"Advisor OSC Finance {VERSION} 인덱싱")
    print("=" * 60)
    print(f"  Document Parser: GPT-4o Vision")
    print(f"  Embedding: OpenAI text-embedding-3-large")
    print(f"  컬렉션: {COLLECTION_NAME}")
    print(f"  PDF 디렉토리: {PDF_DIR}")
    print(f"  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # PDF 파일 목록
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"\n발견된 PDF 파일: {len(pdf_files)}개")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf.name}")

    if not pdf_files:
        print("PDF 파일이 없습니다.")
        return

    settings = get_settings()

    # GPT-4o Vision 파서 초기화
    parser = GPT4oVisionParser(api_key=OPENAI_API_KEY)
    print(f"\n[파서 초기화] GPT-4o Vision")

    # 청커 초기화
    chunker = HierarchicalChunker(
        parent_chunk_size=settings.parent_chunk_size,
        child_chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # 임베딩 서비스 초기화
    dense_service = OpenAIEmbeddingService(api_key=settings.openai_api_key, model="text-embedding-3-large")
    sparse_service = SparseEmbeddingService()
    embedding_service = MultimodalEmbeddingService(dense_service=dense_service, sparse_service=sparse_service)

    # 벡터 스토어 초기화
    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        collection_name=COLLECTION_NAME,
    )
    await vector_store.initialize()

    print(f"\n{'=' * 60}")
    print("인덱싱 시작")
    print("=" * 60)

    total_stats = {'total_pages': 0, 'total_elements': 0, 'parent_chunks': 0, 'child_chunks': 0, 'success': 0, 'failed': 0}

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] {pdf_path.name}")
        try:
            # GPT-4o Vision으로 파싱
            parsed_doc = await parser.parse(pdf_path)
            print(f"  - 페이지: {parsed_doc.total_pages}, 요소: {len(parsed_doc.elements)}")

            # 청킹
            parent_chunks, child_chunks = chunker.chunk_document(document=parsed_doc, caption_results=[])
            print(f"  - 부모 청크: {len(parent_chunks)}, 자식 청크: {len(child_chunks)}")

            # 임베딩
            dense_embeddings, sparse_embeddings = await embedding_service.embed_chunks(
                parent_chunks=parent_chunks, child_chunks=child_chunks)

            # 저장
            await vector_store.add_chunks(
                parent_chunks=parent_chunks, child_chunks=child_chunks,
                dense_embeddings=dense_embeddings, sparse_embeddings=sparse_embeddings)

            total_stats['total_pages'] += parsed_doc.total_pages
            total_stats['total_elements'] += len(parsed_doc.elements)
            total_stats['parent_chunks'] += len(parent_chunks)
            total_stats['child_chunks'] += len(child_chunks)
            total_stats['success'] += 1
            print(f"  ✓ 완료")

        except Exception as e:
            print(f"  ✗ 오류: {e}")
            import traceback
            traceback.print_exc()
            total_stats['failed'] += 1

    print(f"\n{'=' * 60}")
    print(f"인덱싱 완료 ({VERSION})")
    print("=" * 60)
    print(f"  성공: {total_stats['success']}개")
    print(f"  실패: {total_stats['failed']}개")
    print(f"  총 페이지: {total_stats['total_pages']}")
    print(f"  부모 청크: {total_stats['parent_chunks']}")
    print(f"  자식 청크: {total_stats['child_chunks']}")
    print(f"  컬렉션: {COLLECTION_NAME}")
    print(f"  완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    await vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())
