# -*- coding: utf-8 -*-
"""
Advisor OSC Finance v1.6.4 인덱싱 스크립트

파이프라인 (OpenAI 권장 구성):
- Document Parser: PyMuPDF4LLM (마크다운 출력)
- Image Captioning: GPT-4o
- Embedding: OpenAI text-embedding-3-large (3072 dim)
- Collection: advisor_osc_finance_v164_pymupdf4llm
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional
import uuid

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
local_env = Path(__file__).parent / ".env"
if local_env.exists():
    load_dotenv(local_env, override=True)
else:
    load_dotenv(PROJECT_ROOT / ".env.docker", override=True)

import pymupdf4llm
import fitz
from openai import AsyncOpenAI
from config import get_settings
from rag.chunkers import HierarchicalChunker
from rag.embeddings import OpenAIEmbeddingService, SparseEmbeddingService, MultimodalEmbeddingService
from rag.vectorstore import QdrantVectorStore
from rag.parsers.document_parser import ParsedDocument, ParsedElement, ElementType

# v1.6.4 설정
VERSION = "v1.6.4"
COLLECTION_NAME = "advisor_osc_finance_v164_pymupdf4llm"
PDF_DIR = Path(__file__).parent / "files" / "finance"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@dataclass
class PyMuPDF4LLMParser:
    """PyMuPDF4LLM 기반 PDF 파서 (마크다운 출력)"""

    async def parse(self, pdf_path: Path) -> ParsedDocument:
        """PDF를 PyMuPDF4LLM으로 파싱 (마크다운 형식)"""
        # PyMuPDF4LLM으로 마크다운 추출
        md_text = pymupdf4llm.to_markdown(str(pdf_path))

        # 페이지별로 분리
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        # 전체 마크다운을 하나의 요소로 생성
        elements = [ParsedElement(
            element_id=str(uuid.uuid4()),
            element_type=ElementType.PARAGRAPH,
            content=md_text,
            page=1,
            markdown_content=md_text,
            metadata={"parser": "pymupdf4llm", "source": pdf_path.name}
        )]

        return ParsedDocument(
            source=str(pdf_path),
            filename=pdf_path.name,
            total_pages=total_pages,
            elements=elements,
            metadata={"parser": "pymupdf4llm", "version": VERSION}
        )


@dataclass
class GPT4oImageCaptioner:
    """GPT-4o 기반 이미지 캡셔닝"""
    api_key: str
    model: str = "gpt-4o"

    async def caption_images(self, pdf_path: Path) -> List[dict]:
        """PDF에서 이미지 추출 후 GPT-4o로 캡셔닝"""
        import base64

        doc = fitz.open(pdf_path)
        captions = []
        client = AsyncOpenAI(api_key=self.api_key)

        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images()

            for img_idx, img in enumerate(images):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_bytes = pix.tobytes("png")
                    img_base64 = base64.b64encode(img_bytes).decode()

                    # GPT-4o로 이미지 설명 생성
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "이 이미지를 한국어로 상세히 설명해주세요. 차트나 그래프인 경우 데이터와 트렌드를 포함해주세요."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}", "detail": "high"}}
                            ]
                        }],
                        max_tokens=1024
                    )

                    caption = response.choices[0].message.content
                    captions.append({
                        "page": page_num + 1,
                        "image_idx": img_idx,
                        "caption": caption
                    })

                except Exception as e:
                    print(f"    이미지 캡셔닝 실패 (p{page_num+1}, img{img_idx}): {e}")

        doc.close()
        return captions


async def main():
    """Finance v1.6.4 인덱싱 실행"""
    print("=" * 60)
    print(f"Advisor OSC Finance {VERSION} 인덱싱")
    print("=" * 60)
    print(f"  Document Parser: PyMuPDF4LLM (마크다운)")
    print(f"  Image Captioning: GPT-4o")
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

    # 파서 및 캡셔너 초기화
    parser = PyMuPDF4LLMParser()
    captioner = GPT4oImageCaptioner(api_key=OPENAI_API_KEY)
    print(f"\n[파서 초기화] PyMuPDF4LLM + GPT-4o ImageCaptioner")

    # 청커 초기화
    chunker = HierarchicalChunker(
        parent_chunk_size=settings.parent_chunk_size,
        child_chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # 임베딩 서비스 초기화 (OpenAI)
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

    total_stats = {'total_pages': 0, 'total_elements': 0, 'parent_chunks': 0, 'child_chunks': 0, 'success': 0, 'failed': 0, 'images_captioned': 0}

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] {pdf_path.name}")
        try:
            # PyMuPDF4LLM으로 파싱
            parsed_doc = await parser.parse(pdf_path)
            print(f"  - 페이지: {parsed_doc.total_pages}, 요소: {len(parsed_doc.elements)}")

            # GPT-4o로 이미지 캡셔닝
            captions = await captioner.caption_images(pdf_path)
            print(f"  - 이미지 캡션: {len(captions)}개")
            total_stats['images_captioned'] += len(captions)

            # 청킹 (caption_results는 빈 리스트로 전달 - 형식 호환성 문제)
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
    print(f"  이미지 캡션: {total_stats['images_captioned']}개")
    print(f"  부모 청크: {total_stats['parent_chunks']}")
    print(f"  자식 청크: {total_stats['child_chunks']}")
    print(f"  컬렉션: {COLLECTION_NAME}")
    print(f"  완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    await vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())
