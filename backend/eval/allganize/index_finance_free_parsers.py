# -*- coding: utf-8 -*-
"""
Advisor OSC Finance 인덱싱 스크립트 (무료 파서 버전)

파서 옵션:
- docling: IBM 오픈소스, 레이아웃 분석 특화 (무료)
- pymupdf: PyMuPDF4LLM (무료, 빠름)
- gemini: Gemini 3 Flash OCR (유료, 고품질)
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
# 로컬 .env 먼저 로드, 없으면 .env.docker 사용
local_env = Path(__file__).parent / ".env"
if local_env.exists():
    load_dotenv(local_env, override=True)
else:
    load_dotenv(PROJECT_ROOT / ".env.docker", override=True)

from config import get_settings, Settings
from rag.parsers import get_document_parser, PARSER_INFO
from rag.parsers.image_captioner import OpenAIImageCaptioner, BatchImageCaptioner
from rag.chunkers import HierarchicalChunker
from rag.embeddings import (
    OpenAIEmbeddingService,
    SparseEmbeddingService,
    MultimodalEmbeddingService,
)
from rag.vectorstore import QdrantVectorStore

# 설정
PARSER_TYPE = os.getenv("DOCUMENT_PARSER", "pymupdf")  # docling, pymupdf, gemini
COLLECTION_NAME = f"advisor_osc_finance_{PARSER_TYPE}"
PDF_DIR = Path(__file__).parent / "files" / "finance"


async def main():
    """Finance 인덱싱 실행"""
    print("=" * 60)
    print(f"Advisor OSC Finance 인덱싱 ({PARSER_TYPE.upper()})")
    print("=" * 60)
    print(f"  컬렉션: {COLLECTION_NAME}")
    print(f"  PDF 디렉토리: {PDF_DIR}")
    print(f"  파서: {PARSER_INFO.get(PARSER_TYPE, {}).get('name', PARSER_TYPE)}")
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

    # 설정 로드
    settings = get_settings()

    # 파서 초기화
    parser = get_document_parser(PARSER_TYPE)
    print(f"\n[파서 초기화] {PARSER_TYPE}")

    # 이미지 캡셔너 초기화
    image_captioner = OpenAIImageCaptioner(
        api_key=settings.openai_api_key,
        model=settings.vlm_model,
    )
    captioner = BatchImageCaptioner(captioner=image_captioner)

    # 청커 초기화
    chunker = HierarchicalChunker(
        parent_chunk_size=settings.parent_chunk_size,
        child_chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # 임베딩 서비스 초기화
    dense_service = OpenAIEmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
    sparse_service = SparseEmbeddingService()
    embedding_service = MultimodalEmbeddingService(
        dense_service=dense_service,
        sparse_service=sparse_service,
    )

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

    # 문서 인덱싱
    total_stats = {
        'total_pages': 0,
        'total_elements': 0,
        'parent_chunks': 0,
        'child_chunks': 0,
        'captioned_images': 0,
        'success': 0,
        'failed': 0,
    }

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] {pdf_path.name}")
        try:
            # 문서 파싱
            parsed_doc = await parser.parse(pdf_path)
            print(f"  - 페이지: {parsed_doc.total_pages}, 요소: {len(parsed_doc.elements)}")

            # 이미지 캡셔닝
            caption_results = []
            images = parsed_doc.get_images_and_charts()
            if images:
                print(f"  - 이미지/차트 캡셔닝: {len(images)}개")
                caption_results = await captioner.caption_elements(
                    elements=parsed_doc.elements,
                )

            # 청킹
            parent_chunks, child_chunks = chunker.chunk_document(
                document=parsed_doc,
                caption_results=caption_results,
            )
            print(f"  - 부모 청크: {len(parent_chunks)}, 자식 청크: {len(child_chunks)}")

            # 임베딩
            dense_embeddings, sparse_embeddings = await embedding_service.embed_chunks(
                parent_chunks=parent_chunks,
                child_chunks=child_chunks,
            )

            # 저장
            await vector_store.add_chunks(
                parent_chunks=parent_chunks,
                child_chunks=child_chunks,
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings,
            )

            total_stats['total_pages'] += parsed_doc.total_pages
            total_stats['total_elements'] += len(parsed_doc.elements)
            total_stats['parent_chunks'] += len(parent_chunks)
            total_stats['child_chunks'] += len(child_chunks)
            total_stats['captioned_images'] += len(caption_results)
            total_stats['success'] += 1
            print(f"  ✓ 완료")

        except Exception as e:
            print(f"  ✗ 오류: {e}")
            total_stats['failed'] += 1

    # 결과 출력
    print(f"\n{'=' * 60}")
    print(f"인덱싱 완료 ({PARSER_TYPE.upper()})")
    print("=" * 60)
    print(f"  성공: {total_stats['success']}개")
    print(f"  실패: {total_stats['failed']}개")
    print(f"  총 페이지: {total_stats['total_pages']}")
    print(f"  총 요소: {total_stats['total_elements']}")
    print(f"  부모 청크: {total_stats['parent_chunks']}")
    print(f"  자식 청크: {total_stats['child_chunks']}")
    print(f"  캡셔닝 이미지: {total_stats['captioned_images']}")
    print(f"  컬렉션: {COLLECTION_NAME}")
    print(f"  완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    await vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())
