# -*- coding: utf-8 -*-
"""
Finance 문서 인덱싱 스크립트 (V7.6.1 Azure)

Azure Document Intelligence 파서를 사용하여
mh_rag_finance_v7_6_azure 컬렉션에 인덱싱
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

# .bashrc에서 Azure 키 로드
import subprocess
result = subprocess.run(
    ['bash', '-c', 'source ~/.bashrc && echo "$AZURE_DOCUMENT_INTELLEGENCE_KEY|$AZURE_DOCUMENT_INTELLEGENCE_END"'],
    capture_output=True, text=True
)
parts = result.stdout.strip().split('|')
if len(parts) == 2:
    os.environ['AZURE_DOCUMENT_INTELLIGENCE_KEY'] = parts[0]
    os.environ['AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'] = parts[1]
    print(f"Azure API Key loaded: {parts[0][:15]}...")
    print(f"Azure Endpoint: {parts[1]}")

from dotenv import load_dotenv
load_dotenv()

# MH_rag의 Azure 파서와 임베딩 서비스 사용
sys.path.insert(0, '/home/aiedu/workspace/MH_rag')
from src.parsers.azure_document_parser import AzureDocumentParser
from src.parsers.hybrid_image_processor import HybridImageProcessor
from src.embeddings import OpenAIEmbeddingService, SparseEmbeddingService, MultimodalEmbeddingService
from src.vectorstore import QdrantVectorStore
from src.chunkers import HierarchicalChunker

# 설정
COLLECTION_NAME = "mh_rag_finance_v7_6_azure"
QDRANT_HOST = "localhost"
QDRANT_PORT = 10304  # advisor production Qdrant
FINANCE_DIR = Path("/home/aiedu/workspace/MH_rag/eval/allganize/files/finance")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def create_collection_if_not_exists(vector_store: QdrantVectorStore):
    """컬렉션이 없으면 생성"""
    from qdrant_client.models import Distance, VectorParams

    try:
        info = await vector_store.get_collection_info()
        print(f"컬렉션 '{COLLECTION_NAME}' 이미 존재 - {info.get('points_count', 0)} points")
        return True
    except:
        print(f"컬렉션 '{COLLECTION_NAME}' 생성 중...")
        # 3072 차원 (text-embedding-3-large)
        await vector_store.create_collection(
            vectors_config={
                "dense": VectorParams(size=3072, distance=Distance.COSINE)
            }
        )
        print(f"컬렉션 '{COLLECTION_NAME}' 생성 완료")
        return True


async def index_document(
    pdf_path: Path,
    parser: AzureDocumentParser,
    chunker: HierarchicalChunker,
    embedding_service: MultimodalEmbeddingService,
    vector_store: QdrantVectorStore,
    image_processor: HybridImageProcessor = None,
) -> dict:
    """단일 문서 인덱싱"""
    print(f"\n파싱 중: {pdf_path.name}")

    # 1. Azure Document Intelligence로 파싱
    parsed_doc = await parser.parse(str(pdf_path))
    print(f"  - 총 {len(parsed_doc.elements)}개 요소, {parsed_doc.total_pages}페이지")

    # 2. 이미지/차트 캡셔닝 (옵션)
    caption_results = []
    if image_processor:
        images = parsed_doc.get_images_and_charts()
        if images:
            print(f"  - 이미지/차트 {len(images)}개 캡셔닝 중...")
            for img_elem in images:
                try:
                    caption_result = await image_processor.process_image_element(
                        img_elem,
                        elements=parsed_doc.elements,
                    )
                    if caption_result and caption_result.caption:
                        img_elem.caption = caption_result.caption
                        caption_results.append(caption_result)
                except Exception as e:
                    print(f"    캡셔닝 실패: {e}")

    # 3. 계층적 청킹
    chunks = chunker.chunk(
        document=parsed_doc,
        source=pdf_path.name,
    )

    parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
    child_chunks = [c for c in chunks if c.chunk_type == "child"]
    print(f"  - Parent: {len(parent_chunks)}, Child: {len(child_chunks)}")

    # 4. 임베딩 생성
    dense_embeddings, sparse_embeddings = await embedding_service.embed_chunks(
        [c.content for c in chunks]
    )

    # 5. Qdrant에 저장
    await vector_store.upsert_chunks(
        chunks=chunks,
        dense_embeddings=dense_embeddings,
        sparse_embeddings=sparse_embeddings,
    )

    return {
        "filename": pdf_path.name,
        "total_pages": parsed_doc.total_pages,
        "total_elements": len(parsed_doc.elements),
        "parent_chunks": len(parent_chunks),
        "child_chunks": len(child_chunks),
        "captioned_images": len(caption_results),
    }


async def main():
    print("=" * 70)
    print("Finance 문서 인덱싱 - V7.6.1 Azure")
    print("=" * 70)
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"컬렉션: {COLLECTION_NAME}")
    print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print("=" * 70)

    # PDF 파일 목록
    pdf_files = list(FINANCE_DIR.glob("*.pdf"))
    print(f"\nPDF 파일: {len(pdf_files)}개")
    for f in pdf_files:
        print(f"  - {f.name}")

    # 서비스 초기화
    print("\n서비스 초기화 중...")

    # Azure Document Parser
    parser = AzureDocumentParser()

    # 임베딩 서비스
    dense_service = OpenAIEmbeddingService(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-large"
    )
    sparse_service = SparseEmbeddingService()
    embedding_service = MultimodalEmbeddingService(dense_service, sparse_service)

    # 벡터 스토어
    vector_store = QdrantVectorStore(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=COLLECTION_NAME,
    )

    # 청커
    chunker = HierarchicalChunker(
        parent_chunk_size=2000,
        child_chunk_size=500,
        chunk_overlap=50,
    )

    # 이미지 프로세서 (Azure OCR + VLM 하이브리드)
    try:
        image_processor = HybridImageProcessor(
            azure_api_key=os.environ.get('AZURE_DOCUMENT_INTELLIGENCE_KEY', ''),
            openai_api_key=OPENAI_API_KEY,
        )
        print("  - 이미지 프로세서: Azure OCR + VLM 하이브리드")
    except Exception as e:
        image_processor = None
        print(f"  - 이미지 프로세서 초기화 실패: {e}")

    # 컬렉션 생성/확인
    await create_collection_if_not_exists(vector_store)

    # 인덱싱 실행
    print("\n인덱싱 시작...")
    stats = {
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
            result = await index_document(
                pdf_path=pdf_path,
                parser=parser,
                chunker=chunker,
                embedding_service=embedding_service,
                vector_store=vector_store,
                image_processor=image_processor,
            )
            stats['total_pages'] += result.get('total_pages', 0)
            stats['total_elements'] += result.get('total_elements', 0)
            stats['parent_chunks'] += result.get('parent_chunks', 0)
            stats['child_chunks'] += result.get('child_chunks', 0)
            stats['captioned_images'] += result.get('captioned_images', 0)
            stats['success'] += 1
            print(f"  ✓ 완료 - 페이지: {result.get('total_pages', 0)}, 청크: {result.get('parent_chunks', 0)}")
        except Exception as e:
            stats['failed'] += 1
            print(f"  ✗ 실패: {str(e)[:100]}")

    # 결과 출력
    print("\n" + "=" * 70)
    print("인덱싱 완료")
    print("=" * 70)
    print(f"성공: {stats['success']}/{len(pdf_files)}")
    print(f"실패: {stats['failed']}/{len(pdf_files)}")
    print(f"총 페이지: {stats['total_pages']}")
    print(f"총 요소: {stats['total_elements']}")
    print(f"Parent 청크: {stats['parent_chunks']}")
    print(f"Child 청크: {stats['child_chunks']}")
    print(f"캡셔닝 이미지: {stats['captioned_images']}")
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main())
