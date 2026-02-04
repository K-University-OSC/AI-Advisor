#!/usr/bin/env python3
"""
문서 재인덱싱 스크립트
수정된 index_document_task를 사용하여 올바른 컬렉션에 인덱싱
"""
import asyncio
import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, '/app')
os.chdir('/app')

from dotenv import load_dotenv
load_dotenv('/app/.env.docker', override=True)

async def reindex_all_documents():
    """모든 문서 재인덱싱"""
    from database import get_db
    from sqlalchemy import text
    from routers.data_management import index_document_task

    print(f"=== 문서 재인덱싱 시작 ===")
    print(f"Collection: {os.getenv('QDRANT_COLLECTION_NAME')}")
    print(f"Embedding: {os.getenv('EMBEDDING_PROVIDER')} / {os.getenv('EMBEDDING_MODEL')}")
    print()

    async with get_db() as session:
        result = await session.execute(
            text("SELECT id, filename, original_filename FROM rag_documents ORDER BY uploaded_at DESC")
        )
        docs = result.mappings().all()

    print(f"총 {len(docs)}개 문서 발견\n")

    for doc in docs:
        doc_id = doc['id']
        filename = doc['filename']
        original = doc['original_filename']
        file_path = f"/app/data/documents/{filename}"

        print(f"📄 {original}")
        print(f"   ID: {doc_id}")
        print(f"   Path: {file_path}")

        if not os.path.exists(file_path):
            print(f"   ❌ 파일 없음, 건너뜀")
            continue

        try:
            await index_document_task(doc_id, file_path)
            print(f"   ✅ 인덱싱 완료")
        except Exception as e:
            print(f"   ❌ 오류: {e}")
        print()

    print("=== 재인덱싱 완료 ===")

if __name__ == "__main__":
    asyncio.run(reindex_all_documents())
