"""
V7.2 Multimodal Hierarchical RAG

V7.1 대비 변경점:
- Azure OCR → Upstage Document OCR
- 이미지 처리 속도 9배 향상 (4524ms → 486ms)
- 동일한 신뢰도 유지 (0.95)

Collection: mh_rag_finance_v7_2
"""

import asyncio
from pathlib import Path
from typing import Optional, AsyncIterator
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, Settings

from rag.parsers import (
    UpstageDocumentParser,
    OpenAIImageCaptioner,
    BatchImageCaptioner,
    ParsedDocument,
    CaptionResult,
)
from rag.parsers.hybrid_image_processor_v72 import HybridImageProcessorV72, BatchHybridProcessorV72
from rag.chunkers import (
    HierarchicalChunker,
    ParentChunk,
    ChildChunk,
)
from rag.embeddings import (
    OpenAIEmbeddingService,
    SparseEmbeddingService,
    MultimodalEmbeddingService,
)
from rag.vectorstore import (
    QdrantVectorStore,
    HybridSearchConfig,
)
from rag.retriever import (
    HierarchicalRetriever,
    RetrievalConfig,
    RetrievalResult,
    EnhancedHierarchicalRetriever,
    EnhancedRetrievalConfig,
    EnhancedRetrievalResult,
)
from rag.chain import (
    RAGChain,
    RAGResponse,
    ChatMessage,
)


# V7.2 Collection 이름
V72_COLLECTION_NAME = "mh_rag_finance_v7_2"


class MultimodalHierarchicalRAGV72:
    """V7.2 멀티모달 계층적 RAG 시스템

    V7.1 대비 변경점:
    - Upstage Document OCR 사용 (Azure 대신)
    - 이미지 처리 속도 9배 향상
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        collection_name: str = V72_COLLECTION_NAME,
    ):
        """
        Args:
            settings: 설정 객체 (없으면 환경 변수에서 로드)
            collection_name: Qdrant Collection 이름
        """
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self._initialized = False

        self._parser: Optional[UpstageDocumentParser] = None
        self._captioner: Optional[BatchImageCaptioner] = None
        self._hybrid_processor: Optional[BatchHybridProcessorV72] = None
        self._use_hybrid_image: bool = True
        self._chunker: Optional[HierarchicalChunker] = None
        self._embedding_service: Optional[MultimodalEmbeddingService] = None
        self._vector_store: Optional[QdrantVectorStore] = None
        self._retriever: Optional[HierarchicalRetriever] = None
        self._enhanced_retriever: Optional[EnhancedHierarchicalRetriever] = None
        self._chain: Optional[RAGChain] = None
        self._use_enhanced: bool = True

    async def initialize(self) -> None:
        """시스템 초기화"""
        if self._initialized:
            return

        print("=" * 60)
        print("V7.2 Multimodal Hierarchical RAG 초기화")
        print("  - OCR: Upstage Document OCR (Azure 대체)")
        print("  - VLM: GPT-4o")
        print(f"  - Collection: {self.collection_name}")
        print("=" * 60)

        self._parser = UpstageDocumentParser(
            api_key=self.settings.upstage_api_key,
        )

        image_captioner = OpenAIImageCaptioner(
            api_key=self.settings.openai_api_key,
            model=self.settings.vlm_model,
        )
        self._captioner = BatchImageCaptioner(captioner=image_captioner)

        # V7.2: Upstage Document OCR + VLM 하이브리드 프로세서
        hybrid_processor = HybridImageProcessorV72(
            openai_api_key=self.settings.openai_api_key,
            upstage_api_key=self.settings.upstage_api_key,
            vlm_model=self.settings.vlm_model,
        )
        if hybrid_processor.initialize():
            self._hybrid_processor = BatchHybridProcessorV72(
                processor=hybrid_processor,
                max_concurrent=5,  # Upstage OCR이 빠르므로 동시성 증가
            )
            print("  ✓ Upstage Document OCR 하이브리드 이미지 프로세서 활성화 (V7.2)")
        else:
            self._use_hybrid_image = False
            print("  ✗ Upstage OCR 미설정, 기본 VLM 캡셔너 사용")

        self._chunker = HierarchicalChunker(
            parent_chunk_size=self.settings.parent_chunk_size,
            child_chunk_size=self.settings.child_chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        dense_service = OpenAIEmbeddingService(
            api_key=self.settings.openai_api_key,
            model=self.settings.embedding_model,
        )
        sparse_service = SparseEmbeddingService()
        self._embedding_service = MultimodalEmbeddingService(
            dense_service=dense_service,
            sparse_service=sparse_service,
        )

        # V7.2 전용 Collection 사용
        self._vector_store = QdrantVectorStore(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            api_key=self.settings.qdrant_api_key,
            collection_name=self.collection_name,
        )
        await self._vector_store.initialize()

        # 기본 Retriever
        self._retriever = HierarchicalRetriever(
            vector_store=self._vector_store,
            embedding_service=self._embedding_service,
            config=RetrievalConfig(
                top_k=8,
                use_hybrid=False,
                expand_to_parent=True,
                rerank=True,
                rerank_top_k=25,
            ),
        )

        # Enhanced Retriever V7 (V7.1과 동일한 설정)
        self._enhanced_retriever = EnhancedHierarchicalRetriever(
            vector_store=self._vector_store,
            embedding_service=self._embedding_service,
            api_key=self.settings.openai_api_key,
            config=EnhancedRetrievalConfig(
                top_k=20,
                rerank_top_k=60,
                expand_to_parent=True,
                rerank=True,
                enable_query_expansion=True,
                enable_multi_query=False,
                enable_adaptive_search=True,
                enable_rrf=True,
                enable_hyde=False,
                num_sub_queries=3,
                rrf_k=60,
                rrf_dense_weight=1.5,
                rrf_sparse_weight=0.5,
                enable_table_adaptive=True,
            ),
        )
        print("  ✓ Enhanced Retriever V7 활성화")

        self._chain = RAGChain(
            retriever=self._retriever,
            api_key=self.settings.openai_api_key,
            model=self.settings.llm_model,
        )

        self._initialized = True
        print(f"\n✓ V7.2 Multimodal Hierarchical RAG 시스템 초기화 완료")

    async def ingest_document(
        self,
        file_path: str | Path,
        caption_images: bool = True,
    ) -> dict:
        """문서를 인덱싱"""
        if not self._initialized:
            await self.initialize()

        file_path = Path(file_path)
        print(f"\n📄 문서 파싱 중: {file_path.name}")

        parsed_doc = await self._parser.parse(file_path)
        print(f"  - 총 {len(parsed_doc.elements)}개 요소, {parsed_doc.total_pages}페이지")

        caption_results = []
        hybrid_results = []
        if caption_images:
            images = parsed_doc.get_images_and_charts()
            if images:
                print(f"🖼️ 이미지/차트 처리 중: {len(images)}개")

                # V7.2: Upstage Document OCR + VLM 하이브리드 처리
                if self._use_hybrid_image and self._hybrid_processor:
                    print("  - V7.2 하이브리드 모드 (Upstage Document OCR + VLM)")
                    hybrid_results = await self._hybrid_processor.process_elements(
                        elements=parsed_doc.elements,
                    )
                    # 하이브리드 결과를 CaptionResult로 변환
                    for hr in hybrid_results:
                        caption_results.append(CaptionResult(
                            element_id=hr.element_id,
                            original_content="",
                            caption=hr.combined_content,
                            summary=hr.vlm_summary,
                            key_values={},
                            metadata=hr.metadata,
                        ))
                    print(f"  - {len(hybrid_results)}개 V7.2 하이브리드 처리 완료")
                else:
                    # 기본 VLM 캡셔너
                    print("  - VLM 캡셔너 모드")
                    caption_results = await self._captioner.caption_elements(
                        elements=parsed_doc.elements,
                    )
                    print(f"  - {len(caption_results)}개 캡셔닝 완료")

        print(f"📝 계층적 청킹 중...")
        parent_chunks, child_chunks = self._chunker.chunk_document(
            document=parsed_doc,
            caption_results=caption_results,
        )
        print(f"  - 부모 청크: {len(parent_chunks)}개")
        print(f"  - 자식 청크: {len(child_chunks)}개")

        print(f"🔢 임베딩 생성 중...")
        dense_embeddings, sparse_embeddings = await self._embedding_service.embed_chunks(
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
        )
        print(f"  - {len(dense_embeddings)}개 임베딩 생성 완료")

        print(f"💾 벡터 DB에 저장 중...")
        await self._vector_store.add_chunks(
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )
        print(f"✓ 문서 인덱싱 완료: {file_path.name}")

        return {
            "filename": file_path.name,
            "total_pages": parsed_doc.total_pages,
            "total_elements": len(parsed_doc.elements),
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
            "captioned_images": len(caption_results),
            "version": "v7.2",
            "ocr_provider": "upstage_document_ocr",
        }

    async def ingest_documents(
        self,
        file_paths: list[str | Path],
        caption_images: bool = True,
    ) -> list[dict]:
        """여러 문서를 순차적으로 인덱싱"""
        results = []
        for file_path in file_paths:
            try:
                result = await self.ingest_document(file_path, caption_images)
                results.append(result)
            except Exception as e:
                results.append({
                    "filename": str(file_path),
                    "error": str(e),
                })
        return results

    async def chat(
        self,
        query: str,
        conversation_history: Optional[list[ChatMessage]] = None,
        use_enhanced: Optional[bool] = None,
    ) -> RAGResponse:
        """RAG 기반 채팅"""
        if not self._initialized:
            await self.initialize()

        should_use_enhanced = use_enhanced if use_enhanced is not None else self._use_enhanced

        if should_use_enhanced and self._enhanced_retriever:
            return await self._chat_enhanced(query, conversation_history)
        else:
            return await self._chain.chat(
                query=query,
                conversation_history=conversation_history,
            )

    async def _chat_enhanced(
        self,
        query: str,
        conversation_history: Optional[list[ChatMessage]] = None,
    ) -> RAGResponse:
        """Enhanced Retriever를 사용한 채팅"""
        enhanced_result = await self._enhanced_retriever.retrieve(query)

        answer = await self._generate_answer(
            query=query,
            context=enhanced_result.context,
            conversation_history=conversation_history,
        )

        return RAGResponse(
            answer=answer,
            sources=enhanced_result.sources,
            retrieval_result=None,
            metadata={
                "model": self.settings.llm_model,
                "enhanced": True,
                "version": "v7.2",
                "ocr_provider": "upstage_document_ocr",
                "enhancements": enhanced_result.metadata.get("enhancements_applied", []),
                "keywords": enhanced_result.enhanced_query.keywords,
            },
        )

    async def _generate_answer(
        self,
        query: str,
        context: str,
        conversation_history: Optional[list[ChatMessage]] = None,
    ) -> str:
        """LLM으로 답변 생성"""
        import httpx

        system_prompt = """당신은 문서를 분석하는 전문가 AI 어시스턴트입니다.
제공된 컨텍스트를 기반으로 사용자의 질문에 정확하게 답변해주세요.

다음 지침을 따라주세요:
1. 컨텍스트에 있는 정보만을 사용하여 답변하세요.
2. 수치나 데이터를 인용할 때는 정확하게 인용하세요.
3. 컨텍스트에서 답을 찾을 수 없으면 솔직하게 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
4. 차트나 그래프 분석 내용이 포함되어 있다면 그 내용도 활용하세요.
5. 답변 마지막에 참조한 출처(문서명, 페이지)를 간단히 언급해주세요.

답변은 명확하고 구조화되어야 하며, 필요시 불릿 포인트나 번호 매기기를 사용하세요."""

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({"role": msg.role, "content": msg.content})

        user_message = f"""다음 컨텍스트를 참고하여 질문에 답변해주세요.

## 컨텍스트
{context}

## 질문
{query}"""

        messages.append({"role": "user", "content": user_message})

        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": 0.3,
            "max_completion_tokens": 2000,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )

        if response.status_code != 200:
            raise Exception(f"OpenAI API 오류: {response.status_code} - {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_source: Optional[str] = None,
    ) -> RetrievalResult:
        """문서 검색만 수행"""
        if not self._initialized:
            await self.initialize()

        return await self._retriever.retrieve(
            query=query,
            config=RetrievalConfig(top_k=top_k),
            filter_source=filter_source,
        )

    async def delete_collection(self) -> None:
        """Collection 삭제 (재인덱싱 전)"""
        if not self._initialized:
            await self.initialize()

        await self._vector_store.delete_collection()
        print(f"✓ Collection 삭제 완료: {self.collection_name}")

    async def close(self) -> None:
        """리소스 정리"""
        if self._vector_store:
            await self._vector_store.close()
        self._initialized = False


async def create_rag_system_v72(
    settings: Optional[Settings] = None,
    collection_name: str = V72_COLLECTION_NAME,
) -> MultimodalHierarchicalRAGV72:
    """V7.2 RAG 시스템 팩토리 함수"""
    rag = MultimodalHierarchicalRAGV72(settings=settings, collection_name=collection_name)
    await rag.initialize()
    return rag
