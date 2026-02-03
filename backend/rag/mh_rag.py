"""
Multimodal Hierarchical RAG 메인 인터페이스
모든 컴포넌트를 통합하여 쉽게 사용할 수 있는 API 제공
"""

import asyncio
from pathlib import Path
from typing import Optional, AsyncIterator
import os
import sys

# advisor/backend를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, Settings

from rag.parsers import (
    UpstageDocumentParser,
    OpenAIImageCaptioner,
    GeminiImageCaptioner,
    BatchImageCaptioner,
    ParsedDocument,
    CaptionResult,
)
from rag.parsers.hybrid_image_processor import HybridImageProcessor, BatchHybridProcessor
from rag.chunkers import (
    HierarchicalChunker,
    ParentChunk,
    ChildChunk,
)
from rag.embeddings import (
    OpenAIEmbeddingService,
    GeminiEmbeddingService,
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


class MultimodalHierarchicalRAG:
    """멀티모달 계층적 RAG 시스템"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ):
        """
        Args:
            settings: 설정 객체 (없으면 환경 변수에서 로드)
        """
        self.settings = settings or get_settings()
        self._initialized = False

        self._parser: Optional[UpstageDocumentParser] = None
        self._captioner: Optional[BatchImageCaptioner] = None
        self._hybrid_processor: Optional[BatchHybridProcessor] = None
        self._use_hybrid_image: bool = True  # 하이브리드 이미지 처리 사용
        self._chunker: Optional[HierarchicalChunker] = None
        self._embedding_service: Optional[MultimodalEmbeddingService] = None
        self._vector_store: Optional[QdrantVectorStore] = None
        self._retriever: Optional[HierarchicalRetriever] = None
        self._enhanced_retriever: Optional[EnhancedHierarchicalRetriever] = None
        self._chain: Optional[RAGChain] = None
        self._use_enhanced: bool = True  # Enhanced Retriever 사용 여부

    async def initialize(self) -> None:
        """시스템 초기화"""
        if self._initialized:
            return

        self._parser = UpstageDocumentParser(
            api_key=self.settings.upstage_api_key,
        )

        # 이미지 캡셔너 선택 (VLM_PROVIDER 환경변수에 따라)
        vlm_provider = os.getenv("VLM_PROVIDER", "openai").lower()
        if vlm_provider == "google":
            google_api_key = os.getenv("GOOGLE_API_KEY", "")
            image_captioner = GeminiImageCaptioner(
                api_key=google_api_key,
                model=self.settings.vlm_model or "gemini-3-flash-preview",
            )
            print(f"  - Gemini 이미지 캡셔너 활성화 ({self.settings.vlm_model or 'gemini-3-flash-preview'})")
        else:
            image_captioner = OpenAIImageCaptioner(
                api_key=self.settings.openai_api_key,
                model=self.settings.vlm_model,
            )
            print(f"  - OpenAI 이미지 캡셔너 활성화 ({self.settings.vlm_model})")
        self._captioner = BatchImageCaptioner(captioner=image_captioner)

        # 하이브리드 이미지 프로세서 (Azure OCR + VLM)
        hybrid_processor = HybridImageProcessor(
            openai_api_key=self.settings.openai_api_key,
            vlm_model=self.settings.vlm_model,
        )
        if hybrid_processor.initialize():
            self._hybrid_processor = BatchHybridProcessor(processor=hybrid_processor)
            print("  - Azure OCR 하이브리드 이미지 프로세서 활성화")
        else:
            self._use_hybrid_image = False
            print("  - Azure OCR 미설정, 기본 VLM 캡셔너 사용")

        self._chunker = HierarchicalChunker(
            parent_chunk_size=self.settings.parent_chunk_size,
            child_chunk_size=self.settings.child_chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        # 임베딩 서비스 선택 (EMBEDDING_PROVIDER 환경변수에 따라)
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        if embedding_provider == "google":
            google_api_key = os.getenv("GOOGLE_API_KEY", "")
            dense_service = GeminiEmbeddingService(
                api_key=google_api_key,
                model=self.settings.embedding_model,
            )
            print(f"  - Gemini 임베딩 서비스 활성화 ({self.settings.embedding_model})")
        else:
            dense_service = OpenAIEmbeddingService(
                api_key=self.settings.openai_api_key,
                model=self.settings.embedding_model,
            )
            print(f"  - OpenAI 임베딩 서비스 활성화 ({self.settings.embedding_model})")
        sparse_service = SparseEmbeddingService()
        self._embedding_service = MultimodalEmbeddingService(
            dense_service=dense_service,
            sparse_service=sparse_service,
        )

        self._vector_store = QdrantVectorStore(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            api_key=self.settings.qdrant_api_key,
            collection_name=self.settings.qdrant_collection_name,
        )
        await self._vector_store.initialize()

        # 기본 Retriever (하위 호환성)
        self._retriever = HierarchicalRetriever(
            vector_store=self._vector_store,
            embedding_service=self._embedding_service,
            config=RetrievalConfig(
                top_k=8,
                use_hybrid=False,  # BM25 비활성화 (한국어 RAG 성능 향상)
                expand_to_parent=True,
                rerank=True,  # BGE Reranker 활성화
                rerank_top_k=25,  # 리랭킹 최적화
            ),
        )

        # Enhanced Retriever V7 (V6 + Query Expansion 다양화 + Fallback 검색 + top_k 증가)
        # V7 개선사항:
        # - Query Expansion: 3개 이상 다양한 관점의 대안 쿼리 생성
        # - Fallback Search: 검색 결과 부족/낮은 점수 시 대안 쿼리로 재검색
        # - top_k 증가: 15 → 20 (더 많은 후보 확보)
        self._enhanced_retriever = EnhancedHierarchicalRetriever(
            vector_store=self._vector_store,
            embedding_service=self._embedding_service,
            api_key=self.settings.openai_api_key,
            config=EnhancedRetrievalConfig(
                top_k=20,  # 15→20: 더 많은 후보 확보 (V7)
                rerank_top_k=60,  # 리랭킹 후보 유지
                expand_to_parent=True,
                rerank=True,
                # 핵심 기능 활성화
                enable_query_expansion=True,  # V7: 다양화된 쿼리 확장
                enable_multi_query=False,  # RRF와 함께 사용 시 비활성화
                enable_adaptive_search=True,
                enable_rrf=True,  # RRF 하이브리드 검색 활성화
                enable_hyde=False,  # HyDE는 실험적
                num_sub_queries=3,  # V7: 3개 대안 쿼리 생성
                # RRF 파라미터
                rrf_k=60,
                rrf_dense_weight=1.5,
                rrf_sparse_weight=0.5,
                # 테이블/이미지 쿼리 적응형 검색
                enable_table_adaptive=True,
            ),
        )
        print("  - Enhanced Retriever V7 활성화 (top_k=20, fallback_search, query_expansion_v2)")

        self._chain = RAGChain(
            retriever=self._retriever,
            api_key=self.settings.openai_api_key,
            model=self.settings.llm_model,
        )

        self._initialized = True
        print(f"✓ Multimodal Hierarchical RAG 시스템 초기화 완료")

    async def ingest_document(
        self,
        file_path: str | Path,
        caption_images: bool = True,
    ) -> dict:
        """
        문서를 인덱싱

        Args:
            file_path: 문서 파일 경로
            caption_images: 이미지/차트 캡셔닝 여부

        Returns:
            인덱싱 결과 정보
        """
        if not self._initialized:
            await self.initialize()

        file_path = Path(file_path)
        print(f"📄 문서 파싱 중: {file_path.name}")

        parsed_doc = await self._parser.parse(file_path)
        print(f"  - 총 {len(parsed_doc.elements)}개 요소, {parsed_doc.total_pages}페이지")

        caption_results = []
        hybrid_results = []
        if caption_images:
            images = parsed_doc.get_images_and_charts()
            if images:
                print(f"🖼️ 이미지/차트 처리 중: {len(images)}개")

                # 하이브리드 처리 (Azure OCR + VLM)
                if self._use_hybrid_image and self._hybrid_processor:
                    print("  - 하이브리드 모드 (Azure OCR + VLM)")
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
                    print(f"  - {len(hybrid_results)}개 하이브리드 처리 완료")
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
        }

    async def ingest_documents(
        self,
        file_paths: list[str | Path],
        caption_images: bool = True,
    ) -> list[dict]:
        """
        여러 문서를 순차적으로 인덱싱

        Args:
            file_paths: 문서 파일 경로 리스트
            caption_images: 이미지/차트 캡셔닝 여부

        Returns:
            각 문서의 인덱싱 결과
        """
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
        """
        RAG 기반 채팅

        Args:
            query: 사용자 질문
            conversation_history: 이전 대화 기록
            use_enhanced: Enhanced Retriever 사용 여부 (None이면 기본값 사용)

        Returns:
            RAG 응답
        """
        if not self._initialized:
            await self.initialize()

        # Enhanced Retriever 사용 여부 결정
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
        # Enhanced Retrieval 수행
        enhanced_result = await self._enhanced_retriever.retrieve(query)

        # LLM 응답 생성
        answer = await self._generate_answer(
            query=query,
            context=enhanced_result.context,
            conversation_history=conversation_history,
        )

        return RAGResponse(
            answer=answer,
            sources=enhanced_result.sources,
            retrieval_result=None,  # Enhanced 결과는 별도 타입
            metadata={
                "model": self.settings.llm_model,
                "enhanced": True,
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
3. 차트나 그래프 분석 내용이 포함되어 있다면 그 내용도 활용하세요.
4. 출처나 참고 문서는 별도로 언급하지 마세요. UI에서 자동으로 표시됩니다.

## 정보 부족 시 대응
- 질문에 대한 **직접적인 답변**이 컨텍스트에 없으면:
  1. "제공된 문서에서 [질문 주제]에 대한 직접적인 정보는 찾을 수 없습니다."라고 명시
  2. 단, 관련된 내용이 조금이라도 있다면 "다만, 관련하여 다음 정보가 있습니다:"로 시작하여 간략히 안내
  3. 완전히 관련 없는 내용만 있다면 그냥 정보 없음만 안내
- 부분적인 정보만 있는 경우: 있는 정보는 답변하고, 없는 부분은 명확히 구분

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

    async def chat_stream(
        self,
        query: str,
        conversation_history: Optional[list[ChatMessage]] = None,
    ) -> AsyncIterator[str]:
        """
        스트리밍 RAG 채팅

        Args:
            query: 사용자 질문
            conversation_history: 이전 대화 기록

        Yields:
            응답 텍스트 청크
        """
        if not self._initialized:
            await self.initialize()

        async for chunk in self._chain.chat_stream(
            query=query,
            conversation_history=conversation_history,
        ):
            yield chunk

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_source: Optional[str] = None,
    ) -> RetrievalResult:
        """
        문서 검색만 수행 (LLM 응답 없이)

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            filter_source: 특정 소스만 검색

        Returns:
            검색 결과
        """
        if not self._initialized:
            await self.initialize()

        return await self._retriever.retrieve(
            query=query,
            config=RetrievalConfig(top_k=top_k),
            filter_source=filter_source,
        )

    async def delete_document(self, source: str) -> None:
        """
        특정 문서의 모든 청크 삭제

        Args:
            source: 삭제할 문서명
        """
        if not self._initialized:
            await self.initialize()

        await self._vector_store.delete_by_source(source)
        print(f"✓ 문서 삭제 완료: {source}")

    async def close(self) -> None:
        """리소스 정리"""
        if self._vector_store:
            await self._vector_store.close()
        self._initialized = False


async def create_rag_system(
    settings: Optional[Settings] = None,
) -> MultimodalHierarchicalRAG:
    """
    RAG 시스템 팩토리 함수

    Args:
        settings: 설정 객체

    Returns:
        초기화된 RAG 시스템
    """
    rag = MultimodalHierarchicalRAG(settings=settings)
    await rag.initialize()
    return rag
