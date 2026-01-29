"""
계층적 검색기 (Hierarchical Retriever)
자식 청크로 검색하고 부모 청크로 문맥을 확장
"""

from dataclasses import dataclass, field
from typing import Optional
import asyncio
import os

from rag.vectorstore import QdrantVectorStore, SearchResult, HybridSearchConfig
from rag.embeddings import MultimodalEmbeddingService
from rag.retriever.reranker import BGEReranker
from rag.retriever.query_enhancer import MultiQueryGenerator


@dataclass
class RetrievalConfig:
    """검색 설정 (V7.6.1 기반)"""
    top_k: int = 10  # V7.6.1: 8 → 10 (테이블 검색 개선을 위해 더 많은 컨텍스트 제공)
    use_hybrid: bool = False  # BM25 비활성화 (한국어 RAG에서 성능 저하 방지)
    expand_to_parent: bool = True
    include_siblings: bool = False
    rerank: bool = True  # BGE Reranker 활성화 (+24.6%p 성능 향상)
    rerank_top_k: int = 30  # V7.6.1: 25 → 30 (테이블 청크가 더 많이 포함되도록 후보군 확대)
    use_multi_query: bool = False  # V7.6.1: Multi-Query 비활성화 (단순화)
    num_queries: int = 4  # Multi-Query 사용 시 쿼리 수


@dataclass
class RetrievalResult:
    """검색 결과"""
    query: str
    child_results: list[SearchResult]
    parent_contents: dict[str, str]  # V7.6.1: 단순 dict (tuple → str)
    context: str
    sources: list[dict]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "child_count": len(self.child_results),
            "parent_count": len(self.parent_contents),
            "context_length": len(self.context),
            "sources": self.sources,
        }


class HierarchicalRetriever:
    """계층적 검색기"""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_service: MultimodalEmbeddingService,
        config: Optional[RetrievalConfig] = None,
    ):
        """
        Args:
            vector_store: Qdrant 벡터 스토어
            embedding_service: 임베딩 서비스
            config: 검색 설정
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.config = config or RetrievalConfig()

        # BGE Reranker 초기화
        self.reranker = BGEReranker()
        self._reranker_initialized = False

        # Multi-Query Generator 초기화
        api_key = os.getenv("OPENAI_API_KEY", "")
        self.multi_query_generator = MultiQueryGenerator(api_key=api_key, model="gpt-4o-mini") if api_key else None

    def _ensure_reranker(self) -> bool:
        """Reranker 초기화 확인"""
        if not self._reranker_initialized:
            self._reranker_initialized = self.reranker.initialize()
        return self._reranker_initialized

    async def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        filter_source: Optional[str] = None,
    ) -> RetrievalResult:
        """
        쿼리로 관련 문서 검색

        Args:
            query: 검색 쿼리
            config: 검색 설정 (없으면 기본값 사용)
            filter_source: 특정 소스만 검색

        Returns:
            검색 결과
        """
        config = config or self.config

        # Multi-Query Retrieval
        queries = [query]
        if config.use_multi_query and self.multi_query_generator:
            try:
                queries = await self.multi_query_generator.generate(query, config.num_queries)
            except Exception as e:
                print(f"Multi-query 생성 실패, 원본 쿼리 사용: {e}")

        # 각 쿼리로 검색 후 결과 병합
        all_results = []
        seen_chunk_ids = set()

        for q in queries:
            dense_embedding, sparse_embedding = await self.embedding_service.embed_query(q)

            # 리랭킹이 활성화된 경우 더 많은 문서를 먼저 검색
            search_top_k = config.rerank_top_k if config.rerank else config.top_k

            if config.use_hybrid:
                child_results = await self.vector_store.hybrid_search(
                    query_dense_vector=dense_embedding,
                    query_sparse_vector=sparse_embedding,
                    config=HybridSearchConfig(top_k=search_top_k),
                    filter_source=filter_source,
                    only_children=True,
                )
            else:
                child_results = await self.vector_store.search(
                    query_vector=dense_embedding,
                    config=HybridSearchConfig(top_k=search_top_k),
                    filter_source=filter_source,
                    only_children=True,
                )

            # 중복 제거하며 결과 병합
            for result in child_results:
                if result.chunk_id not in seen_chunk_ids:
                    all_results.append(result)
                    seen_chunk_ids.add(result.chunk_id)

        # BGE Reranker로 병합된 결과를 리랭킹 (원본 쿼리 기준)
        if config.rerank and all_results and self._ensure_reranker():
            child_results = self._rerank_results(query, all_results, config.top_k)
        else:
            # 리랭킹 없으면 점수순 정렬 후 top_k 선택
            child_results = sorted(all_results, key=lambda x: x.score, reverse=True)[:config.top_k]

        parent_contents = {}
        if config.expand_to_parent:
            parent_contents = await self._fetch_parent_contents(child_results)

        context = self._build_context(child_results, parent_contents, config)

        sources = self._extract_sources(child_results)

        return RetrievalResult(
            query=query,
            child_results=child_results,
            parent_contents=parent_contents,
            context=context,
            sources=sources,
        )

    def _rerank_results(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """BGE Reranker로 검색 결과 리랭킹"""
        if not results:
            return results

        # 문서 컨텐츠 추출
        documents = [r.content for r in results]

        # 리랭킹 수행
        reranked = self.reranker.rerank(query, documents, top_k=top_k)

        # 리랭킹된 순서대로 결과 재정렬
        content_to_result = {r.content: r for r in results}
        reranked_results = []

        for doc, score in reranked:
            if doc in content_to_result:
                result = content_to_result[doc]
                # 리랭킹 점수로 업데이트
                result.score = float(score)
                reranked_results.append(result)

        return reranked_results

    async def retrieve_with_context(
        self,
        query: str,
        conversation_history: Optional[list[dict]] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> RetrievalResult:
        """
        대화 이력을 고려한 검색

        Args:
            query: 검색 쿼리
            conversation_history: 이전 대화 기록
            config: 검색 설정

        Returns:
            검색 결과
        """
        enhanced_query = query
        if conversation_history:
            recent_context = self._extract_recent_context(conversation_history)
            if recent_context:
                enhanced_query = f"{recent_context}\n\n현재 질문: {query}"

        return await self.retrieve(enhanced_query, config)

    async def _fetch_parent_contents(
        self, child_results: list[SearchResult]
    ) -> dict[str, str]:
        """자식 청크들의 부모 컨텐츠 조회 (V7.6.1 원본)

        Returns:
            dict[parent_id, content]
        """
        parent_ids = set()
        for result in child_results:
            if result.parent_id:
                parent_ids.add(result.parent_id)

        parent_contents = {}

        async def fetch_parent(parent_id: str):
            parent = await self.vector_store.get_by_id(parent_id)
            if parent:
                return parent_id, parent.content
            return parent_id, None

        tasks = [fetch_parent(pid) for pid in parent_ids]
        results = await asyncio.gather(*tasks)

        for parent_id, content in results:
            if content:
                parent_contents[parent_id] = content

        return parent_contents

    def _build_context(
        self,
        child_results: list[SearchResult],
        parent_contents: dict[str, str],
        config: RetrievalConfig,
    ) -> str:
        """검색 결과로부터 LLM 컨텍스트 구성 (V7.6.1 원본)

        V7.6.1: Parent+Child 콘텐츠 병합 (정보 손실 방지)
        - Parent 확장 시 Parent 컨텐츠만 사용
        - Parent가 없으면 Child 컨텐츠 사용
        """
        context_parts = []
        used_parents = set()

        if config.expand_to_parent:
            for result in child_results:
                if result.parent_id and result.parent_id not in used_parents:
                    parent_content = parent_contents.get(result.parent_id)
                    if parent_content:
                        section_header = f"[출처: {result.source}, 섹션: {result.heading or '알 수 없음'}]"
                        context_parts.append(f"{section_header}\n{parent_content}")
                        used_parents.add(result.parent_id)
        else:
            for result in child_results:
                section_header = f"[출처: {result.source}, 페이지: {result.page}]"
                context_parts.append(f"{section_header}\n{result.content}")

        return "\n\n---\n\n".join(context_parts)

    def _extract_sources(self, child_results: list[SearchResult]) -> list[dict]:
        """검색 결과에서 출처 정보 추출"""
        sources = []
        seen = set()

        for result in child_results:
            source_key = f"{result.source}_{result.page}"
            if source_key not in seen:
                source_info = {
                    "chunk_id": result.chunk_id,
                    "source": result.source,
                    "page": result.page,
                    "score": result.score,
                    "heading": result.heading,
                    "bbox": result.bbox,
                    "content_preview": result.content[:200] if result.content else "",
                }
                sources.append(source_info)
                seen.add(source_key)

        return sources

    def _extract_recent_context(
        self, conversation_history: list[dict], max_turns: int = 3
    ) -> str:
        """대화 기록에서 최근 컨텍스트 추출"""
        recent = conversation_history[-max_turns * 2:]
        context_parts = []

        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                context_parts.append(f"사용자: {content[:200]}")
            elif role == "assistant":
                context_parts.append(f"어시스턴트: {content[:200]}")

        return "\n".join(context_parts)


class LangChainRetrieverWrapper:
    """LangChain 호환 리트리버 래퍼"""

    def __init__(self, retriever: HierarchicalRetriever):
        self.retriever = retriever

    async def aget_relevant_documents(self, query: str) -> list[dict]:
        """LangChain 비동기 검색 인터페이스"""
        result = await self.retriever.retrieve(query)

        documents = []
        for child in result.child_results:
            doc = {
                "page_content": child.content,
                "metadata": {
                    "source": child.source,
                    "page": child.page,
                    "chunk_id": child.chunk_id,
                    "bbox": child.bbox,
                    "heading": child.heading,
                    "score": child.score,
                },
            }
            documents.append(doc)

        return documents

    def get_relevant_documents(self, query: str) -> list[dict]:
        """LangChain 동기 검색 인터페이스"""
        import asyncio
        return asyncio.run(self.aget_relevant_documents(query))
