"""
Qdrant 벡터 스토어 모듈
Hybrid Search (Vector + Sparse/BM25) 지원
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import asyncio
import uuid

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    NamedVector,
    NamedSparseVector,
    SparseVector,
    Prefetch,
    Query,
    FusionQuery,
    Fusion,
)

from rag.chunkers import ParentChunk, ChildChunk


@dataclass
class HybridSearchConfig:
    """하이브리드 검색 설정"""
    alpha: float = 0.7  # Vector search weight (1-alpha = sparse weight)
    top_k: int = 10
    use_reranking: bool = True


@dataclass
class SearchResult:
    """검색 결과"""
    chunk_id: str
    content: str
    score: float
    source: str
    page: int
    bbox: Optional[dict] = None
    heading: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class QdrantVectorStore:
    """Qdrant 벡터 스토어"""

    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        collection_name: str = "multimodal_hierarchical_rag",
        embedding_dim: int = 3072,  # Baseline V1: text-embedding-3-large
    ):
        """
        Args:
            host: Qdrant 서버 호스트
            port: Qdrant 서버 포트
            api_key: API 키 (클라우드용)
            collection_name: 컬렉션 이름
            embedding_dim: 임베딩 차원
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        self._sync_client: Optional[QdrantClient] = None
        self._async_client: Optional[AsyncQdrantClient] = None

    @property
    def sync_client(self) -> QdrantClient:
        """동기 클라이언트"""
        if self._sync_client is None:
            if self.api_key:
                self._sync_client = QdrantClient(
                    url=f"https://{self.host}",
                    api_key=self.api_key,
                )
            else:
                self._sync_client = QdrantClient(
                    host=self.host,
                    port=self.port,
                )
        return self._sync_client

    @property
    def async_client(self) -> AsyncQdrantClient:
        """비동기 클라이언트"""
        if self._async_client is None:
            if self.api_key:
                self._async_client = AsyncQdrantClient(
                    url=f"https://{self.host}",
                    api_key=self.api_key,
                )
            else:
                self._async_client = AsyncQdrantClient(
                    host=self.host,
                    port=self.port,
                )
        return self._async_client

    async def initialize(self) -> None:
        """컬렉션 초기화"""
        collections = await self.async_client.get_collections()
        existing_names = [c.name for c in collections.collections]

        if self.collection_name not in existing_names:
            await self._create_collection()

    async def _create_collection(self) -> None:
        """컬렉션 생성"""
        await self.async_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.DENSE_VECTOR_NAME: VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )

        await self.async_client.create_payload_index(
            collection_name=self.collection_name,
            field_name="chunk_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await self.async_client.create_payload_index(
            collection_name=self.collection_name,
            field_name="source",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await self.async_client.create_payload_index(
            collection_name=self.collection_name,
            field_name="parent_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    async def add_chunks(
        self,
        parent_chunks: list[ParentChunk],
        child_chunks: list[ChildChunk],
        dense_embeddings: dict[str, list[float]],
        sparse_embeddings: Optional[dict[str, tuple[list[int], list[float]]]] = None,
    ) -> None:
        """청크 추가"""
        points = []

        for chunk in parent_chunks:
            if chunk.chunk_id not in dense_embeddings:
                continue

            vectors = {
                self.DENSE_VECTOR_NAME: dense_embeddings[chunk.chunk_id],
            }

            sparse_vector = None
            if sparse_embeddings and chunk.chunk_id in sparse_embeddings:
                indices, values = sparse_embeddings[chunk.chunk_id]
                sparse_vector = {
                    self.SPARSE_VECTOR_NAME: SparseVector(
                        indices=indices,
                        values=values,
                    )
                }

            point = PointStruct(
                id=chunk.chunk_id,
                vector=vectors if not sparse_vector else {**vectors, **sparse_vector},
                payload={
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "chunk_type": chunk.chunk_type.value,
                    "source": chunk.source,
                    "page": chunk.page,
                    "heading": chunk.heading,
                    "child_ids": chunk.child_ids,
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                    **chunk.metadata,
                },
            )
            points.append(point)

        for chunk in child_chunks:
            if chunk.chunk_id not in dense_embeddings:
                continue

            vectors = {
                self.DENSE_VECTOR_NAME: dense_embeddings[chunk.chunk_id],
            }

            sparse_vector = None
            if sparse_embeddings and chunk.chunk_id in sparse_embeddings:
                indices, values = sparse_embeddings[chunk.chunk_id]
                sparse_vector = {
                    self.SPARSE_VECTOR_NAME: SparseVector(
                        indices=indices,
                        values=values,
                    )
                }

            point = PointStruct(
                id=chunk.chunk_id,
                vector=vectors if not sparse_vector else {**vectors, **sparse_vector},
                payload={
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "chunk_type": chunk.chunk_type.value,
                    "source": chunk.source,
                    "page": chunk.page,
                    "parent_id": chunk.parent_id,
                    "bbox": chunk.bbox,
                    "element_type": chunk.element_type,
                    "heading": chunk.heading,
                    **chunk.metadata,
                },
            )
            points.append(point)

        if points:
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await self.async_client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )

    async def search(
        self,
        query_vector: list[float],
        config: Optional[HybridSearchConfig] = None,
        filter_source: Optional[str] = None,
        only_children: bool = True,
    ) -> list[SearchResult]:
        """벡터 검색"""
        config = config or HybridSearchConfig()

        filter_conditions = []
        if only_children:
            filter_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value="child"),
                )
            )
        if filter_source:
            filter_conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=filter_source),
                )
            )

        query_filter = None
        if filter_conditions:
            query_filter = Filter(must=filter_conditions)

        # 최신 Qdrant 클라이언트 API 사용 (query_points)
        results = await self.async_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=self.DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=config.top_k,
            with_payload=True,
        )

        return self._convert_query_results(results.points)

    async def sparse_search(
        self,
        query_sparse_vector: tuple[list[int], list[float]],
        config: Optional[HybridSearchConfig] = None,
        filter_source: Optional[str] = None,
        only_children: bool = True,
    ) -> list[SearchResult]:
        """Sparse(BM25) 검색만 수행"""
        config = config or HybridSearchConfig()

        filter_conditions = []
        if only_children:
            filter_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value="child"),
                )
            )
        if filter_source:
            filter_conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=filter_source),
                )
            )

        query_filter = None
        if filter_conditions:
            query_filter = Filter(must=filter_conditions)

        indices, values = query_sparse_vector

        results = await self.async_client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(indices=indices, values=values),
            using=self.SPARSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=config.top_k,
            with_payload=True,
        )

        return self._convert_query_results(results.points)

    async def hybrid_search(
        self,
        query_dense_vector: list[float],
        query_sparse_vector: tuple[list[int], list[float]],
        config: Optional[HybridSearchConfig] = None,
        filter_source: Optional[str] = None,
        only_children: bool = True,
    ) -> list[SearchResult]:
        """하이브리드 검색 (Vector + Sparse) - Qdrant 내장 RRF 사용"""
        config = config or HybridSearchConfig()

        filter_conditions = []
        if only_children:
            filter_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value="child"),
                )
            )
        if filter_source:
            filter_conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=filter_source),
                )
            )

        query_filter = None
        if filter_conditions:
            query_filter = Filter(must=filter_conditions)

        indices, values = query_sparse_vector

        results = await self.async_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=query_dense_vector,
                    using=self.DENSE_VECTOR_NAME,
                    limit=config.top_k * 2,
                ),
                Prefetch(
                    query=SparseVector(indices=indices, values=values),
                    using=self.SPARSE_VECTOR_NAME,
                    limit=config.top_k * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=config.top_k,
            with_payload=True,
        )

        return self._convert_query_results(results.points)

    async def get_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """ID로 청크 조회"""
        try:
            results = await self.async_client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_payload=True,
            )
            if results:
                point = results[0]
                payload = point.payload
                return SearchResult(
                    chunk_id=payload.get("chunk_id", str(point.id)),
                    content=payload.get("content", ""),
                    score=1.0,
                    source=payload.get("source", ""),
                    page=payload.get("page", 1),
                    bbox=payload.get("bbox"),
                    heading=payload.get("heading"),
                    parent_id=payload.get("parent_id"),
                    metadata={
                        k: v for k, v in payload.items()
                        if k not in ["chunk_id", "content", "source", "page", "bbox", "heading", "parent_id"]
                    },
                )
        except Exception:
            pass
        return None

    async def get_parent(self, child_chunk_id: str) -> Optional[SearchResult]:
        """자식 청크의 부모 조회"""
        child = await self.get_by_id(child_chunk_id)
        if child and child.parent_id:
            return await self.get_by_id(child.parent_id)
        return None

    async def delete_by_source(self, source: str) -> None:
        """특정 소스의 모든 청크 삭제"""
        await self.async_client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source),
                        )
                    ]
                )
            ),
        )

    async def close(self) -> None:
        """클라이언트 종료"""
        if self._async_client:
            await self._async_client.close()
        if self._sync_client:
            self._sync_client.close()

    def _convert_results(self, results: list) -> list[SearchResult]:
        """검색 결과 변환"""
        search_results = []
        for result in results:
            payload = result.payload
            search_results.append(SearchResult(
                chunk_id=payload.get("chunk_id", str(result.id)),
                content=payload.get("content", ""),
                score=result.score,
                source=payload.get("source", ""),
                page=payload.get("page", 1),
                bbox=payload.get("bbox"),
                heading=payload.get("heading"),
                parent_id=payload.get("parent_id"),
                metadata={
                    k: v for k, v in payload.items()
                    if k not in ["chunk_id", "content", "source", "page", "bbox", "heading", "parent_id"]
                },
            ))
        return search_results

    def _convert_query_results(self, points: list) -> list[SearchResult]:
        """쿼리 결과 변환"""
        search_results = []
        for point in points:
            payload = point.payload
            search_results.append(SearchResult(
                chunk_id=payload.get("chunk_id", str(point.id)),
                content=payload.get("content", ""),
                score=point.score if point.score else 0.0,
                source=payload.get("source", ""),
                page=payload.get("page", 1),
                bbox=payload.get("bbox"),
                heading=payload.get("heading"),
                parent_id=payload.get("parent_id"),
                metadata={
                    k: v for k, v in payload.items()
                    if k not in ["chunk_id", "content", "source", "page", "bbox", "heading", "parent_id"]
                },
            ))
        return search_results
