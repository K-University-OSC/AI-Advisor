# -*- coding: utf-8 -*-
"""
V9 Dual Collection Qdrant Store
Context Type별 임베딩 분리:
- Paragraph → text-embedding-3-large (3072차원)
- Table/Image → text-embedding-3-small (1536차원)
"""

from dataclasses import dataclass, field
from typing import Optional
import asyncio

from qdrant_client import AsyncQdrantClient
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
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
)

from rag.chunkers import ParentChunk, ChildChunk
from .qdrant_store import SearchResult, HybridSearchConfig


@dataclass
class DualCollectionConfig:
    """듀얼 컬렉션 설정"""
    base_name: str = "mh_rag_finance_v9"
    paragraph_embedding_dim: int = 3072  # text-embedding-3-large
    table_image_embedding_dim: int = 1536  # text-embedding-3-small

    @property
    def paragraph_collection(self) -> str:
        return f"{self.base_name}_paragraph"

    @property
    def table_image_collection(self) -> str:
        return f"{self.base_name}_table_image"


class DualCollectionQdrantStore:
    """
    V9 Dual Collection Qdrant Store

    Context Type별 최적화된 임베딩 사용:
    - Paragraph: text-embedding-3-large (3072차원)
    - Table/Image: text-embedding-3-small (1536차원)
    """

    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        config: Optional[DualCollectionConfig] = None,
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.config = config or DualCollectionConfig()

        self._async_client: Optional[AsyncQdrantClient] = None

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
        """두 컬렉션 초기화"""
        collections = await self.async_client.get_collections()
        existing_names = [c.name for c in collections.collections]

        # Paragraph 컬렉션 (3072차원)
        if self.config.paragraph_collection not in existing_names:
            await self._create_collection(
                self.config.paragraph_collection,
                self.config.paragraph_embedding_dim,
            )
            print(f"  - Paragraph 컬렉션 생성: {self.config.paragraph_collection} (3072차원)")

        # Table/Image 컬렉션 (1536차원)
        if self.config.table_image_collection not in existing_names:
            await self._create_collection(
                self.config.table_image_collection,
                self.config.table_image_embedding_dim,
            )
            print(f"  - Table/Image 컬렉션 생성: {self.config.table_image_collection} (1536차원)")

    async def _create_collection(self, name: str, embedding_dim: int) -> None:
        """컬렉션 생성"""
        await self.async_client.create_collection(
            collection_name=name,
            vectors_config={
                self.DENSE_VECTOR_NAME: VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )

        # 인덱스 생성
        for field_name in ["chunk_type", "source", "parent_id", "element_type"]:
            await self.async_client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def _get_collection_for_element_type(self, element_type: str) -> str:
        """element_type에 따라 적절한 컬렉션 반환"""
        if element_type in ("paragraph", "text", "title", "heading"):
            return self.config.paragraph_collection
        else:  # table, image, chart, figure
            return self.config.table_image_collection

    async def add_chunks(
        self,
        parent_chunks: list[ParentChunk],
        child_chunks: list[ChildChunk],
        paragraph_dense_embeddings: dict[str, list[float]],  # 3072차원
        table_image_dense_embeddings: dict[str, list[float]],  # 1536차원
        sparse_embeddings: Optional[dict[str, tuple[list[int], list[float]]]] = None,
    ) -> dict:
        """
        청크 추가 - Context Type별로 다른 컬렉션에 저장

        Returns:
            저장 통계
        """
        paragraph_points = []
        table_image_points = []
        sparse_embeddings = sparse_embeddings or {}

        # Parent chunks 분류
        for chunk in parent_chunks:
            # Parent는 주로 paragraph로 취급
            if chunk.chunk_id in paragraph_dense_embeddings:
                point = self._create_point(
                    chunk,
                    paragraph_dense_embeddings[chunk.chunk_id],
                    sparse_embeddings.get(chunk.chunk_id),
                    is_parent=True,
                )
                paragraph_points.append(point)

        # Child chunks 분류
        for chunk in child_chunks:
            element_type = chunk.element_type or "paragraph"

            if element_type in ("paragraph", "text", "title", "heading"):
                if chunk.chunk_id in paragraph_dense_embeddings:
                    point = self._create_point(
                        chunk,
                        paragraph_dense_embeddings[chunk.chunk_id],
                        sparse_embeddings.get(chunk.chunk_id),
                        is_parent=False,
                    )
                    paragraph_points.append(point)
            else:  # table, image, chart, figure
                if chunk.chunk_id in table_image_dense_embeddings:
                    point = self._create_point(
                        chunk,
                        table_image_dense_embeddings[chunk.chunk_id],
                        sparse_embeddings.get(chunk.chunk_id),
                        is_parent=False,
                    )
                    table_image_points.append(point)

        # 배치 저장
        stats = {"paragraph": 0, "table_image": 0}

        if paragraph_points:
            await self._batch_upsert(self.config.paragraph_collection, paragraph_points)
            stats["paragraph"] = len(paragraph_points)

        if table_image_points:
            await self._batch_upsert(self.config.table_image_collection, table_image_points)
            stats["table_image"] = len(table_image_points)

        return stats

    def _create_point(
        self,
        chunk,
        dense_embedding: list[float],
        sparse_embedding: Optional[tuple[list[int], list[float]]],
        is_parent: bool,
    ) -> PointStruct:
        """포인트 생성"""
        vectors = {self.DENSE_VECTOR_NAME: dense_embedding}

        if sparse_embedding:
            indices, values = sparse_embedding
            vectors[self.SPARSE_VECTOR_NAME] = SparseVector(
                indices=indices,
                values=values,
            )

        if is_parent:
            payload = {
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
            }
        else:
            payload = {
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
            }

        return PointStruct(
            id=chunk.chunk_id,
            vector=vectors,
            payload=payload,
        )

    async def _batch_upsert(self, collection_name: str, points: list[PointStruct]) -> None:
        """배치 업서트"""
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await self.async_client.upsert(
                collection_name=collection_name,
                points=batch,
            )

    async def search_paragraph(
        self,
        query_vector: list[float],  # 3072차원
        config: Optional[HybridSearchConfig] = None,
        filter_source: Optional[str] = None,
        only_children: bool = True,
    ) -> list[SearchResult]:
        """Paragraph 컬렉션 검색"""
        return await self._search_collection(
            self.config.paragraph_collection,
            query_vector,
            config,
            filter_source,
            only_children,
        )

    async def search_table_image(
        self,
        query_vector: list[float],  # 1536차원
        config: Optional[HybridSearchConfig] = None,
        filter_source: Optional[str] = None,
        only_children: bool = True,
    ) -> list[SearchResult]:
        """Table/Image 컬렉션 검색"""
        return await self._search_collection(
            self.config.table_image_collection,
            query_vector,
            config,
            filter_source,
            only_children,
        )

    async def _search_collection(
        self,
        collection_name: str,
        query_vector: list[float],
        config: Optional[HybridSearchConfig] = None,
        filter_source: Optional[str] = None,
        only_children: bool = True,
    ) -> list[SearchResult]:
        """단일 컬렉션 검색"""
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

        results = await self.async_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=self.DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=config.top_k,
            with_payload=True,
        )

        return self._convert_query_results(results.points)

    async def hybrid_search_paragraph(
        self,
        query_dense_vector: list[float],  # 3072차원
        query_sparse_vector: tuple[list[int], list[float]],
        config: Optional[HybridSearchConfig] = None,
        filter_source: Optional[str] = None,
        only_children: bool = True,
    ) -> list[SearchResult]:
        """Paragraph 하이브리드 검색"""
        return await self._hybrid_search_collection(
            self.config.paragraph_collection,
            query_dense_vector,
            query_sparse_vector,
            config,
            filter_source,
            only_children,
        )

    async def hybrid_search_table_image(
        self,
        query_dense_vector: list[float],  # 1536차원
        query_sparse_vector: tuple[list[int], list[float]],
        config: Optional[HybridSearchConfig] = None,
        filter_source: Optional[str] = None,
        only_children: bool = True,
    ) -> list[SearchResult]:
        """Table/Image 하이브리드 검색"""
        return await self._hybrid_search_collection(
            self.config.table_image_collection,
            query_dense_vector,
            query_sparse_vector,
            config,
            filter_source,
            only_children,
        )

    async def _hybrid_search_collection(
        self,
        collection_name: str,
        query_dense_vector: list[float],
        query_sparse_vector: tuple[list[int], list[float]],
        config: Optional[HybridSearchConfig] = None,
        filter_source: Optional[str] = None,
        only_children: bool = True,
    ) -> list[SearchResult]:
        """단일 컬렉션 하이브리드 검색"""
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
            collection_name=collection_name,
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

    async def search_all(
        self,
        paragraph_query_vector: list[float],  # 3072차원
        table_image_query_vector: list[float],  # 1536차원
        config: Optional[HybridSearchConfig] = None,
        filter_source: Optional[str] = None,
        only_children: bool = True,
    ) -> list[SearchResult]:
        """
        두 컬렉션에서 동시 검색 후 RRF로 병합
        """
        config = config or HybridSearchConfig()

        # 병렬 검색
        paragraph_results, table_image_results = await asyncio.gather(
            self.search_paragraph(
                paragraph_query_vector,
                HybridSearchConfig(top_k=config.top_k),
                filter_source,
                only_children,
            ),
            self.search_table_image(
                table_image_query_vector,
                HybridSearchConfig(top_k=config.top_k),
                filter_source,
                only_children,
            ),
        )

        # RRF 병합
        return self._rrf_merge(
            [paragraph_results, table_image_results],
            k=60,
            top_k=config.top_k,
        )

    def _rrf_merge(
        self,
        result_lists: list[list[SearchResult]],
        k: int = 60,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """RRF (Reciprocal Rank Fusion) 병합"""
        scores = {}
        result_map = {}

        for results in result_lists:
            for rank, result in enumerate(results, 1):
                chunk_id = result.chunk_id
                rrf_score = 1.0 / (k + rank)

                if chunk_id in scores:
                    scores[chunk_id] += rrf_score
                else:
                    scores[chunk_id] = rrf_score
                    result_map[chunk_id] = result

        # 점수 순으로 정렬
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        merged_results = []
        for chunk_id in sorted_ids[:top_k]:
            result = result_map[chunk_id]
            # RRF 점수로 업데이트
            merged_results.append(SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=scores[chunk_id],
                source=result.source,
                page=result.page,
                bbox=result.bbox,
                heading=result.heading,
                parent_id=result.parent_id,
                metadata=result.metadata,
            ))

        return merged_results

    async def get_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """ID로 청크 조회 (두 컬렉션에서 검색)"""
        for collection_name in [
            self.config.paragraph_collection,
            self.config.table_image_collection,
        ]:
            try:
                results = await self.async_client.retrieve(
                    collection_name=collection_name,
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
                continue
        return None

    async def get_parent(self, child_chunk_id: str) -> Optional[SearchResult]:
        """자식 청크의 부모 조회"""
        child = await self.get_by_id(child_chunk_id)
        if child and child.parent_id:
            return await self.get_by_id(child.parent_id)
        return None

    async def delete_by_source(self, source: str) -> None:
        """특정 소스의 모든 청크 삭제 (두 컬렉션에서)"""
        for collection_name in [
            self.config.paragraph_collection,
            self.config.table_image_collection,
        ]:
            await self.async_client.delete(
                collection_name=collection_name,
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
