"""
Qdrant VectorDB Provider
"""

import logging
from typing import List, Dict, Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

from providers.vectordb.base import BaseVectorDBProvider, VectorSearchResult

logger = logging.getLogger(__name__)


class QdrantProvider(BaseVectorDBProvider):
    """
    Qdrant VectorDB Provider

    사용 예시:
        provider = QdrantProvider(host="localhost", port=6333)
        await provider.create_collection("my_collection", dimension=3072)
        await provider.upsert("my_collection", [{"id": "1", "vector": [...], "payload": {...}}])
        results = await provider.search("my_collection", query_vector, top_k=5)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.client = AsyncQdrantClient(host=host, port=port)

    @property
    def provider_name(self) -> str:
        return "qdrant"

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance: str = "cosine",
        **kwargs
    ) -> bool:
        """컬렉션 생성"""
        try:
            distance_map = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "dot": Distance.DOT,
            }

            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=distance_map.get(distance, Distance.COSINE)
                )
            )
            logger.info(f"Qdrant 컬렉션 생성: {collection_name} (dim={dimension})")
            return True

        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug(f"컬렉션 이미 존재: {collection_name}")
                return True
            logger.error(f"컬렉션 생성 오류: {e}")
            raise

    async def delete_collection(self, collection_name: str) -> bool:
        """컬렉션 삭제"""
        try:
            await self.client.delete_collection(collection_name)
            logger.info(f"Qdrant 컬렉션 삭제: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"컬렉션 삭제 오류: {e}")
            return False

    async def collection_exists(self, collection_name: str) -> bool:
        """컬렉션 존재 여부 확인"""
        try:
            collections = await self.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception as e:
            logger.error(f"컬렉션 확인 오류: {e}")
            return False

    async def upsert(
        self,
        collection_name: str,
        vectors: List[Dict[str, Any]]
    ) -> int:
        """벡터 삽입/업데이트"""
        try:
            points = [
                PointStruct(
                    id=v["id"],
                    vector=v["vector"],
                    payload=v.get("payload", {})
                )
                for v in vectors
            ]

            await self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            logger.debug(f"Qdrant upsert: {len(points)} 벡터 -> {collection_name}")
            return len(points)

        except Exception as e:
            logger.error(f"Qdrant upsert 오류: {e}")
            raise

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filter_conditions: Optional[Dict] = None,
        score_threshold: float = 0.0,
        **kwargs
    ) -> List[VectorSearchResult]:
        """벡터 유사도 검색"""
        try:
            # 필터 변환
            qdrant_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    if isinstance(value, list):
                        # 리스트는 MatchAny 사용
                        from qdrant_client.models import MatchAny
                        conditions.append(
                            FieldCondition(key=key, match=MatchAny(any=value))
                        )
                    else:
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                qdrant_filter = Filter(must=conditions)

            # qdrant-client 1.x query_points API 사용
            search_result = await self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=score_threshold if score_threshold > 0 else None,
                with_payload=True
            )

            return [
                VectorSearchResult(
                    id=str(point.id),
                    score=point.score,
                    payload=point.payload or {},
                    vector=None
                )
                for point in search_result.points
            ]

        except Exception as e:
            logger.error(f"Qdrant 검색 오류: {e}")
            raise

    async def delete(
        self,
        collection_name: str,
        ids: List[str]
    ) -> int:
        """벡터 삭제"""
        try:
            await self.client.delete(
                collection_name=collection_name,
                points_selector=ids
            )
            logger.debug(f"Qdrant 삭제: {len(ids)} 벡터")
            return len(ids)

        except Exception as e:
            logger.error(f"Qdrant 삭제 오류: {e}")
            raise

    async def count(self, collection_name: str) -> int:
        """컬렉션의 벡터 수 조회"""
        try:
            info = await self.client.get_collection(collection_name)
            return info.points_count
        except Exception as e:
            logger.error(f"Qdrant count 오류: {e}")
            return 0
