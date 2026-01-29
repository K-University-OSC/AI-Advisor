"""
Reranker Benchmark 평가 스크립트
3개 리랭커 비교:
1. BAAI/bge-reranker-v2-m3 (로컬, 560MB, 다국어)
2. BAAI/bge-reranker-v2.5-gemma2-lightweight (로컬, 2.5GB, 고성능)
3. Cohere rerank-v3.5 (API, 유료)

평가 지표: Hit Rate@1, MRR, NDCG@5, Precision@5, Recall@5, Latency
"""

import asyncio
import json
import os
import sys
import time
import math
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import uuid
import numpy as np

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)

# Cohere for reranking
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    print("⚠️ Cohere not installed.")

# BGE Reranker
try:
    from FlagEmbedding import FlagReranker
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False
    print("⚠️ FlagEmbedding not installed.")


@dataclass
class TestQuery:
    """테스트 쿼리"""
    query_id: str
    user_id: str
    query_text: str
    relevant_doc_ids: List[str]


class RerankerBenchmark:
    """리랭커 벤치마크 평가"""

    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.collection_name = "reranker_benchmark"
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dim = 3072

        # Cohere 클라이언트
        self.cohere_client = None
        if COHERE_AVAILABLE and os.getenv("COHERE_API_KEY"):
            self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
            print("✅ Cohere client initialized")

        # BGE Rerankers (lazy loading)
        self.bge_m3 = None
        self.bge_gemma = None

        self.documents = []
        self.document_ids = []

    def load_bge_m3(self):
        """BGE Reranker v2-m3 로드"""
        if self.bge_m3 is None and BGE_AVAILABLE:
            print("🔄 Loading BGE-Reranker-v2-m3...")
            start = time.time()
            self.bge_m3 = FlagReranker(
                "BAAI/bge-reranker-v2-m3",
                use_fp16=True
            )
            print(f"   ✅ Loaded in {time.time() - start:.1f}s")
        return self.bge_m3

    def load_bge_gemma(self):
        """BGE Reranker v2.5-gemma2-lightweight 로드"""
        if self.bge_gemma is None and BGE_AVAILABLE:
            print("🔄 Loading BGE-Reranker-v2.5-gemma2-lightweight...")
            start = time.time()
            try:
                # gemma2 모델은 LLM 기반이므로 LayerWiseFlagLLMReranker 사용
                from FlagEmbedding import LayerWiseFlagLLMReranker
                self.bge_gemma = LayerWiseFlagLLMReranker(
                    "BAAI/bge-reranker-v2.5-gemma2-lightweight",
                    use_fp16=True,
                    trust_remote_code=True
                )
                print(f"   ✅ Loaded in {time.time() - start:.1f}s")
            except Exception as e:
                print(f"   ⚠️ Failed to load gemma2: {e}")
                self.bge_gemma = None
        return self.bge_gemma

    def generate_test_data(self, num_users: int = 10) -> Tuple[List[Dict], List[TestQuery]]:
        """테스트 데이터 생성"""
        documents = []
        queries = []

        # 다양한 도메인의 테스트 데이터
        test_scenarios = [
            {
                "theme": "개발자",
                "docs": [
                    "나는 소프트웨어 개발자로 일하고 있어요",
                    "Python과 JavaScript를 주로 사용해요",
                    "주말에는 오픈소스 프로젝트에 기여해요",
                    "커피를 마시면서 코딩하는 것을 좋아해요",
                    "최근에 AI와 머신러닝에 관심이 많아요"
                ],
                "queries": [
                    ("어떤 일을 하세요?", [0]),
                    ("프로그래밍 언어 뭐 써요?", [1]),
                    ("취미가 뭐예요?", [2, 3]),
                    ("요즘 뭐에 관심있어요?", [4]),
                    ("개발할 때 뭐 마셔요?", [3])
                ]
            },
            {
                "theme": "학생",
                "docs": [
                    "저는 대학에서 컴퓨터공학을 전공하고 있어요",
                    "올해 졸업 예정이에요",
                    "동아리에서 앱 개발을 하고 있어요",
                    "장학금을 받으며 공부하고 있어요",
                    "졸업 후에는 대기업에 취업하고 싶어요"
                ],
                "queries": [
                    ("전공이 뭐예요?", [0]),
                    ("몇 학년이에요?", [1]),
                    ("동아리 활동 하세요?", [2]),
                    ("학비는 어떻게 해결해요?", [3]),
                    ("졸업 후 계획이 있어요?", [4])
                ]
            },
            {
                "theme": "여행가",
                "docs": [
                    "여행을 정말 좋아해요",
                    "작년에 유럽 5개국을 다녀왔어요",
                    "다음 목표는 남미 여행이에요",
                    "사진 찍는 것을 좋아해서 여행 중 많이 찍어요",
                    "현지 음식 먹는 것이 여행의 묘미라고 생각해요"
                ],
                "queries": [
                    ("취미가 뭐예요?", [0]),
                    ("최근에 어디 다녀왔어요?", [1]),
                    ("다음에 어디 가고 싶어요?", [2]),
                    ("여행 중에 뭐 해요?", [3, 4]),
                    ("여행의 재미가 뭐예요?", [4])
                ]
            },
            {
                "theme": "요리사",
                "docs": [
                    "요리사로 일하고 있어요",
                    "이탈리안 요리를 전문으로 해요",
                    "나만의 레스토랑을 여는 것이 꿈이에요",
                    "신선한 재료에 집착하는 편이에요",
                    "주말에는 집에서 새로운 레시피를 실험해요"
                ],
                "queries": [
                    ("무슨 일 하세요?", [0]),
                    ("어떤 요리를 주로 해요?", [1]),
                    ("꿈이 뭐예요?", [2]),
                    ("요리할 때 중요하게 생각하는 것은?", [3]),
                    ("쉬는 날에는 뭐 해요?", [4])
                ]
            },
            {
                "theme": "음악가",
                "docs": [
                    "기타를 치는 것을 좋아해요",
                    "밴드에서 기타리스트로 활동하고 있어요",
                    "주로 락과 블루스를 연주해요",
                    "음악은 10살 때부터 시작했어요",
                    "언젠가 앨범을 내고 싶어요"
                ],
                "queries": [
                    ("악기 할 줄 알아요?", [0]),
                    ("밴드 활동 하세요?", [1]),
                    ("어떤 장르를 좋아해요?", [2]),
                    ("음악은 언제부터 했어요?", [3]),
                    ("음악 관련 목표가 있어요?", [4])
                ]
            },
            {
                "theme": "피트니스",
                "docs": [
                    "헬스장에서 웨이트 트레이닝을 해요",
                    "매일 아침 5시에 일어나서 운동해요",
                    "건강한 식단 관리도 함께 하고 있어요",
                    "마라톤 대회에 참가하는 것이 목표예요",
                    "운동 후 단백질 쉐이크를 꼭 마셔요"
                ],
                "queries": [
                    ("운동 하세요?", [0]),
                    ("언제 운동해요?", [1]),
                    ("식단도 관리해요?", [2]),
                    ("운동 목표가 있어요?", [3]),
                    ("운동 후에 뭐 먹어요?", [4])
                ]
            },
            {
                "theme": "게이머",
                "docs": [
                    "게임을 정말 좋아해요",
                    "FPS와 RPG 장르를 주로 해요",
                    "주말에는 친구들과 온라인으로 게임해요",
                    "e스포츠 경기 보는 것도 좋아해요",
                    "게임용 PC를 직접 조립했어요"
                ],
                "queries": [
                    ("취미가 뭐예요?", [0]),
                    ("어떤 게임 좋아해요?", [1]),
                    ("주말에 뭐 해요?", [2]),
                    ("e스포츠 관심 있어요?", [3]),
                    ("PC 사양이 어떻게 돼요?", [4])
                ]
            },
            {
                "theme": "예술가",
                "docs": [
                    "그림 그리는 것을 좋아해요",
                    "주로 수채화와 유화를 그려요",
                    "전시회에 작품을 출품한 적이 있어요",
                    "자연 풍경을 그리는 것을 좋아해요",
                    "미술관 가는 것을 즐겨요"
                ],
                "queries": [
                    ("취미가 있어요?", [0]),
                    ("어떤 그림 그려요?", [1]),
                    ("전시회 해본 적 있어요?", [2]),
                    ("주로 뭘 그려요?", [3]),
                    ("주말에 뭐 해요?", [4])
                ]
            },
            {
                "theme": "독서가",
                "docs": [
                    "독서를 정말 좋아해요",
                    "한 달에 책을 4-5권 읽어요",
                    "추리소설과 SF를 주로 읽어요",
                    "도서관에 자주 가요",
                    "독서 모임에 참여하고 있어요"
                ],
                "queries": [
                    ("취미가 뭐예요?", [0]),
                    ("책 많이 읽어요?", [1]),
                    ("어떤 장르 좋아해요?", [2]),
                    ("책은 어디서 빌려요?", [3]),
                    ("독서 모임 같은 거 해요?", [4])
                ]
            },
            {
                "theme": "반려동물",
                "docs": [
                    "고양이 두 마리를 키우고 있어요",
                    "고양이 이름은 나비와 콩이에요",
                    "매일 아침저녁으로 밥을 챙겨줘요",
                    "주말에는 함께 놀아줘요",
                    "고양이 용품에 돈을 많이 써요"
                ],
                "queries": [
                    ("반려동물 있어요?", [0]),
                    ("이름이 뭐예요?", [1]),
                    ("돌보는 게 힘들지 않아요?", [2]),
                    ("주말에 뭐 해요?", [3]),
                    ("비용이 많이 들어요?", [4])
                ]
            }
        ]

        for user_idx in range(min(num_users, len(test_scenarios))):
            scenario = test_scenarios[user_idx]
            user_id = f"user_{user_idx + 1}"

            # 문서 생성
            for doc_idx, doc_text in enumerate(scenario["docs"]):
                doc_id = f"{user_id}_doc_{doc_idx}"
                documents.append({
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "text": doc_text
                })

            # 쿼리 생성
            for q_idx, (query_text, relevant_indices) in enumerate(scenario["queries"]):
                queries.append(TestQuery(
                    query_id=f"{user_id}_query_{q_idx}",
                    user_id=user_id,
                    query_text=query_text,
                    relevant_doc_ids=[f"{user_id}_doc_{i}" for i in relevant_indices]
                ))

        return documents, queries

    def get_embedding(self, text: str) -> List[float]:
        """OpenAI 임베딩 생성"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def setup_collection(self):
        """Qdrant 컬렉션 설정"""
        collections = self.qdrant.get_collections().collections
        if any(c.name == self.collection_name for c in collections):
            self.qdrant.delete_collection(self.collection_name)

        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Collection '{self.collection_name}' 생성 완료")

    def index_documents(self, documents: List[Dict]):
        """문서 인덱싱"""
        points = []
        self.documents = []
        self.document_ids = []

        for doc in documents:
            embedding = self.get_embedding(doc["text"])

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "doc_id": doc["doc_id"],
                    "user_id": doc["user_id"],
                    "text": doc["text"]
                }
            )
            points.append(point)
            self.documents.append(doc["text"])
            self.document_ids.append(doc["doc_id"])

        # Qdrant에 업로드
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points[i:i+batch_size]
            )

        print(f"✅ {len(points)}개 문서 인덱싱 완료")

    def vector_search(self, query: str, user_id: str, top_k: int = 15) -> List[Dict]:
        """Vector Search (공통)"""
        query_embedding = self.get_embedding(query)

        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k
        )

        return [
            {
                "doc_id": r.payload["doc_id"],
                "text": r.payload["text"],
                "score": r.score
            }
            for r in search_result.points
        ]

    def rerank_with_cohere(self, query: str, candidates: List[Dict], top_k: int = 5) -> Tuple[List[Dict], float]:
        """Cohere Reranking"""
        if not self.cohere_client or not candidates:
            return candidates[:top_k], 0.0

        start_time = time.time()
        try:
            # Rate limiting
            time.sleep(6.5)

            rerank_response = self.cohere_client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=[c["text"] for c in candidates],
                top_n=top_k
            )

            reranked = []
            for r in rerank_response.results:
                candidate = candidates[r.index].copy()
                candidate["rerank_score"] = r.relevance_score
                reranked.append(candidate)

            latency = (time.time() - start_time) * 1000
            return reranked, latency
        except Exception as e:
            print(f"   Cohere error: {e}")
            return candidates[:top_k], 0.0

    def rerank_with_bge_m3(self, query: str, candidates: List[Dict], top_k: int = 5) -> Tuple[List[Dict], float]:
        """BGE Reranker v2-m3 Reranking"""
        reranker = self.load_bge_m3()
        if not reranker or not candidates:
            return candidates[:top_k], 0.0

        start_time = time.time()
        try:
            pairs = [[query, c["text"]] for c in candidates]
            scores = reranker.compute_score(pairs, normalize=True)

            if isinstance(scores, (int, float)):
                scores = [scores]

            scored = list(zip(range(len(candidates)), scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for idx, score in scored[:top_k]:
                candidate = candidates[idx].copy()
                candidate["rerank_score"] = float(score)
                reranked.append(candidate)

            latency = (time.time() - start_time) * 1000
            return reranked, latency
        except Exception as e:
            print(f"   BGE-m3 error: {e}")
            return candidates[:top_k], 0.0

    def rerank_with_bge_gemma(self, query: str, candidates: List[Dict], top_k: int = 5) -> Tuple[List[Dict], float]:
        """BGE Reranker v2.5-gemma2-lightweight Reranking"""
        reranker = self.load_bge_gemma()
        if not reranker or not candidates:
            return candidates[:top_k], 0.0

        start_time = time.time()
        try:
            pairs = [[query, c["text"]] for c in candidates]
            scores = reranker.compute_score(pairs, normalize=True)

            if isinstance(scores, (int, float)):
                scores = [scores]

            scored = list(zip(range(len(candidates)), scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for idx, score in scored[:top_k]:
                candidate = candidates[idx].copy()
                candidate["rerank_score"] = float(score)
                reranked.append(candidate)

            latency = (time.time() - start_time) * 1000
            return reranked, latency
        except Exception as e:
            print(f"   BGE-gemma error: {e}")
            return candidates[:top_k], 0.0

    def evaluate_reranker(
        self,
        reranker_name: str,
        rerank_func,
        queries: List[TestQuery],
        top_k: int = 5
    ) -> Dict[str, float]:
        """리랭커 평가"""
        results = {
            "hits": [],
            "mrr": [],
            "ndcg": [],
            "precision": [],
            "recall": [],
            "latency": []
        }

        total = len(queries)
        print(f"\n📊 {reranker_name} 평가 중...")

        for idx, query in enumerate(queries):
            print(f"\r   진행: {idx+1}/{total}", end="", flush=True)

            # Vector Search
            candidates = self.vector_search(query.query_text, query.user_id, top_k=15)

            # Reranking
            reranked, latency = rerank_func(query.query_text, candidates, top_k)

            # 메트릭 계산
            retrieved_ids = [r["doc_id"] for r in reranked]
            relevant_ids = set(query.relevant_doc_ids)

            metrics = self._calculate_metrics(retrieved_ids, relevant_ids)
            results["hits"].append(metrics["hit@1"])
            results["mrr"].append(metrics["mrr"])
            results["ndcg"].append(metrics["ndcg@5"])
            results["precision"].append(metrics["precision@5"])
            results["recall"].append(metrics["recall@5"])
            results["latency"].append(latency)

        print()

        # 평균 계산
        return {
            "hit_rate@1": np.mean(results["hits"]),
            "mrr": np.mean(results["mrr"]),
            "ndcg@5": np.mean(results["ndcg"]),
            "precision@5": np.mean(results["precision"]),
            "recall@5": np.mean(results["recall"]),
            "avg_latency_ms": np.mean(results["latency"]),
            "p95_latency_ms": np.percentile(results["latency"], 95) if results["latency"] else 0
        }

    def _calculate_metrics(self, retrieved: List[str], relevant: set) -> Dict[str, float]:
        """메트릭 계산"""
        # Hit@1
        hit_at_1 = 1.0 if retrieved and retrieved[0] in relevant else 0.0

        # MRR
        mrr = 0.0
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                mrr = 1.0 / (i + 1)
                break

        # NDCG@5
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:5]):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 2)

        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), 5)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        # Precision@5
        relevant_in_top5 = sum(1 for doc_id in retrieved[:5] if doc_id in relevant)
        precision = relevant_in_top5 / 5

        # Recall@5
        recall = relevant_in_top5 / len(relevant) if relevant else 0.0

        return {
            "hit@1": hit_at_1,
            "mrr": mrr,
            "ndcg@5": ndcg,
            "precision@5": precision,
            "recall@5": recall
        }

    def run_benchmark(self) -> Dict[str, Any]:
        """벤치마크 실행"""
        print("=" * 70)
        print("Reranker Benchmark")
        print("BGE-v2-m3 vs BGE-v2.5-gemma2 vs Cohere rerank-v3.5")
        print("=" * 70)
        print()

        # 1. 데이터 생성
        print("📊 테스트 데이터 생성 중...")
        documents, queries = self.generate_test_data(num_users=10)
        print(f"   - 문서 수: {len(documents)}")
        print(f"   - 쿼리 수: {len(queries)}")
        print()

        # 2. 인덱싱
        print("🔧 Qdrant 컬렉션 설정 및 인덱싱...")
        self.setup_collection()
        self.index_documents(documents)
        print()

        # 3. 각 리랭커 평가
        results = {}

        # Vector Only (baseline)
        print("📊 Vector Only (baseline) 평가 중...")
        vector_results = {"hits": [], "mrr": [], "ndcg": [], "precision": [], "recall": [], "latency": []}
        for idx, query in enumerate(queries):
            print(f"\r   진행: {idx+1}/{len(queries)}", end="", flush=True)
            start = time.time()
            candidates = self.vector_search(query.query_text, query.user_id, top_k=5)
            latency = (time.time() - start) * 1000

            retrieved_ids = [r["doc_id"] for r in candidates]
            relevant_ids = set(query.relevant_doc_ids)
            metrics = self._calculate_metrics(retrieved_ids, relevant_ids)

            vector_results["hits"].append(metrics["hit@1"])
            vector_results["mrr"].append(metrics["mrr"])
            vector_results["ndcg"].append(metrics["ndcg@5"])
            vector_results["precision"].append(metrics["precision@5"])
            vector_results["recall"].append(metrics["recall@5"])
            vector_results["latency"].append(latency)
        print()

        results["Vector Only"] = {
            "hit_rate@1": np.mean(vector_results["hits"]),
            "mrr": np.mean(vector_results["mrr"]),
            "ndcg@5": np.mean(vector_results["ndcg"]),
            "precision@5": np.mean(vector_results["precision"]),
            "recall@5": np.mean(vector_results["recall"]),
            "avg_latency_ms": np.mean(vector_results["latency"]),
            "p95_latency_ms": np.percentile(vector_results["latency"], 95)
        }

        # BGE Reranker v2-m3
        if BGE_AVAILABLE:
            results["BGE-v2-m3"] = self.evaluate_reranker(
                "BGE-Reranker-v2-m3",
                self.rerank_with_bge_m3,
                queries
            )

        # BGE Reranker v2.5-gemma2-lightweight (transformers 버전 호환성 문제로 비활성화)
        # 현재 transformers 버전에서 Gemma2FlashAttention2 import 오류 발생
        # 향후 FlagEmbedding 또는 transformers 업데이트 후 활성화 필요
        # if BGE_AVAILABLE:
        #     results["BGE-v2.5-gemma2"] = self.evaluate_reranker(
        #         "BGE-Reranker-v2.5-gemma2-lightweight",
        #         self.rerank_with_bge_gemma,
        #         queries
        #     )

        # Cohere rerank-v3.5
        if self.cohere_client:
            results["Cohere"] = self.evaluate_reranker(
                "Cohere rerank-v3.5",
                self.rerank_with_cohere,
                queries
            )

        return {
            "results": results,
            "config": {
                "num_documents": len(documents),
                "num_queries": len(queries),
                "embedding_model": self.embedding_model
            }
        }


def print_results(benchmark_results: Dict[str, Any]):
    """결과 출력"""
    results = benchmark_results["results"]

    print("\n" + "=" * 90)
    print("📈 Reranker Benchmark 결과")
    print("=" * 90)
    print()

    # 헤더
    rerankers = list(results.keys())
    header = "│ Metric              │"
    for r in rerankers:
        header += f" {r:^18} │"
    separator = "├" + "─" * 21 + "┼" + "┼".join(["─" * 20] * len(rerankers)) + "┤"

    print("┌" + "─" * 21 + "┬" + "┬".join(["─" * 20] * len(rerankers)) + "┐")
    print(header)
    print(separator)

    # 메트릭
    metrics_config = [
        ("Hit Rate@1", "hit_rate@1", "{:.1%}"),
        ("MRR", "mrr", "{:.3f}"),
        ("NDCG@5", "ndcg@5", "{:.3f}"),
        ("Precision@5", "precision@5", "{:.3f}"),
        ("Recall@5", "recall@5", "{:.3f}"),
        ("Avg Latency (ms)", "avg_latency_ms", "{:.1f}"),
        ("P95 Latency (ms)", "p95_latency_ms", "{:.1f}")
    ]

    for label, key, fmt in metrics_config:
        row = f"│ {label:<19} │"
        values = [results[r].get(key, 0) for r in rerankers]
        best_idx = values.index(max(values)) if key != "avg_latency_ms" and key != "p95_latency_ms" else values.index(min(values))

        for idx, (r, val) in enumerate(zip(rerankers, values)):
            formatted = fmt.format(val)
            if idx == best_idx and len(rerankers) > 1:
                row += f" {formatted:>14} 🏆  │"
            else:
                row += f" {formatted:>18} │"
        print(row)

    print("└" + "─" * 21 + "┴" + "┴".join(["─" * 20] * len(rerankers)) + "┘")
    print()

    # 분석
    print("📋 리랭커 비교:")
    print("   • Vector Only: 벡터 검색만 (baseline)")
    print("   • BGE-v2-m3: BAAI/bge-reranker-v2-m3 (560MB, 다국어, 무료)")
    print("   • BGE-v2.5-gemma2: BAAI/bge-reranker-v2.5-gemma2-lightweight (2.5GB, 고성능, 무료)")
    print("   • Cohere: rerank-v3.5 (API, $2/1000 searches)")
    print()

    # 최고 성능 분석
    if len(results) > 1:
        print("📊 분석:")
        best_hit = max(results.items(), key=lambda x: x[1].get("hit_rate@1", 0))
        best_mrr = max(results.items(), key=lambda x: x[1].get("mrr", 0))
        fastest = min(results.items(), key=lambda x: x[1].get("avg_latency_ms", float("inf")))

        print(f"   • 최고 Hit Rate@1: {best_hit[0]} ({best_hit[1]['hit_rate@1']:.1%})")
        print(f"   • 최고 MRR: {best_mrr[0]} ({best_mrr[1]['mrr']:.3f})")
        print(f"   • 최저 Latency: {fastest[0]} ({fastest[1]['avg_latency_ms']:.1f}ms)")
        print()


async def main():
    benchmark = RerankerBenchmark()
    results = benchmark.run_benchmark()

    print_results(results)

    # 결과 저장
    output_file = "/tmp/reranker_benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "Reranker Comparison",
            "timestamp": datetime.now().isoformat(),
            **results
        }, f, indent=2, ensure_ascii=False)

    print(f"📁 결과 저장: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
