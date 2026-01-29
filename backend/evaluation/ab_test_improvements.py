"""
A/B 테스트: 성능 개선 방안 비교
동일한 랜덤 시드를 사용하여 각 개선사항의 효과를 정확히 비교

테스트 시나리오:
1. 베이스라인: 기존 3-stage (동의어 사전 확장 전)
2. 개선 A: 동의어 사전 확장 (41→80개)
3. 개선 B: Semantic Cache 임계값 최적화 (0.95→0.85)
4. 개선 C: A + B 조합
"""

import asyncio
import json
import os
import sys
import time
import random
import numpy as np
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Provider imports
from config import settings
from providers.embedding import get_embedding_provider
from providers.vectordb import get_vectordb_provider
from providers.reranker import get_reranker_provider
from providers.llm import get_llm_provider
from services.search_enhancements import (
    SearchEnhancer, RRFusion, get_search_enhancer,
    LocalQueryExpander, HybridQueryExpander, SemanticCache
)

# 랜덤 시드 고정
RANDOM_SEED = 42

@dataclass
class TestResult:
    """테스트 결과"""
    test_name: str
    ndcg_5: float
    precision_5: float
    recall_5: float
    mrr: float
    hit_rate_1: float
    avg_latency_ms: float
    p95_latency_ms: float
    config: Dict[str, Any]

@dataclass
class LaMP_Profile:
    user_id: str
    profile_items: List[Dict[str, Any]]
    task_type: str

@dataclass
class LaMP_Query:
    query_id: str
    user_id: str
    query_text: str
    task_type: str
    ground_truth: str
    relevant_profile_ids: List[str]


class BM25:
    """BM25 검색 알고리즘"""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.doc_lens = []
        self.avg_doc_len = 0
        self.corpus_size = 0
        self.documents = []
        self.tokenized_docs = []

    def tokenize(self, text: str) -> List[str]:
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def fit(self, documents: List[str]):
        self.documents = documents
        self.corpus_size = len(documents)
        self.tokenized_docs = [self.tokenize(doc) for doc in documents]
        self.doc_lens = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_len = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs = defaultdict(int)
        for doc in self.tokenized_docs:
            for term in set(doc):
                self.doc_freqs[term] += 1

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_tokens = self.tokenize(query)
        scores = []
        for idx, doc in enumerate(self.tokenized_docs):
            score = 0
            doc_len = self.doc_lens[idx]
            for term in query_tokens:
                if term in doc:
                    tf = doc.count(term)
                    df = self.doc_freqs.get(term, 0)
                    idf = np.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
                    score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class ABTestBenchmark:
    """A/B 테스트 벤치마크"""

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.embedding_provider = None
        self.vectordb_provider = None
        self.reranker_provider = None
        self.llm_provider = None
        self.collection_name = "ab_test_benchmark"
        self.bm25 = BM25()
        self.rrf = RRFusion(k=60)
        self.documents = []
        self.document_ids = []
        self._initialized = False

    def _set_seed(self):
        """랜덤 시드 고정"""
        random.seed(self.seed)
        np.random.seed(self.seed)

    async def initialize(self):
        """Provider 초기화"""
        if self._initialized:
            return

        print("🔧 Provider 초기화 중...")

        self.embedding_provider = get_embedding_provider()
        print(f"   ✅ Embedding: {settings.providers.embedding_provider}")

        self.vectordb_provider = get_vectordb_provider()
        print(f"   ✅ VectorDB: {settings.providers.vectordb_provider}")

        try:
            self.reranker_provider = get_reranker_provider()
            print(f"   ✅ Reranker: {settings.providers.reranker_provider}")
        except:
            self.reranker_provider = None

        try:
            self.llm_provider = get_llm_provider(model="gpt5-mini")
            print(f"   ✅ LLM: gpt5-mini (Query Expansion)")
        except:
            self.llm_provider = None

        self._initialized = True
        print()

    def generate_test_data(self, num_users: int = 10, items_per_user: int = 20) -> Tuple[List[LaMP_Profile], List[LaMP_Query]]:
        """고정 시드로 테스트 데이터 생성"""
        self._set_seed()

        profiles = []
        queries = []

        user_personas = [
            {"name": "tech_enthusiast", "interests": ["AI", "programming"], "products": ["laptop", "smartphone", "headphones"], "rating_bias": 3.5},
            {"name": "casual_user", "interests": ["movies", "music"], "products": ["camera", "speakers"], "rating_bias": 4.2},
            {"name": "professional", "interests": ["productivity", "business"], "products": ["office equipment", "software"], "rating_bias": 3.8},
            {"name": "creative", "interests": ["art", "design"], "products": ["graphics tablet", "camera"], "rating_bias": 4.5},
            {"name": "student", "interests": ["studying", "entertainment"], "products": ["textbooks", "laptop"], "rating_bias": 3.5},
            {"name": "health_focused", "interests": ["fitness", "nutrition"], "products": ["fitness tracker", "supplements"], "rating_bias": 4.0},
            {"name": "minimalist", "interests": ["simple living"], "products": ["essential items", "quality tools"], "rating_bias": 4.0},
            {"name": "gamer", "interests": ["video games", "esports"], "products": ["gaming PC", "monitor", "keyboard"], "rating_bias": 4.3},
            {"name": "parent", "interests": ["family", "children"], "products": ["toys", "educational items"], "rating_bias": 3.9},
            {"name": "senior", "interests": ["simplicity", "health"], "products": ["easy devices", "health monitors"], "rating_bias": 4.1},
        ]

        products_db = {
            "laptop": ["MacBook Pro M3", "Dell XPS 15", "ThinkPad X1", "ASUS ROG", "Surface Laptop"],
            "smartphone": ["iPhone 15 Pro", "Galaxy S24 Ultra", "Pixel 8 Pro", "OnePlus 12"],
            "headphones": ["Sony WH-1000XM5", "AirPods Pro 2", "Bose QC Ultra", "Sennheiser Momentum"],
            "camera": ["Sony A7 IV", "Canon EOS R6", "Nikon Z6", "Fujifilm X-T5"],
            "gaming PC": ["ROG Strix", "Alienware Aurora", "HP Omen", "MSI Trident"],
            "monitor": ["LG UltraFine", "Dell UltraSharp", "ASUS ProArt", "Samsung Odyssey"],
            "keyboard": ["Keychron Q1", "Logitech MX Keys", "HHKB Professional", "Corsair K100"],
            "fitness tracker": ["Apple Watch Ultra", "Garmin Fenix", "Fitbit Sense", "Samsung Galaxy Watch"],
        }

        styles = ["technical", "casual", "formal", "expressive", "informal", "motivational", "concise", "enthusiastic", "practical", "clear"]

        for user_idx in range(min(num_users, len(user_personas))):
            persona = user_personas[user_idx]
            user_id = f"user_{user_idx + 1}"
            profile_items = []
            style = styles[user_idx % len(styles)]

            for item_idx in range(items_per_user):
                category = random.choice(persona["products"])
                products = products_db.get(category, products_db["laptop"])
                product = random.choice(products)
                rating = min(5, max(1, int(random.gauss(persona["rating_bias"], 0.8))))

                review = self._generate_review(product, rating, style)

                profile_items.append({
                    "id": f"{user_id}_item_{item_idx}",
                    "product": product,
                    "category": category,
                    "rating": rating,
                    "review": review,
                    "date": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                })

            profiles.append(LaMP_Profile(user_id=user_id, profile_items=profile_items, task_type="LaMP-3"))

            # 쿼리 생성
            for q_idx in range(3):
                query_category = random.choice(persona["products"])
                products = products_db.get(query_category, products_db["laptop"])
                query_product = random.choice(products)

                relevant_ids = [item["id"] for item in profile_items if item["category"] == query_category][:5]
                related_ratings = [item["rating"] for item in profile_items if item["category"] == query_category]
                expected_rating = round(sum(related_ratings) / len(related_ratings)) if related_ratings else 4

                queries.append(LaMP_Query(
                    query_id=f"{user_id}_query_{q_idx}",
                    user_id=user_id,
                    query_text=f"{query_product}에 대해 이 사용자는 어떤 평점을 줄까요?",
                    task_type="LaMP-3",
                    ground_truth=str(expected_rating),
                    relevant_profile_ids=relevant_ids
                ))

        return profiles, queries

    def _generate_review(self, product: str, rating: int, style: str) -> str:
        templates = {
            "technical": [
                f"{product}의 스펙은 인상적입니다. 성능 테스트 결과 {rating}점을 주고 싶네요.",
                f"기술적으로 분석했을 때 {product}는 {rating}점 수준입니다. 벤치마크 성능이 좋습니다.",
            ],
            "casual": [
                f"{product} 정말 좋아요! {rating}점 드립니다~",
                f"오 {product} 진짜 괜찮네요. {rating}점이요!",
            ],
            "formal": [
                f"{product}에 대해 {rating}점을 부여합니다. 품질이 우수합니다.",
                f"검토 결과 {product}는 {rating}점으로 평가됩니다.",
            ],
            "expressive": [
                f"와! {product} 너무 좋아요! ⭐{rating}점!",
                f"{product} 정말 최고예요! {rating}점 드려요!",
            ],
            "informal": [
                f"{product} ㅋㅋ 괜찮음 {rating}점",
                f"음 {product} {rating}점 줄만함",
            ],
            "motivational": [
                f"{product}로 목표 달성! {rating}점 강력 추천!",
                f"건강한 선택 {product}! {rating}점입니다!",
            ],
            "concise": [
                f"{product}: {rating}점. 만족.",
                f"{rating}점. {product} 좋음.",
            ],
            "enthusiastic": [
                f"{product} 대박! 게임할 때 최고! {rating}점!!!",
                f"와 {product} 진짜 좋아요 {rating}점 드림!",
            ],
            "practical": [
                f"가성비 좋은 {product}, {rating}점 추천합니다.",
                f"{product} 실용적이에요. {rating}점.",
            ],
            "clear": [
                f"{product}은 {rating}점입니다. 사용하기 쉬워요.",
                f"쉽게 쓸 수 있는 {product}. {rating}점.",
            ],
        }
        reviews = templates.get(style, templates["casual"])
        return random.choice(reviews)

    def _str_to_uuid(self, s: str) -> str:
        """문자열을 UUID로 변환 (일관된 해시 기반)"""
        return str(uuid.UUID(hashlib.md5(s.encode()).hexdigest()))

    async def index_profiles(self, profiles: List[LaMP_Profile]):
        """프로필 인덱싱"""
        # 컬렉션 생성/재생성
        try:
            await self.vectordb_provider.delete_collection(self.collection_name)
        except:
            pass

        dim = self.embedding_provider.dimension
        await self.vectordb_provider.create_collection(self.collection_name, dimension=dim)

        self.documents = []
        self.document_ids = []
        self.id_mapping = {}  # 원본 ID -> UUID 매핑
        all_items = []

        for profile in profiles:
            for item in profile.profile_items:
                text = f"{item['product']} {item['category']} {item['review']}"
                uuid_id = self._str_to_uuid(item["id"])
                self.documents.append(text)
                self.document_ids.append(item["id"])  # 원본 ID 저장
                self.id_mapping[item["id"]] = uuid_id
                all_items.append({
                    "id": uuid_id,  # UUID 사용
                    "original_id": item["id"],
                    "text": text,
                    "user_id": profile.user_id,
                    "category": item["category"],
                    "product": item["product"],
                    "rating": item["rating"],
                })

        # BM25 인덱싱
        self.bm25.fit(self.documents)

        # 벡터 임베딩
        batch_size = 50
        for i in range(0, len(all_items), batch_size):
            batch = all_items[i:i+batch_size]
            texts = [item["text"] for item in batch]
            embeddings = await self.embedding_provider.embed(texts)

            points = []
            for j, item in enumerate(batch):
                points.append({
                    "id": item["id"],  # UUID
                    "vector": embeddings[j],
                    "payload": {
                        "text": item["text"],
                        "original_id": item["original_id"],
                        "user_id": item["user_id"],
                        "category": item["category"],
                        "product": item["product"],
                        "rating": item["rating"],
                    }
                })
            await self.vectordb_provider.upsert(self.collection_name, points)

        print(f"✅ {len(all_items)}개 프로필 인덱싱 완료")

    async def evaluate_3stage(
        self,
        queries: List[LaMP_Query],
        use_local_expansion: bool = True,
        test_name: str = "3-stage"
    ) -> TestResult:
        """3-stage 파이프라인 평가"""
        hit_rate_scores = []
        mrr_scores = []
        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        latencies = []

        # Local Expander 설정
        local_expander = LocalQueryExpander(use_fasttext=False) if use_local_expansion else None

        for query in queries:
            start_time = time.time()

            # 1. Query Expansion (로컬)
            query_text = query.query_text
            queries_to_search = [query_text]

            if local_expander:
                expanded = local_expander.expand(query_text)
                # ExpandedQuery는 dataclass이므로 get_all_queries() 사용
                queries_to_search = expanded.get_all_queries()[:3]

            # 2. Vector Search
            query_embedding = await self.embedding_provider.embed_single(queries_to_search[0])
            vector_results = await self.vectordb_provider.search(
                self.collection_name, query_embedding, top_k=20
            )

            # Vector 결과에서 original_id 추출
            vector_items = []
            for r in vector_results:
                payload = r.payload if hasattr(r, 'payload') else {}
                orig_id = payload.get("original_id") if isinstance(payload, dict) else None
                if orig_id:
                    vector_items.append({"id": orig_id})

            # 3. BM25 Hybrid
            bm25_results_raw = self.bm25.search(query_text, top_k=20)
            bm25_items = [{"id": self.document_ids[idx]} for idx, _ in bm25_results_raw]

            # RRF Fusion (original_id 기준)
            fused_results = self.rrf.fuse([vector_items, bm25_items], top_k=10)
            fused_ids = [r["id"] for r in fused_results]

            # 4. Reranking
            if self.reranker_provider:
                docs_to_rerank = []
                fused_ids_for_rerank = []
                for fid in fused_ids[:10]:
                    try:
                        idx = self.document_ids.index(fid)
                        docs_to_rerank.append(self.documents[idx])
                        fused_ids_for_rerank.append(fid)
                    except:
                        pass

                if docs_to_rerank:
                    reranked = await self.reranker_provider.rerank(query_text, docs_to_rerank, top_k=5)
                    final_ids = []
                    for result in reranked:
                        # RerankResult는 document 속성을 가진 dataclass
                        doc = result.document if hasattr(result, 'document') else result
                        try:
                            idx = self.documents.index(doc)
                            final_ids.append(self.document_ids[idx])
                        except:
                            pass
                else:
                    final_ids = fused_ids[:5]
            else:
                final_ids = fused_ids[:5]

            latency = (time.time() - start_time) * 1000
            latencies.append(latency)

            # 메트릭 계산
            relevant_set = set(query.relevant_profile_ids)

            # Hit Rate@1
            hit_rate_scores.append(1.0 if final_ids and final_ids[0] in relevant_set else 0.0)

            # MRR
            mrr = 0.0
            for i, doc_id in enumerate(final_ids):
                if doc_id in relevant_set:
                    mrr = 1.0 / (i + 1)
                    break
            mrr_scores.append(mrr)

            # NDCG@5
            dcg = 0.0
            for i, doc_id in enumerate(final_ids[:5]):
                if doc_id in relevant_set:
                    dcg += 1.0 / np.log2(i + 2)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), 5)))
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

            # Precision@5
            hits = sum(1 for doc_id in final_ids[:5] if doc_id in relevant_set)
            precision_scores.append(hits / 5)

            # Recall@5
            recall_scores.append(hits / len(relevant_set) if relevant_set else 0.0)

        return TestResult(
            test_name=test_name,
            ndcg_5=np.mean(ndcg_scores),
            precision_5=np.mean(precision_scores),
            recall_5=np.mean(recall_scores),
            mrr=np.mean(mrr_scores),
            hit_rate_1=np.mean(hit_rate_scores),
            avg_latency_ms=np.mean(latencies),
            p95_latency_ms=np.percentile(latencies, 95),
            config={"use_local_expansion": use_local_expansion}
        )


async def run_ab_tests():
    """A/B 테스트 실행"""
    print("=" * 70)
    print("A/B 테스트: 성능 개선 방안 비교")
    print("=" * 70)
    print()

    benchmark = ABTestBenchmark(seed=RANDOM_SEED)
    await benchmark.initialize()

    # 테스트 데이터 생성 (고정 시드)
    print("📊 테스트 데이터 생성 중 (시드: 42)...")
    profiles, queries = benchmark.generate_test_data(num_users=10, items_per_user=20)
    print(f"   - 사용자: {len(profiles)}")
    print(f"   - 프로필 아이템: {sum(len(p.profile_items) for p in profiles)}")
    print(f"   - 쿼리: {len(queries)}")
    print()

    # 인덱싱
    await benchmark.index_profiles(profiles)
    print()

    # 테스트 실행
    results = []

    print("🧪 테스트 실행 중...")
    print("-" * 70)

    # Test 1: 베이스라인 (RRF K=60)
    print("\n[1/6] 베이스라인 (RRF K=60)...")
    benchmark.rrf = RRFusion(k=60)
    result1 = await benchmark.evaluate_3stage(queries, use_local_expansion=True, test_name="K=60 (기본)")
    results.append(result1)
    print(f"   NDCG@5: {result1.ndcg_5:.4f}, Latency: {result1.avg_latency_ms:.1f}ms")

    # Test 2: RRF K=40
    print("\n[2/6] RRF K=40...")
    benchmark.rrf = RRFusion(k=40)
    result2 = await benchmark.evaluate_3stage(queries, use_local_expansion=True, test_name="K=40")
    results.append(result2)
    print(f"   NDCG@5: {result2.ndcg_5:.4f}, Latency: {result2.avg_latency_ms:.1f}ms")

    # Test 3: RRF K=50
    print("\n[3/6] RRF K=50...")
    benchmark.rrf = RRFusion(k=50)
    result3 = await benchmark.evaluate_3stage(queries, use_local_expansion=True, test_name="K=50")
    results.append(result3)
    print(f"   NDCG@5: {result3.ndcg_5:.4f}, Latency: {result3.avg_latency_ms:.1f}ms")

    # Test 4: RRF K=70
    print("\n[4/6] RRF K=70...")
    benchmark.rrf = RRFusion(k=70)
    result4 = await benchmark.evaluate_3stage(queries, use_local_expansion=True, test_name="K=70")
    results.append(result4)
    print(f"   NDCG@5: {result4.ndcg_5:.4f}, Latency: {result4.avg_latency_ms:.1f}ms")

    # Test 5: RRF K=80
    print("\n[5/6] RRF K=80...")
    benchmark.rrf = RRFusion(k=80)
    result5 = await benchmark.evaluate_3stage(queries, use_local_expansion=True, test_name="K=80")
    results.append(result5)
    print(f"   NDCG@5: {result5.ndcg_5:.4f}, Latency: {result5.avg_latency_ms:.1f}ms")

    # Test 6: RRF K=30
    print("\n[6/6] RRF K=30...")
    benchmark.rrf = RRFusion(k=30)
    result6 = await benchmark.evaluate_3stage(queries, use_local_expansion=True, test_name="K=30")
    results.append(result6)
    print(f"   NDCG@5: {result6.ndcg_5:.4f}, Latency: {result6.avg_latency_ms:.1f}ms")

    # 결과 출력
    print()
    print("=" * 70)
    print("📈 A/B 테스트 결과")
    print("=" * 70)
    print()
    print(f"{'테스트':<20} {'NDCG@5':<12} {'MRR':<12} {'Hit@1':<12} {'Latency':<12}")
    print("-" * 70)

    for r in results:
        print(f"{r.test_name:<20} {r.ndcg_5:.4f}       {r.mrr:.4f}       {r.hit_rate_1:.1%}        {r.avg_latency_ms:.1f}ms")

    # 최고 성능 찾기
    if len(results) >= 2:
        baseline = results[0]
        best_ndcg = max(results, key=lambda x: x.ndcg_5)

        print()
        print("-" * 70)
        print(f"📊 분석 결과:")
        print(f"   베이스라인 (K=60): NDCG@5 = {baseline.ndcg_5:.4f}")
        print(f"   최고 성능: {best_ndcg.test_name} (NDCG@5 = {best_ndcg.ndcg_5:.4f})")

        if best_ndcg.ndcg_5 > baseline.ndcg_5:
            improve = ((best_ndcg.ndcg_5 - baseline.ndcg_5) / baseline.ndcg_5) * 100
            print(f"   개선율: +{improve:.1f}%")
        else:
            print(f"   → K=60이 이미 최적 또는 차이 없음")

    # 결과 저장
    result_path = "/tmp/ab_test_results.json"
    with open(result_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\n📁 결과 저장: {result_path}")


if __name__ == "__main__":
    asyncio.run(run_ab_tests())
