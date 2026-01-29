"""
Reranker 비교 벤치마크 스크립트

BGE (GPU/로컬) vs Cohere (API/No GPU) 성능 비교
이전 결과와 함께 종합 분석
"""

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Provider imports
from config import settings
from providers.embedding import get_embedding_provider
from providers.vectordb import get_vectordb_provider
from providers.reranker.bge_provider import BGERerankerProvider
from providers.reranker.cohere_provider import CohereRerankerProvider
from providers.llm import get_llm_provider

# Search enhancements
from services.search_enhancements import RRFusion, get_search_enhancer

try:
    import numpy as np
except ImportError:
    print("numpy 설치 필요: pip install numpy")
    sys.exit(1)


class BM25:
    """BM25 검색 알고리즘"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.doc_lens = []
        self.avg_doc_len = 0
        self.corpus_size = 0
        self.tokenized_docs = []

    def _tokenize(self, text: str) -> List[str]:
        import re
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def fit(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        self.tokenized_docs = [self._tokenize(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0

        for doc in self.tokenized_docs:
            seen = set()
            for token in doc:
                if token not in seen:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen.add(token)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        import math
        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc in enumerate(self.tokenized_docs):
            score = 0.0
            doc_len = self.doc_lens[idx]

            for token in query_tokens:
                if token in self.doc_freqs:
                    df = self.doc_freqs[token]
                    idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
                    tf = doc.count(token)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                    score += idf * numerator / denominator

            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class RerankerComparisonBenchmark:
    """Reranker 비교 벤치마크"""

    def __init__(self):
        self.embedding_provider = None
        self.vectordb_provider = None
        self.bge_reranker = None
        self.cohere_reranker = None
        self.llm_provider = None
        self.search_enhancer = None
        self.collection_name = "reranker_comparison_v1"

        # BM25 인덱스
        self.bm25 = BM25()
        self.documents = []
        self.document_ids = []

        # RRF Fusion
        self.rrf = RRFusion(k=60)

        self._initialized = False

    async def initialize(self):
        """Provider 초기화"""
        if self._initialized:
            return

        print("🔧 Provider 초기화 중...")

        # Embedding Provider
        self.embedding_provider = get_embedding_provider()
        print(f"   ✅ Embedding: {settings.providers.embedding_provider}")

        # VectorDB Provider
        self.vectordb_provider = get_vectordb_provider()
        print(f"   ✅ VectorDB: {settings.providers.vectordb_provider}")

        # BGE Reranker (GPU/로컬)
        try:
            self.bge_reranker = BGERerankerProvider(model="BAAI/bge-reranker-v2-m3")
            # 미리 로드하여 초기화 시간 측정에서 제외
            print("   ⏳ BGE Reranker 로딩 중...")
            self.bge_reranker._load_model()
            print(f"   ✅ BGE Reranker: BAAI/bge-reranker-v2-m3 (GPU/Local)")
        except Exception as e:
            print(f"   ⚠️ BGE Reranker 초기화 실패: {e}")
            self.bge_reranker = None

        # Cohere Reranker (API)
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if cohere_api_key:
            try:
                self.cohere_reranker = CohereRerankerProvider(
                    api_key=cohere_api_key,
                    model="rerank-multilingual-v3.0"
                )
                print(f"   ✅ Cohere Reranker: rerank-multilingual-v3.0 (API)")
            except Exception as e:
                print(f"   ⚠️ Cohere Reranker 초기화 실패: {e}")
                self.cohere_reranker = None
        else:
            print("   ⚠️ Cohere Reranker: COHERE_API_KEY 없음 (건너뜀)")
            self.cohere_reranker = None

        # LLM Provider (Query Expansion용) - gpt5-mini 사용
        try:
            self.llm_provider = get_llm_provider(model="gpt5-mini")
            self.search_enhancer = get_search_enhancer(self.llm_provider)
            print(f"   ✅ Search Enhancer: Query Expansion + RRF Fusion (gpt5-mini)")
        except Exception as e:
            print(f"   ⚠️ Search Enhancer 초기화 실패: {e}")
            self.search_enhancer = None

        self._initialized = True
        print()

    def generate_test_data(self, num_users: int = 10, items_per_user: int = 20):
        """테스트 데이터 생성 (LaMP 스타일)"""
        import random

        profiles = []
        queries = []

        # 카테고리별 제품 데이터
        products = {
            "전자기기": ["스마트폰", "노트북", "태블릿", "스마트워치", "무선이어폰", "모니터", "키보드"],
            "도서": ["프로그래밍 입문", "데이터 분석", "인공지능 기초", "웹 개발", "알고리즘", "디자인 패턴"],
            "의류": ["티셔츠", "청바지", "후드티", "재킷", "운동화", "백팩", "모자"],
            "식품": ["커피", "차", "초콜릿", "건강식품", "간식", "음료"],
            "취미": ["운동기구", "악기", "캠핑용품", "보드게임", "DIY 키트", "원예용품"]
        }

        sentiments = {
            5: ["정말 좋아요", "최고입니다", "강력 추천합니다", "완벽해요", "대만족"],
            4: ["괜찮아요", "좋은 편이에요", "추천합니다", "만족스러워요"],
            3: ["보통이에요", "그저 그래요", "나쁘지 않아요"],
            2: ["별로예요", "기대 이하예요", "아쉬워요"],
            1: ["실망입니다", "추천하지 않아요", "안 좋아요"]
        }

        for user_idx in range(num_users):
            user_id = f"user_{user_idx:03d}"
            profile_items = []

            # 사용자별 선호 카테고리 (2-3개)
            preferred_categories = random.sample(list(products.keys()), random.randint(2, 3))

            for item_idx in range(items_per_user):
                # 80% 확률로 선호 카테고리에서 선택
                if random.random() < 0.8:
                    category = random.choice(preferred_categories)
                else:
                    category = random.choice(list(products.keys()))

                product = random.choice(products[category])

                # 선호 카테고리면 높은 평점 확률
                if category in preferred_categories:
                    rating = random.choices([5, 4, 3, 2, 1], weights=[0.4, 0.35, 0.15, 0.07, 0.03])[0]
                else:
                    rating = random.choices([5, 4, 3, 2, 1], weights=[0.15, 0.25, 0.35, 0.15, 0.1])[0]

                sentiment = random.choice(sentiments[rating])
                review_text = f"{product} 구매 후기: {sentiment}. 카테고리는 {category}이고 평점은 {rating}점입니다."

                profile_items.append({
                    "item_id": f"{user_id}_item_{item_idx:03d}",
                    "product": product,
                    "category": category,
                    "rating": rating,
                    "text": review_text
                })

            profiles.append({
                "user_id": user_id,
                "profile_items": profile_items,
                "preferred_categories": preferred_categories
            })

            # 쿼리 생성 (사용자당 3개)
            for q_idx in range(3):
                # 관련 있는 프로필 아이템 선택 (1-3개)
                relevant_items = [
                    item for item in profile_items
                    if item["category"] in preferred_categories and item["rating"] >= 4
                ]
                if not relevant_items:
                    relevant_items = profile_items[:3]

                selected_relevant = random.sample(relevant_items, min(random.randint(1, 3), len(relevant_items)))

                # 쿼리 텍스트 생성
                target_category = random.choice(preferred_categories)
                query_templates = [
                    f"{target_category} 관련 추천해주세요",
                    f"좋아하는 {target_category} 제품이 뭐였나요?",
                    f"이전에 구매한 {target_category} 중에 괜찮은 거 있나요?",
                    f"{target_category}에서 만족한 제품 알려주세요"
                ]

                queries.append({
                    "query_id": f"{user_id}_query_{q_idx}",
                    "user_id": user_id,
                    "query_text": random.choice(query_templates),
                    "relevant_profile_ids": [item["item_id"] for item in selected_relevant]
                })

        return profiles, queries

    async def setup_collection(self):
        """컬렉션 설정"""
        exists = await self.vectordb_provider.collection_exists(self.collection_name)
        if exists:
            await self.vectordb_provider.delete_collection(self.collection_name)

        await self.vectordb_provider.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedding_provider.dimension
        )
        print(f"   ✅ 컬렉션 생성: {self.collection_name}")

    async def index_profiles(self, profiles: List[Dict]):
        """프로필 인덱싱"""
        all_texts = []
        all_ids = []
        vectors_to_upsert = []

        for profile in profiles:
            user_id = profile["user_id"]
            for item in profile["profile_items"]:
                all_texts.append(item["text"])
                all_ids.append(item["item_id"])

        # 임베딩 생성
        print(f"   ⏳ {len(all_texts)}개 문서 임베딩 생성 중...")
        embeddings = await self.embedding_provider.embed(all_texts)

        for i, (text, item_id) in enumerate(zip(all_texts, all_ids)):
            user_id = item_id.split("_item_")[0]
            profile = next(p for p in profiles if p["user_id"] == user_id)
            item = next(item for item in profile["profile_items"] if item["item_id"] == item_id)

            # Qdrant는 UUID 형식의 ID 필요
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item_id))

            vectors_to_upsert.append({
                "id": point_id,
                "vector": embeddings[i],
                "payload": {
                    "user_id": user_id,
                    "item_id": item_id,
                    "text": text,
                    "product": item["product"],
                    "category": item["category"],
                    "rating": item["rating"]
                }
            })

        # VectorDB에 저장
        await self.vectordb_provider.upsert(self.collection_name, vectors_to_upsert)

        # BM25 인덱스 구축
        self.documents = all_texts
        self.document_ids = all_ids
        self.bm25.fit(all_texts)

        print(f"   ✅ {len(vectors_to_upsert)}개 벡터 인덱싱 완료")

    async def search_with_reranker(
        self,
        query: str,
        user_id: str,
        reranker,
        reranker_name: str,
        top_k: int = 5
    ) -> Tuple[List[Dict], float]:
        """특정 Reranker로 검색"""
        start_time = time.time()

        # 1. Vector Search
        query_embedding = (await self.embedding_provider.embed([query]))[0]

        results = await self.vectordb_provider.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            top_k=top_k * 3,
            filter_conditions={"user_id": user_id}
        )

        vector_candidates = [
            {
                "id": r.payload["item_id"],
                "item_id": r.payload["item_id"],
                "text": r.payload["text"],
                "vector_score": r.score,
                "product": r.payload["product"],
                "category": r.payload["category"],
                "rating": r.payload["rating"]
            }
            for r in results
        ]

        # 2. BM25 Search
        bm25_results = self.bm25.search(query, top_k=top_k * 3)
        user_doc_indices = [i for i, doc_id in enumerate(self.document_ids) if doc_id.startswith(user_id)]
        bm25_filtered = [(idx, score) for idx, score in bm25_results if idx in user_doc_indices]

        bm25_candidates = []
        for idx, bm25_score in bm25_filtered:
            item_id = self.document_ids[idx]
            bm25_candidates.append({
                "id": item_id,
                "item_id": item_id,
                "text": self.documents[idx],
                "bm25_score": bm25_score
            })

        # 3. RRF Fusion
        candidates = self.rrf.fuse_with_scores(
            vector_candidates, bm25_candidates, id_key="id", top_k=top_k * 2
        )

        # 4. Reranking
        if reranker and candidates:
            try:
                rerank_results = await reranker.rerank(
                    query=query,
                    documents=[c["text"] for c in candidates],
                    top_k=top_k
                )
                reranked = []
                for r in rerank_results:
                    candidate = candidates[r.index]
                    candidate["rerank_score"] = r.score
                    reranked.append(candidate)
                candidates = reranked
            except Exception as e:
                print(f"   {reranker_name} Reranking 오류: {e}")
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        latency = (time.time() - start_time) * 1000
        return candidates, latency

    async def search_no_reranker(
        self,
        query: str,
        user_id: str,
        top_k: int = 5
    ) -> Tuple[List[Dict], float]:
        """Reranker 없이 검색 (baseline)"""
        start_time = time.time()

        # Vector Search
        query_embedding = (await self.embedding_provider.embed([query]))[0]

        results = await self.vectordb_provider.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            top_k=top_k * 3,
            filter_conditions={"user_id": user_id}
        )

        vector_candidates = [
            {
                "id": r.payload["item_id"],
                "item_id": r.payload["item_id"],
                "text": r.payload["text"],
                "vector_score": r.score
            }
            for r in results
        ]

        # BM25 Search
        bm25_results = self.bm25.search(query, top_k=top_k * 3)
        user_doc_indices = [i for i, doc_id in enumerate(self.document_ids) if doc_id.startswith(user_id)]
        bm25_filtered = [(idx, score) for idx, score in bm25_results if idx in user_doc_indices]

        bm25_candidates = []
        for idx, bm25_score in bm25_filtered:
            item_id = self.document_ids[idx]
            bm25_candidates.append({
                "id": item_id,
                "item_id": item_id,
                "text": self.documents[idx],
                "bm25_score": bm25_score
            })

        # RRF Fusion only (no reranking)
        candidates = self.rrf.fuse_with_scores(
            vector_candidates, bm25_candidates, id_key="id", top_k=top_k
        )

        latency = (time.time() - start_time) * 1000
        return candidates, latency

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
        dcg = sum(1.0 / np.log2(i + 2) for i, doc_id in enumerate(retrieved[:5]) if doc_id in relevant)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), 5)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        # Precision@5, Recall@5
        relevant_in_top5 = sum(1 for doc_id in retrieved[:5] if doc_id in relevant)
        precision = relevant_in_top5 / 5
        recall = relevant_in_top5 / len(relevant) if relevant else 0.0

        return {"hits": hit_at_1, "mrr": mrr, "ndcg": ndcg, "precision": precision, "recall": recall}

    async def evaluate(self, queries: List[Dict]) -> Dict[str, Any]:
        """벤치마크 평가"""
        methods = ["no-reranker"]
        if self.bge_reranker:
            methods.append("bge-reranker")
        if self.cohere_reranker:
            methods.append("cohere-reranker")

        results = {
            method: {"hits": [], "mrr": [], "ndcg": [], "precision": [], "recall": [], "latency": []}
            for method in methods
        }

        total = len(queries)

        for idx, query_data in enumerate(queries):
            print(f"\r평가 중... {idx+1}/{total}", end="", flush=True)
            relevant_ids = set(query_data["relevant_profile_ids"])
            query_text = query_data["query_text"]
            user_id = query_data["user_id"]

            # No Reranker (baseline)
            res_no, lat_no = await self.search_no_reranker(query_text, user_id)
            retrieved_no = [r["item_id"] for r in res_no]
            metrics_no = self._calculate_metrics(retrieved_no, relevant_ids)
            for key, value in metrics_no.items():
                results["no-reranker"][key].append(value)
            results["no-reranker"]["latency"].append(lat_no)

            # BGE Reranker
            if self.bge_reranker:
                res_bge, lat_bge = await self.search_with_reranker(
                    query_text, user_id, self.bge_reranker, "BGE"
                )
                retrieved_bge = [r["item_id"] for r in res_bge]
                metrics_bge = self._calculate_metrics(retrieved_bge, relevant_ids)
                for key, value in metrics_bge.items():
                    results["bge-reranker"][key].append(value)
                results["bge-reranker"]["latency"].append(lat_bge)

            # Cohere Reranker (rate limit 대응: 6초 대기)
            if self.cohere_reranker:
                try:
                    res_coh, lat_coh = await self.search_with_reranker(
                        query_text, user_id, self.cohere_reranker, "Cohere"
                    )
                    retrieved_coh = [r["item_id"] for r in res_coh]
                    metrics_coh = self._calculate_metrics(retrieved_coh, relevant_ids)
                    for key, value in metrics_coh.items():
                        results["cohere-reranker"][key].append(value)
                    results["cohere-reranker"]["latency"].append(lat_coh)
                    # Rate limit 방지 (Trial: 10/분)
                    await asyncio.sleep(6.5)
                except Exception as e:
                    print(f"\n   Cohere 오류: {e}")

        print("\n")

        # 평균 계산
        summary = {}
        for method in methods:
            if results[method]["hits"]:
                summary[method] = {
                    "hit_rate@1": np.mean(results[method]["hits"]),
                    "mrr": np.mean(results[method]["mrr"]),
                    "ndcg@5": np.mean(results[method]["ndcg"]),
                    "precision@5": np.mean(results[method]["precision"]),
                    "recall@5": np.mean(results[method]["recall"]),
                    "avg_latency_ms": np.mean(results[method]["latency"]),
                    "p95_latency_ms": np.percentile(results[method]["latency"], 95)
                }

        return summary


async def main():
    print("=" * 80)
    print("🔬 Reranker 비교 벤치마크")
    print("   BGE (GPU/Local) vs Cohere (API/No GPU) 성능 비교")
    print("=" * 80)
    print()

    benchmark = RerankerComparisonBenchmark()

    # 초기화
    await benchmark.initialize()

    # 데이터 생성 (Cohere rate limit 고려하여 10개 쿼리만)
    print("📊 테스트 데이터 생성 중...")
    profiles, queries = benchmark.generate_test_data(num_users=5, items_per_user=15)
    queries = queries[:10]  # 10개 쿼리만 (Cohere rate limit 고려)
    print(f"   - 사용자: {len(profiles)}")
    print(f"   - 프로필 아이템: {sum(len(p['profile_items']) for p in profiles)}")
    print(f"   - 쿼리: {len(queries)}")
    print()

    # 인덱싱
    await benchmark.setup_collection()
    await benchmark.index_profiles(profiles)
    print()

    # 평가
    print("🧪 벤치마크 평가 실행...")
    print("-" * 80)
    results = await benchmark.evaluate(queries)

    # 결과 출력
    print("=" * 80)
    print("📈 Reranker 비교 결과")
    print("=" * 80)
    print()

    methods = list(results.keys())
    num_methods = len(methods)

    # 동적 테이블 헤더 생성
    col_width = 16
    header = "│ Metric              │"
    for method in methods:
        header += f" {method:^{col_width}} │"

    border_top = "┌─────────────────────┬" + "─" * (col_width + 2) + "┬" * (num_methods - 1)
    border_top = "┌─────────────────────┬" + "┬".join(["─" * (col_width + 2)] * num_methods) + "┐"
    border_mid = "├─────────────────────┼" + "┼".join(["─" * (col_width + 2)] * num_methods) + "┤"
    border_bot = "└─────────────────────┴" + "┴".join(["─" * (col_width + 2)] * num_methods) + "┘"

    print(border_top)
    print(header)
    print(border_mid)

    metrics_list = [
        ("Hit Rate@1", "hit_rate@1", "{:.1%}"),
        ("MRR", "mrr", "{:.3f}"),
        ("NDCG@5", "ndcg@5", "{:.3f}"),
        ("Precision@5", "precision@5", "{:.3f}"),
        ("Recall@5", "recall@5", "{:.3f}"),
        ("Avg Latency (ms)", "avg_latency_ms", "{:.1f}"),
        ("P95 Latency (ms)", "p95_latency_ms", "{:.1f}")
    ]

    for label, key, fmt in metrics_list:
        values = [results[m][key] for m in methods]

        if key.endswith("latency_ms"):
            best_idx = values.index(min(values))
        else:
            best_idx = values.index(max(values))

        row = f"│ {label:<19} │"
        for i, val in enumerate(values):
            val_str = fmt.format(val)
            if i == best_idx:
                val_str = val_str + " ✓"
            row += f" {val_str:>{col_width}} │"
        print(row)

    print(border_bot)
    print()

    # 개선율 분석
    baseline = results.get("no-reranker", {})
    if baseline:
        print("📊 Reranker 적용 개선율 (no-reranker 대비):")
        for method in methods:
            if method == "no-reranker":
                continue
            r = results[method]
            ndcg_diff = ((r["ndcg@5"] - baseline["ndcg@5"]) / baseline["ndcg@5"]) * 100 if baseline["ndcg@5"] > 0 else 0
            hit_diff = ((r["hit_rate@1"] - baseline["hit_rate@1"]) / baseline["hit_rate@1"]) * 100 if baseline["hit_rate@1"] > 0 else 0
            mrr_diff = ((r["mrr"] - baseline["mrr"]) / baseline["mrr"]) * 100 if baseline["mrr"] > 0 else 0
            latency_diff = ((r["avg_latency_ms"] - baseline["avg_latency_ms"]) / baseline["avg_latency_ms"]) * 100

            print(f"\n   {method}:")
            print(f"      NDCG@5:     {ndcg_diff:+.1f}%")
            print(f"      Hit Rate@1: {hit_diff:+.1f}%")
            print(f"      MRR:        {mrr_diff:+.1f}%")
            print(f"      Latency:    {latency_diff:+.1f}% {'(더 느림)' if latency_diff > 0 else '(더 빠름)'}")

    print()

    # BGE vs Cohere 직접 비교
    if "bge-reranker" in results and "cohere-reranker" in results:
        bge = results["bge-reranker"]
        cohere = results["cohere-reranker"]

        print("=" * 80)
        print("🔍 BGE vs Cohere 직접 비교")
        print("=" * 80)

        comparisons = [
            ("NDCG@5", "ndcg@5"),
            ("Hit Rate@1", "hit_rate@1"),
            ("MRR", "mrr"),
            ("Latency (ms)", "avg_latency_ms")
        ]

        for label, key in comparisons:
            bge_val = bge[key]
            cohere_val = cohere[key]

            if key == "avg_latency_ms":
                diff = ((cohere_val - bge_val) / bge_val) * 100
                winner = "BGE" if bge_val < cohere_val else "Cohere"
                print(f"   {label:20}: BGE {bge_val:.1f}ms vs Cohere {cohere_val:.1f}ms → {winner} 승 ({abs(diff):.1f}% 차이)")
            else:
                diff = ((cohere_val - bge_val) / bge_val) * 100 if bge_val > 0 else 0
                winner = "Cohere" if cohere_val > bge_val else "BGE"
                print(f"   {label:20}: BGE {bge_val:.3f} vs Cohere {cohere_val:.3f} → {winner} 승 ({abs(diff):.1f}% 차이)")

        print()
        print("💡 결론:")
        if cohere["ndcg@5"] > bge["ndcg@5"]:
            ndcg_improvement = ((cohere["ndcg@5"] - bge["ndcg@5"]) / bge["ndcg@5"]) * 100
            print(f"   - Cohere Reranker가 정확도에서 {ndcg_improvement:.1f}% 우수")
        else:
            ndcg_improvement = ((bge["ndcg@5"] - cohere["ndcg@5"]) / cohere["ndcg@5"]) * 100
            print(f"   - BGE Reranker가 정확도에서 {ndcg_improvement:.1f}% 우수")

        if bge["avg_latency_ms"] < cohere["avg_latency_ms"]:
            latency_improvement = ((cohere["avg_latency_ms"] - bge["avg_latency_ms"]) / cohere["avg_latency_ms"]) * 100
            print(f"   - BGE Reranker가 속도에서 {latency_improvement:.1f}% 빠름 (로컬 GPU 활용)")
        else:
            latency_improvement = ((bge["avg_latency_ms"] - cohere["avg_latency_ms"]) / bge["avg_latency_ms"]) * 100
            print(f"   - Cohere Reranker가 속도에서 {latency_improvement:.1f}% 빠름 (API)")

    print()
    print("=" * 80)
    print("🔧 테스트 환경:")
    print(f"   - Embedding: {settings.providers.embedding_provider} ({settings.providers.embedding_model})")
    print(f"   - VectorDB: {settings.providers.vectordb_provider}")
    print(f"   - BGE Model: BAAI/bge-reranker-v2-m3")
    print(f"   - Cohere Model: rerank-multilingual-v3.0")
    print("=" * 80)

    # 결과 저장
    output_file = "/tmp/reranker_comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "Reranker Comparison",
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "config": {
                "num_users": len(profiles),
                "num_queries": len(queries)
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📄 결과 저장: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
