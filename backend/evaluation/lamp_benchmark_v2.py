"""
LaMP (Language Model Personalization) Benchmark 평가 스크립트 v2
Provider 패턴을 사용하여 리팩토링된 버전

LaMP 벤치마크 태스크:
- LaMP-1: Personalized Citation Identification (논문 인용 예측)
- LaMP-2: Personalized Movie Tagging (영화 태깅)
- LaMP-3: Personalized Product Rating (제품 평점 예측)
- LaMP-4: Personalized News Headline Generation (뉴스 헤드라인 생성)
- LaMP-5: Personalized Scholarly Title Generation (논문 제목 생성)
- LaMP-6: Personalized Email Subject Generation (이메일 제목 생성)
- LaMP-7: Personalized Tweet Paraphrasing (트윗 패러프레이징)

이 스크립트는 Provider 패턴을 사용하여 개인화 시스템을 평가합니다.
"""

import asyncio
import json
import os
import sys
import time
import math
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import uuid

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

# Search enhancements (Query Expansion, RRF)
from services.search_enhancements import SearchEnhancer, RRFusion, get_search_enhancer

try:
    import numpy as np
except ImportError:
    print("numpy 설치 필요: pip install numpy")
    sys.exit(1)


@dataclass
class LaMP_Profile:
    """LaMP 사용자 프로필 데이터"""
    user_id: str
    profile_items: List[Dict[str, Any]]
    task_type: str


@dataclass
class LaMP_Query:
    """LaMP 쿼리 데이터"""
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

    def get_scores(self, query: str) -> List[float]:
        query_tokens = self.tokenize(query)
        scores = []
        for idx, doc in enumerate(self.tokenized_docs):
            score = 0.0
            doc_len = self.doc_lens[idx]
            term_freqs = defaultdict(int)
            for term in doc:
                term_freqs[term] += 1
            for term in query_tokens:
                if term not in term_freqs:
                    continue
                tf = term_freqs[term]
                df = self.doc_freqs.get(term, 0)
                if df == 0:
                    continue
                idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
                score += idf * tf_norm
            scores.append(score)
        return scores

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        scores = self.get_scores(query)
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]


class LaMP_Benchmark_V2:
    """Provider 패턴을 사용하는 LaMP 벤치마크"""

    def __init__(self):
        self.embedding_provider = None
        self.vectordb_provider = None
        self.reranker_provider = None
        self.llm_provider = None
        self.search_enhancer = None
        self.collection_name = "lamp_benchmark_v2"

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
        try:
            self.embedding_provider = get_embedding_provider()
            print(f"   ✅ Embedding: {settings.providers.embedding_provider} ({settings.providers.embedding_model})")
        except Exception as e:
            print(f"   ❌ Embedding 초기화 실패: {e}")
            raise

        # VectorDB Provider
        try:
            self.vectordb_provider = get_vectordb_provider()
            print(f"   ✅ VectorDB: {settings.providers.vectordb_provider}")
        except Exception as e:
            print(f"   ❌ VectorDB 초기화 실패: {e}")
            raise

        # Reranker Provider
        try:
            self.reranker_provider = get_reranker_provider()
            print(f"   ✅ Reranker: {settings.providers.reranker_provider}")
        except Exception as e:
            print(f"   ⚠️ Reranker 초기화 실패 (비활성화): {e}")
            self.reranker_provider = None

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

    def generate_lamp_data(self, num_users: int = 10, items_per_user: int = 20, seed: int = 42) -> Tuple[List[LaMP_Profile], List[LaMP_Query]]:
        """LaMP 스타일 데이터 생성 (seed로 재현 가능)"""
        # 재현 가능한 결과를 위해 seed 설정
        random.seed(seed)
        np.random.seed(seed)

        profiles = []
        queries = []

        # 사용자 페르소나 정의
        user_personas = [
            {"name": "tech_enthusiast", "interests": ["AI", "programming", "gadgets"], "style": "technical",
             "products": ["laptop", "smartphone", "headphones"], "rating_bias": 3.5},
            {"name": "casual_user", "interests": ["movies", "music", "travel"], "style": "casual",
             "products": ["camera", "speakers", "travel gear"], "rating_bias": 4.2},
            {"name": "professional", "interests": ["productivity", "business"], "style": "formal",
             "products": ["office equipment", "software", "books"], "rating_bias": 3.8},
            {"name": "creative", "interests": ["art", "design", "photography"], "style": "expressive",
             "products": ["graphics tablet", "camera", "software"], "rating_bias": 4.5},
            {"name": "student", "interests": ["studying", "entertainment"], "style": "informal",
             "products": ["textbooks", "laptop", "headphones"], "rating_bias": 3.5},
            {"name": "health_focused", "interests": ["fitness", "nutrition"], "style": "motivational",
             "products": ["fitness tracker", "supplements", "workout gear"], "rating_bias": 4.0},
            {"name": "minimalist", "interests": ["simple living", "quality"], "style": "concise",
             "products": ["essential items", "quality tools"], "rating_bias": 4.0},
            {"name": "gamer", "interests": ["video games", "esports"], "style": "enthusiastic",
             "products": ["gaming PC", "monitor", "keyboard"], "rating_bias": 4.3},
            {"name": "parent", "interests": ["family", "children", "home"], "style": "practical",
             "products": ["toys", "educational items", "appliances"], "rating_bias": 3.9},
            {"name": "senior", "interests": ["simplicity", "health"], "style": "clear",
             "products": ["easy devices", "health monitors"], "rating_bias": 4.1},
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

        for user_idx in range(min(num_users, len(user_personas))):
            persona = user_personas[user_idx]
            user_id = f"user_{user_idx + 1}"
            profile_items = []

            for item_idx in range(items_per_user):
                category = random.choice(persona["products"])
                products = products_db.get(category, products_db["laptop"])
                product = random.choice(products)

                # 평점 (페르소나 편향 반영)
                rating = min(5, max(1, int(random.gauss(persona["rating_bias"], 0.8))))

                # 리뷰 생성
                review = self._generate_review(product, rating, persona["style"])

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
        """리뷰 생성"""
        templates = {
            "technical": f"{product} 성능 테스트 결과 만족스럽습니다. {rating}/5점.",
            "casual": f"{product} 진짜 좋아요! {rating}점 드려요~",
            "formal": f"{product}에 대한 평가: 전반적으로 {'우수' if rating >= 4 else '보통'}합니다. {rating}/5",
            "expressive": f"와! {product} 완전 사랑해요 💕 {rating}점!",
            "informal": f"ㅋㅋ {product} {'개이득' if rating >= 4 else '그저그럼'} {rating}점",
            "motivational": f"{product}로 목표 달성 중! {rating}/5점",
            "concise": f"{product}: {rating}/5",
            "enthusiastic": f"{product} 최고!!! {rating}점!!!",
            "practical": f"{product} {'추천' if rating >= 4 else '보통'}: {rating}/5",
            "clear": f"{product} - {rating}점, {'좋음' if rating >= 4 else '보통'}",
        }
        return templates.get(style, templates["casual"])

    async def setup_collection(self):
        """벡터 DB 컬렉션 설정"""
        dimension = self.embedding_provider.dimension

        # 기존 컬렉션 삭제 후 재생성
        try:
            await self.vectordb_provider.delete_collection(self.collection_name)
        except:
            pass

        await self.vectordb_provider.create_collection(
            collection_name=self.collection_name,
            dimension=dimension
        )
        print(f"✅ Collection '{self.collection_name}' 생성 완료 (dim={dimension})")

    async def index_profiles(self, profiles: List[LaMP_Profile]):
        """프로필 인덱싱"""
        vectors = []
        self.documents = []
        self.document_ids = []

        print("📊 프로필 임베딩 생성 중...")
        all_texts = []
        all_payloads = []

        for profile in profiles:
            for item in profile.profile_items:
                text = f"제품: {item['product']} | 카테고리: {item['category']} | 평점: {item['rating']}/5 | 리뷰: {item['review']}"
                all_texts.append(text)
                all_payloads.append({
                    "item_id": item["id"],
                    "user_id": profile.user_id,
                    "product": item["product"],
                    "category": item["category"],
                    "rating": item["rating"],
                    "review": item["review"],
                    "text": text,
                })
                self.documents.append(text)
                self.document_ids.append(item["id"])

        # 배치 임베딩
        batch_size = 50
        all_embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            embeddings = await self.embedding_provider.embed(batch)
            all_embeddings.extend(embeddings)
            print(f"   임베딩: {min(i+batch_size, len(all_texts))}/{len(all_texts)}")

        # 벡터 구성
        for idx, (embedding, payload) in enumerate(zip(all_embeddings, all_payloads)):
            vectors.append({
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": payload
            })

        # 업로드
        await self.vectordb_provider.upsert(self.collection_name, vectors)

        # BM25 인덱스
        self.bm25.fit(self.documents)

        print(f"✅ {len(vectors)}개 프로필 인덱싱 완료")

    async def search_2stage(self, query: str, user_id: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """2-stage: Vector Search → Reranking"""
        start_time = time.time()

        # 1. Vector Search
        query_embedding = (await self.embedding_provider.embed([query]))[0]

        results = await self.vectordb_provider.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            top_k=top_k * 3,
            filter_conditions={"user_id": user_id}
        )

        candidates = [
            {
                "item_id": r.payload["item_id"],
                "text": r.payload["text"],
                "score": r.score,
                "product": r.payload["product"],
                "category": r.payload["category"],
                "rating": r.payload["rating"]
            }
            for r in results
        ]

        # 2. Reranking
        if self.reranker_provider and candidates:
            try:
                rerank_results = await self.reranker_provider.rerank(
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
                print(f"   Reranking 오류: {e}")
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        latency = (time.time() - start_time) * 1000
        return candidates, latency

    async def search_3stage(self, query: str, user_id: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """3-stage: Vector Search → BM25 Hybrid → Reranking"""
        start_time = time.time()

        # 1. Vector Search
        query_embedding = (await self.embedding_provider.embed([query]))[0]

        results = await self.vectordb_provider.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            top_k=top_k * 3,
            filter_conditions={"user_id": user_id}
        )

        vector_candidates = {
            r.payload["item_id"]: {
                "item_id": r.payload["item_id"],
                "text": r.payload["text"],
                "vector_score": r.score,
                "product": r.payload["product"],
                "category": r.payload["category"],
                "rating": r.payload["rating"]
            }
            for r in results
        }

        # 2. BM25 Hybrid
        bm25_results = self.bm25.search(query, top_k=top_k * 3)
        user_doc_indices = [i for i, doc_id in enumerate(self.document_ids) if doc_id.startswith(user_id)]
        bm25_filtered = [(idx, score) for idx, score in bm25_results if idx in user_doc_indices]

        if bm25_filtered:
            max_bm25 = max(score for _, score in bm25_filtered) or 1
            for idx, bm25_score in bm25_filtered:
                item_id = self.document_ids[idx]
                if item_id in vector_candidates:
                    vector_candidates[item_id]["bm25_score"] = bm25_score / max_bm25

        # Hybrid score
        for item_id, candidate in vector_candidates.items():
            vector_score = candidate.get("vector_score", 0)
            bm25_score = candidate.get("bm25_score", 0)
            candidate["hybrid_score"] = 0.7 * vector_score + 0.3 * bm25_score

        candidates = sorted(vector_candidates.values(), key=lambda x: x.get("hybrid_score", 0), reverse=True)[:top_k * 2]

        # 3. Reranking
        if self.reranker_provider and candidates:
            try:
                rerank_results = await self.reranker_provider.rerank(
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
                print(f"   Reranking 오류: {e}")
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        latency = (time.time() - start_time) * 1000
        return candidates, latency

    async def search_rrf_hybrid(self, query: str, user_id: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """RRF Hybrid: Vector + BM25 → RRF Fusion → Reranking"""
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
        if self.reranker_provider and candidates:
            try:
                rerank_results = await self.reranker_provider.rerank(
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
                print(f"   Reranking 오류: {e}")
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        latency = (time.time() - start_time) * 1000
        return candidates, latency

    async def search_4stage(self, query: str, user_id: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """4-stage: Query Expansion → Vector Search → RRF Fusion → Reranking"""
        start_time = time.time()

        # 0. Query Expansion
        queries_to_search = [query]
        if self.search_enhancer:
            try:
                expanded = await self.search_enhancer.expand_query(query)
                queries_to_search = expanded.get_all_queries()
            except Exception as e:
                print(f"   Query expansion 오류: {e}")

        # 1. Vector Search for each expanded query
        all_vector_results = []
        for q in queries_to_search:
            q_embedding = (await self.embedding_provider.embed([q]))[0]
            results = await self.vectordb_provider.search(
                collection_name=self.collection_name,
                query_vector=q_embedding,
                top_k=top_k * 2,
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
            if vector_candidates:
                all_vector_results.append(vector_candidates)

        # 2. RRF Fusion across expanded queries
        if len(all_vector_results) > 1:
            candidates = self.rrf.fuse(all_vector_results, id_key="id", top_k=top_k * 2)
        elif all_vector_results:
            candidates = all_vector_results[0][:top_k * 2]
        else:
            return [], (time.time() - start_time) * 1000

        # 3. BM25 + RRF for hybrid
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

        if bm25_candidates:
            candidates = self.rrf.fuse_with_scores(
                candidates, bm25_candidates, id_key="id", top_k=top_k * 2
            )

        # 4. Reranking
        if self.reranker_provider and candidates:
            try:
                rerank_results = await self.reranker_provider.rerank(
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
                print(f"   Reranking 오류: {e}")
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        latency = (time.time() - start_time) * 1000
        return candidates, latency

    async def evaluate(self, queries: List[LaMP_Query], include_enhanced: bool = True) -> Dict[str, Any]:
        """벤치마크 평가"""
        results = {
            "2-stage": {"hits": [], "mrr": [], "ndcg": [], "precision": [], "recall": [], "latency": []},
            "3-stage": {"hits": [], "mrr": [], "ndcg": [], "precision": [], "recall": [], "latency": []},
            "rrf-hybrid": {"hits": [], "mrr": [], "ndcg": [], "precision": [], "recall": [], "latency": []},
            "4-stage": {"hits": [], "mrr": [], "ndcg": [], "precision": [], "recall": [], "latency": []}
        }

        total = len(queries)

        for idx, query in enumerate(queries):
            print(f"\r평가 중... {idx+1}/{total}", end="", flush=True)
            relevant_ids = set(query.relevant_profile_ids)

            # 2-stage
            results_2, latency_2 = await self.search_2stage(query.query_text, query.user_id)
            retrieved_2 = [r["item_id"] for r in results_2]
            metrics_2 = self._calculate_metrics(retrieved_2, relevant_ids)
            for key, value in metrics_2.items():
                results["2-stage"][key].append(value)
            results["2-stage"]["latency"].append(latency_2)

            # 3-stage
            results_3, latency_3 = await self.search_3stage(query.query_text, query.user_id)
            retrieved_3 = [r["item_id"] for r in results_3]
            metrics_3 = self._calculate_metrics(retrieved_3, relevant_ids)
            for key, value in metrics_3.items():
                results["3-stage"][key].append(value)
            results["3-stage"]["latency"].append(latency_3)

            # RRF Hybrid (NEW)
            if include_enhanced:
                results_rrf, latency_rrf = await self.search_rrf_hybrid(query.query_text, query.user_id)
                retrieved_rrf = [r["item_id"] for r in results_rrf]
                metrics_rrf = self._calculate_metrics(retrieved_rrf, relevant_ids)
                for key, value in metrics_rrf.items():
                    results["rrf-hybrid"][key].append(value)
                results["rrf-hybrid"]["latency"].append(latency_rrf)

                # 4-stage (Query Expansion + RRF) (NEW)
                results_4, latency_4 = await self.search_4stage(query.query_text, query.user_id)
                retrieved_4 = [r["item_id"] for r in results_4]
                metrics_4 = self._calculate_metrics(retrieved_4, relevant_ids)
                for key, value in metrics_4.items():
                    results["4-stage"][key].append(value)
                results["4-stage"]["latency"].append(latency_4)

        print("\n")

        # 평균 계산
        summary = {}
        methods = ["2-stage", "3-stage"]
        if include_enhanced:
            methods.extend(["rrf-hybrid", "4-stage"])

        for method in methods:
            if results[method]["hits"]:  # 결과가 있는 경우만
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


async def main():
    print("=" * 70)
    print("LaMP (Language Model Personalization) Benchmark v2")
    print("Provider 패턴 기반 개인화 시스템 평가")
    print("=" * 70)
    print()

    benchmark = LaMP_Benchmark_V2()

    # 초기화
    await benchmark.initialize()

    # 데이터 생성
    print("📊 LaMP 데이터 생성 중...")
    profiles, queries = benchmark.generate_lamp_data(num_users=10, items_per_user=20)
    print(f"   - 사용자: {len(profiles)}")
    print(f"   - 프로필 아이템: {sum(len(p.profile_items) for p in profiles)}")
    print(f"   - 쿼리: {len(queries)}")
    print()

    # 인덱싱
    await benchmark.setup_collection()
    await benchmark.index_profiles(profiles)
    print()

    # 평가
    print("🧪 벤치마크 평가 실행...")
    print("-" * 70)
    results = await benchmark.evaluate(queries)

    # 결과 출력
    print("=" * 70)
    print("📈 LaMP Benchmark 결과")
    print("=" * 70)
    print()

    # 4개 파이프라인 비교 테이블
    methods = list(results.keys())
    has_enhanced = "4-stage" in results

    if has_enhanced:
        print("┌─────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
        print("│ Metric              │ 2-stage      │ 3-stage      │ rrf-hybrid   │ 4-stage      │")
        print("├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
    else:
        print("┌─────────────────────┬──────────────────┬──────────────────┐")
        print("│ Metric              │ 2-stage          │ 3-stage          │")
        print("├─────────────────────┼──────────────────┼──────────────────┤")

    metrics = [
        ("Hit Rate@1", "hit_rate@1", "{:.1%}"),
        ("MRR", "mrr", "{:.3f}"),
        ("NDCG@5", "ndcg@5", "{:.3f}"),
        ("Precision@5", "precision@5", "{:.3f}"),
        ("Recall@5", "recall@5", "{:.3f}"),
        ("Avg Latency (ms)", "avg_latency_ms", "{:.1f}"),
        ("P95 Latency (ms)", "p95_latency_ms", "{:.1f}")
    ]

    for label, key, fmt in metrics:
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
            if has_enhanced:
                row += f" {val_str:>12} │"
            else:
                row += f" {val_str:>16} │"
        print(row)

    if has_enhanced:
        print("└─────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
    else:
        print("└─────────────────────┴──────────────────┴──────────────────┘")
    print()

    # 3-stage 대비 개선율 계산
    baseline = results["3-stage"]
    print("📊 3-stage 대비 개선율:")
    for method in methods:
        if method == "3-stage":
            continue
        r = results[method]
        ndcg_diff = ((r["ndcg@5"] - baseline["ndcg@5"]) / baseline["ndcg@5"]) * 100
        latency_diff = ((baseline["avg_latency_ms"] - r["avg_latency_ms"]) / baseline["avg_latency_ms"]) * 100
        print(f"   {method}: NDCG {ndcg_diff:+.1f}%, Latency {latency_diff:+.1f}%")
    print()

    print("📋 파이프라인 구성:")
    print("   2-stage:    Vector Search → Reranking")
    print("   3-stage:    Vector Search → BM25 Hybrid → Reranking")
    print("   rrf-hybrid: Vector + BM25 → RRF Fusion → Reranking (NEW)")
    print("   4-stage:    Query Expansion → Vector → RRF Fusion → Reranking (NEW)")
    print()

    print("🔧 Provider 설정:")
    print(f"   - Embedding: {settings.providers.embedding_provider} ({settings.providers.embedding_model})")
    print(f"   - VectorDB: {settings.providers.vectordb_provider}")
    print(f"   - Reranker: {settings.providers.reranker_provider}")
    print()

    # 결과 저장
    output_file = "/tmp/lamp_benchmark_v2_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "LaMP-v2",
            "timestamp": datetime.now().isoformat(),
            "providers": {
                "embedding": settings.providers.embedding_provider,
                "vectordb": settings.providers.vectordb_provider,
                "reranker": settings.providers.reranker_provider
            },
            "config": {
                "num_users": len(profiles),
                "items_per_user": 20,
                "num_queries": len(queries)
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"📁 결과 저장: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
