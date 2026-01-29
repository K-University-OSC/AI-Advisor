"""
LaMP (Language Model Personalization) Benchmark 평가 스크립트
2-stage vs 3-stage 파이프라인 비교

LaMP 벤치마크 태스크:
- LaMP-1: Personalized Citation Identification (논문 인용 예측)
- LaMP-2: Personalized News Categorization (뉴스 카테고리 분류)
- LaMP-3: Personalized Product Rating (제품 평점 예측)
- LaMP-4: Personalized News Headline Generation (뉴스 헤드라인 생성)
- LaMP-5: Personalized Scholarly Title Generation (논문 제목 생성)
- LaMP-6: Personalized Email Subject Generation (이메일 제목 생성)
- LaMP-7: Personalized Tweet Paraphrasing (트윗 패러프레이징)

이 스크립트는 LaMP 스타일의 개인화 검색 태스크를 시뮬레이션합니다.
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
import numpy as np

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchParams
)

# Cohere for reranking
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    print("⚠️ Cohere not installed. Reranking will be disabled.")


@dataclass
class LaMP_Profile:
    """LaMP 사용자 프로필 데이터"""
    user_id: str
    profile_items: List[Dict[str, Any]]  # 과거 활동 기록
    task_type: str  # LaMP-1 ~ LaMP-7


@dataclass
class LaMP_Query:
    """LaMP 쿼리 데이터"""
    query_id: str
    user_id: str
    query_text: str
    task_type: str
    ground_truth: str  # 정답
    relevant_profile_ids: List[str]  # 관련 프로필 아이템 ID


class BM25:
    """BM25 검색 알고리즘 구현"""

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
        """간단한 토크나이저"""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def fit(self, documents: List[str]):
        """문서 컬렉션 인덱싱"""
        self.documents = documents
        self.corpus_size = len(documents)
        self.tokenized_docs = [self.tokenize(doc) for doc in documents]

        # 문서 길이 계산
        self.doc_lens = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_len = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0

        # 문서 빈도 계산
        self.doc_freqs = defaultdict(int)
        for doc in self.tokenized_docs:
            for term in set(doc):
                self.doc_freqs[term] += 1

    def get_scores(self, query: str) -> List[float]:
        """쿼리에 대한 모든 문서의 BM25 점수 계산"""
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
        """상위 k개 문서 검색"""
        scores = self.get_scores(query)
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]


class LaMP_Benchmark:
    """LaMP 벤치마크 평가 클래스"""

    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.collection_name = "lamp_benchmark"
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dim = 3072

        # Cohere client for reranking
        self.cohere_client = None
        if COHERE_AVAILABLE and os.getenv("COHERE_API_KEY"):
            self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
            print("✅ Cohere Reranking 활성화")
        else:
            print("⚠️ Cohere Reranking 비활성화")

        # BM25 인덱스
        self.bm25 = BM25()
        self.documents = []
        self.document_ids = []

    def generate_lamp_data(self, num_users: int = 10, items_per_user: int = 20) -> Tuple[List[LaMP_Profile], List[LaMP_Query]]:
        """
        LaMP 스타일의 개인화 데이터 생성

        LaMP 태스크 시뮬레이션:
        - LaMP-3 스타일: 제품 리뷰/평점 예측
        - LaMP-4 스타일: 콘텐츠 제목 생성
        - LaMP-6 스타일: 이메일/메시지 스타일
        """

        profiles = []
        queries = []

        # 사용자별 관심사/스타일 정의
        user_personas = [
            {
                "name": "tech_enthusiast",
                "interests": ["AI", "machine learning", "programming", "tech gadgets"],
                "style": "technical",
                "preferred_products": ["laptop", "smartphone", "headphones", "smartwatch"],
                "rating_tendency": "critical"  # 평균 3.5/5
            },
            {
                "name": "casual_user",
                "interests": ["movies", "music", "travel", "food"],
                "style": "casual",
                "preferred_products": ["camera", "speakers", "travel gear", "kitchen appliances"],
                "rating_tendency": "generous"  # 평균 4.2/5
            },
            {
                "name": "professional",
                "interests": ["productivity", "business", "finance", "networking"],
                "style": "formal",
                "preferred_products": ["office equipment", "professional software", "books", "courses"],
                "rating_tendency": "balanced"  # 평균 3.8/5
            },
            {
                "name": "creative",
                "interests": ["art", "design", "photography", "music production"],
                "style": "expressive",
                "preferred_products": ["graphics tablet", "camera", "software", "instruments"],
                "rating_tendency": "enthusiastic"  # 평균 4.5/5
            },
            {
                "name": "student",
                "interests": ["studying", "budget products", "entertainment", "social"],
                "style": "informal",
                "preferred_products": ["textbooks", "laptop", "headphones", "snacks"],
                "rating_tendency": "varied"  # 2-5점 다양
            },
            {
                "name": "health_focused",
                "interests": ["fitness", "nutrition", "wellness", "outdoor activities"],
                "style": "motivational",
                "preferred_products": ["fitness tracker", "supplements", "workout gear", "healthy food"],
                "rating_tendency": "positive"
            },
            {
                "name": "minimalist",
                "interests": ["simple living", "quality over quantity", "sustainable products"],
                "style": "concise",
                "preferred_products": ["essential items", "quality tools", "durable goods"],
                "rating_tendency": "selective"
            },
            {
                "name": "gamer",
                "interests": ["video games", "esports", "gaming hardware", "streaming"],
                "style": "enthusiastic",
                "preferred_products": ["gaming PC", "monitor", "keyboard", "gaming chair"],
                "rating_tendency": "passionate"
            },
            {
                "name": "parent",
                "interests": ["family", "children", "home", "safety"],
                "style": "practical",
                "preferred_products": ["toys", "educational items", "home appliances", "safety equipment"],
                "rating_tendency": "thorough"
            },
            {
                "name": "senior",
                "interests": ["simplicity", "reliability", "health", "hobbies"],
                "style": "clear",
                "preferred_products": ["easy-to-use devices", "health monitors", "hobby supplies"],
                "rating_tendency": "appreciative"
            }
        ]

        # 제품 카테고리 및 예시
        product_templates = {
            "laptop": [
                "MacBook Pro 14인치 M3", "Dell XPS 15", "ThinkPad X1 Carbon",
                "ASUS ROG Zephyrus", "HP Spectre x360", "Surface Laptop 5"
            ],
            "smartphone": [
                "iPhone 15 Pro", "Samsung Galaxy S24 Ultra", "Google Pixel 8 Pro",
                "OnePlus 12", "Xiaomi 14 Pro"
            ],
            "headphones": [
                "Sony WH-1000XM5", "AirPods Pro 2", "Bose QuietComfort Ultra",
                "Sennheiser Momentum 4", "Audio-Technica ATH-M50x"
            ],
            "camera": [
                "Sony A7 IV", "Canon EOS R6 Mark II", "Nikon Z6 III",
                "Fujifilm X-T5", "Panasonic Lumix S5 II"
            ],
            "smartwatch": [
                "Apple Watch Ultra 2", "Samsung Galaxy Watch 6", "Garmin Fenix 7",
                "Fitbit Sense 2", "Google Pixel Watch 2"
            ],
            "keyboard": [
                "Keychron Q1 Pro", "Logitech MX Keys", "HHKB Professional",
                "Das Keyboard 4", "Corsair K100 RGB"
            ],
            "monitor": [
                "LG UltraFine 5K", "Dell UltraSharp U2723QE", "ASUS ProArt PA32UCG",
                "Samsung Odyssey G9", "BenQ PD3220U"
            ],
            "books": [
                "Atomic Habits", "Deep Work", "The Psychology of Money",
                "Clean Code", "Thinking, Fast and Slow"
            ]
        }

        # 리뷰 템플릿
        review_templates = {
            "technical": [
                "성능 벤치마크 결과 {product}는 {metric}에서 {score}를 기록했습니다. {detail}",
                "기술적 관점에서 {product}의 {feature}는 {assessment}. 특히 {highlight}가 인상적입니다.",
                "{product} 사용 후 {duration} 경과. {technical_analysis}. 종합 평점: {rating}/5"
            ],
            "casual": [
                "{product} 진짜 좋아요! {reason} 덕분에 {benefit}. 강추!",
                "이거 사길 잘했다~ {product} {positive}하고 {positive2}해서 만족!",
                "{product} 쓴 지 {duration}됐는데 {experience}. {conclusion}"
            ],
            "formal": [
                "{product}에 대한 평가입니다. {overview}. 장점: {pros}. 단점: {cons}. 결론: {verdict}",
                "업무용으로 {product}를 {duration} 사용했습니다. {professional_assessment}",
                "{product}의 가성비를 분석하면 {analysis}. 추천 대상: {target}"
            ],
            "expressive": [
                "와! {product} 완전 사랑해요 💕 {emotional_response} {creative_use}",
                "{product}로 {creative_work} 했는데 결과물이 {result}! {enthusiasm}",
                "예술가 관점에서 {product}는 {artistic_assessment}. 영감을 주는 제품!"
            ],
            "informal": [
                "ㅋㅋ {product} 가성비 미쳤음 {benefit} {slang_positive}",
                "{product} 솔직 후기: {honest_opinion} 근데 {but} {conclusion}",
                "학생 입장에서 {product}는 {student_perspective} {emoji}"
            ]
        }

        for user_idx in range(min(num_users, len(user_personas))):
            persona = user_personas[user_idx]
            user_id = f"user_{user_idx + 1}"

            profile_items = []

            # 사용자별 프로필 아이템 생성 (과거 리뷰/활동)
            for item_idx in range(items_per_user):
                # 제품 카테고리 선택 (사용자 선호 반영)
                if random.random() < 0.7:  # 70% 확률로 선호 카테고리
                    category = random.choice(persona["preferred_products"])
                    if category not in product_templates:
                        category = random.choice(list(product_templates.keys()))
                else:
                    category = random.choice(list(product_templates.keys()))

                product = random.choice(product_templates.get(category, product_templates["laptop"]))

                # 평점 결정 (사용자 경향 반영)
                if persona["rating_tendency"] == "critical":
                    rating = random.choices([2, 3, 4, 5], weights=[10, 30, 40, 20])[0]
                elif persona["rating_tendency"] == "generous":
                    rating = random.choices([3, 4, 5], weights=[10, 40, 50])[0]
                elif persona["rating_tendency"] == "enthusiastic":
                    rating = random.choices([4, 5], weights=[30, 70])[0]
                elif persona["rating_tendency"] == "varied":
                    rating = random.randint(2, 5)
                else:
                    rating = random.choices([3, 4, 5], weights=[20, 50, 30])[0]

                # 리뷰 생성
                style = persona["style"]
                template = random.choice(review_templates.get(style, review_templates["casual"]))

                review = self._generate_review(template, product, rating, persona)

                profile_item = {
                    "id": f"{user_id}_item_{item_idx}",
                    "product": product,
                    "category": category,
                    "rating": rating,
                    "review": review,
                    "date": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                    "style": style
                }
                profile_items.append(profile_item)

            profiles.append(LaMP_Profile(
                user_id=user_id,
                profile_items=profile_items,
                task_type="LaMP-3"  # Product rating prediction
            ))

            # 쿼리 생성 (새 제품에 대한 평점 예측)
            for q_idx in range(3):  # 사용자당 3개 쿼리
                # 관련 있는 카테고리에서 새 제품 선택
                query_category = random.choice(persona["preferred_products"])
                if query_category not in product_templates:
                    query_category = random.choice(list(product_templates.keys()))

                query_product = random.choice(product_templates[query_category])

                # 실제 관련 프로필 아이템 찾기 (같은 카테고리)
                relevant_ids = [
                    item["id"] for item in profile_items
                    if item["category"] == query_category
                ][:5]

                # 예상 평점 계산 (관련 아이템 평균)
                related_ratings = [
                    item["rating"] for item in profile_items
                    if item["category"] == query_category
                ]
                expected_rating = round(sum(related_ratings) / len(related_ratings)) if related_ratings else 4

                query = LaMP_Query(
                    query_id=f"{user_id}_query_{q_idx}",
                    user_id=user_id,
                    query_text=f"{query_product}에 대해 이 사용자는 어떤 평점을 줄까요? 사용자의 과거 리뷰 스타일과 선호도를 고려하세요.",
                    task_type="LaMP-3",
                    ground_truth=str(expected_rating),
                    relevant_profile_ids=relevant_ids
                )
                queries.append(query)

        return profiles, queries

    def _generate_review(self, template: str, product: str, rating: int, persona: Dict) -> str:
        """리뷰 텍스트 생성"""
        # 간단한 템플릿 채우기
        replacements = {
            "{product}": product,
            "{rating}": str(rating),
            "{duration}": random.choice(["1주일", "2주", "한 달", "3개월", "6개월"]),
            "{metric}": random.choice(["속도", "배터리", "성능", "품질"]),
            "{score}": random.choice(["우수", "양호", "평균 이상", "최상위권"]),
            "{detail}": random.choice(["전반적으로 만족", "일부 개선 필요", "기대 이상"]),
            "{feature}": random.choice(["디자인", "성능", "가격", "내구성"]),
            "{assessment}": random.choice(["훌륭합니다", "괜찮습니다", "아쉽습니다"]),
            "{highlight}": random.choice(["마감", "속도", "편의성", "디자인"]),
            "{technical_analysis}": "성능 대비 가격 합리적",
            "{reason}": random.choice(["디자인", "성능", "가격"]),
            "{benefit}": random.choice(["매일 사용 중", "완전 편해짐", "생산성 향상"]),
            "{positive}": random.choice(["예쁘", "빠르", "편하"]),
            "{positive2}": random.choice(["가벼워", "조용해", "오래가"]),
            "{experience}": "만족스럽게 사용 중",
            "{conclusion}": random.choice(["추천!", "괜찮아요", "가성비 좋음"]),
            "{overview}": "전반적으로 우수한 제품",
            "{pros}": "성능, 디자인",
            "{cons}": "가격",
            "{verdict}": "추천함",
            "{professional_assessment}": "업무 효율성 향상에 도움",
            "{analysis}": "가격 대비 성능 우수",
            "{target}": "전문가/일반 사용자",
            "{emotional_response}": "너무 좋아서 행복해요",
            "{creative_use}": "창작 활동에 활용 중",
            "{creative_work}": "작품",
            "{result}": "대만족",
            "{enthusiasm}": "최고!",
            "{artistic_assessment}": "영감을 주는 도구",
            "{honest_opinion}": "나쁘지 않음",
            "{but}": "가격이 좀...",
            "{student_perspective}": "학생 예산엔 부담",
            "{slang_positive}": "개이득",
            "{emoji}": "👍"
        }

        review = template
        for key, value in replacements.items():
            review = review.replace(key, value)

        return review

    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
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

    def index_profiles(self, profiles: List[LaMP_Profile]):
        """프로필 데이터 인덱싱"""
        points = []
        self.documents = []
        self.document_ids = []

        for profile in profiles:
            for item in profile.profile_items:
                # 검색용 텍스트 구성
                text = f"제품: {item['product']} | 카테고리: {item['category']} | 평점: {item['rating']}/5 | 리뷰: {item['review']}"

                embedding = self.get_embedding(text)

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "item_id": item["id"],
                        "user_id": profile.user_id,
                        "product": item["product"],
                        "category": item["category"],
                        "rating": item["rating"],
                        "review": item["review"],
                        "text": text,
                        "style": item["style"]
                    }
                )
                points.append(point)
                self.documents.append(text)
                self.document_ids.append(item["id"])

        # Qdrant에 업로드
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points[i:i+batch_size]
            )

        # BM25 인덱스 구축
        self.bm25.fit(self.documents)

        print(f"✅ {len(points)}개 프로필 아이템 인덱싱 완료")

    def search_before(self, query: str, user_id: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """
        Before (2-stage): Vector Search → Cohere Reranking
        """
        start_time = time.time()

        # 1단계: Vector Search
        query_embedding = self.get_embedding(query)

        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k * 3
        )
        results = search_result.points

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

        # 2단계: Cohere Reranking
        if self.cohere_client and candidates:
            try:
                # Rate limiting for Cohere Trial API (10 calls/min)
                time.sleep(3.5)  # ~17 calls/min max, safe margin

                rerank_response = self.cohere_client.rerank(
                    model="rerank-v3.5",
                    query=query,
                    documents=[c["text"] for c in candidates],
                    top_n=top_k
                )

                reranked = []
                for r in rerank_response.results:
                    candidate = candidates[r.index]
                    candidate["rerank_score"] = r.relevance_score
                    reranked.append(candidate)
                candidates = reranked
            except Exception as e:
                if "429" not in str(e):
                    print(f"Reranking error: {e}")
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        latency = (time.time() - start_time) * 1000
        return candidates, latency

    def search_after(self, query: str, user_id: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """
        After (3-stage): Vector Search → BM25 Hybrid → Cohere Reranking
        """
        start_time = time.time()

        # 1단계: Vector Search
        query_embedding = self.get_embedding(query)

        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k * 3
        )
        vector_results = search_result.points

        vector_candidates = {
            r.payload["item_id"]: {
                "item_id": r.payload["item_id"],
                "text": r.payload["text"],
                "vector_score": r.score,
                "product": r.payload["product"],
                "category": r.payload["category"],
                "rating": r.payload["rating"]
            }
            for r in vector_results
        }

        # 2단계: BM25 Hybrid
        bm25_results = self.bm25.search(query, top_k=top_k * 3)

        # 해당 사용자의 문서만 필터링
        user_doc_indices = [
            i for i, doc_id in enumerate(self.document_ids)
            if doc_id.startswith(user_id)
        ]

        bm25_filtered = [
            (idx, score) for idx, score in bm25_results
            if idx in user_doc_indices
        ]

        # BM25 점수 정규화
        if bm25_filtered:
            max_bm25 = max(score for _, score in bm25_filtered) if bm25_filtered else 1
            for idx, bm25_score in bm25_filtered:
                item_id = self.document_ids[idx]
                if item_id in vector_candidates:
                    vector_candidates[item_id]["bm25_score"] = bm25_score / max_bm25 if max_bm25 > 0 else 0

        # 하이브리드 점수 계산 (Vector 0.7, BM25 0.3)
        for item_id, candidate in vector_candidates.items():
            vector_score = candidate.get("vector_score", 0)
            bm25_score = candidate.get("bm25_score", 0)
            candidate["hybrid_score"] = 0.7 * vector_score + 0.3 * bm25_score

        # 하이브리드 점수로 정렬
        candidates = sorted(
            vector_candidates.values(),
            key=lambda x: x.get("hybrid_score", 0),
            reverse=True
        )[:top_k * 2]

        # 3단계: Cohere Reranking
        if self.cohere_client and candidates:
            try:
                # Rate limiting for Cohere Trial API (10 calls/min)
                time.sleep(3.5)  # ~17 calls/min max, safe margin

                rerank_response = self.cohere_client.rerank(
                    model="rerank-v3.5",
                    query=query,
                    documents=[c["text"] for c in candidates],
                    top_n=top_k
                )

                reranked = []
                for r in rerank_response.results:
                    candidate = candidates[r.index]
                    candidate["rerank_score"] = r.relevance_score
                    reranked.append(candidate)
                candidates = reranked
            except Exception as e:
                if "429" not in str(e):
                    print(f"Reranking error: {e}")
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        latency = (time.time() - start_time) * 1000
        return candidates, latency

    def evaluate(self, queries: List[LaMP_Query], profiles: List[LaMP_Profile]) -> Dict[str, Any]:
        """벤치마크 평가 실행"""
        results = {
            "before": {"hits": [], "mrr": [], "ndcg": [], "precision": [], "recall": [], "latency": []},
            "after": {"hits": [], "mrr": [], "ndcg": [], "precision": [], "recall": [], "latency": []}
        }

        total = len(queries)

        for idx, query in enumerate(queries):
            print(f"\r평가 중... {idx+1}/{total}", end="", flush=True)

            relevant_ids = set(query.relevant_profile_ids)

            # Before (2-stage) 평가
            before_results, before_latency = self.search_before(query.query_text, query.user_id)
            before_retrieved = [r["item_id"] for r in before_results]

            before_metrics = self._calculate_metrics(before_retrieved, relevant_ids)
            results["before"]["hits"].append(before_metrics["hit@1"])
            results["before"]["mrr"].append(before_metrics["mrr"])
            results["before"]["ndcg"].append(before_metrics["ndcg@5"])
            results["before"]["precision"].append(before_metrics["precision@5"])
            results["before"]["recall"].append(before_metrics["recall@5"])
            results["before"]["latency"].append(before_latency)

            # After (3-stage) 평가
            after_results, after_latency = self.search_after(query.query_text, query.user_id)
            after_retrieved = [r["item_id"] for r in after_results]

            after_metrics = self._calculate_metrics(after_retrieved, relevant_ids)
            results["after"]["hits"].append(after_metrics["hit@1"])
            results["after"]["mrr"].append(after_metrics["mrr"])
            results["after"]["ndcg"].append(after_metrics["ndcg@5"])
            results["after"]["precision"].append(after_metrics["precision@5"])
            results["after"]["recall"].append(after_metrics["recall@5"])
            results["after"]["latency"].append(after_latency)

        print("\n")

        # 평균 계산
        summary = {}
        for method in ["before", "after"]:
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
        """검색 메트릭 계산"""
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


async def main():
    print("=" * 70)
    print("LaMP (Language Model Personalization) Benchmark")
    print("2-stage vs 3-stage 파이프라인 비교")
    print("=" * 70)
    print()

    benchmark = LaMP_Benchmark()

    # 1. 데이터 생성
    print("📊 LaMP 스타일 개인화 데이터 생성 중...")
    profiles, queries = benchmark.generate_lamp_data(num_users=10, items_per_user=20)
    print(f"   - 사용자 수: {len(profiles)}")
    print(f"   - 총 프로필 아이템: {sum(len(p.profile_items) for p in profiles)}")
    print(f"   - 쿼리 수: {len(queries)}")
    print()

    # 2. 인덱싱
    print("🔧 Qdrant 컬렉션 설정 및 인덱싱...")
    benchmark.setup_collection()
    benchmark.index_profiles(profiles)
    print()

    # 3. 평가 실행
    print("🧪 벤치마크 평가 실행...")
    print("-" * 70)
    results = benchmark.evaluate(queries, profiles)

    # 4. 결과 출력
    print("=" * 70)
    print("📈 LaMP Benchmark 결과")
    print("=" * 70)
    print()

    print("┌─────────────────────┬──────────────────┬──────────────────┐")
    print("│ Metric              │ Before (2-stage) │ After (3-stage)  │")
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
        before_val = results["before"][key]
        after_val = results["after"][key]

        if key.endswith("latency_ms"):
            # 낮을수록 좋음
            diff = before_val - after_val
            indicator = "⬇️" if diff > 0 else "⬆️"
        else:
            # 높을수록 좋음
            diff = after_val - before_val
            indicator = "⬆️" if diff > 0 else "⬇️" if diff < 0 else "➡️"

        before_str = fmt.format(before_val)
        after_str = fmt.format(after_val) + f" {indicator}"

        print(f"│ {label:<19} │ {before_str:>16} │ {after_str:>16} │")

    print("└─────────────────────┴──────────────────┴──────────────────┘")
    print()

    # 파이프라인 설명
    print("📋 파이프라인 구성:")
    print("   Before (2-stage): Vector Search → Cohere Reranking")
    print("   After (3-stage):  Vector Search → BM25 Hybrid → Cohere Reranking")
    print()

    # 분석
    print("📊 분석:")
    hit_diff = (results["after"]["hit_rate@1"] - results["before"]["hit_rate@1"]) * 100
    mrr_diff = (results["after"]["mrr"] - results["before"]["mrr"]) * 100
    latency_diff = results["after"]["avg_latency_ms"] - results["before"]["avg_latency_ms"]

    if hit_diff > 0:
        print(f"   ✅ Hit Rate@1이 {hit_diff:.1f}%p 향상되었습니다.")
    elif hit_diff < 0:
        print(f"   ⚠️ Hit Rate@1이 {abs(hit_diff):.1f}%p 감소했습니다.")
    else:
        print(f"   ➡️ Hit Rate@1은 동일합니다.")

    if mrr_diff > 0:
        print(f"   ✅ MRR이 {mrr_diff:.1f}%p 향상되었습니다.")
    elif mrr_diff < 0:
        print(f"   ⚠️ MRR이 {abs(mrr_diff):.1f}%p 감소했습니다.")

    print(f"   ⏱️ Latency는 {latency_diff:+.1f}ms 차이납니다.")
    print()

    print("💡 LaMP 벤치마크 특성:")
    print("   - 개인화 검색 태스크로, 사용자별 과거 활동 기록을 기반으로 검색")
    print("   - 제품 카테고리/키워드 기반 필터링이 중요한 태스크")
    print("   - BM25가 카테고리/제품명 매칭에서 강점을 보일 수 있음")
    print()

    # 결과 저장
    output_file = "/tmp/lamp_benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "LaMP",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_users": len(profiles),
                "items_per_user": 20,
                "num_queries": len(queries),
                "embedding_model": "text-embedding-3-large",
                "reranking": "cohere/rerank-v3.5"
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"📁 결과 저장: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
