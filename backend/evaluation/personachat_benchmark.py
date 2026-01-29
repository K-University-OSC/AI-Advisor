"""
PersonaChat Benchmark 평가 스크립트
2-stage vs 3-stage 파이프라인 비교

PersonaChat (Facebook AI Research):
- 페르소나 기반 대화 데이터셋
- 각 대화자가 4-5개의 페르소나 문장을 가짐
- 페르소나를 기반으로 일관된 대화를 생성
- 검색 증강 생성(RAG)에서 올바른 페르소나 검색이 핵심

평가 태스크: 주어진 대화 컨텍스트에서 관련 페르소나 문장 검색
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
    Filter, FieldCondition, MatchValue
)

# Cohere for reranking
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    print("⚠️ Cohere not installed. Reranking will be disabled.")


@dataclass
class Persona:
    """페르소나 데이터"""
    persona_id: str
    user_id: str
    statements: List[str]  # 페르소나 문장들 (4-5개)


@dataclass
class PersonaChatQuery:
    """PersonaChat 쿼리"""
    query_id: str
    user_id: str
    dialogue_context: str  # 대화 컨텍스트
    query_utterance: str   # 현재 발화
    relevant_persona_ids: List[str]  # 관련 페르소나 문장 인덱스


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
        tokens = re.findall(r'[가-힣]+|[a-z0-9]+', text)
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


class PersonaChatBenchmark:
    """PersonaChat 벤치마크 평가"""

    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.collection_name = "personachat_benchmark"
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dim = 3072

        self.cohere_client = None
        if COHERE_AVAILABLE and os.getenv("COHERE_API_KEY"):
            self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
            print("✅ Cohere Reranking 활성화")
        else:
            print("⚠️ Cohere Reranking 비활성화")

        self.bm25 = BM25()
        self.documents = []
        self.document_ids = []

    def generate_personachat_data(self, num_users: int = 15) -> Tuple[List[Persona], List[PersonaChatQuery]]:
        """
        PersonaChat 스타일의 페르소나 기반 대화 데이터 생성
        """
        personas = []
        queries = []

        # 다양한 페르소나 템플릿
        persona_templates = [
            {
                "theme": "developer",
                "statements": [
                    "나는 소프트웨어 개발자로 일하고 있어요",
                    "Python과 JavaScript를 주로 사용해요",
                    "주말에는 오픈소스 프로젝트에 기여해요",
                    "커피를 마시면서 코딩하는 것을 좋아해요",
                    "최근에 AI와 머신러닝에 관심이 많아요"
                ],
                "dialogues": [
                    ("어떤 일을 하세요?", [0]),
                    ("프로그래밍 언어 뭐 써요?", [1]),
                    ("취미가 뭐예요?", [2, 3]),
                    ("요즘 뭐에 관심있어요?", [4]),
                    ("개발할 때 뭐 마셔요?", [3])
                ]
            },
            {
                "theme": "student",
                "statements": [
                    "저는 대학에서 컴퓨터공학을 전공하고 있어요",
                    "올해 졸업 예정이에요",
                    "동아리에서 앱 개발을 하고 있어요",
                    "장학금을 받으며 공부하고 있어요",
                    "졸업 후에는 대기업에 취업하고 싶어요"
                ],
                "dialogues": [
                    ("전공이 뭐예요?", [0]),
                    ("몇 학년이에요?", [1]),
                    ("동아리 활동 하세요?", [2]),
                    ("학비는 어떻게 해결해요?", [3]),
                    ("졸업 후 계획이 있어요?", [4])
                ]
            },
            {
                "theme": "traveler",
                "statements": [
                    "여행을 정말 좋아해요",
                    "작년에 유럽 5개국을 다녀왔어요",
                    "다음 목표는 남미 여행이에요",
                    "사진 찍는 것을 좋아해서 여행 중 많이 찍어요",
                    "현지 음식 먹는 것이 여행의 묘미라고 생각해요"
                ],
                "dialogues": [
                    ("취미가 뭐예요?", [0]),
                    ("최근에 어디 다녀왔어요?", [1]),
                    ("다음에 어디 가고 싶어요?", [2]),
                    ("여행 중에 뭐 해요?", [3, 4]),
                    ("여행의 재미가 뭐예요?", [4])
                ]
            },
            {
                "theme": "chef",
                "statements": [
                    "요리사로 일하고 있어요",
                    "이탈리안 요리를 전문으로 해요",
                    "나만의 레스토랑을 여는 것이 꿈이에요",
                    "신선한 재료에 집착하는 편이에요",
                    "주말에는 집에서 새로운 레시피를 실험해요"
                ],
                "dialogues": [
                    ("무슨 일 하세요?", [0]),
                    ("어떤 요리를 주로 해요?", [1]),
                    ("꿈이 뭐예요?", [2]),
                    ("요리할 때 중요하게 생각하는 것은?", [3]),
                    ("쉬는 날에는 뭐 해요?", [4])
                ]
            },
            {
                "theme": "musician",
                "statements": [
                    "기타를 치는 것을 좋아해요",
                    "밴드에서 기타리스트로 활동하고 있어요",
                    "주로 락과 블루스를 연주해요",
                    "음악은 10살 때부터 시작했어요",
                    "언젠가 앨범을 내고 싶어요"
                ],
                "dialogues": [
                    ("악기 할 줄 알아요?", [0]),
                    ("밴드 활동 하세요?", [1]),
                    ("어떤 장르를 좋아해요?", [2]),
                    ("음악은 언제부터 했어요?", [3]),
                    ("음악 관련 목표가 있어요?", [4])
                ]
            },
            {
                "theme": "fitness",
                "statements": [
                    "헬스장에서 웨이트 트레이닝을 해요",
                    "매일 아침 5시에 일어나서 운동해요",
                    "건강한 식단 관리도 함께 하고 있어요",
                    "마라톤 대회에 참가하는 것이 목표예요",
                    "운동 후 단백질 쉐이크를 꼭 마셔요"
                ],
                "dialogues": [
                    ("운동 하세요?", [0]),
                    ("언제 운동해요?", [1]),
                    ("식단도 관리해요?", [2]),
                    ("운동 목표가 있어요?", [3]),
                    ("운동 후에 뭐 먹어요?", [4])
                ]
            },
            {
                "theme": "gamer",
                "statements": [
                    "게임을 정말 좋아해요",
                    "FPS와 RPG 장르를 주로 해요",
                    "주말에는 친구들과 온라인으로 게임해요",
                    "e스포츠 경기 보는 것도 좋아해요",
                    "게임용 PC를 직접 조립했어요"
                ],
                "dialogues": [
                    ("취미가 뭐예요?", [0]),
                    ("어떤 게임 좋아해요?", [1]),
                    ("주말에 뭐 해요?", [2]),
                    ("e스포츠 관심 있어요?", [3]),
                    ("PC 사양이 어떻게 돼요?", [4])
                ]
            },
            {
                "theme": "artist",
                "statements": [
                    "그림 그리는 것을 좋아해요",
                    "주로 수채화와 유화를 그려요",
                    "전시회에 작품을 출품한 적이 있어요",
                    "자연 풍경을 그리는 것을 좋아해요",
                    "미술관 가는 것을 즐겨요"
                ],
                "dialogues": [
                    ("취미가 있어요?", [0]),
                    ("어떤 그림 그려요?", [1]),
                    ("전시회 해본 적 있어요?", [2]),
                    ("주로 뭘 그려요?", [3]),
                    ("주말에 뭐 해요?", [4])
                ]
            },
            {
                "theme": "reader",
                "statements": [
                    "독서를 정말 좋아해요",
                    "한 달에 책을 4-5권 읽어요",
                    "추리소설과 SF를 주로 읽어요",
                    "도서관에 자주 가요",
                    "독서 모임에 참여하고 있어요"
                ],
                "dialogues": [
                    ("취미가 뭐예요?", [0]),
                    ("책 많이 읽어요?", [1]),
                    ("어떤 장르 좋아해요?", [2]),
                    ("책은 어디서 빌려요?", [3]),
                    ("독서 모임 같은 거 해요?", [4])
                ]
            },
            {
                "theme": "pet_owner",
                "statements": [
                    "고양이 두 마리를 키우고 있어요",
                    "고양이 이름은 나비와 콩이에요",
                    "매일 아침저녁으로 밥을 챙겨줘요",
                    "주말에는 함께 놀아줘요",
                    "고양이 용품에 돈을 많이 써요"
                ],
                "dialogues": [
                    ("반려동물 있어요?", [0]),
                    ("이름이 뭐예요?", [1]),
                    ("돌보는 게 힘들지 않아요?", [2]),
                    ("주말에 뭐 해요?", [3]),
                    ("비용이 많이 들어요?", [4])
                ]
            },
            {
                "theme": "coffee_lover",
                "statements": [
                    "커피를 정말 좋아해요",
                    "하루에 3잔은 꼭 마셔요",
                    "집에서 직접 원두를 갈아서 내려요",
                    "카페 투어를 즐겨요",
                    "라떼아트를 배우고 있어요"
                ],
                "dialogues": [
                    ("커피 좋아해요?", [0]),
                    ("하루에 몇 잔 마셔요?", [1]),
                    ("집에서 커피 내려요?", [2]),
                    ("카페 자주 가요?", [3]),
                    ("바리스타에 관심 있어요?", [4])
                ]
            },
            {
                "theme": "movie_buff",
                "statements": [
                    "영화를 정말 좋아해요",
                    "일주일에 2-3편은 봐요",
                    "스릴러와 SF 장르를 좋아해요",
                    "영화관에서 보는 것을 선호해요",
                    "영화 리뷰 블로그를 운영해요"
                ],
                "dialogues": [
                    ("취미가 뭐예요?", [0]),
                    ("영화 많이 봐요?", [1]),
                    ("어떤 장르 좋아해요?", [2]),
                    ("OTT로 봐요, 영화관 가요?", [3]),
                    ("영화 관련 활동 해요?", [4])
                ]
            },
            {
                "theme": "entrepreneur",
                "statements": [
                    "스타트업을 운영하고 있어요",
                    "IT 서비스 분야에서 일해요",
                    "팀원이 10명 정도 있어요",
                    "투자를 받아서 성장 중이에요",
                    "워라밸보다는 일에 집중하고 있어요"
                ],
                "dialogues": [
                    ("무슨 일 하세요?", [0]),
                    ("어떤 분야예요?", [1]),
                    ("회사 규모가 어떻게 돼요?", [2]),
                    ("사업은 잘 되고 있어요?", [3]),
                    ("일하느라 바쁘시겠어요", [4])
                ]
            },
            {
                "theme": "language_learner",
                "statements": [
                    "외국어 배우는 것을 좋아해요",
                    "지금 일본어를 공부하고 있어요",
                    "영어, 중국어는 이미 할 줄 알아요",
                    "언어 교환 앱을 사용해요",
                    "목표는 5개 국어를 하는 거예요"
                ],
                "dialogues": [
                    ("취미가 뭐예요?", [0]),
                    ("지금 뭐 배우고 있어요?", [1]),
                    ("외국어 몇 개 해요?", [2]),
                    ("어떻게 공부해요?", [3]),
                    ("언어 관련 목표가 있어요?", [4])
                ]
            },
            {
                "theme": "gardener",
                "statements": [
                    "정원 가꾸는 것을 좋아해요",
                    "베란다에서 채소를 키워요",
                    "토마토와 허브를 주로 키워요",
                    "매일 아침 물을 줘요",
                    "직접 키운 채소로 요리하는 게 보람있어요"
                ],
                "dialogues": [
                    ("취미가 있어요?", [0]),
                    ("집에서 뭔가 키워요?", [1]),
                    ("어떤 식물 키워요?", [2]),
                    ("관리하기 힘들지 않아요?", [3]),
                    ("수확하면 어떻게 해요?", [4])
                ]
            }
        ]

        for user_idx in range(min(num_users, len(persona_templates))):
            template = persona_templates[user_idx]
            user_id = f"user_{user_idx + 1}"

            # 페르소나 생성
            persona_statements = []
            for stmt_idx, stmt in enumerate(template["statements"]):
                persona_id = f"{user_id}_persona_{stmt_idx}"
                persona_statements.append({
                    "id": persona_id,
                    "text": stmt
                })

            personas.append(Persona(
                persona_id=f"{user_id}_persona",
                user_id=user_id,
                statements=[s["text"] for s in persona_statements]
            ))

            # 쿼리 생성
            for q_idx, (dialogue, relevant_indices) in enumerate(template["dialogues"]):
                query = PersonaChatQuery(
                    query_id=f"{user_id}_query_{q_idx}",
                    user_id=user_id,
                    dialogue_context=f"대화 상대: {dialogue}",
                    query_utterance=dialogue,
                    relevant_persona_ids=[f"{user_id}_persona_{i}" for i in relevant_indices]
                )
                queries.append(query)

        return personas, queries

    def get_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def setup_collection(self):
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

    def index_personas(self, personas: List[Persona]):
        points = []
        self.documents = []
        self.document_ids = []

        for persona in personas:
            for stmt_idx, statement in enumerate(persona.statements):
                persona_id = f"{persona.user_id}_persona_{stmt_idx}"
                embedding = self.get_embedding(statement)

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "persona_id": persona_id,
                        "user_id": persona.user_id,
                        "text": statement,
                        "statement_index": stmt_idx
                    }
                )
                points.append(point)
                self.documents.append(statement)
                self.document_ids.append(persona_id)

        # Qdrant에 업로드
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points[i:i+batch_size]
            )

        # BM25 인덱스 구축
        self.bm25.fit(self.documents)

        print(f"✅ {len(points)}개 페르소나 문장 인덱싱 완료")

    def search_before(self, query: str, user_id: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """Before (2-stage): Vector Search → Cohere Reranking"""
        start_time = time.time()

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
                "persona_id": r.payload["persona_id"],
                "text": r.payload["text"],
                "score": r.score
            }
            for r in results
        ]

        # Cohere Reranking
        if self.cohere_client and candidates:
            try:
                time.sleep(3.5)  # Rate limiting

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
        """After (3-stage): Vector Search → BM25 Hybrid → Cohere Reranking"""
        start_time = time.time()

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
            r.payload["persona_id"]: {
                "persona_id": r.payload["persona_id"],
                "text": r.payload["text"],
                "vector_score": r.score
            }
            for r in vector_results
        }

        # BM25 검색
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

        # BM25 점수 정규화 및 하이브리드 점수
        if bm25_filtered:
            max_bm25 = max(score for _, score in bm25_filtered) if bm25_filtered else 1
            for idx, bm25_score in bm25_filtered:
                persona_id = self.document_ids[idx]
                if persona_id in vector_candidates:
                    vector_candidates[persona_id]["bm25_score"] = bm25_score / max_bm25 if max_bm25 > 0 else 0

        for persona_id, candidate in vector_candidates.items():
            vector_score = candidate.get("vector_score", 0)
            bm25_score = candidate.get("bm25_score", 0)
            candidate["hybrid_score"] = 0.7 * vector_score + 0.3 * bm25_score

        candidates = sorted(
            vector_candidates.values(),
            key=lambda x: x.get("hybrid_score", 0),
            reverse=True
        )[:top_k * 2]

        # Cohere Reranking
        if self.cohere_client and candidates:
            try:
                time.sleep(3.5)  # Rate limiting

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

    def evaluate(self, queries: List[PersonaChatQuery], personas: List[Persona]) -> Dict[str, Any]:
        """벤치마크 평가"""
        results = {
            "before": {"hits": [], "mrr": [], "ndcg": [], "precision": [], "recall": [], "latency": []},
            "after": {"hits": [], "mrr": [], "ndcg": [], "precision": [], "recall": [], "latency": []}
        }

        total = len(queries)

        for idx, query in enumerate(queries):
            print(f"\r평가 중... {idx+1}/{total}", end="", flush=True)

            relevant_ids = set(query.relevant_persona_ids)

            # Before (2-stage)
            before_results, before_latency = self.search_before(query.query_utterance, query.user_id)
            before_retrieved = [r["persona_id"] for r in before_results]

            before_metrics = self._calculate_metrics(before_retrieved, relevant_ids)
            results["before"]["hits"].append(before_metrics["hit@1"])
            results["before"]["mrr"].append(before_metrics["mrr"])
            results["before"]["ndcg"].append(before_metrics["ndcg@5"])
            results["before"]["precision"].append(before_metrics["precision@5"])
            results["before"]["recall"].append(before_metrics["recall@5"])
            results["before"]["latency"].append(before_latency)

            # After (3-stage)
            after_results, after_latency = self.search_after(query.query_utterance, query.user_id)
            after_retrieved = [r["persona_id"] for r in after_results]

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
    print("PersonaChat Benchmark")
    print("2-stage vs 3-stage 파이프라인 비교")
    print("=" * 70)
    print()

    benchmark = PersonaChatBenchmark()

    # 1. 데이터 생성
    print("📊 PersonaChat 스타일 데이터 생성 중...")
    personas, queries = benchmark.generate_personachat_data(num_users=15)
    print(f"   - 사용자 수: {len(personas)}")
    print(f"   - 총 페르소나 문장: {sum(len(p.statements) for p in personas)}")
    print(f"   - 쿼리 수: {len(queries)}")
    print()

    # 2. 인덱싱
    print("🔧 Qdrant 컬렉션 설정 및 인덱싱...")
    benchmark.setup_collection()
    benchmark.index_personas(personas)
    print()

    # 3. 평가
    print("🧪 벤치마크 평가 실행...")
    print("-" * 70)
    results = benchmark.evaluate(queries, personas)

    # 4. 결과 출력
    print("=" * 70)
    print("📈 PersonaChat Benchmark 결과")
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
            diff = before_val - after_val
            indicator = "⬇️" if diff > 0 else "⬆️"
        else:
            diff = after_val - before_val
            indicator = "⬆️" if diff > 0 else "⬇️" if diff < 0 else "➡️"

        before_str = fmt.format(before_val)
        after_str = fmt.format(after_val) + f" {indicator}"

        print(f"│ {label:<19} │ {before_str:>16} │ {after_str:>16} │")

    print("└─────────────────────┴──────────────────┴──────────────────┘")
    print()

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

    print("💡 PersonaChat 벤치마크 특성:")
    print("   - 페르소나 기반 대화에서 올바른 페르소나 문장 검색")
    print("   - 대화 컨텍스트와 페르소나 간의 의미적 연결이 중요")
    print("   - 키워드 매칭도 페르소나 검색에 도움이 될 수 있음")
    print()

    # 결과 저장
    output_file = "/tmp/personachat_benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "PersonaChat",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_users": len(personas),
                "num_persona_statements": sum(len(p.statements) for p in personas),
                "num_queries": len(queries),
                "embedding_model": "text-embedding-3-large",
                "reranking": "cohere/rerank-v3.5"
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"📁 결과 저장: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
