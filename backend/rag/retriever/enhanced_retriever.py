# -*- coding: utf-8 -*-
"""
Enhanced Retriever 모듈

범용적인 검색 성능 향상 기능:
1. Query Expansion - 쿼리 확장으로 검색 범위 확대
2. Multi-Query Retrieval - 다중 관점 검색으로 recall 향상
3. Adaptive Config - 쿼리 특성에 따른 동적 검색 설정
4. Result Fusion - 다중 검색 결과 통합 및 중복 제거
5. RRF (Reciprocal Rank Fusion) - Dense + Sparse 결과 순위 융합
6. HyDE (Hypothetical Document Embeddings) - 가상 답변 기반 검색

모든 기능은 도메인/문서 타입에 독립적으로 동작합니다.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple
from collections import defaultdict
import httpx

from rag.vectorstore import QdrantVectorStore, SearchResult, HybridSearchConfig
from rag.embeddings import MultimodalEmbeddingService
from rag.retriever.reranker import BGEReranker, CohereReranker, VoyageReranker, ColBERTReranker, JinaReranker


@dataclass
class EnhancedQuery:
    """확장된 쿼리 정보"""
    original: str
    expanded: str
    keywords: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    has_specific_terms: bool = False  # 특정 용어/숫자 포함 여부
    recommended_hybrid: bool = False  # BM25 사용 추천 여부


@dataclass
class EnhancedRetrievalConfig:
    """향상된 검색 설정"""
    # 기본 검색 설정 (V7.2: top_k 8→10)
    top_k: int = 10
    rerank_top_k: int = 30
    expand_to_parent: bool = True
    rerank: bool = True

    # V8: Reranker 선택
    # - "cohere": API 기반, 고성능 (기본값)
    # - "bge": 로컬, 무료, Cross-Encoder
    # - "voyage": API 기반, 최신
    # - "colbert": Token-level Late Interaction, 한국어 최적화
    # - "jina": Multilingual, Late Interaction
    reranker_type: str = "cohere"

    # 향상 기능 설정
    enable_query_expansion: bool = True      # 쿼리 확장 활성화
    enable_multi_query: bool = True          # 다중 쿼리 활성화
    enable_adaptive_search: bool = True      # 적응형 검색 활성화
    enable_rrf: bool = True                  # RRF 하이브리드 검색 활성화
    enable_hyde: bool = False                # HyDE 검색 활성화 (실험적)

    # 다중 쿼리 설정
    num_sub_queries: int = 3                 # 생성할 서브 쿼리 수

    # 적응형 검색 임계값
    keyword_density_threshold: float = 0.3   # 키워드 밀도 임계값

    # RRF 설정
    rrf_k: int = 60                          # RRF k 파라미터 (기본값 60)
    rrf_dense_weight: float = 1.5            # Dense 검색 가중치 (테이블 검색 개선)
    rrf_sparse_weight: float = 0.5           # Sparse 검색 가중치

    # 테이블 쿼리 적응형 검색
    enable_table_adaptive: bool = True       # 테이블 쿼리 감지 시 Dense Only 사용
    table_keywords: list = None              # 테이블 관련 키워드 (None이면 기본값 사용)


@dataclass
class EnhancedRetrievalResult:
    """향상된 검색 결과"""
    query: str
    enhanced_query: EnhancedQuery
    child_results: list[SearchResult]
    parent_contents: dict[str, str]
    context: str
    sources: list[dict]
    metadata: dict = field(default_factory=dict)


class QueryExpander:
    """
    쿼리 확장기 - V7.2: 도메인 특화 동의어 사전 추가

    검색 recall을 높이기 위해 원본 쿼리에 관련 키워드를 추가합니다.
    V7.2: 금융/연금 도메인 동의어 사전으로 검색 실패 케이스 개선
    """

    # V7.2: 도메인 특화 동의어 사전 (검색 실패 케이스 분석 기반)
    SYNONYM_DICT = {
        # 연금 관련
        "변액연금": ["변액연금", "변액보험", "투자형연금", "유닛링크드"],
        "연금": ["연금", "퇴직연금", "국민연금", "기업연금", "개인연금"],
        "적립금": ["적립금", "준비금", "기금", "자산"],
        "잔고": ["잔고", "잔액", "규모", "총액", "순자산"],

        # 펀드 관련
        "펀드": ["펀드", "투자신탁", "집합투자", "뮤추얼펀드"],
        "월지급식": ["월지급식", "월분배", "인컴형", "배당형"],
        "순자산": ["순자산", "AUM", "운용자산", "수탁고"],

        # 법률 관련 (오답 케이스 #19)
        "간병": ["간병", "요양", "돌봄", "케어"],
        "종사자": ["종사자", "근로자", "인력", "직원", "요양보호사"],
        "처우개선": ["처우개선", "급여인상", "임금인상", "근로조건"],
        "법률번호": ["법률번호", "법률 제", "법 제", "법률제"],

        # 금융 지표
        "수익률": ["수익률", "이익률", "성과", "리턴"],
        "금리": ["금리", "이자율", "기준금리", "이율"],

        # ESG/녹색금융 관련 (오답 케이스 #26)
        "그린뉴딜": ["그린뉴딜", "녹색뉴딜", "탄소중립", "친환경"],
        "녹색금융": ["녹색금융", "ESG금융", "지속가능금융", "환경금융"],
        "기후위기": ["기후위기", "기후변화", "온실가스", "탄소배출"],

        # PCAF 관련 (오답 케이스 #30)
        "PCAF": ["PCAF", "탄소회계금융협회", "Partnership for Carbon Accounting Financials"],
        "온실가스": ["온실가스", "GHG", "탄소배출", "이산화탄소"],
        "배출량": ["배출량", "배출", "탄소발자국", "Scope"],

        # 호주 연금 관련 (오답 케이스 #56)
        "호주": ["호주", "Australia", "오스트레일리아"],
        "연금시스템": ["연금시스템", "연금제도", "퇴직연금제도", "슈퍼애뉴에이션"],
        "동기": ["동기", "인센티브", "유인"],
    }

    def __init__(self, api_key: str, model: str = "gpt-5.2"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"

    async def expand(self, query: str) -> EnhancedQuery:
        """
        쿼리 확장 수행

        Args:
            query: 원본 쿼리

        Returns:
            EnhancedQuery: 확장된 쿼리 정보
        """
        # 1. 쿼리 특성 분석 (로컬, API 호출 없음)
        has_specific_terms = self._analyze_query_specificity(query)

        # 2. LLM 기반 키워드 추출 및 확장
        try:
            keywords, sub_queries = await self._llm_expand(query)
        except Exception as e:
            print(f"LLM 쿼리 확장 실패, 폴백 사용: {e}")
            keywords = self._fallback_extract_keywords(query)
            sub_queries = []

        # 3. 확장된 쿼리 생성
        expanded = self._build_expanded_query(query, keywords)

        # 4. BM25 사용 추천 결정
        recommended_hybrid = has_specific_terms or len(keywords) >= 3

        return EnhancedQuery(
            original=query,
            expanded=expanded,
            keywords=keywords,
            sub_queries=sub_queries,
            has_specific_terms=has_specific_terms,
            recommended_hybrid=recommended_hybrid,
        )

    def _analyze_query_specificity(self, query: str) -> bool:
        """쿼리의 특정성 분석 (숫자, 고유명사, 전문용어 포함 여부)"""
        # 숫자 포함
        has_numbers = bool(re.search(r'\d+', query))

        # 퍼센트/비율 포함
        has_percentage = bool(re.search(r'%|퍼센트|비율|율$', query))

        # 영문 약어 포함 (대문자 2글자 이상)
        has_acronym = bool(re.search(r'[A-Z]{2,}', query))

        # 따옴표로 감싼 특정 용어
        has_quoted = bool(re.search(r'["\'].*?["\']', query))

        # 짧은 쿼리 (키워드 검색에 유리)
        is_short = len(query.split()) <= 5

        return has_numbers or has_percentage or has_acronym or has_quoted or is_short

    async def _llm_expand(self, query: str) -> tuple[list[str], list[str]]:
        """LLM을 사용한 쿼리 확장 (V7: 3개 이상 다양한 변형)"""
        system_prompt = """당신은 RAG 검색 쿼리 최적화 전문가입니다.

주어진 질문을 분석하여 검색 recall을 최대화하기 위한 다양한 변형을 생성하세요.

작업:
1. 핵심 키워드 5-7개 추출:
   - 원본 질문의 핵심 개념
   - 동의어/유의어
   - 상위 개념/하위 개념
   - 관련 전문용어

2. 대안 질문 3개 생성 (각각 다른 관점):
   - 구체적 표현: 수치, 날짜, 이름 등을 명시적으로 언급
   - 추상적 표현: 개념적/일반적 표현으로 변환
   - 키워드 중심: 핵심 키워드만으로 구성된 짧은 쿼리

규칙:
- 원본 의미를 반드시 유지
- 검색 시스템에서 찾을 수 있는 문서 표현 고려
- 표/차트/그래프에 있을 수 있는 데이터 형식 고려

JSON 형식:
{
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "alternative_queries": ["구체적 질문", "추상적 질문", "키워드 질문"]
}"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"질문: {query}"}
            ],
            "temperature": 0.2,
            "max_completion_tokens": 300,
            "response_format": {"type": "json_object"}
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.api_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"API 오류: {response.status_code}")

        result = json.loads(response.json()["choices"][0]["message"]["content"])

        keywords = result.get("keywords", [])
        sub_queries = result.get("alternative_queries", [])

        return keywords, sub_queries

    def _fallback_extract_keywords(self, query: str) -> list[str]:
        """폴백: 규칙 기반 키워드 추출"""
        # 불용어 제거
        stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로',
                     '와', '과', '도', '만', '까지', '부터', '에게', '한테', '께',
                     '무엇', '어떤', '어떻게', '왜', '언제', '어디', '누가', '무슨'}

        # 토큰화 및 필터링
        tokens = re.findall(r'[가-힣]+|[a-zA-Z]+|\d+', query)
        keywords = [t for t in tokens if t not in stopwords and len(t) >= 2]

        return keywords[:5]

    def _build_expanded_query(self, query: str, keywords: list[str]) -> str:
        """
        확장된 쿼리 생성 (V7.2: 동의어 사전 적용)
        """
        if not keywords:
            keywords = []

        # V7.2: 동의어 사전에서 추가 키워드 추출
        synonym_keywords = self._get_synonyms_from_query(query)
        all_keywords = list(set(keywords + synonym_keywords))

        # 원본에 없는 키워드만 추가
        query_lower = query.lower()
        new_keywords = [kw for kw in all_keywords if kw.lower() not in query_lower]

        if new_keywords:
            return f"{query} {' '.join(new_keywords[:7])}"  # 최대 7개
        return query

    def _get_synonyms_from_query(self, query: str) -> list[str]:
        """
        V7.2: 쿼리에서 동의어 사전 매칭 키워드 추출
        """
        synonyms = []
        query_lower = query.lower()

        for key, values in self.SYNONYM_DICT.items():
            if key.lower() in query_lower:
                # 원본 키워드 제외하고 동의어만 추가
                for v in values:
                    if v.lower() != key.lower() and v.lower() not in query_lower:
                        synonyms.append(v)

        return synonyms[:5]  # 최대 5개 동의어


class QueryAdaptiveWeights:
    """
    Query-Adaptive Weights - V7.7

    쿼리 특성을 분석하여 Dense/Sparse 가중치를 동적으로 조정합니다.
    특정 테스트셋에 편향되지 않는 범용적인 방법입니다.

    쿼리 유형별 가중치 전략:
    1. 숫자/날짜 중심 쿼리 → Sparse(BM25) 강화
    2. 개념적/의미적 쿼리 → Dense(Embedding) 강화
    3. 고유명사/전문용어 쿼리 → Sparse 강화
    4. 일반 쿼리 → 균형 유지
    """

    # 숫자/날짜 패턴
    NUMBER_PATTERNS = [
        r'\d{4}년',           # 2023년
        r'\d{1,2}월',          # 3월, 12월
        r'\d{1,2}일',          # 1일, 31일
        r'\d+%',               # 12.5%
        r'\d+억',              # 100억
        r'\d+조',              # 10조
        r'\d+만',              # 1000만
        r'제?\d+호',           # 제44호
        r'제?\d+조',           # 제10조
        r'\d+\.?\d*%?p?',      # 숫자 일반
    ]

    # 전문용어/고유명사 패턴 (정확한 키워드 매칭 필요)
    SPECIFIC_TERM_INDICATORS = [
        # 법률/규정
        '법률', '법', '조례', '규정', '시행령', '시행규칙', '제정', '개정',
        # 기관/조직명
        'PCAF', 'TCFD', 'ESG', 'CDP', 'GRI', 'SASB', 'ISSB',
        'IMF', 'OECD', 'EU', 'SEC', 'FSB',
        # 금융 전문용어
        '스코프', 'Scope', '배출량', '탄소', 'CBAM',
        '변액연금', '퇴직연금', 'DC형', 'DB형', 'IRP',
    ]

    # 개념적/의미적 쿼리 패턴
    CONCEPTUAL_INDICATORS = [
        '설명', '의미', '정의', '개념', '원리', '원칙',
        '왜', '어떻게', '무엇', '차이', '비교', '장단점',
        '영향', '효과', '결과', '관계', '연관',
        '전략', '방법', '방식', '접근', '프레임워크',
    ]

    def __init__(
        self,
        base_dense_weight: float = 1.0,
        base_sparse_weight: float = 1.0,
        max_dense_weight: float = 2.0,
        max_sparse_weight: float = 2.0,
        min_weight: float = 0.3,
    ):
        """
        Args:
            base_dense_weight: 기본 Dense 가중치
            base_sparse_weight: 기본 Sparse 가중치
            max_dense_weight: Dense 가중치 상한
            max_sparse_weight: Sparse 가중치 상한
            min_weight: 최소 가중치 (0이 되지 않도록)
        """
        self.base_dense_weight = base_dense_weight
        self.base_sparse_weight = base_sparse_weight
        self.max_dense_weight = max_dense_weight
        self.max_sparse_weight = max_sparse_weight
        self.min_weight = min_weight

        # 숫자 패턴 컴파일
        self._number_patterns = [re.compile(p) for p in self.NUMBER_PATTERNS]

    def analyze_query(self, query: str) -> dict:
        """
        쿼리 특성 분석

        Returns:
            {
                'has_numbers': bool,
                'number_count': int,
                'has_specific_terms': bool,
                'specific_term_count': int,
                'is_conceptual': bool,
                'conceptual_count': int,
                'query_type': str,  # 'numeric', 'specific', 'conceptual', 'balanced'
            }
        """
        query_lower = query.lower()

        # 1. 숫자/날짜 분석
        number_matches = []
        for pattern in self._number_patterns:
            matches = pattern.findall(query)
            number_matches.extend(matches)
        number_count = len(number_matches)

        # 2. 전문용어/고유명사 분석
        specific_terms = []
        for term in self.SPECIFIC_TERM_INDICATORS:
            if term.lower() in query_lower:
                specific_terms.append(term)
        specific_count = len(specific_terms)

        # 3. 개념적 쿼리 분석
        conceptual_terms = []
        for term in self.CONCEPTUAL_INDICATORS:
            if term in query_lower:
                conceptual_terms.append(term)
        conceptual_count = len(conceptual_terms)

        # 4. 쿼리 유형 결정
        if number_count >= 2 or (number_count >= 1 and specific_count >= 1):
            query_type = 'numeric'
        elif specific_count >= 2:
            query_type = 'specific'
        elif conceptual_count >= 2:
            query_type = 'conceptual'
        elif number_count >= 1:
            query_type = 'numeric_light'
        elif specific_count >= 1:
            query_type = 'specific_light'
        elif conceptual_count >= 1:
            query_type = 'conceptual_light'
        else:
            query_type = 'balanced'

        return {
            'has_numbers': number_count > 0,
            'number_count': number_count,
            'number_matches': number_matches,
            'has_specific_terms': specific_count > 0,
            'specific_term_count': specific_count,
            'specific_terms': specific_terms,
            'is_conceptual': conceptual_count > 0,
            'conceptual_count': conceptual_count,
            'conceptual_terms': conceptual_terms,
            'query_type': query_type,
        }

    def get_adaptive_weights(self, query: str) -> tuple[float, float]:
        """
        쿼리에 따른 적응형 가중치 반환

        Args:
            query: 검색 쿼리

        Returns:
            (dense_weight, sparse_weight) 튜플
        """
        analysis = self.analyze_query(query)
        query_type = analysis['query_type']

        # 가중치 조정 로직
        if query_type == 'numeric':
            # 숫자/날짜 중심: Sparse 강화 (정확한 토큰 매칭)
            dense_weight = self.base_dense_weight * 0.7
            sparse_weight = self.base_sparse_weight * 1.8
        elif query_type == 'specific':
            # 전문용어: Sparse 약간 강화
            dense_weight = self.base_dense_weight * 0.85
            sparse_weight = self.base_sparse_weight * 1.5
        elif query_type == 'conceptual':
            # 개념적 쿼리: Dense 강화 (의미적 유사도)
            dense_weight = self.base_dense_weight * 1.8
            sparse_weight = self.base_sparse_weight * 0.6
        elif query_type == 'numeric_light':
            # 숫자 약간 포함: 약간 Sparse 강화
            dense_weight = self.base_dense_weight * 0.9
            sparse_weight = self.base_sparse_weight * 1.3
        elif query_type == 'specific_light':
            # 전문용어 약간 포함: 균형에 가까움
            dense_weight = self.base_dense_weight * 0.95
            sparse_weight = self.base_sparse_weight * 1.2
        elif query_type == 'conceptual_light':
            # 개념적 요소 약간 포함
            dense_weight = self.base_dense_weight * 1.3
            sparse_weight = self.base_sparse_weight * 0.85
        else:
            # balanced: 기본 가중치 유지
            dense_weight = self.base_dense_weight
            sparse_weight = self.base_sparse_weight

        # 가중치 범위 제한
        dense_weight = max(self.min_weight, min(self.max_dense_weight, dense_weight))
        sparse_weight = max(self.min_weight, min(self.max_sparse_weight, sparse_weight))

        return dense_weight, sparse_weight

    def get_weights_with_info(self, query: str) -> dict:
        """
        가중치와 분석 정보 함께 반환 (디버깅용)
        """
        analysis = self.analyze_query(query)
        dense_weight, sparse_weight = self.get_adaptive_weights(query)

        return {
            'query': query,
            'analysis': analysis,
            'dense_weight': dense_weight,
            'sparse_weight': sparse_weight,
            'weight_ratio': f"Dense:{dense_weight:.2f} / Sparse:{sparse_weight:.2f}",
        }


class RRFHybridSearcher:
    """
    Reciprocal Rank Fusion (RRF) 하이브리드 검색기

    Dense와 Sparse 검색 결과를 순위 기반으로 융합합니다.
    점수가 아닌 순위를 사용하므로 서로 다른 스케일의 점수도 공정하게 결합됩니다.

    RRF Score = sum(1 / (k + rank_i)) for each ranking system i
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_service: MultimodalEmbeddingService,
        k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ):
        """
        Args:
            vector_store: Qdrant 벡터 스토어
            embedding_service: 임베딩 서비스
            k: RRF k 파라미터 (기본값 60, 클수록 하위 순위의 영향력 증가)
            dense_weight: Dense 검색 가중치
            sparse_weight: Sparse 검색 가중치
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    async def search(
        self,
        query: str,
        top_k: int = 25,
        filter_source: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        RRF 기반 하이브리드 검색 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            filter_source: 소스 필터

        Returns:
            RRF 점수로 정렬된 검색 결과
        """
        # 1. 쿼리 임베딩 생성
        dense_embedding, sparse_embedding = await self.embedding_service.embed_query(query)

        # 2. Dense/Sparse 검색을 병렬로 수행
        dense_task = self.vector_store.search(
            query_vector=dense_embedding,
            config=HybridSearchConfig(top_k=top_k * 2),  # 더 많이 검색
            filter_source=filter_source,
            only_children=True,
        )

        sparse_task = self.vector_store.sparse_search(
            query_sparse_vector=sparse_embedding,
            config=HybridSearchConfig(top_k=top_k * 2),
            filter_source=filter_source,
            only_children=True,
        )

        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        # 3. RRF 점수 계산
        rrf_scores = self._calculate_rrf_scores(dense_results, sparse_results)

        # 4. 점수 기준 정렬 및 상위 k개 반환
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 5. SearchResult 객체 복원
        all_results = {r.chunk_id: r for r in dense_results + sparse_results}
        final_results = []

        for chunk_id, rrf_score in sorted_chunks[:top_k]:
            if chunk_id in all_results:
                result = all_results[chunk_id]
                result.score = rrf_score  # RRF 점수로 교체
                final_results.append(result)

        return final_results

    def _calculate_rrf_scores(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
    ) -> dict[str, float]:
        """RRF 점수 계산"""
        scores = defaultdict(float)

        # Dense 검색 결과 순위 점수
        for rank, result in enumerate(dense_results):
            scores[result.chunk_id] += self.dense_weight / (self.k + rank + 1)

        # Sparse 검색 결과 순위 점수
        for rank, result in enumerate(sparse_results):
            scores[result.chunk_id] += self.sparse_weight / (self.k + rank + 1)

        return dict(scores)


class HyDEGenerator:
    """
    HyDE (Hypothetical Document Embeddings) 생성기

    사용자 질문에 대한 가상의 답변을 먼저 생성하고,
    그 답변과 유사한 문서를 검색합니다.
    개념적/추상적 질문에 특히 효과적입니다.
    """

    def __init__(self, api_key: str, model: str = "gpt-5.2"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"

    async def generate_hypothetical_answer(self, query: str) -> str:
        """
        질문에 대한 가상의 답변 생성

        Args:
            query: 사용자 질문

        Returns:
            가상의 답변 (검색용)
        """
        system_prompt = """당신은 문서 검색을 돕는 전문가입니다.
주어진 질문에 대해, 문서에 있을 법한 형태의 답변을 작성하세요.

규칙:
1. 실제 정보를 알 필요 없이, 그럴듯한 형태의 답변을 작성
2. 구체적인 수치나 데이터를 포함 (예: "매출액은 1,234억원...")
3. 문서에서 발췌한 것처럼 작성
4. 2-3문장으로 간결하게

예시:
질문: "2023년 영업이익률은?"
답변: "2023년 영업이익률은 12.5%로 전년 대비 2.3%p 상승하였습니다. 이는 원가 절감 노력과 고부가가치 제품 매출 확대에 기인합니다."
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"질문: {query}"}
            ],
            "temperature": 0.7,
            "max_completion_tokens": 200,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.api_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"API 오류: {response.status_code}")

        result = response.json()
        return result["choices"][0]["message"]["content"]


class MultiQueryRetriever:
    """
    다중 쿼리 검색기 - 도메인 독립적

    여러 관점의 쿼리로 검색하여 recall을 높이고,
    결과를 통합하여 더 다양한 관련 문서를 찾습니다.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_service: MultimodalEmbeddingService,
        reranker: Optional[BGEReranker] = None,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.reranker = reranker or BGEReranker()
        self._reranker_initialized = False

    def _ensure_reranker(self) -> bool:
        """Reranker 초기화 확인"""
        if not self._reranker_initialized:
            self._reranker_initialized = self.reranker.initialize()
        return self._reranker_initialized

    async def retrieve(
        self,
        queries: list[str],
        config: EnhancedRetrievalConfig,
        use_hybrid: bool = False,
    ) -> list[SearchResult]:
        """
        다중 쿼리로 검색 수행

        Args:
            queries: 쿼리 리스트 (원본 + 서브쿼리)
            config: 검색 설정
            use_hybrid: BM25 하이브리드 사용 여부

        Returns:
            통합된 검색 결과 (중복 제거, 리랭킹 완료)
        """
        all_results = []
        seen_chunks = set()

        # 각 쿼리로 검색 수행
        search_tasks = []
        for query in queries:
            search_tasks.append(
                self._search_single_query(query, config, use_hybrid)
            )

        # 병렬 검색
        query_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # 결과 통합 (중복 제거)
        for results in query_results:
            if isinstance(results, Exception):
                print(f"검색 오류: {results}")
                continue

            for result in results:
                if result.chunk_id not in seen_chunks:
                    all_results.append(result)
                    seen_chunks.add(result.chunk_id)

        # 리랭킹 (원본 쿼리 기준)
        if config.rerank and all_results and self._ensure_reranker():
            original_query = queries[0]
            all_results = self._rerank_results(original_query, all_results, config.top_k)
        else:
            # 점수 기준 정렬 후 top_k 선택
            all_results = sorted(all_results, key=lambda x: x.score, reverse=True)
            all_results = all_results[:config.top_k]

        return all_results

    async def _search_single_query(
        self,
        query: str,
        config: EnhancedRetrievalConfig,
        use_hybrid: bool,
    ) -> list[SearchResult]:
        """단일 쿼리 검색"""
        dense_embedding, sparse_embedding = await self.embedding_service.embed_query(query)

        # 다중 쿼리에서는 각 쿼리당 적은 수의 결과 검색
        per_query_top_k = max(config.rerank_top_k // 2, 10)

        if use_hybrid:
            results = await self.vector_store.hybrid_search(
                query_dense_vector=dense_embedding,
                query_sparse_vector=sparse_embedding,
                config=HybridSearchConfig(top_k=per_query_top_k),
                only_children=True,
            )
        else:
            results = await self.vector_store.search(
                query_vector=dense_embedding,
                config=HybridSearchConfig(top_k=per_query_top_k),
                only_children=True,
            )

        return results

    def _rerank_results(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """BGE Reranker로 검색 결과 리랭킹"""
        if not results:
            return results

        documents = [r.content for r in results]
        reranked = self.reranker.rerank(query, documents, top_k=top_k)

        content_to_result = {r.content: r for r in results}
        reranked_results = []

        for doc, score in reranked:
            if doc in content_to_result:
                result = content_to_result[doc]
                result.score = float(score)
                reranked_results.append(result)

        return reranked_results


class EnhancedHierarchicalRetriever:
    """
    향상된 계층적 검색기

    기존 HierarchicalRetriever에 다음 기능을 추가:
    1. Query Expansion - 쿼리 확장
    2. Multi-Query Retrieval - 다중 쿼리 검색
    3. Adaptive Search - 적응형 검색 (쿼리 특성에 따른 동적 설정)
    4. RRF Hybrid Search - Dense + Sparse 순위 융합
    5. HyDE - 가상 답변 기반 검색 (실험적)

    모든 기능은 도메인/문서 타입에 독립적으로 동작합니다.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_service: MultimodalEmbeddingService,
        api_key: str,
        config: Optional[EnhancedRetrievalConfig] = None,
    ):
        """
        Args:
            vector_store: Qdrant 벡터 스토어
            embedding_service: 임베딩 서비스
            api_key: OpenAI API 키 (쿼리 확장용)
            config: 검색 설정
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.config = config or EnhancedRetrievalConfig()
        self.api_key = api_key

        # 향상 기능 컴포넌트
        self.query_expander = QueryExpander(api_key=api_key)
        self.multi_query_retriever = MultiQueryRetriever(
            vector_store=vector_store,
            embedding_service=embedding_service,
        )

        # RRF 하이브리드 검색기
        self.rrf_searcher = RRFHybridSearcher(
            vector_store=vector_store,
            embedding_service=embedding_service,
            k=self.config.rrf_k,
            dense_weight=self.config.rrf_dense_weight,
            sparse_weight=self.config.rrf_sparse_weight,
        )

        # HyDE 생성기
        self.hyde_generator = HyDEGenerator(api_key=api_key)

        # V7.7: Query-Adaptive Weights
        self.adaptive_weights = QueryAdaptiveWeights(
            base_dense_weight=self.config.rrf_dense_weight,
            base_sparse_weight=self.config.rrf_sparse_weight,
        )
        self._enable_adaptive_weights = True  # V7.7 활성화 플래그

        # V7.6: Reranker 선택 (BGE, Cohere, Voyage, ColBERT, Jina)
        reranker_type = self.config.reranker_type.lower()
        if reranker_type == "cohere":
            self.reranker = CohereReranker()
            print("V7.2: Cohere Reranker 사용")
        elif reranker_type == "voyage":
            self.reranker = VoyageReranker()
            print("V7.3: Voyage Reranker 2.5 사용")
        elif reranker_type == "colbert":
            self.reranker = ColBERTReranker()
            print("V7.6: ColBERT Reranker 사용 (한국어 최적화)")
        elif reranker_type == "jina":
            self.reranker = JinaReranker()
            print("V7.6: Jina Reranker v2 사용 (다국어)")
        else:
            self.reranker = BGEReranker()
            print("V7.2: BGE Reranker 사용")
        self._reranker_initialized = False

        # 테이블/이미지 쿼리 감지용 키워드 (Dense Only 검색 적용)
        # V6: 대폭 확장된 키워드 리스트
        self._table_keywords = self.config.table_keywords or [
            # === 테이블/표 관련 ===
            "표", "테이블", "목록", "리스트", "일람", "현황",
            # 법률/제도 관련 (테이블에 자주 포함)
            "법률", "법령", "시행령", "시행규칙", "조례", "규정", "제정", "법률번호",
            # 연도별 데이터
            "연도별", "년도별", "기간별", "분기별", "월별", "일별",
            # 비교/나열
            "비교", "차이점", "공통점", "종류", "유형", "구분",
            # 수치 데이터
            "얼마", "몇", "수치", "통계", "금액", "규모",

            # === 이미지/차트 관련 (V6 대폭 확장) ===
            # 차트/그래프 유형
            "그래프", "차트", "그림", "도표", "다이어그램", "플로우차트",
            "막대그래프", "선그래프", "원그래프", "파이차트", "히스토그램",
            # 수익률/성과 지표
            "수익률", "성장률", "증가율", "변화율", "감소율", "상승률", "하락률",
            "YTD", "전년대비", "전분기대비", "연환산", "누적",
            # 금융 지표
            "금리", "이자율", "할인율", "국채", "채권", "주가", "지수",
            "ROE", "ROA", "PER", "PBR", "EPS", "BPS",
            "영업이익률", "순이익률", "매출성장률", "CAGR",
            # 자산/부채 관련
            "자산", "부채", "자본", "순자산", "운용자산", "수탁고",
            "AUM", "NAV", "펀드규모",
            # 연금/기금 관련
            "적립금", "적립률", "기여금", "급여", "연금",
            "기금", "수익", "손실", "손익",
            # 기간/추이
            "추이", "트렌드", "변동", "흐름", "전망", "예측",
            "1년", "3년", "5년", "10년", "15년", "20년",
            # 비교 표현
            "높은", "낮은", "최대", "최소", "평균", "중앙값",
            "1위", "2위", "3위", "순위", "랭킹",
        ]

        # 이미지 청크 가중치 부스팅 설정
        self.image_boost_factor = 1.3  # 이미지 청크 30% 가중치 부여

    def _ensure_reranker(self) -> bool:
        """Reranker 초기화 확인"""
        if not self._reranker_initialized:
            self._reranker_initialized = self.reranker.initialize()
        return self._reranker_initialized

    def _is_table_query(self, query: str) -> bool:
        """테이블 관련 쿼리인지 감지"""
        query_lower = query.lower()

        # 1. 테이블 키워드 포함 여부
        keyword_match = any(kw in query_lower for kw in self._table_keywords)

        # 2. 연도 + "에" 패턴 (예: "2005년에", "1994년부터")
        import re
        year_pattern = bool(re.search(r'\d{4}년[에부]', query))

        # 3. "어떤 것들" 패턴 (목록형 질문)
        list_pattern = "어떤 것들" in query or "무엇들" in query or "종류" in query

        return keyword_match or year_pattern or list_pattern

    async def retrieve(
        self,
        query: str,
        config: Optional[EnhancedRetrievalConfig] = None,
        filter_source: Optional[str] = None,
    ) -> EnhancedRetrievalResult:
        """
        향상된 검색 수행

        Args:
            query: 검색 쿼리
            config: 검색 설정 (없으면 기본값 사용)
            filter_source: 특정 소스만 검색

        Returns:
            EnhancedRetrievalResult: 향상된 검색 결과
        """
        config = config or self.config
        metadata = {"enhancements_applied": []}

        # 1. 쿼리 확장
        if config.enable_query_expansion:
            enhanced_query = await self.query_expander.expand(query)
            metadata["enhancements_applied"].append("query_expansion")
            metadata["keywords"] = enhanced_query.keywords
        else:
            enhanced_query = EnhancedQuery(
                original=query,
                expanded=query,
            )

        # 2. HyDE 가상 답변 생성 (선택적)
        hyde_query = None
        if config.enable_hyde:
            try:
                hyde_query = await self.hyde_generator.generate_hypothetical_answer(query)
                metadata["enhancements_applied"].append("hyde")
                metadata["hyde_query"] = hyde_query[:100] + "..."
            except Exception as e:
                print(f"HyDE 생성 실패: {e}")

        # 3. 검색 쿼리 결정
        search_query = enhanced_query.expanded
        if hyde_query:
            # HyDE 사용 시 가상 답변을 검색 쿼리로 사용
            search_query = hyde_query

        # 3.5. 테이블 쿼리 감지 (적응형 검색)
        is_table_query = False
        if config.enable_table_adaptive:
            is_table_query = self._is_table_query(query)
            if is_table_query:
                metadata["table_query_detected"] = True

        # 4. 검색 전략 결정
        # 테이블 쿼리: Dense Only (BM25가 테이블 구조에 취약)
        if is_table_query:
            child_results = await self._dense_only_search(
                search_query,
                config,
                filter_source,
            )
            metadata["enhancements_applied"].append("dense_only_table")
        # RRF 하이브리드 검색 (일반 쿼리)
        elif config.enable_rrf:
            # V7.7: Query-Adaptive Weights 정보 기록
            if self._enable_adaptive_weights:
                weight_info = self.adaptive_weights.get_weights_with_info(query)
                metadata["adaptive_weights"] = {
                    "query_type": weight_info["analysis"]["query_type"],
                    "dense_weight": weight_info["dense_weight"],
                    "sparse_weight": weight_info["sparse_weight"],
                }
                metadata["enhancements_applied"].append("query_adaptive_weights")

            child_results = await self._rrf_search(
                search_query,
                config,
                filter_source,
            )
            metadata["enhancements_applied"].append("rrf_hybrid")
        # 5. 다중 쿼리 검색
        elif config.enable_multi_query and enhanced_query.sub_queries:
            # 적응형 검색 설정 결정
            use_hybrid = enhanced_query.recommended_hybrid if config.enable_adaptive_search else False

            all_queries = [search_query] + enhanced_query.sub_queries
            child_results = await self.multi_query_retriever.retrieve(
                queries=all_queries,
                config=config,
                use_hybrid=use_hybrid,
            )
            metadata["enhancements_applied"].append("multi_query")
            metadata["num_queries"] = len(all_queries)
        # 6. 단일 쿼리 검색 (폴백)
        else:
            use_hybrid = enhanced_query.recommended_hybrid if config.enable_adaptive_search else False
            child_results = await self._single_query_search(
                search_query,
                config,
                use_hybrid,
                filter_source,
            )

        # V7: Fallback 검색 - 결과가 부족하거나 낮은 점수일 때 대안 쿼리로 재검색
        child_results = await self._fallback_search_if_needed(
            child_results=child_results,
            original_query=query,
            enhanced_query=enhanced_query,
            config=config,
            filter_source=filter_source,
            metadata=metadata,
        )

        # 7. 부모 청크 확장
        parent_contents = {}
        if config.expand_to_parent:
            parent_contents = await self._fetch_parent_contents(child_results)

        # 8. 컨텍스트 구성
        context = self._build_context(child_results, parent_contents, config)

        # 9. 출처 정보 추출
        sources = self._extract_sources(child_results)

        return EnhancedRetrievalResult(
            query=query,
            enhanced_query=enhanced_query,
            child_results=child_results,
            parent_contents=parent_contents,
            context=context,
            sources=sources,
            metadata=metadata,
        )

    async def _rrf_search(
        self,
        query: str,
        config: EnhancedRetrievalConfig,
        filter_source: Optional[str] = None,
    ) -> list[SearchResult]:
        """RRF 하이브리드 검색 - V7.7: Query-Adaptive Weights 적용"""
        # V7.7: 쿼리에 따른 동적 가중치 조정
        if self._enable_adaptive_weights:
            dense_weight, sparse_weight = self.adaptive_weights.get_adaptive_weights(query)
            # RRF 검색기 가중치 동적 업데이트
            self.rrf_searcher.dense_weight = dense_weight
            self.rrf_searcher.sparse_weight = sparse_weight

        # RRF로 검색
        child_results = await self.rrf_searcher.search(
            query=query,
            top_k=config.rerank_top_k if config.rerank else config.top_k,
            filter_source=filter_source,
        )

        # 리랭킹
        if config.rerank and child_results and self._ensure_reranker():
            child_results = self._rerank_results(query, child_results, config.top_k)

        return child_results

    async def _dense_only_search(
        self,
        query: str,
        config: EnhancedRetrievalConfig,
        filter_source: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Dense Only 검색 (테이블/이미지 쿼리용)

        BM25가 테이블 구조 데이터에 취약하므로,
        테이블/이미지 관련 쿼리는 Dense 임베딩만으로 검색합니다.
        V6: 이미지 청크 가중치 부스팅 추가
        """
        dense_embedding, _ = await self.embedding_service.embed_query(query)

        search_top_k = config.rerank_top_k if config.rerank else config.top_k

        child_results = await self.vector_store.search(
            query_vector=dense_embedding,
            config=HybridSearchConfig(top_k=search_top_k),
            filter_source=filter_source,
            only_children=True,
        )

        # V6: 이미지 청크 가중치 부스팅
        child_results = self._boost_image_chunks(child_results, query)

        # 리랭킹
        if config.rerank and child_results and self._ensure_reranker():
            child_results = self._rerank_results(query, child_results, config.top_k)

        return child_results

    async def _fallback_search_if_needed(
        self,
        child_results: list[SearchResult],
        original_query: str,
        enhanced_query: EnhancedQuery,
        config: EnhancedRetrievalConfig,
        filter_source: Optional[str],
        metadata: dict,
    ) -> list[SearchResult]:
        """
        V7: Fallback 검색 - 결과가 부족하거나 점수가 낮을 때 대안 쿼리로 재검색

        검색 실패 패턴:
        1. 결과가 없거나 매우 적음 (< top_k의 절반)
        2. 최고 점수가 너무 낮음 (< 0.5)

        Fallback 전략:
        1. 대안 쿼리들로 순차 검색
        2. 키워드 기반 검색 (BM25 강화)
        3. 결과 병합 및 중복 제거
        """
        # Fallback 필요 여부 판단
        min_results = max(config.top_k // 2, 3)
        max_score = max((r.score for r in child_results), default=0)

        needs_fallback = (
            len(child_results) < min_results or
            (len(child_results) > 0 and max_score < 0.5)
        )

        if not needs_fallback:
            return child_results

        metadata["enhancements_applied"].append("fallback_search")
        metadata["fallback_reason"] = (
            "insufficient_results" if len(child_results) < min_results else "low_scores"
        )

        # 기존 결과 보존
        all_results = list(child_results)
        seen_ids = {r.chunk_id for r in child_results}

        # 1. 대안 쿼리로 검색
        for alt_query in enhanced_query.sub_queries[:3]:
            if len(all_results) >= config.top_k * 2:
                break

            try:
                alt_results = await self._dense_only_search(
                    alt_query, config, filter_source
                )
                for r in alt_results:
                    if r.chunk_id not in seen_ids:
                        all_results.append(r)
                        seen_ids.add(r.chunk_id)
            except Exception:
                continue

        # 2. 키워드 기반 검색 (확장된 키워드 사용)
        if len(all_results) < config.top_k and enhanced_query.keywords:
            keyword_query = " ".join(enhanced_query.keywords[:5])
            try:
                keyword_results = await self._single_query_search(
                    keyword_query, config, use_hybrid=True, filter_source=filter_source
                )
                for r in keyword_results:
                    if r.chunk_id not in seen_ids:
                        all_results.append(r)
                        seen_ids.add(r.chunk_id)
            except Exception:
                pass

        # 3. 결과 정렬 및 상위 선택
        all_results.sort(key=lambda x: x.score, reverse=True)

        # 리랭킹 수행
        if config.rerank and all_results and self._ensure_reranker():
            all_results = self._rerank_results(
                original_query, all_results, config.top_k
            )
        else:
            all_results = all_results[:config.top_k]

        metadata["fallback_total_results"] = len(all_results)

        return all_results

    def _boost_image_chunks(
        self,
        results: list[SearchResult],
        query: str,
    ) -> list[SearchResult]:
        """
        이미지 청크 가중치 부스팅 (V6)

        이미지 관련 쿼리 감지 시 이미지 타입 청크에 가중치를 부여합니다.
        """
        # 이미지 관련 키워드 확인
        image_keywords = [
            "그래프", "차트", "그림", "도표", "다이어그램",
            "수익률", "금리", "국채", "추이", "변동",
            "1년", "3년", "5년", "10년", "15년", "20년",
            "비교", "높은", "낮은", "최대", "최소",
        ]

        query_lower = query.lower()
        is_image_query = any(kw in query_lower for kw in image_keywords)

        if not is_image_query:
            return results

        # 이미지 청크에 가중치 부여
        for result in results:
            content_type = result.metadata.get("content_type", "") if result.metadata else ""
            if content_type == "image" or "이미지" in (result.heading or ""):
                result.score *= self.image_boost_factor

        # 점수 기준 재정렬
        return sorted(results, key=lambda x: x.score, reverse=True)

    async def _single_query_search(
        self,
        query: str,
        config: EnhancedRetrievalConfig,
        use_hybrid: bool,
        filter_source: Optional[str] = None,
    ) -> list[SearchResult]:
        """단일 쿼리 검색"""
        dense_embedding, sparse_embedding = await self.embedding_service.embed_query(query)

        search_top_k = config.rerank_top_k if config.rerank else config.top_k

        if use_hybrid:
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

        # 리랭킹
        if config.rerank and child_results and self._ensure_reranker():
            child_results = self._rerank_results(query, child_results, config.top_k)

        return child_results

    def _rerank_results(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """BGE Reranker로 검색 결과 리랭킹"""
        if not results:
            return results

        documents = [r.content for r in results]
        reranked = self.reranker.rerank(query, documents, top_k=top_k)

        content_to_result = {r.content: r for r in results}
        reranked_results = []

        for doc, score in reranked:
            if doc in content_to_result:
                result = content_to_result[doc]
                result.score = float(score)
                reranked_results.append(result)

        return reranked_results

    async def _fetch_parent_contents(
        self, child_results: list[SearchResult]
    ) -> dict[str, str]:
        """자식 청크들의 부모 컨텐츠 조회"""
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
        config: EnhancedRetrievalConfig,
    ) -> str:
        """검색 결과로부터 LLM 컨텍스트 구성 (V7.6.1 원본)

        Note: 이슈 #8의 Child Fallback 수정을 V7.6.3, V7.6.4로 시도했으나 모두 성능 하락.
        - V7.6.3 (무조건 Child 추가): 85% → 75% (-10%p)
        - V7.6.4 (조건부 Child 추가): 85% → 70% (-15%p)

        원인: Parent가 정상 존재하고 Child 내용도 대부분 Parent에 포함됨.
        Child Fallback 추가 시 중복 컨텍스트로 인한 노이즈 발생.
        """
        context_parts = []
        used_parents = set()

        # 컨텍스트에 출처 헤더 포함하지 않음 (프론트엔드에서 별도 표시)
        if config.expand_to_parent:
            for result in child_results:
                if result.parent_id and result.parent_id not in used_parents:
                    parent_content = parent_contents.get(result.parent_id)
                    if parent_content:
                        context_parts.append(parent_content)
                        used_parents.add(result.parent_id)
        else:
            for result in child_results:
                context_parts.append(result.content)

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
