# -*- coding: utf-8 -*-
"""
Query Enhancement 모듈

검색 성능 향상을 위한 쿼리 확장 및 다중 쿼리 생성
- Query Expansion: 동의어/관련 키워드 추가
- Multi-Query: 여러 관점의 쿼리 생성
- Query Classification: 쿼리 유형 분류 (text/table/image)
"""

import json
import re
from dataclasses import dataclass
from typing import Optional
import httpx


@dataclass
class EnhancedQuery:
    """확장된 쿼리 결과"""
    original: str
    expanded: str
    keywords: list[str]
    query_type: str  # 'text', 'table', 'image', 'mixed'
    use_bm25: bool  # BM25 사용 여부 추천


class QueryEnhancer:
    """쿼리 확장 및 분류기"""

    def __init__(self, api_key: str, model: str = "gpt-5.2"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"

    async def enhance(self, query: str) -> EnhancedQuery:
        """
        쿼리 확장 및 분류

        Args:
            query: 원본 쿼리

        Returns:
            EnhancedQuery: 확장된 쿼리 정보
        """
        system_prompt = """당신은 RAG 시스템의 검색 품질을 높이기 위한 쿼리 분석 전문가입니다.

주어진 질문을 분석하여 다음을 수행하세요:

1. **키워드 추출**: 핵심 검색 키워드 3-5개 추출
2. **쿼리 유형 분류**:
   - text: 일반 텍스트/단락에서 답을 찾는 질문
   - table: 표/테이블 데이터가 필요한 질문 (숫자 비교, 목록, 요건 등)
   - image: 차트/그래프/이미지 분석이 필요한 질문
   - mixed: 여러 유형이 복합된 질문
3. **BM25 추천**: 특정 용어/숫자가 중요하면 true, 의미 검색이 중요하면 false

JSON 형식으로 출력:
{
  "keywords": ["키워드1", "키워드2", "키워드3"],
  "query_type": "text|table|image|mixed",
  "use_bm25": true|false,
  "reasoning": "분류 이유 (1문장)"
}"""

        user_prompt = f"질문: {query}"

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0,
                "max_completion_tokens": 200,
                "response_format": {"type": "json_object"}
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                )

            if response.status_code != 200:
                return self._fallback_enhance(query)

            result = json.loads(response.json()["choices"][0]["message"]["content"])

            keywords = result.get("keywords", [])
            query_type = result.get("query_type", "text")
            use_bm25 = result.get("use_bm25", False)

            # 확장된 쿼리 생성
            expanded = self._build_expanded_query(query, keywords)

            return EnhancedQuery(
                original=query,
                expanded=expanded,
                keywords=keywords,
                query_type=query_type,
                use_bm25=use_bm25,
            )

        except Exception as e:
            print(f"쿼리 확장 실패: {e}")
            return self._fallback_enhance(query)

    def _build_expanded_query(self, query: str, keywords: list[str]) -> str:
        """확장된 쿼리 생성"""
        if not keywords:
            return query

        # 원본 쿼리에 키워드 추가
        keyword_str = " ".join(keywords)
        return f"{query} {keyword_str}"

    def _fallback_enhance(self, query: str) -> EnhancedQuery:
        """폴백: 기본 쿼리 분석"""
        # 숫자/비율이 포함되면 table/image 가능성
        has_numbers = bool(re.search(r'\d+', query))
        has_percentage = '%' in query or '비율' in query or '율' in query

        # 특정 키워드로 유형 추정
        table_keywords = ['요건', '조건', '종류', '차이', '비교', '목록', '항목']
        image_keywords = ['차트', '그래프', '추이', '변화', '추세', '현황']

        query_type = 'text'
        if any(kw in query for kw in image_keywords):
            query_type = 'image'
        elif any(kw in query for kw in table_keywords) or has_numbers:
            query_type = 'table'

        # 특정 용어가 있으면 BM25 유리
        use_bm25 = has_numbers or has_percentage or len(query.split()) <= 5

        return EnhancedQuery(
            original=query,
            expanded=query,
            keywords=[],
            query_type=query_type,
            use_bm25=use_bm25,
        )


class MultiQueryGenerator:
    """다중 쿼리 생성기"""

    def __init__(self, api_key: str, model: str = "gpt-5.2"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"

    async def generate(self, query: str, num_queries: int = 4) -> list[str]:
        """
        여러 관점의 쿼리 생성 (Query Decomposition 포함)

        Args:
            query: 원본 쿼리
            num_queries: 생성할 쿼리 수

        Returns:
            list[str]: 다중 쿼리 리스트 (원본 포함)
        """
        system_prompt = f"""당신은 RAG 시스템을 위한 검색 쿼리 최적화 전문가입니다.

주어진 질문을 분석하여 {num_queries-1}개의 검색 쿼리를 생성하세요.

## 생성 규칙
1. **복합 질문 분해**: 질문에 여러 부분이 있으면 각각 별도 쿼리로 분해
   - 예: "A의 명칭과 B의 배경" → "A 명칭", "B 배경"
2. **배경/원인 질문**: "배경", "이유", "원인"이 있으면 관련 통계, 사회적 상황도 검색
   - 예: "X 출시 배경" → "X 출시 배경", "X 관련 통계 현황"
3. **영향/결과 질문**: "영향", "결과", "전망"이 있으면 인과관계 검색
   - 예: "Y의 영향" → "Y 영향", "Y로 인한 결과", "Y 전망"
4. **키워드 변형**: 동의어나 관련 표현으로 변형

## 출력 형식
JSON으로 출력:
{{"queries": ["쿼리1", "쿼리2", "쿼리3"]}}"""

        user_prompt = f"원본 질문: {query}"

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_completion_tokens": 200,
                "response_format": {"type": "json_object"}
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                )

            if response.status_code != 200:
                return [query]

            result = json.loads(response.json()["choices"][0]["message"]["content"])
            queries = result.get("queries", [])

            # 원본 쿼리를 첫 번째로
            all_queries = [query] + queries[:num_queries-1]
            return all_queries

        except Exception as e:
            print(f"다중 쿼리 생성 실패: {e}")
            return [query]
