# -*- coding: utf-8 -*-
"""
Query Enhancement 모듈

검색 성능 향상을 위한 쿼리 확장 및 다중 쿼리 생성
- Query Expansion: 동의어/관련 키워드 추가
- Multi-Query: 여러 관점의 쿼리 생성
- Query Classification: 쿼리 유형 분류 (text/table/image)
- V8: Gemini Flash 지원 추가
"""

import json
import re
import os
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
    """쿼리 확장 및 분류기 (OpenAI / Gemini 지원)"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-3-flash-preview",
        provider: str = "google"
    ):
        self.provider = provider.lower()
        self.model = model

        if self.provider == "google":
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            self.api_url = "https://api.openai.com/v1/chat/completions"

    async def enhance(self, query: str) -> EnhancedQuery:
        """
        쿼리 확장 및 분류 (OpenAI / Gemini 지원)

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

JSON 형식으로만 출력 (다른 텍스트 없이):
{"keywords": ["키워드1", "키워드2", "키워드3"], "query_type": "text", "use_bm25": false}"""

        user_prompt = f"질문: {query}"

        try:
            if self.provider == "google":
                result = await self._call_gemini(system_prompt, user_prompt)
            else:
                result = await self._call_openai(system_prompt, user_prompt)

            if result is None:
                return self._fallback_enhance(query)

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

    async def _call_gemini(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """Gemini API 호출"""
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{
                    "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
                }],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 200,
                }
            }

            url = f"{self.api_url}?key={self.api_key}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                print(f"Gemini API 오류: {response.status_code}")
                return None

            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]

            # JSON 파싱 (```json 블록 제거)
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            return json.loads(text)

        except Exception as e:
            print(f"Gemini 호출 실패: {e}")
            return None

    async def _call_openai(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """OpenAI API 호출"""
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
                response = await client.post(self.api_url, headers=headers, json=payload)

            if response.status_code != 200:
                print(f"OpenAI API 오류: {response.status_code}")
                return None

            return json.loads(response.json()["choices"][0]["message"]["content"])

        except Exception as e:
            print(f"OpenAI 호출 실패: {e}")
            return None

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
    """다중 쿼리 생성기 (OpenAI / Gemini 지원)"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-3-flash-preview",
        provider: str = "google"
    ):
        self.provider = provider.lower()
        self.model = model

        if self.provider == "google":
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
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
2. **배경/원인 질문**: "배경", "이유", "원인"이 있으면 관련 통계도 검색
3. **영향/결과 질문**: "영향", "결과", "전망"이 있으면 인과관계 검색
4. **키워드 변형**: 동의어나 관련 표현으로 변형

JSON으로만 출력 (다른 텍스트 없이):
{{"queries": ["쿼리1", "쿼리2", "쿼리3"]}}"""

        user_prompt = f"원본 질문: {query}"

        try:
            if self.provider == "google":
                result = await self._call_gemini(system_prompt, user_prompt)
            else:
                result = await self._call_openai(system_prompt, user_prompt)

            if result is None:
                return [query]

            queries = result.get("queries", [])
            all_queries = [query] + queries[:num_queries-1]
            return all_queries

        except Exception as e:
            print(f"다중 쿼리 생성 실패: {e}")
            return [query]

    async def _call_gemini(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """Gemini API 호출"""
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{
                    "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 200,
                }
            }

            url = f"{self.api_url}?key={self.api_key}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                return None

            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]

            # JSON 파싱
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            return json.loads(text)

        except Exception as e:
            print(f"Gemini 호출 실패: {e}")
            return None

    async def _call_openai(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """OpenAI API 호출"""
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
                response = await client.post(self.api_url, headers=headers, json=payload)

            if response.status_code != 200:
                return None

            return json.loads(response.json()["choices"][0]["message"]["content"])

        except Exception as e:
            print(f"OpenAI 호출 실패: {e}")
            return None
