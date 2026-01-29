"""
웹 검색 도구 (Web Search Tool)
모든 LLM에서 사용 가능한 독립적인 검색 도구
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Tavily API 키
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


class SearchResult(BaseModel):
    """검색 결과 모델"""
    title: str
    url: str
    content: str
    score: Optional[float] = None


class WebSearchTool:
    """
    Tavily 기반 웹 검색 도구
    GPT-5, Claude, Gemini 등 모든 모델에서 사용 가능
    """

    def __init__(self, api_key: str = None, max_results: int = 5):
        self.api_key = api_key or TAVILY_API_KEY
        self.max_results = max_results
        self._tavily_client = None

    def _get_client(self):
        """Tavily 클라이언트 lazy 초기화"""
        if self._tavily_client is None:
            try:
                from tavily import TavilyClient
                self._tavily_client = TavilyClient(api_key=self.api_key)
            except ImportError:
                raise ImportError("tavily-python 패키지가 설치되지 않았습니다. pip install tavily-python")
        return self._tavily_client

    def search(self, query: str, search_depth: str = "basic") -> Dict[str, Any]:
        """
        웹 검색 수행

        Args:
            query: 검색 쿼리
            search_depth: 검색 깊이 ("basic" 또는 "advanced")

        Returns:
            검색 결과 딕셔너리
        """
        if not self.api_key:
            return {
                "query": query,
                "error": "TAVILY_API_KEY가 설정되지 않았습니다.",
                "results": [],
                "timestamp": datetime.now().isoformat()
            }

        try:
            client = self._get_client()
            response = client.search(
                query=query,
                search_depth=search_depth,
                max_results=self.max_results,
                include_answer=True,
                include_raw_content=False
            )

            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0)
                })

            return {
                "query": query,
                "answer": response.get("answer", ""),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "timestamp": datetime.now().isoformat()
            }

    def format_results_for_llm(self, search_result: Dict[str, Any]) -> str:
        """
        검색 결과를 LLM 프롬프트용으로 포맷팅

        중간 과정(쿼리, 시간, URL 등)은 숨기고 핵심 내용만 전달

        Args:
            search_result: search() 메서드의 반환값

        Returns:
            포맷팅된 문자열
        """
        if "error" in search_result and search_result["error"]:
            return f"검색 오류: {search_result['error']}"

        output = []

        # 요약 답변이 있으면 먼저 표시
        if search_result.get("answer"):
            output.append(f"**요약:** {search_result['answer']}")
            output.append("")

        # 검색 결과 - 내용 중심으로 간결하게
        results = search_result.get("results", [])
        if results:
            output.append("**검색된 정보:**")
            for i, result in enumerate(results, 1):
                title = result['title']
                content = result['content'][:400]
                output.append(f"\n{i}. **{title}**")
                output.append(f"   {content}")

            # 출처 정보는 별도 섹션으로 (LLM이 그대로 사용할 수 있도록)
            output.append("")
            output.append("**참고 출처:**")
            for result in results:
                title = result['title']
                url = result['url']
                output.append(f"- [{title}]({url})")

        return "\n".join(output)


# LangChain Tool로 사용할 수 있는 함수
@tool
def web_search(query: str) -> str:
    """
    실시간 웹 검색을 수행합니다. 최신 정보, 뉴스, 현재 이벤트 등을 검색할 때 사용하세요.

    Args:
        query: 검색할 내용

    Returns:
        검색 결과 문자열
    """
    search_tool = WebSearchTool()
    result = search_tool.search(query)
    return search_tool.format_results_for_llm(result)


# 검색 필요 여부 판단 함수
def needs_web_search(message: str) -> bool:
    """
    메시지가 웹 검색이 필요한지 판단

    Args:
        message: 사용자 메시지

    Returns:
        웹 검색 필요 여부
    """
    # 실시간 정보가 필요한 키워드
    search_keywords = [
        # 시간 관련
        "오늘", "어제", "이번주", "이번달", "올해", "최근", "현재", "지금",
        "today", "yesterday", "this week", "this month", "this year", "recent", "current", "now",
        # 정보 요청
        "뉴스", "소식", "속보", "발표", "news", "update", "announcement",
        # 가격/시세
        "가격", "시세", "환율", "주가", "price", "rate", "stock",
        # 날씨
        "날씨", "기온", "weather", "temperature",
        # 검색 의도
        "검색", "찾아", "알아봐", "search", "find", "look up",
        # 비교/순위
        "비교", "순위", "랭킹", "best", "top", "ranking",
        # 이벤트
        "경기", "결과", "일정", "game", "result", "schedule"
    ]

    message_lower = message.lower()
    return any(keyword in message_lower for keyword in search_keywords)
