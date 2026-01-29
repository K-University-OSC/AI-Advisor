"""
RAG 체인 모듈
검색 결과를 바탕으로 LLM 응답 생성
"""

from dataclasses import dataclass, field
from typing import Optional, AsyncIterator
import httpx

from rag.retriever import HierarchicalRetriever, RetrievalResult, RetrievalConfig


@dataclass
class ChatMessage:
    """채팅 메시지"""
    role: str  # user, assistant, system
    content: str


@dataclass
class RAGResponse:
    """RAG 응답"""
    answer: str
    sources: list[dict]
    retrieval_result: Optional[RetrievalResult] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "metadata": self.metadata,
        }


class RAGChain:
    """RAG 체인"""

    SYSTEM_PROMPT = """당신은 행정 업무를 돕는 친절한 AI 도우미입니다.
일반 직원들이 쉽게 이해할 수 있도록 명확하고 친근하게 답변해주세요.

## 핵심 규칙 (V7.6.1)
1. 컨텍스트 정보만 사용
2. 수치/데이터는 단위와 함께 정확히 인용
3. 테이블에서 조건에 맞는 모든 항목 나열
4. 차트/그래프 분석 내용 활용
5. 출처(문서명, 페이지) 명시

## 답변 스타일
- 전문 용어는 쉬운 말로 풀어서 설명하세요.
- 핵심 내용을 먼저 말하고, 세부 사항은 그 다음에 설명하세요.
- 숫자나 비율은 "~입니다"로 명확하게 전달하세요.
- 비교 질문은 표나 항목별로 한눈에 보기 쉽게 정리하세요.
- "컨텍스트", "문서에 따르면" 같은 기술적 표현을 사용하지 마세요.
- 바로 답변 내용부터 시작하세요.

## 답변 원칙
1. 질문에서 물어본 내용에만 집중해서 답변하세요.
2. 모르는 내용은 "확인되지 않습니다"라고 안내하세요.
3. 여러 항목이 해당되면 모두 답변하세요.

## 응답 포맷
- 제목은 **굵은 글씨**로 표시하세요.
- 여러 항목은 번호(1. 2. 3.)나 글머리 기호(-)로 정리하세요.
- 복잡한 비교는 표로 정리해서 보여주세요.

답변 마지막에 참고 문서명을 간단히 안내해주세요."""

    def __init__(
        self,
        retriever: HierarchicalRetriever,
        api_key: str,
        model: str = "gpt-5.2",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ):
        """
        Args:
            retriever: 계층적 검색기
            api_key: OpenAI API 키
            model: 사용할 LLM 모델
            temperature: 응답 다양성
            max_tokens: 최대 토큰 수
        """
        self.retriever = retriever
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://api.openai.com/v1/chat/completions"

    async def chat(
        self,
        query: str,
        conversation_history: Optional[list[ChatMessage]] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
    ) -> RAGResponse:
        """
        RAG 기반 채팅

        Args:
            query: 사용자 질문
            conversation_history: 이전 대화 기록
            retrieval_config: 검색 설정

        Returns:
            RAG 응답
        """
        history_dicts = None
        if conversation_history:
            history_dicts = [
                {"role": m.role, "content": m.content}
                for m in conversation_history
            ]
        retrieval_result = await self.retriever.retrieve_with_context(
            query=query,
            conversation_history=history_dicts,
            config=retrieval_config,
        )

        answer = await self._generate_answer(
            query=query,
            context=retrieval_result.context,
            conversation_history=conversation_history,
        )

        return RAGResponse(
            answer=answer,
            sources=retrieval_result.sources,
            retrieval_result=retrieval_result,
            metadata={
                "model": self.model,
                "temperature": self.temperature,
            },
        )

    async def chat_stream(
        self,
        query: str,
        conversation_history: Optional[list[ChatMessage]] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
    ) -> AsyncIterator[str]:
        """
        스트리밍 RAG 채팅

        Args:
            query: 사용자 질문
            conversation_history: 이전 대화 기록
            retrieval_config: 검색 설정

        Yields:
            응답 텍스트 청크
        """
        history_dicts = None
        if conversation_history:
            history_dicts = [
                {"role": m.role, "content": m.content}
                for m in conversation_history
            ]

        retrieval_result = await self.retriever.retrieve_with_context(
            query=query,
            conversation_history=history_dicts,
            config=retrieval_config,
        )

        async for chunk in self._generate_answer_stream(
            query=query,
            context=retrieval_result.context,
            conversation_history=conversation_history,
        ):
            yield chunk

    async def _generate_answer(
        self,
        query: str,
        context: str,
        conversation_history: Optional[list[ChatMessage]] = None,
    ) -> str:
        """LLM으로 답변 생성"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]

        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        user_message = f"""다음 컨텍스트를 참고하여 질문에 답변해주세요.

## 컨텍스트
{context}

## 질문
{query}"""

        messages.append({"role": "user", "content": user_message})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )

        if response.status_code != 200:
            raise Exception(
                f"OpenAI API 오류: {response.status_code} - {response.text}"
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def _generate_answer_stream(
        self,
        query: str,
        context: str,
        conversation_history: Optional[list[ChatMessage]] = None,
    ) -> AsyncIterator[str]:
        """LLM으로 스트리밍 답변 생성"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]

        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        user_message = f"""다음 컨텍스트를 참고하여 질문에 답변해주세요.

## 컨텍스트
{context}

## 질문
{query}"""

        messages.append({"role": "user", "content": user_message})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                self.api_url,
                headers=headers,
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
