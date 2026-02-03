'''
LLM Factory 모듈
다양한 LLM 제공자들의 모델을 생성하는 팩토리 패턴 구현

@modified:
2025.07.09 LLM 관련 코드를 별도 파일로 분리
2025.07.28 frequency_penalty 경고 수정
2025.08.27 pplx 모델 변경. gmn15 삭제. gpt5 추가
2025.09.17 Perplexity search_results 출처 처리 추가, O3-Pro 모델을 위한 LangChain 커스텀 LLM 래퍼 삭제
2025.09.19 pplxp 추가
2025.11.29 Perplexity 커스텀 래퍼 추가 (citations 지원)
'''
import os
from dotenv import load_dotenv
# .env 파일의 값을 시스템 환경변수보다 우선하도록 override=True 설정
load_dotenv(override=True)
from typing import Any, Optional, List, Dict, Iterator, AsyncIterator
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatPerplexity
from langchain_anthropic import ChatAnthropic
import openai
import httpx
import json

# 환경 변수 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# EDU KEY
OPENAI_API_KEY_EDU = os.getenv("OPENAI_API_KEY_EDU")
PERPLEXITY_API_KEY_EDU = os.getenv("PERPLEXITY_API_KEY_EDU")
GOOGLE_API_KEY_EDU = os.getenv("GOOGLE_API_KEY_EDU")
OPENAI_API_KEY_SOJANG = os.getenv("OPENAI_API_KEY_SOJANG")
GOOGLE_API_KEY_SOJANG = os.getenv("GOOGLE_API_KEY_SOJANG")

# 명시적으로 export할 함수들
__all__ = ['get_llm', 'get_perplexity_search_results', 'PerplexityChatModel', 'tavily_search']

# 모듈 버전 정보
__version__ = '1.0.0'

import requests


# =============================================================================
# Tavily 웹 검색 함수
# =============================================================================
def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Tavily API를 사용한 웹 검색

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수

    Returns:
        포맷된 검색 결과 문자열
    """
    if not TAVILY_API_KEY:
        return "Tavily API 키가 설정되지 않았습니다."

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False
            },
            timeout=30
        )

        if response.status_code != 200:
            return f"검색 오류: {response.status_code}"

        data = response.json()
        results = data.get("results", [])
        answer = data.get("answer", "")

        # 결과 포맷팅
        formatted = []
        if answer:
            formatted.append(f"요약: {answer}\n")

        for i, r in enumerate(results, 1):
            formatted.append(f"{i}. {r.get('title', 'No title')}")
            formatted.append(f"   URL: {r.get('url', '')}")
            formatted.append(f"   {r.get('content', '')[:300]}...")
            formatted.append("")

        return "\n".join(formatted) if formatted else "검색 결과가 없습니다."

    except Exception as e:
        return f"검색 오류: {str(e)}"


class PerplexityChatModel(BaseChatModel):
    """
    Perplexity API를 직접 호출하는 커스텀 Chat 모델
    스트리밍 + citations(출처) 지원
    """
    model: str = "sonar-pro"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000
    _citations: List[str] = []

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "perplexity"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model, "temperature": self.temperature}

    def _convert_messages_to_perplexity_format(self, messages: List[BaseMessage]) -> List[Dict]:
        """LangChain 메시지를 Perplexity API 형식으로 변환"""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Perplexity API는 system role 지원
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                # 멀티모달 메시지 처리
                if isinstance(msg.content, list):
                    # 이미지 + 텍스트
                    content_parts = []
                    for part in msg.content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                content_parts.append({"type": "text", "text": part.get("text", "")})
                            elif part.get("type") == "image_url":
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": part.get("image_url", {})
                                })
                        else:
                            content_parts.append({"type": "text", "text": str(part)})
                    result.append({"role": "user", "content": content_parts})
                else:
                    result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return result

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> Any:
        """동기 호출 (non-streaming)"""
        from langchain_core.outputs import ChatResult, ChatGeneration

        perplexity_messages = self._convert_messages_to_perplexity_format(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": perplexity_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False
        }

        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=120
        )

        if response.status_code != 200:
            raise Exception(f"Perplexity API 오류: {response.status_code} - {response.text}")

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        self._citations = result.get("citations", [])

        # <think>...</think> 태그 제거 (정규식 사용)
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # citations를 응답 끝에 추가
        if self._citations:
            citations_text = "\n\n---\n**출처:**\n"
            for i, url in enumerate(self._citations, 1):
                citations_text += f"{i}. {url}\n"
            content += citations_text

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> AsyncIterator[ChatGenerationChunk]:
        """비동기 스트리밍 호출 - citations 포함, <think> 태그 필터링"""
        perplexity_messages = self._convert_messages_to_perplexity_format(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": perplexity_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True
        }

        citations = []
        # <think> 태그 필터링을 위한 버퍼
        buffer = ""
        in_think_tag = False

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"Perplexity API 오류: {response.status_code} - {error_text.decode()}")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        json_str = line[6:]
                        if json_str.strip() == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(json_str)

                            # citations 수집 (첫 번째 청크에서 주로 옴)
                            if "citations" in chunk_data and chunk_data["citations"]:
                                citations = chunk_data["citations"]

                            # 컨텐츠 추출
                            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                # <think> 태그 필터링 로직
                                buffer += content

                                # <think> 태그가 시작되면 필터링 모드 진입
                                while True:
                                    if not in_think_tag:
                                        # <think> 시작 태그 찾기
                                        think_start = buffer.find("<think>")
                                        if think_start != -1:
                                            # <think> 이전 내용 출력
                                            if think_start > 0:
                                                output = buffer[:think_start]
                                                yield ChatGenerationChunk(
                                                    message=AIMessageChunk(content=output)
                                                )
                                            buffer = buffer[think_start + 7:]  # "<think>" 제거
                                            in_think_tag = True
                                        else:
                                            # "<think" 부분 문자열이 있을 수 있으니 마지막 7자는 버퍼에 유지
                                            if len(buffer) > 7:
                                                output = buffer[:-7]
                                                buffer = buffer[-7:]
                                                if output:
                                                    yield ChatGenerationChunk(
                                                        message=AIMessageChunk(content=output)
                                                    )
                                            break
                                    else:
                                        # </think> 종료 태그 찾기
                                        think_end = buffer.find("</think>")
                                        if think_end != -1:
                                            buffer = buffer[think_end + 8:]  # "</think>" 제거
                                            in_think_tag = False
                                        else:
                                            # 아직 종료 태그가 없으면 버퍼 유지
                                            break
                        except json.JSONDecodeError:
                            continue

        # 스트리밍 종료 후 남은 버퍼 처리
        if buffer and not in_think_tag:
            # 남은 버퍼에서 <think> 태그 없는지 확인
            if "<think>" not in buffer:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=buffer)
                )

        # 스트리밍 완료 후 citations 추가
        if citations:
            citations_text = "\n\n---\n**출처:**\n"
            for i, url in enumerate(citations, 1):
                citations_text += f"{i}. {url}\n"
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=citations_text)
            )

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> Iterator[ChatGenerationChunk]:
        """동기 스트리밍 호출 - <think> 태그 필터링"""
        perplexity_messages = self._convert_messages_to_perplexity_format(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": perplexity_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True
        }

        citations = []
        # <think> 태그 필터링을 위한 버퍼
        buffer = ""
        in_think_tag = False

        with requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            stream=True,
            timeout=120
        ) as response:
            if response.status_code != 200:
                raise Exception(f"Perplexity API 오류: {response.status_code} - {response.text}")

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        json_str = line[6:]
                        if json_str.strip() == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(json_str)

                            # citations 수집
                            if "citations" in chunk_data and chunk_data["citations"]:
                                citations = chunk_data["citations"]

                            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                # <think> 태그 필터링 로직
                                buffer += content

                                while True:
                                    if not in_think_tag:
                                        think_start = buffer.find("<think>")
                                        if think_start != -1:
                                            if think_start > 0:
                                                output = buffer[:think_start]
                                                yield ChatGenerationChunk(
                                                    message=AIMessageChunk(content=output)
                                                )
                                            buffer = buffer[think_start + 7:]
                                            in_think_tag = True
                                        else:
                                            if len(buffer) > 7:
                                                output = buffer[:-7]
                                                buffer = buffer[-7:]
                                                if output:
                                                    yield ChatGenerationChunk(
                                                        message=AIMessageChunk(content=output)
                                                    )
                                            break
                                    else:
                                        think_end = buffer.find("</think>")
                                        if think_end != -1:
                                            buffer = buffer[think_end + 8:]
                                            in_think_tag = False
                                        else:
                                            break
                        except json.JSONDecodeError:
                            continue

        # 스트리밍 종료 후 남은 버퍼 처리
        if buffer and not in_think_tag:
            if "<think>" not in buffer:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=buffer)
                )

        # 스트리밍 완료 후 citations 추가
        if citations:
            citations_text = "\n\n---\n**출처:**\n"
            for i, url in enumerate(citations, 1):
                citations_text += f"{i}. {url}\n"
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=citations_text)
            )


def get_perplexity_search_results(prompt: str, api_key: str, model: str = "sonar-pro") -> List[Dict]:
    """
    Perplexity API를 직접 호출하여 search_results를 가져오는 함수
    
    Args:
        prompt: 질문 프롬프트
        api_key: Perplexity API 키
        model: 사용할 모델명 (기본값: sonar-pro)
    
    Returns:
        search_results 리스트
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.7,
            "stream": False
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            search_results = result.get("search_results", [])
            return search_results
        else:
            print(f"Perplexity API 오류: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"Perplexity search_results 가져오기 오류: {e}")
        return []

                
def get_llm(user_settings: dict):
    """
    사용자 설정에 따라 적절한 LLM 인스턴스를 생성하는 팩토리 함수

    Args:
        user_settings (dict): 사용자 설정 딕셔너리
            - model_provider: 모델 제공자 (pplx, pplxp, gmn25f, gmn25, cld4o, gpt5, gpt52, gpt5m, gpt5n 등)
            - temperature: 온도 설정 (기본값: 0.7)

    Returns:
        LLM 인스턴스
    """
    model_provider = user_settings.get("model_provider", "default")

    # 환경변수 기반 기본 모델 사용 (Advisor OSC v1.6)
    if model_provider == "default" or model_provider is None:
        llm_provider = os.getenv("LLM_PROVIDER", "google").lower()
        llm_model = os.getenv("LLM_MODEL", "gemini-3-flash-preview")

        if llm_provider == "google":
            return ChatGoogleGenerativeAI(
                model=llm_model,
                google_api_key=GOOGLE_API_KEY,
                temperature=user_settings.get("temperature", 0.7),
                max_output_tokens=5000,
                disable_streaming=False
            )
        elif llm_provider == "openai":
            return ChatOpenAI(
                model=llm_model,
                max_completion_tokens=2000,
                streaming=True,
                openai_api_key=OPENAI_API_KEY
            )
        elif llm_provider == "anthropic":
            return ChatAnthropic(
                model=llm_model,
                api_key=CLAUDE_API_KEY,
                temperature=user_settings.get("temperature", 0.7),
                max_tokens=2000,
                streaming=True
            )
        else:
            # 알 수 없는 provider는 Google Gemini 사용
            return ChatGoogleGenerativeAI(
                model=llm_model,
                google_api_key=GOOGLE_API_KEY,
                temperature=user_settings.get("temperature", 0.7),
                max_output_tokens=5000,
                disable_streaming=False
            )
    
    if model_provider == "pplxp": # perplexity sonar-pro (출처 포함)
        return PerplexityChatModel(
            model="sonar-pro",
            api_key=PERPLEXITY_API_KEY,
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000
        )
    elif model_provider == "pplx":  # perplexity sonar-reasoning (추론 모델, 출처 포함)
        return PerplexityChatModel(
            model="sonar-reasoning",
            api_key=PERPLEXITY_API_KEY,
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000
        )
    elif model_provider == "gmn30":  # Gemini 3.0 Pro
        return ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",
            google_api_key=GOOGLE_API_KEY,
            temperature=user_settings.get("temperature", 0.7),
            max_output_tokens=5000,
            disable_streaming=False
        )
    elif model_provider == "gmn25f":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=user_settings.get("temperature", 0.7),
            max_output_tokens=5000,
            disable_streaming=False  # False로 설정하면 스트리밍 활성화
        )
    elif model_provider == "gmn25":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=user_settings.get("temperature", 0.7),
            max_output_tokens=5000,
            disable_streaming=False  # False로 설정하면 스트리밍 활성화
        )
    elif model_provider == "cld45o":  # Claude 4.5 Opus
        return ChatAnthropic(
            model="claude-opus-4-5-20251101",
            api_key=CLAUDE_API_KEY,
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            streaming=True
        )
    elif model_provider == "cld45s":  # Claude 4.5 Sonnet
        return ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            api_key=CLAUDE_API_KEY,
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            streaming=True
        )
    elif model_provider == "cld4o":
        return ChatAnthropic(
            model="claude-opus-4-20250514",
            api_key=CLAUDE_API_KEY,
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            streaming=True
        )
    elif model_provider == "gpt4o":  # openai gpt-4o
        return ChatOpenAI(
            model="gpt-4o",
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )
    elif model_provider == "gpt4om":  # openai gpt-4o-mini
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )
    elif model_provider == "gpt41":  # openai gpt-4.1
        return ChatOpenAI(
            model="gpt-4.1",
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )
    elif model_provider == "gpto3":  # openai o3
        return ChatOpenAI(
            model="o3",
            max_completion_tokens=2000,
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )
    elif model_provider == "gpto3p":  # openai o3-pro
        return ChatOpenAI(
            model="o3-pro",
            max_completion_tokens=2000,
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )
    elif model_provider == "gpt5":  # openai gpt-5
        return ChatOpenAI(
            model="gpt-5",
            max_completion_tokens=2000,
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )
    elif model_provider == "gpt5m":  # openai gpt-5-mini
        return ChatOpenAI(
            model="gpt-5-mini",
            max_completion_tokens=2000,
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )
    elif model_provider == "gpt5n":  # openai gpt-5-nano
        return ChatOpenAI(
            model="gpt-5-nano",
            max_completion_tokens=2000,
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )
    elif model_provider == "gpt52":  # openai gpt-5.2
        return ChatOpenAI(
            model="gpt-5.2",
            max_completion_tokens=2000,
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )
    ### EDU key 추가
    elif model_provider == "gpt4o-edu": # openai gpt-4o
        return ChatOpenAI(
            model="gpt-4o",
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000,
            frequency_penalty=user_settings.get("frequency_penalty", 0.0),
            streaming=True,
            openai_api_key=OPENAI_API_KEY_SOJANG
        )
    elif model_provider == "pplx-edu": # perplexity (EDU, 출처 포함)
        return PerplexityChatModel(
            model="sonar-pro",
            api_key=PERPLEXITY_API_KEY_EDU,
            temperature=user_settings.get("temperature", 0.7),
            max_tokens=2000
        )
    elif model_provider == "gmn15-edu": 
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            google_api_key=GOOGLE_API_KEY_SOJANG,
            temperature=user_settings.get("temperature", 0.7),
            max_output_tokens=5000,
            disable_streaming=False  # False로 설정하면 스트리밍 활성화
        )
    elif model_provider == "gmn25f-edu": 
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY_SOJANG,
            temperature=user_settings.get("temperature", 0.7),
            max_output_tokens=5000,
            disable_streaming=False  # False로 설정하면 스트리밍 활성화
        )
    elif model_provider == "gmn25-edu":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=GOOGLE_API_KEY_SOJANG,
            temperature=user_settings.get("temperature", 0.7),
            max_output_tokens=5000,
            disable_streaming=False  # False로 설정하면 스트리밍 활성화
        )
    else:
        # 기본값: GPT-5-mini 사용
        return ChatOpenAI(
            model="gpt-5-mini",
            max_completion_tokens=2000,
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )

