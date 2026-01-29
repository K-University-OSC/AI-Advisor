"""
Reranker 모듈
검색 결과를 리랭킹하여 정확도 향상

V7.2: Cohere Reranker 추가
V7.3: Voyage Reranker 추가
V7.6: ColBERT Reranker 추가
- BGE: 로컬 실행, 무료, Cross-Encoder
- Cohere: API 기반, 고성능
- Voyage: API 기반, 최신 (rerank-2.5)
- ColBERT: Token-level 매칭, 정밀한 검색 (RAGatouille)
"""

import os
from typing import Optional, List, Tuple
import httpx


class VoyageReranker:
    """
    Voyage Reranker - V7.3

    Voyage AI의 rerank-2.5 모델을 사용한 고성능 리랭킹
    https://docs.voyageai.com/reference/reranker-api
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-2.5",
    ):
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.model = model
        self.api_url = "https://api.voyageai.com/v1/rerank"
        self._initialized = bool(self.api_key)

    def initialize(self) -> bool:
        """리랭커 초기화 확인"""
        if not self.api_key:
            print("Voyage API Key가 설정되지 않았습니다. VOYAGE_API_KEY 환경변수를 확인하세요.")
            return False
        self._initialized = True
        return True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Voyage API를 사용한 동기 리랭킹
        """
        if not self._initialized or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_k": min(top_k, len(documents)),
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.api_url, headers=headers, json=payload)

            if response.status_code != 200:
                print(f"Voyage API 오류: {response.status_code} - {response.text}")
                return [(doc, 0.0) for doc in documents[:top_k]]

            result = response.json()

            # 결과 파싱
            scored_docs = []
            for item in result.get("data", []):
                idx = item["index"]
                score = item["relevance_score"]
                scored_docs.append((documents[idx], score))

            return scored_docs[:top_k]

        except Exception as e:
            print(f"Voyage 리랭킹 실패: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]

    async def rerank_async(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Voyage API를 사용한 비동기 리랭킹
        """
        if not self._initialized or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_k": min(top_k, len(documents)),
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.api_url, headers=headers, json=payload)

            if response.status_code != 200:
                print(f"Voyage API 오류: {response.status_code}")
                return [(doc, 0.0) for doc in documents[:top_k]]

            result = response.json()

            scored_docs = []
            for item in result.get("data", []):
                idx = item["index"]
                score = item["relevance_score"]
                scored_docs.append((documents[idx], score))

            return scored_docs[:top_k]

        except Exception as e:
            print(f"Voyage 리랭킹 실패: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]


class CohereReranker:
    """
    Cohere Reranker - V7.2

    Cohere의 rerank API를 사용한 고성능 리랭킹
    https://docs.cohere.com/reference/rerank
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-multilingual-v3.0",
    ):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self.api_url = "https://api.cohere.com/v1/rerank"
        self._initialized = bool(self.api_key)

    def initialize(self) -> bool:
        """리랭커 초기화 확인"""
        if not self.api_key:
            print("Cohere API Key가 설정되지 않았습니다. COHERE_API_KEY 환경변수를 확인하세요.")
            return False
        self._initialized = True
        return True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Cohere API를 사용한 동기 리랭킹
        """
        if not self._initialized or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]

        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": min(top_k, len(documents)),
                "return_documents": False,
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.api_url, headers=headers, json=payload)

            if response.status_code != 200:
                print(f"Cohere API 오류: {response.status_code} - {response.text}")
                return [(doc, 0.0) for doc in documents[:top_k]]

            result = response.json()

            # 결과 파싱
            scored_docs = []
            for item in result.get("results", []):
                idx = item["index"]
                score = item["relevance_score"]
                scored_docs.append((documents[idx], score))

            return scored_docs[:top_k]

        except Exception as e:
            print(f"Cohere 리랭킹 실패: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]

    async def rerank_async(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Cohere API를 사용한 비동기 리랭킹
        """
        if not self._initialized or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": min(top_k, len(documents)),
                "return_documents": False,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.api_url, headers=headers, json=payload)

            if response.status_code != 200:
                print(f"Cohere API 오류: {response.status_code}")
                return [(doc, 0.0) for doc in documents[:top_k]]

            result = response.json()

            scored_docs = []
            for item in result.get("results", []):
                idx = item["index"]
                score = item["relevance_score"]
                scored_docs.append((documents[idx], score))

            return scored_docs[:top_k]

        except Exception as e:
            print(f"Cohere 리랭킹 실패: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]


class BGEReranker:
    """BGE Reranker"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialized = False

    def initialize(self) -> bool:
        """리랭커 초기화 - GPU 0 사용"""
        try:
            from sentence_transformers import CrossEncoder
            print(f"BGE Reranker 로딩: {self.model_name} (GPU 0)")
            self.model = CrossEncoder(self.model_name, max_length=512)
            self._initialized = True
            print("BGE Reranker 초기화 완료")
            return True
        except Exception as e:
            print(f"BGE Reranker 초기화 실패: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        문서 리랭킹

        Args:
            query: 검색 쿼리
            documents: 문서 리스트
            top_k: 반환할 상위 문서 수

        Returns:
            (문서, 점수) 튜플 리스트
        """
        if not self._initialized or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]

        try:
            pairs = [(query, doc) for doc in documents]
            scores = self.model.predict(pairs)
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs[:top_k]
        except Exception as e:
            print(f"리랭킹 실패: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]

    def rerank_with_metadata(
        self,
        query: str,
        results: list,
        content_key: str = 'content',
        top_k: int = 5
    ) -> list:
        """
        메타데이터를 유지하면서 리랭킹

        Args:
            query: 검색 쿼리
            results: SearchResult 또는 dict 리스트
            content_key: 컨텐츠 키 이름
            top_k: 반환할 상위 결과 수

        Returns:
            리랭킹된 결과 리스트
        """
        if not self._initialized or not results:
            return results[:top_k]

        try:
            # 컨텐츠 추출
            documents = []
            for r in results:
                if hasattr(r, content_key):
                    documents.append(getattr(r, content_key))
                elif isinstance(r, dict) and content_key in r:
                    documents.append(r[content_key])
                else:
                    documents.append(str(r))

            # 리랭킹
            pairs = [(query, doc) for doc in documents]
            scores = self.model.predict(pairs)

            # 점수와 함께 정렬
            scored_results = list(zip(results, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            return [r for r, s in scored_results[:top_k]]
        except Exception as e:
            print(f"리랭킹 실패: {e}")
            return results[:top_k]


class ColBERTReranker:
    """
    ColBERT Reranker - V7.6

    Token-level Late Interaction을 사용한 정밀 리랭킹
    - 쿼리와 문서의 각 토큰을 개별적으로 비교
    - BGE Cross-Encoder 대비 더 세밀한 매칭 가능
    - 한국어 최적화 모델 지원 (sigridjineth/colbert-small-korean)

    Reference:
    - https://huggingface.co/sigridjineth/colbert-small-korean-20241212
    - https://github.com/AnswerDotAI/rerankers
    """

    def __init__(
        self,
        model_name: str = "sigridjineth/colbert-small-korean-20241212",
        device: str = "cuda:1",
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialized = False

    def initialize(self) -> bool:
        """ColBERT 모델 초기화 - rerankers 라이브러리 사용"""
        try:
            # GPU 1 사용 설정
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

            from rerankers import Reranker
            print(f"ColBERT Reranker 로딩: {self.model_name} (GPU 1)")
            self.model = Reranker(self.model_name, model_type='colbert', verbose=0)
            self._initialized = True
            print("ColBERT Reranker 초기화 완료")
            return True
        except Exception as e:
            print(f"ColBERT Reranker 초기화 실패: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        ColBERT를 사용한 문서 리랭킹

        Args:
            query: 검색 쿼리
            documents: 문서 리스트
            top_k: 반환할 상위 문서 수

        Returns:
            (문서, 점수) 튜플 리스트
        """
        if not self._initialized or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]

        try:
            # rerankers 라이브러리로 리랭킹
            ranked = self.model.rank(query=query, docs=documents)

            # 결과 파싱
            scored_docs = []
            for r in ranked.results:
                doc_text = r.document.text
                score = r.score
                scored_docs.append((doc_text, score))

            return scored_docs[:top_k]

        except Exception as e:
            print(f"ColBERT 리랭킹 실패: {e}")
            import traceback
            traceback.print_exc()
            return [(doc, 0.0) for doc in documents[:top_k]]

    def rerank_with_metadata(
        self,
        query: str,
        results: list,
        content_key: str = 'content',
        top_k: int = 5
    ) -> list:
        """
        메타데이터를 유지하면서 리랭킹

        Args:
            query: 검색 쿼리
            results: SearchResult 또는 dict 리스트
            content_key: 컨텐츠 키 이름
            top_k: 반환할 상위 결과 수

        Returns:
            리랭킹된 결과 리스트
        """
        if not self._initialized or not results:
            return results[:top_k]

        try:
            # 컨텐츠 추출
            documents = []
            for r in results:
                if hasattr(r, content_key):
                    documents.append(getattr(r, content_key))
                elif isinstance(r, dict) and content_key in r:
                    documents.append(r[content_key])
                else:
                    documents.append(str(r))

            # ColBERT rerank
            ranked = self.model.rank(query=query, docs=documents)

            # 인덱스 기반 점수 매핑
            idx_to_score = {}
            for r in ranked.results:
                idx_to_score[r.document.doc_id] = r.score

            # 원본 결과에 점수 매핑하여 정렬
            scored_results = []
            for i, r in enumerate(results):
                score = idx_to_score.get(i, 0.0)
                scored_results.append((r, score))

            scored_results.sort(key=lambda x: x[1], reverse=True)
            return [r for r, s in scored_results[:top_k]]

        except Exception as e:
            print(f"ColBERT 리랭킹 실패: {e}")
            return results[:top_k]


class JinaReranker:
    """
    Jina Reranker v2 - V7.6 대안

    Multilingual 지원, Late Interaction 방식
    - sentence-transformers CrossEncoder 사용
    - 한국어 포함 다국어 지원

    Reference: https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        device: str = "cuda:1",
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialized = False

    def initialize(self) -> bool:
        """Jina Reranker 초기화"""
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

            from sentence_transformers import CrossEncoder
            print(f"Jina Reranker 로딩: {self.model_name} (GPU 1)")
            self.model = CrossEncoder(self.model_name, trust_remote_code=True)
            self._initialized = True
            print("Jina Reranker 초기화 완료")
            return True
        except Exception as e:
            print(f"Jina Reranker 초기화 실패: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """문서 리랭킹"""
        if not self._initialized or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]

        try:
            pairs = [(query, doc) for doc in documents]
            scores = self.model.predict(pairs)
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs[:top_k]
        except Exception as e:
            print(f"Jina 리랭킹 실패: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]

    def rerank_with_metadata(
        self,
        query: str,
        results: list,
        content_key: str = 'content',
        top_k: int = 5
    ) -> list:
        """메타데이터를 유지하면서 리랭킹"""
        if not self._initialized or not results:
            return results[:top_k]

        try:
            documents = []
            for r in results:
                if hasattr(r, content_key):
                    documents.append(getattr(r, content_key))
                elif isinstance(r, dict) and content_key in r:
                    documents.append(r[content_key])
                else:
                    documents.append(str(r))

            pairs = [(query, doc) for doc in documents]
            scores = self.model.predict(pairs)

            scored_results = list(zip(results, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            return [r for r, s in scored_results[:top_k]]
        except Exception as e:
            print(f"Jina 리랭킹 실패: {e}")
            return results[:top_k]
