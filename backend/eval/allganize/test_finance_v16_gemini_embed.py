# -*- coding: utf-8 -*-
"""
Advisor OSC Finance v1.6 테스트 (Gemini Embedding + GPT-5-mini no thinking)

파이프라인:
- 파서: Gemini OCR (기존 인덱스 사용)
- 임베딩: gemini-embedding-001 (3072 dim, MTEB 1위)
- 리랭킹: Cohere
- LLM: GPT-5-mini (no thinking)
- Judge: GPT-5-mini (no thinking)
"""

import asyncio
import json
import os
import sys
import re
import httpx
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
local_env = Path(__file__).parent / ".env"
if local_env.exists():
    load_dotenv(local_env, override=True)
else:
    load_dotenv(PROJECT_ROOT / ".env.docker", override=True)

import pandas as pd
from openai import AsyncOpenAI

from config import get_settings
from rag.embeddings import SparseEmbeddingService
from rag.embeddings.embedding_service import EmbeddingService
from rag.vectorstore import QdrantVectorStore
from rag.retriever import HierarchicalRetriever, RetrievalConfig

# 설정
VERSION = "v1.6"
COLLECTION_NAME = "advisor_osc_finance_gemini_embed"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class GeminiEmbeddingService(EmbeddingService):
    """Gemini 임베딩 서비스 (gemini-embedding-001)"""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        batch_size: int = 100,
    ):
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
        self.batch_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"

    async def embed_text(self, text: str) -> list[float]:
        """단일 텍스트 임베딩"""
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """다중 텍스트 임베딩 (배치 처리)"""
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """배치 임베딩 (batchEmbedContents API 사용)"""
        cleaned_texts = [self._clean_text(t) for t in texts]

        # batchEmbedContents 요청 형식
        requests = []
        for text in cleaned_texts:
            requests.append({
                "model": f"models/{self.model}",
                "content": {
                    "parts": [{"text": text}]
                }
            })

        payload = {"requests": requests}

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.batch_api_url}?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
            )

        if response.status_code != 200:
            raise Exception(
                f"Gemini 임베딩 API 오류: {response.status_code} - {response.text}"
            )

        result = response.json()
        embeddings = []
        for item in result.get("embeddings", []):
            embeddings.append(item.get("values", []))

        return embeddings

    def _clean_text(self, text: str) -> str:
        """임베딩을 위한 텍스트 정제"""
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        if len(text) > 8000:
            text = text[:8000]
        return text


# 시스템 프롬프트
SYSTEM_PROMPT = """당신은 문서 분석 AI 어시스턴트입니다.
제공된 컨텍스트를 기반으로 사용자의 질문에 정확하고 완전하게 답변해주세요.

**핵심 규칙:**
1. 컨텍스트의 모든 관련 정보를 빠짐없이 포함하세요
2. 수치/데이터는 단위와 함께 정확히 인용하세요
3. 질문에 여러 하위 질문이 있으면 각각 답변하세요
4. 테이블에서 조건에 맞는 모든 항목을 나열하세요
5. 구체적인 사례/예시가 컨텍스트에 있으면 반드시 포함하세요
6. 핵심 키워드(고유명사, 법률명, 수치 등)를 누락하지 마세요
7. 출처(문서명, 페이지)를 명시하세요

**주의:**
- 컨텍스트에서 답을 찾을 수 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요."""

JUDGE_PROMPT = """다음 질문에 대한 생성된 답변이 정답과 의미적으로 일치하는지 평가해주세요.

질문: {question}
정답: {target_answer}
생성된 답변: {generated_answer}

평가 기준:
1. 핵심 정보(숫자, 날짜, 기관명, 정책명 등)가 정확히 일치하는가?
2. 의미적으로 동일한 내용을 전달하는가?

JSON 형식으로만 응답: {{"verdict": "O" 또는 "X", "reason": "간단한 이유"}}"""

CONCURRENCY = 5


def load_test_data():
    """테스트 데이터 로드"""
    csv_path = Path(__file__).parent / "dataset" / "allganize_test_dataset_mapped.csv"
    df = pd.read_csv(csv_path)
    df = df[df["domain"] == "finance"]
    data = []
    for idx, row in df.iterrows():
        data.append({
            "idx": len(data) + 1,
            "question": row["question"],
            "target_answer": row["target_answer"],
            "context_type": row.get("context_type", "unknown"),
        })
    return data


async def call_gpt5_mini(prompt: str, system_prompt: str = None) -> str:
    """GPT-5-mini API 호출 (no thinking)"""
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            max_completion_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"GPT-5-mini 호출 오류: {e}")
    return None


async def generate_answer(query: str, context: str) -> str:
    """RAG 답변 생성"""
    user_prompt = f"""다음 컨텍스트를 참고하여 질문에 완전하게 답변해주세요.
모든 관련 정보, 수치, 사례를 빠짐없이 포함해주세요.

## 컨텍스트
{context}

## 질문
{query}"""

    answer = await call_gpt5_mini(user_prompt, SYSTEM_PROMPT)
    return answer if answer else "답변 생성 실패"


async def call_judge(question: str, target: str, answer: str) -> dict:
    """LLM-as-Judge 평가 (GPT-5-mini, no thinking)"""
    prompt = JUDGE_PROMPT.format(
        question=question, target_answer=target, generated_answer=answer
    )
    try:
        text = await call_gpt5_mini(prompt)
        if text:
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                return json.loads(json_match.group())
    except Exception as e:
        pass
    return {"verdict": "X", "reason": "평가 실패"}


async def process_single_case(item, retriever, config, semaphore, progress_counter, total):
    """단일 테스트 케이스 처리 (병렬)"""
    async with semaphore:
        idx = item["idx"]
        question = item["question"]
        target = item["target_answer"]
        context_type = item["context_type"]

        try:
            # 검색 수행
            retrieval_result = await retriever.retrieve_with_context(
                query=question,
                config=config,
            )

            # 답변 생성
            answer = await generate_answer(
                query=question,
                context=retrieval_result.context
            )

            # LLM-as-Judge 평가
            judge_result = await call_judge(question, target, answer)
            verdict = judge_result.get("verdict", "X")
            reason = judge_result.get("reason", "")

            is_correct = verdict == "O"

            progress_counter[0] += 1
            status = "O" if is_correct else "X"
            print(f"  [{progress_counter[0]}/{total}] idx={idx} -> {status}")

            return {
                "idx": idx,
                "context_type": context_type,
                "question": question,
                "target_answer": target,
                "generated_answer": answer,
                "verdict": verdict,
                "reason": reason,
                "is_correct": is_correct,
            }

        except Exception as e:
            progress_counter[0] += 1
            print(f"  [{progress_counter[0]}/{total}] idx={idx} -> 오류: {e}")
            return {
                "idx": idx,
                "context_type": context_type,
                "question": question,
                "target_answer": target,
                "generated_answer": f"오류: {e}",
                "verdict": "X",
                "reason": "실행 오류",
                "is_correct": False,
            }


class GeminiMultimodalEmbeddingService:
    """Gemini 기반 멀티모달 임베딩 서비스"""

    def __init__(
        self,
        dense_service: GeminiEmbeddingService,
        sparse_service: SparseEmbeddingService = None,
    ):
        self.dense_service = dense_service
        self.sparse_service = sparse_service or SparseEmbeddingService()

    async def embed_query(self, query: str):
        """쿼리 임베딩"""
        dense_embedding = await self.dense_service.embed_text(query)
        sparse_embedding = self.sparse_service.embed_text(query)
        return dense_embedding, sparse_embedding


async def run_test():
    """테스트 실행 (병렬 처리)"""
    print("=" * 60)
    print(f"Advisor OSC Finance {VERSION} 테스트")
    print("=" * 60)
    print(f"  컬렉션: {COLLECTION_NAME}")
    print(f"  파서: Gemini OCR")
    print(f"  임베딩: gemini-embedding-001 (MTEB 1위)")
    print(f"  LLM: GPT-5-mini (no thinking)")
    print(f"  Judge: GPT-5-mini (no thinking)")
    print(f"  Reranker: Cohere")
    print(f"  병렬 처리: {CONCURRENCY}개 동시 실행")
    print(f"  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 테스트 데이터 로드
    test_data = load_test_data()
    total = len(test_data)
    print(f"\n테스트 케이스: {total}개")

    # Gemini 임베딩 서비스 초기화
    settings = get_settings()

    dense_service = GeminiEmbeddingService(
        api_key=GOOGLE_API_KEY,
        model="gemini-embedding-001"
    )
    sparse_service = SparseEmbeddingService()
    embedding_service = GeminiMultimodalEmbeddingService(dense_service, sparse_service)

    # 컬렉션 존재 여부 확인
    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=COLLECTION_NAME,
    )

    # 컬렉션이 없으면 오류 메시지 출력
    try:
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if COLLECTION_NAME not in collection_names:
            print(f"\n❌ 오류: 컬렉션 '{COLLECTION_NAME}'이 존재하지 않습니다.")
            print(f"   먼저 인덱싱을 실행하세요: python index_finance_gemini_embed.py")
            return 0
    except Exception as e:
        print(f"Qdrant 연결 오류: {e}")
        return 0

    retriever = HierarchicalRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        config=RetrievalConfig(),
    )

    config = RetrievalConfig(
        top_k=15,
        rerank_top_k=50,
        rerank=True,
        expand_to_parent=True,
        use_hybrid=False,
    )

    # 테스트 실행 (병렬)
    print(f"\n{'=' * 60}")
    print("테스트 실행 중 (병렬 처리)...")
    print("=" * 60)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    progress_counter = [0]

    tasks = [
        process_single_case(item, retriever, config, semaphore, progress_counter, total)
        for item in test_data
    ]
    results = await asyncio.gather(*tasks)

    # 결과 집계
    by_context = {}
    for r in results:
        context_type = r["context_type"]
        if context_type not in by_context:
            by_context[context_type] = {"correct": 0, "total": 0}
        by_context[context_type]["total"] += 1
        if r["is_correct"]:
            by_context[context_type]["correct"] += 1

    # 결과 출력
    print(f"\n{'=' * 60}")
    print(f"테스트 결과 ({VERSION})")
    print("=" * 60)

    total_correct = sum(1 for r in results if r["is_correct"])
    overall_acc = total_correct / total * 100 if total > 0 else 0

    print(f"\n전체 정확도: {total_correct}/{total} ({overall_acc:.1f}%)")
    print(f"\nContext Type별 성능:")
    print("-" * 40)
    for ct in sorted(by_context.keys()):
        stats = by_context[ct]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {ct}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

    # 실패 케이스 출력
    failed = [r for r in results if not r["is_correct"]]
    if failed:
        print(f"\n실패 케이스 ({len(failed)}개):")
        print("-" * 40)
        for r in failed[:10]:
            print(f"  idx={r['idx']} ({r['context_type']}): {r['reason'][:50]}")

    # 결과 저장
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"finance_{VERSION}_gemini_embed_{timestamp}.json"

    output_data = {
        "version": f"Advisor OSC {VERSION} (Gemini OCR + Gemini Embedding + GPT-5-mini, no thinking)",
        "parser": "gemini",
        "embedding_model": "gemini-embedding-001",
        "llm_model": "gpt-5-mini",
        "judge_model": "gpt-5-mini",
        "reranker": "cohere",
        "collection": COLLECTION_NAME,
        "timestamp": timestamp,
        "total": total,
        "correct": total_correct,
        "accuracy": overall_acc,
        "by_context": {
            ct: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            }
            for ct, stats in by_context.items()
        },
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {output_file}")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    await vector_store.close()
    return overall_acc


if __name__ == "__main__":
    asyncio.run(run_test())
