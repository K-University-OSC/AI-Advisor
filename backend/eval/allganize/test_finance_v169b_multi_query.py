# -*- coding: utf-8 -*-
"""
Advisor OSC Finance v1.6.9b 테스트 (MultiQueryRetriever Only)

v1.6.8 (GPT-4o Vision OCR) + MultiQueryRetriever:
- 원본 쿼리 + 3개 대안 쿼리 생성 (LLM)
- 각 쿼리로 병렬 검색
- 결과 통합 및 중복 제거
- BGE Reranker로 리랭킹

파이프라인:
- Parser: GPT-4o Vision OCR (페이지별 병렬)
- Chunking: HierarchicalChunker
- Retriever: MultiQueryRetriever (NEW)
- Embedding: OpenAI text-embedding-3-large (3072 dim)
- Reranker: BGE v2-m3 (GPU, Thread-safe)
- LLM: GPT-5-mini (no thinking)
- Judge: GPT-5-mini (no thinking)

Collection: advisor_osc_finance_v168_gpt4o_ocr (v1.6.8과 공유)
"""

import asyncio
import json
import os
import sys
import re
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# BGE Reranker 설정 (GPU 1번 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["BGE_DEVICE"] = "cuda"
os.environ["BGE_USE_FP16"] = "false"

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
import httpx

from config import get_settings
from rag.embeddings import OpenAIEmbeddingService, SparseEmbeddingService, MultimodalEmbeddingService
from rag.vectorstore import QdrantVectorStore, HybridSearchConfig

# v1.6.9b 설정
VERSION = "v1.6.9b"
COLLECTION_NAME = "advisor_osc_finance_v168_gpt4o_ocr"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 병렬 처리 설정
TEST_CONCURRENCY = 5
NUM_SUB_QUERIES = 3  # 생성할 대안 쿼리 수

SYSTEM_PROMPT = """당신은 문서 분석 AI 어시스턴트입니다.
제공된 컨텍스트를 기반으로 사용자의 질문에 정확하고 완전하게 답변해주세요.

**핵심 규칙:**
1. 컨텍스트의 모든 관련 정보를 빠짐없이 포함하세요
2. 수치/데이터는 단위와 함께 정확히 인용하세요
3. 질문에 여러 하위 질문이 있으면 각각 답변하세요
4. 테이블에서 조건에 맞는 모든 항목을 나열하세요
5. 구체적인 사례/예시가 컨텍스트에 있으면 반드시 포함하세요
6. [IMAGE: 설명] 형식의 이미지/차트 정보도 적극 활용하세요

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

MULTI_QUERY_PROMPT = """주어진 질문의 의미를 유지하면서 다른 관점의 대안 질문 3개를 생성하세요.

규칙:
1. 원본 의미 유지
2. 각각 다른 표현 사용
3. 반드시 JSON 형식으로 응답

응답 예시:
{"queries": ["대안 질문1", "대안 질문2", "대안 질문3"]}"""


# ============================================================================
# BGE Reranker (Thread-safe)
# ============================================================================

class BGERerankerWrapper:
    """BGE Reranker with Thread-safe Lock"""

    def __init__(self):
        self._model = None
        self._initialized = False
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        if self._initialized:
            return True
        try:
            from FlagEmbedding import FlagReranker
            device = os.getenv("BGE_DEVICE", "cuda")
            use_fp16 = os.getenv("BGE_USE_FP16", "false").lower() == "true"
            device_str = "cuda:0" if device == "cuda" else "cpu"
            print(f"BGE Reranker 로딩: BAAI/bge-reranker-v2-m3 (device={device_str}, FP16={use_fp16})")
            self._model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=use_fp16, devices=[device_str])
            self._initialized = True
            print("BGE Reranker 초기화 완료")
            return True
        except Exception as e:
            print(f"BGE Reranker 초기화 실패: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def rerank(self, query: str, documents: List[str], top_k: int = 15) -> List[tuple]:
        if not self._initialized or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]
        try:
            pairs = [[query, doc] for doc in documents]
            with self._lock:
                scores = self._model.compute_score(pairs)
            if not isinstance(scores, list):
                scores = [scores]
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs[:top_k]
        except Exception as e:
            print(f"BGE 리랭킹 실패: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]

    async def rerank_async(self, query: str, documents: List[str], top_k: int = 15) -> List[tuple]:
        return await asyncio.to_thread(self.rerank, query, documents, top_k)


# ============================================================================
# Multi-Query Generator
# ============================================================================

async def generate_sub_queries(query: str) -> List[str]:
    """LLM을 사용하여 대안 쿼리 생성 (gpt-5-mini)"""
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": MULTI_QUERY_PROMPT},
                {"role": "user", "content": f"질문: {query}"}
            ],
            max_completion_tokens=500,
        )
        # JSON 파싱 (더 넓은 패턴)
        text = response.choices[0].message.content
        # 전체 JSON 블록 추출
        json_match = re.search(r'\{[\s\S]*?"queries"[\s\S]*?\[[\s\S]*?\][\s\S]*?\}', text)
        if json_match:
            try:
                result = json.loads(json_match.group())
                queries = result.get("queries", [])
                if queries and isinstance(queries, list):
                    return queries[:3]  # 최대 3개
            except json.JSONDecodeError:
                pass

        # 폴백: 리스트 직접 추출 시도
        list_match = re.search(r'\[([^\]]+)\]', text)
        if list_match:
            items = re.findall(r'"([^"]+)"', list_match.group(1))
            if items:
                return items[:3]

        return []
    except Exception as e:
        print(f"대안 쿼리 생성 실패: {e}")
        return []


# ============================================================================
# Multi-Query Retriever
# ============================================================================

async def multi_query_search(
    query: str,
    vector_store: QdrantVectorStore,
    embedding_service: MultimodalEmbeddingService,
    top_k: int = 50,
) -> List[dict]:
    """
    MultiQuery 검색 수행

    1. 원본 쿼리 + 대안 쿼리 생성
    2. 각 쿼리로 병렬 검색
    3. 결과 통합 및 중복 제거
    """
    # 1. 대안 쿼리 생성
    sub_queries = await generate_sub_queries(query)
    all_queries = [query] + sub_queries[:NUM_SUB_QUERIES]

    # 2. 각 쿼리로 병렬 검색
    async def search_single(q: str):
        dense_embedding, sparse_embedding = await embedding_service.embed_query(q)
        results = await vector_store.search(
            query_vector=dense_embedding,
            config=HybridSearchConfig(top_k=top_k // len(all_queries)),
            only_children=True,
        )
        return results

    # 병렬 검색
    all_results = await asyncio.gather(*[search_single(q) for q in all_queries])

    # 3. 결과 통합 및 중복 제거
    seen_chunks = set()
    merged_results = []

    for results in all_results:
        for result in results:
            if result.chunk_id not in seen_chunks:
                merged_results.append(result)
                seen_chunks.add(result.chunk_id)

    # 점수 기준 정렬
    merged_results.sort(key=lambda x: x.score, reverse=True)

    return merged_results, len(all_queries)


# ============================================================================
# 테스트 함수
# ============================================================================

def load_test_data():
    csv_path = Path(__file__).parent / "dataset" / "allganize_test_dataset_mapped.csv"
    df = pd.read_csv(csv_path)
    df = df[df["domain"] == "finance"]
    return [{"idx": i+1, "question": row["question"], "target_answer": row["target_answer"],
             "context_type": row.get("context_type", "unknown")} for i, row in df.iterrows()]


async def call_gpt5_mini(prompt: str, system_prompt: str = None) -> str:
    """GPT-5-mini (no thinking) 호출"""
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
    user_prompt = f"""다음 컨텍스트를 참고하여 질문에 완전하게 답변해주세요.
[IMAGE: 설명] 형식의 이미지/차트 정보도 적극 활용해주세요.

## 컨텍스트
{context}

## 질문
{query}"""
    answer = await call_gpt5_mini(user_prompt, SYSTEM_PROMPT)
    return answer if answer else "답변 생성 실패"


async def call_judge(question: str, target: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, target_answer=target, generated_answer=answer)
    try:
        text = await call_gpt5_mini(prompt)
        if text:
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                return json.loads(json_match.group())
    except:
        pass
    return {"verdict": "X", "reason": "평가 실패"}


async def process_single_case(item, vector_store, embedding_service, reranker, semaphore, progress_counter, total):
    """단일 테스트 케이스 처리 - MultiQueryRetriever 사용"""
    async with semaphore:
        idx, question, target, context_type = item["idx"], item["question"], item["target_answer"], item["context_type"]
        try:
            # MultiQuery 검색
            search_results, num_queries = await multi_query_search(
                query=question,
                vector_store=vector_store,
                embedding_service=embedding_service,
                top_k=50,
            )

            # BGE Reranker로 리랭킹
            if search_results and reranker.is_initialized:
                documents = [r.content for r in search_results]
                reranked = await reranker.rerank_async(question, documents, top_k=15)
                context = "\n\n---\n\n".join([doc for doc, score in reranked])
            else:
                context = "\n\n---\n\n".join([r.content for r in search_results[:15]])

            answer = await generate_answer(query=question, context=context)
            judge_result = await call_judge(question, target, answer)
            verdict = judge_result.get("verdict", "X")
            is_correct = verdict == "O"
            progress_counter[0] += 1

            print(f"  [{progress_counter[0]}/{total}] idx={idx} -> {'O' if is_correct else 'X'} (queries={num_queries}, results={len(search_results)})")

            return {
                "idx": idx,
                "context_type": context_type,
                "question": question,
                "target_answer": target,
                "generated_answer": answer,
                "verdict": verdict,
                "reason": judge_result.get("reason", ""),
                "is_correct": is_correct,
                "num_queries": num_queries,
                "num_results": len(search_results),
            }
        except Exception as e:
            progress_counter[0] += 1
            print(f"  [{progress_counter[0]}/{total}] idx={idx} -> 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                "idx": idx,
                "context_type": context_type,
                "question": question,
                "target_answer": target,
                "generated_answer": f"오류: {e}",
                "verdict": "X",
                "reason": "실행 오류",
                "is_correct": False,
                "num_queries": 0,
                "num_results": 0,
            }


async def run_test():
    print("=" * 60)
    print(f"Advisor OSC Finance {VERSION} 테스트")
    print("=" * 60)
    print(f"  컬렉션: {COLLECTION_NAME}")
    print(f"  Retriever: MultiQueryRetriever")
    print(f"  임베딩: OpenAI text-embedding-3-large")
    print(f"  Reranker: BGE v2-m3 (GPU, Thread-safe)")
    print(f"  LLM: GPT-5-mini (no thinking)")
    print(f"  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n핵심 변경 (vs v1.6.8):")
    print(f"  - MultiQueryRetriever 적용")
    print(f"  - 원본 쿼리 + {NUM_SUB_QUERIES}개 대안 쿼리")
    print(f"  - 병렬 검색 후 결과 통합")
    print("=" * 60)

    test_data = load_test_data()
    total = len(test_data)
    print(f"\n테스트 케이스: {total}개")

    settings = get_settings()
    dense_service = OpenAIEmbeddingService(api_key=settings.openai_api_key, model="text-embedding-3-large")
    sparse_service = SparseEmbeddingService()
    embedding_service = MultimodalEmbeddingService(dense_service, sparse_service)

    reranker = BGERerankerWrapper()
    if not reranker.initialize():
        print("BGE Reranker 초기화 실패")
        return 0

    try:
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        collections = [c.name for c in qdrant_client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            print(f"\n컬렉션 '{COLLECTION_NAME}'이 존재하지 않습니다.")
            print("v1.6.8 테스트를 먼저 실행하세요.")
            return 0
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        if collection_info.points_count == 0:
            print(f"\n컬렉션이 비어있습니다.")
            return 0
        print(f"\n컬렉션 정보: {collection_info.points_count} points")
    except Exception as e:
        print(f"Qdrant 연결 오류: {e}")
        return 0

    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=COLLECTION_NAME
    )

    print(f"\n{'=' * 60}\n테스트 실행 중 (병렬 {TEST_CONCURRENCY}개)...\n{'=' * 60}")
    semaphore = asyncio.Semaphore(TEST_CONCURRENCY)
    progress_counter = [0]
    tasks = [
        process_single_case(item, vector_store, embedding_service, reranker, semaphore, progress_counter, total)
        for item in test_data
    ]
    results = await asyncio.gather(*tasks)

    by_context = {}
    for r in results:
        ct = r["context_type"]
        if ct not in by_context:
            by_context[ct] = {"correct": 0, "total": 0}
        by_context[ct]["total"] += 1
        if r["is_correct"]:
            by_context[ct]["correct"] += 1

    total_correct = sum(1 for r in results if r["is_correct"])
    overall_acc = total_correct / total * 100 if total > 0 else 0

    print(f"\n{'=' * 60}\n테스트 결과 ({VERSION})\n{'=' * 60}")
    print(f"\n전체 정확도: {total_correct}/{total} ({overall_acc:.1f}%)")
    print(f"\nContext Type별 성능:")
    for ct in sorted(by_context.keys()):
        stats = by_context[ct]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {ct}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

    # 쿼리 수별 통계
    avg_queries = sum(r["num_queries"] for r in results) / len(results)
    avg_results = sum(r["num_results"] for r in results) / len(results)
    print(f"\n검색 통계:")
    print(f"  평균 쿼리 수: {avg_queries:.1f}")
    print(f"  평균 검색 결과 수: {avg_results:.1f}")

    # 이전 버전과 비교
    print(f"\n{'=' * 60}")
    print("이전 버전 대비 변화:")
    prev_results = {
        "v1.6.6": {"overall": 68.3, "image": 40.0, "paragraph": 76.7, "table": 100.0},
        "v1.6.8": {"overall": 73.3, "image": 70.0, "paragraph": 73.3, "table": 80.0},
    }
    for ver, prev in prev_results.items():
        print(f"\n  vs {ver}:")
        print(f"    전체: {prev['overall']:.1f}% → {overall_acc:.1f}% ({overall_acc - prev['overall']:+.1f}%)")
        for ct in ["image", "paragraph", "table"]:
            if ct in by_context:
                new_acc = by_context[ct]["correct"] / by_context[ct]["total"] * 100
                diff = new_acc - prev.get(ct, 0)
                print(f"    {ct}: {prev.get(ct, 0):.1f}% → {new_acc:.1f}% ({diff:+.1f}%)")
    print("=" * 60)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"finance_{VERSION}_multi_query_{timestamp}.json"

    output_data = {
        "version": VERSION,
        "approach": "v1.6.8 (GPT-4o OCR) + MultiQueryRetriever",
        "pipeline": {
            "retriever": "MultiQueryRetriever",
            "num_sub_queries": NUM_SUB_QUERIES,
            "embedding": "OpenAI text-embedding-3-large",
            "reranker": "BGE v2-m3 (GPU)",
            "llm": "GPT-5-mini (no thinking)",
            "judge": "GPT-5-mini (no thinking)"
        },
        "collection": COLLECTION_NAME,
        "timestamp": timestamp,
        "total": total,
        "correct": total_correct,
        "accuracy": overall_acc,
        "accuracy_by_type": {ct: stats["correct"]/stats["total"]*100 if stats["total"] > 0 else 0
                            for ct, stats in by_context.items()},
        "search_stats": {
            "avg_queries": avg_queries,
            "avg_results": avg_results,
        },
        "results": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {output_file}")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    await vector_store.close()
    return overall_acc


if __name__ == "__main__":
    asyncio.run(run_test())
