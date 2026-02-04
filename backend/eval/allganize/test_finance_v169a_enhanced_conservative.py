# -*- coding: utf-8 -*-
"""
Advisor OSC Finance v1.6.9a 테스트 (EnhancedHierarchicalRetriever - 보수적 설정)

v1.6.8 (GPT-4o Vision OCR) + EnhancedHierarchicalRetriever (최소 설정):
- Query Expansion: 활성화 (LLM 기반 쿼리 확장)
- RRF Hybrid Search: 활성화 (Dense + Sparse 순위 융합)
- Query-Adaptive Weights: 활성화
- Table Adaptive: 비활성화 (v1.6.9에서 과도하게 작동)
- Multi-Query: 비활성화 (단순화)
- HyDE: 비활성화

v1.6.9와 차이점:
- enable_table_adaptive=False (테이블 감지 비활성화)
- expand_to_parent=False (child 결과 직접 사용)

파이프라인:
- Parser: GPT-4o Vision OCR (페이지별 병렬)
- Chunking: HierarchicalChunker
- Retriever: EnhancedHierarchicalRetriever (보수적 설정)
- Embedding: OpenAI text-embedding-3-large (3072 dim)
- Reranker: BGE v2-m3 (내장)
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
import uuid
import base64
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
import fitz  # PyMuPDF
from openai import AsyncOpenAI

from config import get_settings
from rag.embeddings import OpenAIEmbeddingService, SparseEmbeddingService, MultimodalEmbeddingService
from rag.vectorstore import QdrantVectorStore
from rag.retriever.enhanced_retriever import (
    EnhancedHierarchicalRetriever,
    EnhancedRetrievalConfig,
)
from rag.parsers.document_parser import ParsedDocument, ParsedElement, ElementType
from rag.chunkers import HierarchicalChunker

# v1.6.9a 설정
VERSION = "v1.6.9a"
COLLECTION_NAME = "advisor_osc_finance_v168_gpt4o_ocr"  # v1.6.8 컬렉션 재사용
PDF_DIR = Path(__file__).parent / "files" / "finance"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 병렬 처리 설정
OCR_CONCURRENCY = 5  # 페이지 OCR 병렬 수
TEST_CONCURRENCY = 5  # 테스트 케이스 병렬 수

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


async def process_single_case(item, retriever, config, semaphore, progress_counter, total):
    """단일 테스트 케이스 처리 - EnhancedHierarchicalRetriever 사용"""
    async with semaphore:
        idx, question, target, context_type = item["idx"], item["question"], item["target_answer"], item["context_type"]
        try:
            # EnhancedHierarchicalRetriever로 검색 (내장 리랭커 사용)
            retrieval_result = await retriever.retrieve(query=question, config=config)

            # child_results에서 컨텍스트 추출 (리랭킹된 결과)
            if retrieval_result.child_results:
                context = "\n\n---\n\n".join([r.content for r in retrieval_result.child_results])
            else:
                context = retrieval_result.context

            # 메타데이터 추출 (디버깅용)
            enhancements = retrieval_result.metadata.get("enhancements_applied", [])
            adaptive_weights = retrieval_result.metadata.get("adaptive_weights", {})

            answer = await generate_answer(query=question, context=context)
            judge_result = await call_judge(question, target, answer)
            verdict = judge_result.get("verdict", "X")
            is_correct = verdict == "O"
            progress_counter[0] += 1

            # 적용된 향상 기능 표시
            enhancement_str = ",".join(enhancements[:3]) if enhancements else "none"
            weight_info = f"D:{adaptive_weights.get('dense_weight', 0):.1f}/S:{adaptive_weights.get('sparse_weight', 0):.1f}" if adaptive_weights else ""
            print(f"  [{progress_counter[0]}/{total}] idx={idx} -> {'O' if is_correct else 'X'} ({enhancement_str}) {weight_info}")

            return {
                "idx": idx,
                "context_type": context_type,
                "question": question,
                "target_answer": target,
                "generated_answer": answer,
                "verdict": verdict,
                "reason": judge_result.get("reason", ""),
                "is_correct": is_correct,
                "enhancements": enhancements,
                "adaptive_weights": adaptive_weights,
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
                "enhancements": [],
                "adaptive_weights": {},
            }


async def run_test():
    print("=" * 60)
    print(f"Advisor OSC Finance {VERSION} 테스트")
    print("=" * 60)
    print(f"  컬렉션: {COLLECTION_NAME}")
    print(f"  Retriever: EnhancedHierarchicalRetriever (보수적 설정)")
    print(f"  임베딩: OpenAI text-embedding-3-large")
    print(f"  Reranker: BGE v2-m3 (내장)")
    print(f"  LLM: GPT-5-mini (no thinking)")
    print(f"  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n설정 (vs v1.6.9):")
    print("  - enable_table_adaptive=False (테이블 감지 비활성화)")
    print("  - expand_to_parent=False (child 결과 직접 사용)")
    print("  - enable_multi_query=False (단순화)")
    print("  - enable_query_expansion=True")
    print("  - enable_rrf=True")
    print("  - enable_adaptive_search=True")
    print("=" * 60)

    test_data = load_test_data()
    total = len(test_data)
    print(f"\n테스트 케이스: {total}개")

    settings = get_settings()
    dense_service = OpenAIEmbeddingService(api_key=settings.openai_api_key, model="text-embedding-3-large")
    sparse_service = SparseEmbeddingService()
    embedding_service = MultimodalEmbeddingService(dense_service, sparse_service)

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
            print("v1.6.8 테스트를 먼저 실행하세요.")
            return 0
        print(f"\n컬렉션 정보: {qdrant_client.get_collection(COLLECTION_NAME).points_count} points")
    except Exception as e:
        print(f"Qdrant 연결 오류: {e}")
        return 0

    # EnhancedHierarchicalRetriever 설정 (보수적)
    enhanced_config = EnhancedRetrievalConfig(
        # 기본 검색 설정
        top_k=15,                          # v1.6.8과 동일
        rerank_top_k=50,                   # v1.6.8과 동일
        expand_to_parent=False,            # child 결과 직접 사용 (v1.6.8과 동일)
        rerank=True,

        # Reranker 선택 (BGE 사용)
        reranker_type="bge",

        # 향상 기능 (보수적 설정)
        enable_query_expansion=True,       # 쿼리 확장 활성화
        enable_multi_query=False,          # 다중 쿼리 비활성화 (단순화)
        enable_adaptive_search=True,       # 적응형 검색 활성화
        enable_rrf=True,                   # RRF 하이브리드 활성화
        enable_hyde=False,                 # HyDE 비활성화

        # RRF 설정 (기본값)
        rrf_k=60,
        rrf_dense_weight=1.0,              # 균형 가중치
        rrf_sparse_weight=1.0,             # 균형 가중치

        # 테이블 적응형 검색 비활성화 (v1.6.9에서 과도하게 작동)
        enable_table_adaptive=False,
    )

    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=COLLECTION_NAME
    )

    # EnhancedHierarchicalRetriever 초기화
    retriever = EnhancedHierarchicalRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        api_key=settings.openai_api_key,
        config=enhanced_config,
    )

    print(f"\n{'=' * 60}\n테스트 실행 중 (병렬 {TEST_CONCURRENCY}개)...\n{'=' * 60}")
    semaphore = asyncio.Semaphore(TEST_CONCURRENCY)
    progress_counter = [0]
    tasks = [
        process_single_case(item, retriever, enhanced_config, semaphore, progress_counter, total)
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

    # 향상 기능별 통계
    enhancement_stats = {}
    for r in results:
        for e in r.get("enhancements", []):
            if e not in enhancement_stats:
                enhancement_stats[e] = {"correct": 0, "total": 0}
            enhancement_stats[e]["total"] += 1
            if r["is_correct"]:
                enhancement_stats[e]["correct"] += 1

    print(f"\n향상 기능 적용 통계:")
    for e, stats in sorted(enhancement_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {e}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

    # Query-Adaptive Weights 분석
    weight_analysis = {"numeric": [], "conceptual": [], "balanced": [], "specific": []}
    for r in results:
        query_type = r.get("adaptive_weights", {}).get("query_type", "unknown")
        for key in weight_analysis:
            if key in query_type:
                weight_analysis[key].append(r["is_correct"])
                break
        else:
            if query_type != "unknown":
                weight_analysis["balanced"].append(r["is_correct"])

    print(f"\nQuery Type별 정확도:")
    for qtype, results_list in weight_analysis.items():
        if results_list:
            acc = sum(results_list) / len(results_list) * 100
            print(f"  {qtype}: {sum(results_list)}/{len(results_list)} ({acc:.1f}%)")

    # 이전 버전과 비교
    print(f"\n{'=' * 60}")
    print("이전 버전 대비 변화:")
    prev_results = {
        "v1.6.6": {"overall": 68.3, "image": 40.0, "paragraph": 76.7, "table": 100.0},
        "v1.6.8": {"overall": 73.3, "image": 70.0, "paragraph": 73.3, "table": 80.0},
        "v1.6.9": {"overall": 15.0, "image": 10.0, "paragraph": 23.3, "table": 0.0},
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
    output_file = output_dir / f"finance_{VERSION}_enhanced_conservative_{timestamp}.json"

    output_data = {
        "version": VERSION,
        "approach": "v1.6.8 (GPT-4o OCR) + EnhancedHierarchicalRetriever (보수적 설정)",
        "pipeline": {
            "retriever": "EnhancedHierarchicalRetriever",
            "enhancements": [
                "Query Expansion (LLM + 동의어 사전)",
                "RRF Hybrid Search (균형 가중치)",
                "Query-Adaptive Weights",
            ],
            "disabled": [
                "Table Adaptive (과도한 감지 문제)",
                "Multi-Query",
                "HyDE",
                "expand_to_parent",
            ],
            "embedding": "OpenAI text-embedding-3-large",
            "reranker": "BGE v2-m3 (내장)",
            "llm": "GPT-5-mini (no thinking)",
            "judge": "GPT-5-mini (no thinking)"
        },
        "config": {
            "top_k": enhanced_config.top_k,
            "rerank_top_k": enhanced_config.rerank_top_k,
            "rrf_k": enhanced_config.rrf_k,
            "rrf_dense_weight": enhanced_config.rrf_dense_weight,
            "rrf_sparse_weight": enhanced_config.rrf_sparse_weight,
            "enable_query_expansion": enhanced_config.enable_query_expansion,
            "enable_multi_query": enhanced_config.enable_multi_query,
            "enable_rrf": enhanced_config.enable_rrf,
            "enable_table_adaptive": enhanced_config.enable_table_adaptive,
            "expand_to_parent": enhanced_config.expand_to_parent,
        },
        "collection": COLLECTION_NAME,
        "timestamp": timestamp,
        "total": total,
        "correct": total_correct,
        "accuracy": overall_acc,
        "accuracy_by_type": {ct: stats["correct"]/stats["total"]*100 if stats["total"] > 0 else 0
                            for ct, stats in by_context.items()},
        "enhancement_stats": {e: {"correct": s["correct"], "total": s["total"],
                                  "accuracy": s["correct"]/s["total"]*100 if s["total"] > 0 else 0}
                             for e, s in enhancement_stats.items()},
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
