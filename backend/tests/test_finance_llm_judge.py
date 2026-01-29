# -*- coding: utf-8 -*-
"""
Finance 60개 테스트 - LLM-as-Judge 방식
advisor V7.6.1 적용 버전 테스트

평가 모델:
1. GPT-4o-mini
2. Gemini 2.5 Flash
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from dotenv import load_dotenv
load_dotenv()

# advisor production Qdrant 포트 설정
import os
os.environ['QDRANT_PORT'] = '10304'

import pandas as pd
import httpx
from tqdm import tqdm

from config.settings import Settings
from rag.embeddings import OpenAIEmbeddingService, SparseEmbeddingService, MultimodalEmbeddingService
from rag.vectorstore import QdrantVectorStore
from rag.retriever import HierarchicalRetriever, RetrievalConfig
from rag.chain import RAGChain

# advisor production의 finance 컬렉션 - V7.6.1 Azure
COLLECTION_NAME = "mh_rag_finance_v7_6_azure"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# 테스트셋 경로
DATASET_PATH = "/home/aiedu/workspace/MH_rag/eval/allganize/dataset/allganize_test_dataset_mapped.csv"


def load_test_data():
    """테스트 데이터 로드"""
    df = pd.read_csv(DATASET_PATH)
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


# ============ LLM-as-Judge 평가 ============

JUDGE_PROMPT = """다음 질문에 대한 생성된 답변이 정답과 의미적으로 일치하는지 평가해주세요.

질문: {question}

정답: {target_answer}

생성된 답변: {generated_answer}

평가 기준:
1. 핵심 정보(숫자, 날짜, 법률명 등)가 정확히 일치하는가?
2. 의미적으로 동일한 내용을 전달하는가?

JSON 형식으로만 응답: {{"verdict": "O" 또는 "X", "reason": "간단한 이유"}}"""


async def judge_gpt4o_mini(question: str, generated_answer: str, target_answer: str) -> dict:
    """GPT-4o-mini로 평가"""
    prompt = JUDGE_PROMPT.format(
        question=question,
        target_answer=target_answer,
        generated_answer=generated_answer
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0
                }
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                try:
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    result = json.loads(content.strip())
                    return {"verdict": result.get("verdict", "X"), "reason": result.get("reason", "")}
                except:
                    return {"verdict": "X", "reason": "파싱 실패"}
    except Exception as e:
        return {"verdict": "X", "reason": f"API 오류: {str(e)[:50]}"}

    return {"verdict": "X", "reason": "Unknown error"}


async def judge_gemini_flash(question: str, generated_answer: str, target_answer: str) -> dict:
    """Gemini 2.5 Flash로 평가"""
    prompt = JUDGE_PROMPT.format(
        question=question,
        target_answer=target_answer,
        generated_answer=generated_answer
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.0}
                }
            )

            if response.status_code == 200:
                result = response.json()
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                try:
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    parsed = json.loads(content.strip())
                    return {"verdict": parsed.get("verdict", "X"), "reason": parsed.get("reason", "")}
                except:
                    # JSON 파싱 실패시 텍스트에서 O/X 찾기
                    if '"O"' in content or "'O'" in content or "verdict\": \"O" in content:
                        return {"verdict": "O", "reason": "텍스트에서 추출"}
                    return {"verdict": "X", "reason": "파싱 실패"}
    except Exception as e:
        return {"verdict": "X", "reason": f"API 오류: {str(e)[:50]}"}

    return {"verdict": "X", "reason": "Unknown error"}


async def main():
    print("=" * 70)
    print("Finance 60개 테스트 - LLM-as-Judge (advisor V7.6.1)")
    print("=" * 70)
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n평가 모델:")
    print("  1. GPT-4o-mini")
    print("  2. Gemini 2.0 Flash")
    print(f"\n컬렉션: {COLLECTION_NAME}")
    print("=" * 70)

    settings = Settings()
    test_data = load_test_data()
    print(f"\n테스트 문항: {len(test_data)}개")

    # 서비스 초기화
    print("\n서비스 초기화 중...")
    dense_service = OpenAIEmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.embedding_model
    )
    sparse_service = SparseEmbeddingService()
    embedding_service = MultimodalEmbeddingService(dense_service, sparse_service)

    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=10304,  # advisor production Qdrant
        collection_name=COLLECTION_NAME,
    )

    # V7.6.1 설정
    config = RetrievalConfig(
        top_k=10,  # V7.6.1
        rerank_top_k=30,  # V7.6.1
        rerank=True,
        expand_to_parent=True,
        use_multi_query=False,  # V7.6.1
    )
    print(f"\nRetrievalConfig: top_k={config.top_k}, rerank_top_k={config.rerank_top_k}, use_multi_query={config.use_multi_query}")

    retriever = HierarchicalRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        config=config,
    )

    rag_chain = RAGChain(
        retriever=retriever,
        api_key=settings.openai_api_key,
        model=settings.llm_model,
    )

    # 테스트 실행
    results = []
    gpt4o_mini_correct = 0
    gemini_correct = 0
    correct_by_type = {"gpt4o_mini": {}, "gemini": {}}
    total_by_type = {}

    print("\n테스트 시작...")
    for q in tqdm(test_data, desc="테스트 진행"):
        try:
            # 답변 생성
            response = await rag_chain.chat(
                query=q["question"],
                retrieval_config=config,
            )

            # 두 모델로 병렬 평가
            gpt_result, gemini_result = await asyncio.gather(
                judge_gpt4o_mini(q["question"], response.answer, q["target_answer"]),
                judge_gemini_flash(q["question"], response.answer, q["target_answer"]),
            )

            context_type = q.get("context_type", "unknown")
            total_by_type[context_type] = total_by_type.get(context_type, 0) + 1

            if gpt_result["verdict"] == "O":
                gpt4o_mini_correct += 1
                correct_by_type["gpt4o_mini"][context_type] = correct_by_type["gpt4o_mini"].get(context_type, 0) + 1

            if gemini_result["verdict"] == "O":
                gemini_correct += 1
                correct_by_type["gemini"][context_type] = correct_by_type["gemini"].get(context_type, 0) + 1

            results.append({
                "idx": q["idx"],
                "question": q["question"],
                "target_answer": q["target_answer"],
                "generated_answer": response.answer,
                "context_type": context_type,
                "gpt4o_mini_verdict": gpt_result["verdict"],
                "gpt4o_mini_reason": gpt_result["reason"],
                "gemini_verdict": gemini_result["verdict"],
                "gemini_reason": gemini_result["reason"],
            })

        except Exception as e:
            print(f"\n[오류] Q{q['idx']}: {str(e)[:100]}")
            results.append({
                "idx": q["idx"],
                "question": q["question"],
                "error": str(e)[:200],
                "gpt4o_mini_verdict": "X",
                "gemini_verdict": "X"
            })
            ctx_type = q.get("context_type", "unknown")
            total_by_type[ctx_type] = total_by_type.get(ctx_type, 0) + 1

    # 결과 출력
    print("\n" + "=" * 70)
    print("결과 요약 - LLM-as-Judge (advisor V7.6.1)")
    print("=" * 70)

    gpt_acc = 100 * gpt4o_mini_correct / len(test_data)
    gemini_acc = 100 * gemini_correct / len(test_data)

    print(f"\n[전체 정확도]")
    print(f"  GPT-4o-mini  : {gpt4o_mini_correct}/{len(test_data)} ({gpt_acc:.1f}%)")
    print(f"  Gemini Flash : {gemini_correct}/{len(test_data)} ({gemini_acc:.1f}%)")

    print("\n[유형별 정확도]")
    print("-" * 60)
    print(f"{'유형':12s} | {'GPT-4o-mini':15s} | {'Gemini Flash':15s}")
    print("-" * 60)
    for ctx_type in sorted(total_by_type.keys()):
        t = total_by_type[ctx_type]
        gpt_c = correct_by_type["gpt4o_mini"].get(ctx_type, 0)
        gem_c = correct_by_type["gemini"].get(ctx_type, 0)
        gpt_a = 100 * gpt_c / t if t > 0 else 0
        gem_a = 100 * gem_c / t if t > 0 else 0
        print(f"{ctx_type:12s} | {gpt_c:2d}/{t:2d} ({gpt_a:5.1f}%) | {gem_c:2d}/{t:2d} ({gem_a:5.1f}%)")

    # 저장
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"finance_60_llm_judge_v761_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "collection": COLLECTION_NAME,
            "version": "advisor V7.6.1",
            "evaluation_method": "LLM-as-Judge",
            "judges": ["GPT-4o-mini", "Gemini-2.0-Flash"],
            "config": {
                "top_k": config.top_k,
                "rerank_top_k": config.rerank_top_k,
                "use_multi_query": config.use_multi_query,
            },
            "summary": {
                "total": len(test_data),
                "gpt4o_mini": {"correct": gpt4o_mini_correct, "accuracy": gpt_acc},
                "gemini_flash": {"correct": gemini_correct, "accuracy": gemini_acc},
                "by_type": {
                    ctx_type: {
                        "total": total_by_type[ctx_type],
                        "gpt4o_mini": correct_by_type["gpt4o_mini"].get(ctx_type, 0),
                        "gemini_flash": correct_by_type["gemini"].get(ctx_type, 0)
                    }
                    for ctx_type in total_by_type
                }
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {output_file}")
    print(f"완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main())
