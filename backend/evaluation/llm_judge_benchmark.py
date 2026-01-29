"""
LLM-as-Judge 벤치마크 테스트
개인화 향상 모듈 (시맨틱 라우터, Self-RAG, LLM-Judge) 성능 평가

평가 항목:
1. 시맨틱 라우터 정확도
2. Self-RAG 관련성 판단 정확도
3. LLM-as-Judge 평가 일관성
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# 개인화 향상 모듈 import
from services.personalization_enhancer import (
    get_personalization_enhancer,
    QueryIntent,
    SemanticRouter,
    SelfRAG,
    LLMJudge,
    HierarchicalMemoryManager
)


@dataclass
class TestCase:
    """테스트 케이스"""
    query: str
    expected_intent: QueryIntent
    context: str = ""
    response: str = ""
    is_relevant: bool = True


class LLMJudgeBenchmark:
    """LLM-as-Judge 벤치마크"""

    def __init__(self):
        self.enhancer = None
        self.semantic_router = None
        self.self_rag = None
        self.llm_judge = None
        self.results = {}

    async def initialize(self):
        """초기화"""
        print("🔧 개인화 향상 모듈 초기화 중...")
        self.enhancer = get_personalization_enhancer()
        # 개별 컴포넌트 생성
        self.semantic_router = SemanticRouter()
        await self.semantic_router.initialize()
        self.self_rag = SelfRAG()
        self.llm_judge = LLMJudge()
        print("   ✅ 초기화 완료")

    def generate_test_cases(self) -> List[TestCase]:
        """테스트 케이스 생성"""
        test_cases = [
            # GREETING 테스트
            TestCase("안녕하세요!", QueryIntent.GREETING),
            TestCase("좋은 아침이에요", QueryIntent.GREETING),
            TestCase("반갑습니다", QueryIntent.GREETING),
            TestCase("오랜만이에요", QueryIntent.GREETING),
            TestCase("Hi there!", QueryIntent.GREETING),

            # PERSONAL 테스트
            TestCase("내가 지난번에 물어본 거 기억나?", QueryIntent.PERSONAL),
            TestCase("저번에 추천해준 책 뭐였지?", QueryIntent.PERSONAL),
            TestCase("내 프로필 설정 바꿔줘", QueryIntent.PERSONAL),
            TestCase("나한테 맞는 추천해줘", QueryIntent.PERSONAL),
            TestCase("내 학습 기록 보여줘", QueryIntent.PERSONAL),

            # GENERAL 테스트
            TestCase("Python에서 리스트 정렬하는 방법이 뭐야?", QueryIntent.GENERAL),
            TestCase("머신러닝이 뭐야?", QueryIntent.GENERAL),
            TestCase("React와 Vue의 차이점이 뭐야?", QueryIntent.GENERAL),
            TestCase("SQL 조인 종류 알려줘", QueryIntent.GENERAL),
            TestCase("객체지향 프로그래밍이 뭐야?", QueryIntent.GENERAL),

            # ACTION 테스트
            TestCase("이 코드 분석해줘", QueryIntent.ACTION),
            TestCase("이 함수 최적화해줘", QueryIntent.ACTION),
            TestCase("버그 찾아줘", QueryIntent.ACTION),
            TestCase("테스트 케이스 작성해줘", QueryIntent.ACTION),
            TestCase("리팩토링 해줘", QueryIntent.ACTION),

            # CLARIFICATION 테스트
            TestCase("좀 더 자세히 설명해줘", QueryIntent.CLARIFICATION),
            TestCase("예시를 들어줄래?", QueryIntent.CLARIFICATION),
            TestCase("다른 방법은 없어?", QueryIntent.CLARIFICATION),
            TestCase("왜 그렇게 되는 거야?", QueryIntent.CLARIFICATION),
            TestCase("무슨 말이야?", QueryIntent.CLARIFICATION),
        ]
        return test_cases

    def generate_relevance_cases(self) -> List[Dict]:
        """관련성 테스트 케이스 생성"""
        cases = [
            # 관련성 높음
            {
                "query": "Python에서 리스트 정렬하는 방법",
                "context": "Python에서 리스트를 정렬하는 방법은 여러 가지가 있습니다. sort() 메서드를 사용하면 원본 리스트가 정렬되고, sorted() 함수를 사용하면 새로운 정렬된 리스트가 반환됩니다.",
                "expected_relevant": True
            },
            {
                "query": "머신러닝 기초 개념",
                "context": "머신러닝은 데이터로부터 패턴을 학습하여 예측이나 결정을 내리는 인공지능의 한 분야입니다. 지도학습, 비지도학습, 강화학습 등의 방법론이 있습니다.",
                "expected_relevant": True
            },
            {
                "query": "React 상태 관리",
                "context": "React에서 상태 관리는 useState, useReducer 훅을 사용하거나 Redux, MobX 같은 외부 라이브러리를 활용할 수 있습니다.",
                "expected_relevant": True
            },
            # 관련성 낮음
            {
                "query": "Python에서 리스트 정렬하는 방법",
                "context": "오늘 날씨가 좋습니다. 산책하기 좋은 날이네요.",
                "expected_relevant": False
            },
            {
                "query": "데이터베이스 인덱싱",
                "context": "요리를 할 때는 재료 준비가 중요합니다. 신선한 재료를 사용하면 맛이 좋습니다.",
                "expected_relevant": False
            },
            {
                "query": "API 설계 베스트 프랙티스",
                "context": "운동을 규칙적으로 하면 건강에 좋습니다. 하루 30분 걷기를 추천합니다.",
                "expected_relevant": False
            },
            # 부분적 관련성
            {
                "query": "웹 개발 프레임워크",
                "context": "소프트웨어 개발에는 다양한 도구가 있습니다. 프레임워크는 개발 생산성을 높여줍니다.",
                "expected_relevant": True  # 부분적 관련성
            },
            {
                "query": "클라우드 서비스",
                "context": "IT 인프라 관리는 복잡합니다. 서버 운영에는 많은 리소스가 필요합니다.",
                "expected_relevant": True  # 부분적 관련성
            },
        ]
        return cases

    def generate_judge_cases(self) -> List[Dict]:
        """LLM-as-Judge 테스트 케이스"""
        cases = [
            # 좋은 응답
            {
                "query": "Python에서 리스트를 정렬하는 방법은?",
                "response": """Python에서 리스트를 정렬하는 방법은 크게 두 가지가 있습니다:

1. **sort() 메서드**: 원본 리스트를 직접 정렬
```python
my_list = [3, 1, 4, 1, 5]
my_list.sort()  # [1, 1, 3, 4, 5]
```

2. **sorted() 함수**: 새로운 정렬된 리스트 반환
```python
my_list = [3, 1, 4, 1, 5]
new_list = sorted(my_list)  # [1, 1, 3, 4, 5]
```

역순 정렬은 `reverse=True` 옵션을 사용합니다.""",
                "expected_score_min": 4.0
            },
            # 보통 응답
            {
                "query": "머신러닝이 뭐야?",
                "response": "머신러닝은 데이터를 학습하는 AI입니다.",
                "expected_score_min": 2.5
            },
            # 나쁜 응답
            {
                "query": "React 상태 관리 방법",
                "response": "잘 모르겠어요.",
                "expected_score_min": 1.0
            },
        ]
        return cases

    async def test_semantic_router(self) -> Dict[str, Any]:
        """시맨틱 라우터 테스트"""
        print("\n📊 시맨틱 라우터 테스트")
        print("-" * 50)

        test_cases = self.generate_test_cases()
        correct = 0
        total = len(test_cases)
        results_detail = []
        latencies = []

        for tc in test_cases:
            start = time.time()
            intent, score, config = await self.semantic_router.route_async(tc.query)
            latency = (time.time() - start) * 1000

            is_correct = intent == tc.expected_intent
            if is_correct:
                correct += 1

            latencies.append(latency)
            results_detail.append({
                "query": tc.query,
                "expected": tc.expected_intent.value,
                "predicted": intent.value,
                "correct": is_correct,
                "latency_ms": latency
            })

            status = "✓" if is_correct else "✗"
            print(f"   {status} [{tc.expected_intent.value}→{intent.value}] {tc.query[:30]}...")

        accuracy = correct / total
        avg_latency = sum(latencies) / len(latencies)

        print(f"\n   정확도: {accuracy:.1%} ({correct}/{total})")
        print(f"   평균 레이턴시: {avg_latency:.1f}ms")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_latency_ms": avg_latency,
            "details": results_detail
        }

    async def test_self_rag(self) -> Dict[str, Any]:
        """Self-RAG 관련성 테스트"""
        print("\n📊 Self-RAG 관련성 테스트")
        print("-" * 50)

        test_cases = self.generate_relevance_cases()
        correct = 0
        total = len(test_cases)
        results_detail = []
        latencies = []

        for tc in test_cases:
            start = time.time()
            result = await self.self_rag.check_relevance(tc["query"], tc["context"])
            latency = (time.time() - start) * 1000

            is_correct = result.is_relevant == tc["expected_relevant"]
            if is_correct:
                correct += 1

            latencies.append(latency)
            results_detail.append({
                "query": tc["query"],
                "expected_relevant": tc["expected_relevant"],
                "predicted_relevant": result.is_relevant,
                "confidence": result.confidence,
                "correct": is_correct,
                "latency_ms": latency
            })

            status = "✓" if is_correct else "✗"
            exp_str = "관련" if tc["expected_relevant"] else "무관"
            pred_str = "관련" if result.is_relevant else "무관"
            print(f"   {status} [{exp_str}→{pred_str}] (conf={result.confidence:.2f}) {tc['query'][:25]}...")

        accuracy = correct / total
        avg_latency = sum(latencies) / len(latencies)

        print(f"\n   정확도: {accuracy:.1%} ({correct}/{total})")
        print(f"   평균 레이턴시: {avg_latency:.1f}ms")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_latency_ms": avg_latency,
            "details": results_detail
        }

    async def test_llm_judge(self) -> Dict[str, Any]:
        """LLM-as-Judge 테스트"""
        print("\n📊 LLM-as-Judge 평가 테스트")
        print("-" * 50)

        test_cases = self.generate_judge_cases()
        scores = []
        results_detail = []
        latencies = []

        for tc in test_cases:
            start = time.time()
            judge_score = await self.llm_judge.evaluate(
                query=tc["query"],
                response=tc["response"]
            )
            latency = (time.time() - start) * 1000

            meets_expectation = judge_score.overall >= tc["expected_score_min"]
            latencies.append(latency)
            scores.append(judge_score.overall)

            results_detail.append({
                "query": tc["query"],
                "response": tc["response"][:50] + "...",
                "accuracy": judge_score.accuracy,
                "helpfulness": judge_score.helpfulness,
                "personalization": judge_score.personalization,
                "friendliness": judge_score.friendliness,
                "overall": judge_score.overall,
                "expected_min": tc["expected_score_min"],
                "meets_expectation": meets_expectation,
                "latency_ms": latency
            })

            status = "✓" if meets_expectation else "✗"
            print(f"   {status} Overall: {judge_score.overall:.2f}/5 (기대: >={tc['expected_score_min']:.1f})")
            print(f"      정확성:{judge_score.accuracy} 도움:{judge_score.helpfulness} "
                  f"개인화:{judge_score.personalization} 친절:{judge_score.friendliness}")

        avg_score = sum(scores) / len(scores)
        avg_latency = sum(latencies) / len(latencies)

        print(f"\n   평균 점수: {avg_score:.2f}/5")
        print(f"   평균 레이턴시: {avg_latency:.1f}ms")

        return {
            "avg_score": avg_score,
            "scores": scores,
            "avg_latency_ms": avg_latency,
            "details": results_detail
        }

    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        results = {}

        # 시맨틱 라우터 테스트
        results["semantic_router"] = await self.test_semantic_router()

        # Self-RAG 테스트
        results["self_rag"] = await self.test_self_rag()

        # LLM-as-Judge 테스트
        results["llm_judge"] = await self.test_llm_judge()

        return results


async def main():
    print("=" * 70)
    print("LLM-as-Judge 벤치마크")
    print("개인화 향상 모듈 성능 평가")
    print("=" * 70)

    benchmark = LLMJudgeBenchmark()
    await benchmark.initialize()

    results = await benchmark.run_all_tests()

    # 결과 요약
    print("\n" + "=" * 70)
    print("📈 벤치마크 결과 요약")
    print("=" * 70)

    print("\n┌─────────────────────────┬──────────────┬──────────────┐")
    print("│ Component               │ Accuracy     │ Latency (ms) │")
    print("├─────────────────────────┼──────────────┼──────────────┤")

    sr = results["semantic_router"]
    print(f"│ Semantic Router         │ {sr['accuracy']:>10.1%} │ {sr['avg_latency_ms']:>10.1f} │")

    srag = results["self_rag"]
    print(f"│ Self-RAG                │ {srag['accuracy']:>10.1%} │ {srag['avg_latency_ms']:>10.1f} │")

    judge = results["llm_judge"]
    print(f"│ LLM-as-Judge            │ {judge['avg_score']/5:>10.1%} │ {judge['avg_latency_ms']:>10.1f} │")

    print("└─────────────────────────┴──────────────┴──────────────┘")

    # 결과 저장
    output = {
        "benchmark": "LLM-as-Judge",
        "timestamp": datetime.now().isoformat(),
        "results": results
    }

    output_path = "/tmp/llm_judge_benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n📁 결과 저장: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
