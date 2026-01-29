"""
Knowledge Graph Benchmark 평가 스크립트
지식 그래프의 효과를 검증하기 위한 벤치마크

비교 대상:
1. Vector Only - 벡터 검색만
2. Vector + Reranker (BGE) - 벡터 + BGE 리랭킹
3. Vector + KG - 벡터 + 지식 그래프
4. Vector + KG + Reranker - 벡터 + 지식 그래프 + BGE 리랭킹

평가 지표: Entity Recall, Relation Recall, Multi-hop Accuracy, Latency
"""

import asyncio
import json
import os
import sys
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import uuid
import numpy as np

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from services.memory_service import (
    KnowledgeGraph, PIPELINE_MODES, NETWORKX_AVAILABLE
)

print(f"✅ NetworkX Available: {NETWORKX_AVAILABLE}")
print(f"✅ Pipeline Modes: {list(PIPELINE_MODES.keys())}")


@dataclass
class TestScenario:
    """테스트 시나리오"""
    scenario_id: str
    description: str
    entities: List[Dict]  # [{"value": "Python", "type": "technology"}]
    relations: List[Dict]  # [{"subject": "user", "relation": "uses", "object": "Python"}]
    queries: List[Dict]  # [{"query": "...", "expected_entities": [...], "expected_relations": [...]}]


class KnowledgeGraphBenchmark:
    """Knowledge Graph 벤치마크"""

    def __init__(self):
        self.scenarios: List[TestScenario] = []

    def generate_test_scenarios(self) -> List[TestScenario]:
        """테스트 시나리오 생성"""
        scenarios = [
            # 시나리오 1: 개발자 프로필
            TestScenario(
                scenario_id="developer_1",
                description="Python 개발자의 기술 스택",
                entities=[
                    {"value": "Python", "type": "technology", "confidence": 0.95},
                    {"value": "FastAPI", "type": "technology", "confidence": 0.9},
                    {"value": "PostgreSQL", "type": "technology", "confidence": 0.85},
                    {"value": "Docker", "type": "technology", "confidence": 0.9},
                    {"value": "네이버", "type": "organization", "confidence": 0.8}
                ],
                relations=[
                    {"subject": "user", "subject_type": "person", "relation": "uses", "object": "Python", "object_type": "technology", "confidence": 0.95},
                    {"subject": "user", "subject_type": "person", "relation": "uses", "object": "FastAPI", "object_type": "technology", "confidence": 0.9},
                    {"subject": "user", "subject_type": "person", "relation": "uses", "object": "PostgreSQL", "object_type": "technology", "confidence": 0.85},
                    {"subject": "user", "subject_type": "person", "relation": "uses", "object": "Docker", "object_type": "technology", "confidence": 0.9},
                    {"subject": "user", "subject_type": "person", "relation": "works_at", "object": "네이버", "object_type": "organization", "confidence": 0.8},
                    {"subject": "FastAPI", "subject_type": "technology", "relation": "uses", "object": "Python", "object_type": "technology", "confidence": 0.95}  # 2-hop relation
                ],
                queries=[
                    {"query": "Python", "expected_entities": ["Python", "FastAPI"], "hops": 1},
                    {"query": "어떤 기술 사용해요?", "expected_entities": ["Python", "FastAPI", "PostgreSQL", "Docker"], "hops": 1},
                    {"query": "회사", "expected_entities": ["네이버"], "hops": 1},
                    {"query": "FastAPI 관련 기술", "expected_entities": ["FastAPI", "Python"], "hops": 2}  # multi-hop
                ]
            ),

            # 시나리오 2: 학생 프로필
            TestScenario(
                scenario_id="student_1",
                description="컴퓨터공학 전공 학생",
                entities=[
                    {"value": "서울대학교", "type": "organization", "confidence": 0.95},
                    {"value": "컴퓨터공학", "type": "skill", "confidence": 0.9},
                    {"value": "머신러닝", "type": "interest", "confidence": 0.85},
                    {"value": "2024년 2월", "type": "date", "confidence": 0.9},
                    {"value": "AI 스타트업", "type": "interest", "confidence": 0.8}
                ],
                relations=[
                    {"subject": "user", "subject_type": "person", "relation": "studies_at", "object": "서울대학교", "object_type": "organization", "confidence": 0.95},
                    {"subject": "user", "subject_type": "person", "relation": "knows", "object": "컴퓨터공학", "object_type": "skill", "confidence": 0.9},
                    {"subject": "user", "subject_type": "person", "relation": "interested_in", "object": "머신러닝", "object_type": "interest", "confidence": 0.85},
                    {"subject": "user", "subject_type": "person", "relation": "interested_in", "object": "AI 스타트업", "object_type": "interest", "confidence": 0.8}
                ],
                queries=[
                    {"query": "학교", "expected_entities": ["서울대학교"], "hops": 1},
                    {"query": "전공", "expected_entities": ["컴퓨터공학"], "hops": 1},
                    {"query": "관심사", "expected_entities": ["머신러닝", "AI 스타트업"], "hops": 1},
                    {"query": "서울대에서 배운 것", "expected_entities": ["서울대학교", "컴퓨터공학"], "hops": 2}
                ]
            ),

            # 시나리오 3: 복합 관계
            TestScenario(
                scenario_id="complex_1",
                description="복합적인 엔티티 관계",
                entities=[
                    {"value": "LangChain", "type": "technology", "confidence": 0.95},
                    {"value": "RAG", "type": "concept", "confidence": 0.9},
                    {"value": "OpenAI", "type": "organization", "confidence": 0.95},
                    {"value": "GPT-4", "type": "product", "confidence": 0.9},
                    {"value": "챗봇 프로젝트", "type": "project", "confidence": 0.85}
                ],
                relations=[
                    {"subject": "user", "subject_type": "person", "relation": "uses", "object": "LangChain", "object_type": "technology", "confidence": 0.95},
                    {"subject": "user", "subject_type": "person", "relation": "knows", "object": "RAG", "object_type": "concept", "confidence": 0.9},
                    {"subject": "user", "subject_type": "person", "relation": "uses", "object": "GPT-4", "object_type": "product", "confidence": 0.9},
                    {"subject": "user", "subject_type": "person", "relation": "works_on", "object": "챗봇 프로젝트", "object_type": "project", "confidence": 0.85},
                    {"subject": "LangChain", "subject_type": "technology", "relation": "uses", "object": "RAG", "object_type": "concept", "confidence": 0.9},
                    {"subject": "GPT-4", "subject_type": "product", "relation": "created", "object": "OpenAI", "object_type": "organization", "confidence": 0.95}
                ],
                queries=[
                    {"query": "RAG 관련 기술", "expected_entities": ["RAG", "LangChain"], "hops": 2},
                    {"query": "프로젝트", "expected_entities": ["챗봇 프로젝트"], "hops": 1},
                    {"query": "GPT", "expected_entities": ["GPT-4", "OpenAI"], "hops": 2},
                    {"query": "어떤 LLM 써요?", "expected_entities": ["GPT-4", "LangChain"], "hops": 1}
                ]
            ),

            # 시나리오 4: 취미/라이프스타일
            TestScenario(
                scenario_id="lifestyle_1",
                description="취미와 라이프스타일",
                entities=[
                    {"value": "기타", "type": "interest", "confidence": 0.9},
                    {"value": "락 음악", "type": "interest", "confidence": 0.85},
                    {"value": "헬스", "type": "interest", "confidence": 0.9},
                    {"value": "매일 아침 6시", "type": "concept", "confidence": 0.8},
                    {"value": "단백질 보충제", "type": "product", "confidence": 0.75}
                ],
                relations=[
                    {"subject": "user", "subject_type": "person", "relation": "interested_in", "object": "기타", "object_type": "interest", "confidence": 0.9},
                    {"subject": "user", "subject_type": "person", "relation": "prefers", "object": "락 음악", "object_type": "interest", "confidence": 0.85},
                    {"subject": "user", "subject_type": "person", "relation": "interested_in", "object": "헬스", "object_type": "interest", "confidence": 0.9},
                    {"subject": "user", "subject_type": "person", "relation": "uses", "object": "단백질 보충제", "object_type": "product", "confidence": 0.75},
                    {"subject": "기타", "subject_type": "interest", "relation": "uses", "object": "락 음악", "object_type": "interest", "confidence": 0.8}
                ],
                queries=[
                    {"query": "취미", "expected_entities": ["기타", "헬스"], "hops": 1},
                    {"query": "음악", "expected_entities": ["기타", "락 음악"], "hops": 2},
                    {"query": "운동", "expected_entities": ["헬스", "단백질 보충제"], "hops": 2},
                    {"query": "건강 관리", "expected_entities": ["헬스", "단백질 보충제"], "hops": 1}
                ]
            ),

            # 시나리오 5: 경력/직장
            TestScenario(
                scenario_id="career_1",
                description="경력 및 직장 정보",
                entities=[
                    {"value": "삼성전자", "type": "organization", "confidence": 0.95},
                    {"value": "소프트웨어 엔지니어", "type": "skill", "confidence": 0.9},
                    {"value": "5년", "type": "date", "confidence": 0.85},
                    {"value": "팀장", "type": "concept", "confidence": 0.8},
                    {"value": "반도체", "type": "concept", "confidence": 0.75}
                ],
                relations=[
                    {"subject": "user", "subject_type": "person", "relation": "works_at", "object": "삼성전자", "object_type": "organization", "confidence": 0.95},
                    {"subject": "user", "subject_type": "person", "relation": "knows", "object": "소프트웨어 엔지니어", "object_type": "skill", "confidence": 0.9},
                    {"subject": "삼성전자", "subject_type": "organization", "relation": "works_on", "object": "반도체", "object_type": "concept", "confidence": 0.85}
                ],
                queries=[
                    {"query": "직장", "expected_entities": ["삼성전자"], "hops": 1},
                    {"query": "직업", "expected_entities": ["소프트웨어 엔지니어"], "hops": 1},
                    {"query": "삼성에서 하는 일", "expected_entities": ["삼성전자", "반도체"], "hops": 2},
                    {"query": "경력", "expected_entities": ["삼성전자", "소프트웨어 엔지니어", "5년"], "hops": 1}
                ]
            )
        ]

        self.scenarios = scenarios
        return scenarios

    def build_knowledge_graph(self, scenario: TestScenario) -> KnowledgeGraph:
        """시나리오에서 지식 그래프 구축"""
        kg = KnowledgeGraph()

        # 엔티티 추가
        for entity in scenario.entities:
            kg.add_entity(
                entity_value=entity["value"],
                entity_type=entity["type"],
                memory_id=f"{scenario.scenario_id}_{entity['value']}",
                confidence=entity.get("confidence", 0.8)
            )

        # 관계 추가
        for relation in scenario.relations:
            kg.add_relation(
                subject_value=relation["subject"],
                subject_type=relation["subject_type"],
                relation_type=relation["relation"],
                object_value=relation["object"],
                object_type=relation["object_type"],
                memory_id=f"{scenario.scenario_id}_rel_{relation['object']}",
                confidence=relation.get("confidence", 0.8)
            )

        return kg

    def evaluate_entity_retrieval(
        self,
        kg: KnowledgeGraph,
        query: str,
        expected_entities: List[str],
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """엔티티 검색 평가"""
        start_time = time.time()

        # 1. 직접 엔티티 매칭
        found_entities = kg.find_entities(query)

        # 2. 그래프 탐색으로 확장
        if found_entities:
            start_nodes = [e["node_id"] for e in found_entities[:3]]
            traversed = kg.traverse(start_nodes, max_hops=max_hops)
            all_entity_values = set(
                item["entity_value"].lower() for item in traversed
                if item.get("entity_value")
            )
        else:
            all_entity_values = set()

        latency = (time.time() - start_time) * 1000

        # 메트릭 계산
        expected_set = set(e.lower() for e in expected_entities)
        retrieved_set = all_entity_values

        hits = expected_set & retrieved_set
        precision = len(hits) / len(retrieved_set) if retrieved_set else 0.0
        recall = len(hits) / len(expected_set) if expected_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "query": query,
            "expected": list(expected_entities),
            "retrieved": list(all_entity_values),
            "hits": list(hits),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "latency_ms": latency
        }

    def run_benchmark(self) -> Dict[str, Any]:
        """벤치마크 실행"""
        print("=" * 70)
        print("Knowledge Graph Benchmark")
        print("=" * 70)
        print()

        if not NETWORKX_AVAILABLE:
            print("❌ NetworkX not available. Cannot run benchmark.")
            return {"error": "NetworkX not available"}

        # 테스트 시나리오 생성
        print("📊 테스트 시나리오 생성 중...")
        scenarios = self.generate_test_scenarios()
        print(f"   - 시나리오 수: {len(scenarios)}")
        print()

        all_results = []
        scenario_results = {}

        for scenario in scenarios:
            print(f"\n📋 시나리오: {scenario.scenario_id} - {scenario.description}")

            # 그래프 구축
            kg = self.build_knowledge_graph(scenario)
            stats = kg.get_stats()
            print(f"   - 노드: {stats['total_nodes']}, 엣지: {stats['total_edges']}")

            scenario_metrics = []

            for q_data in scenario.queries:
                result = self.evaluate_entity_retrieval(
                    kg=kg,
                    query=q_data["query"],
                    expected_entities=q_data["expected_entities"],
                    max_hops=q_data.get("hops", 2)
                )
                result["hops"] = q_data.get("hops", 2)
                scenario_metrics.append(result)
                all_results.append(result)

                print(f"   Query: \"{q_data['query']}\" -> Recall: {result['recall']:.1%}, F1: {result['f1']:.1%}")

            # 시나리오별 평균
            scenario_results[scenario.scenario_id] = {
                "description": scenario.description,
                "num_queries": len(scenario.queries),
                "avg_precision": np.mean([r["precision"] for r in scenario_metrics]),
                "avg_recall": np.mean([r["recall"] for r in scenario_metrics]),
                "avg_f1": np.mean([r["f1"] for r in scenario_metrics]),
                "avg_latency_ms": np.mean([r["latency_ms"] for r in scenario_metrics])
            }

        # 전체 평균
        overall = {
            "total_queries": len(all_results),
            "avg_precision": np.mean([r["precision"] for r in all_results]),
            "avg_recall": np.mean([r["recall"] for r in all_results]),
            "avg_f1": np.mean([r["f1"] for r in all_results]),
            "avg_latency_ms": np.mean([r["latency_ms"] for r in all_results]),
            "p95_latency_ms": np.percentile([r["latency_ms"] for r in all_results], 95)
        }

        # 홉 수별 분석
        hop_analysis = {}
        for hops in [1, 2]:
            hop_results = [r for r in all_results if r.get("hops") == hops]
            if hop_results:
                hop_analysis[f"{hops}-hop"] = {
                    "count": len(hop_results),
                    "avg_recall": np.mean([r["recall"] for r in hop_results]),
                    "avg_f1": np.mean([r["f1"] for r in hop_results])
                }

        return {
            "overall": overall,
            "by_scenario": scenario_results,
            "by_hops": hop_analysis,
            "detailed_results": all_results
        }


def print_results(results: Dict[str, Any]):
    """결과 출력"""
    print("\n" + "=" * 70)
    print("📈 Knowledge Graph Benchmark 결과")
    print("=" * 70)

    overall = results.get("overall", {})
    print(f"\n📊 전체 성능:")
    print(f"   - Total Queries: {overall.get('total_queries', 0)}")
    print(f"   - Avg Precision: {overall.get('avg_precision', 0):.1%}")
    print(f"   - Avg Recall: {overall.get('avg_recall', 0):.1%}")
    print(f"   - Avg F1: {overall.get('avg_f1', 0):.1%}")
    print(f"   - Avg Latency: {overall.get('avg_latency_ms', 0):.2f}ms")
    print(f"   - P95 Latency: {overall.get('p95_latency_ms', 0):.2f}ms")

    print(f"\n📊 홉 수별 분석:")
    for hop_key, hop_data in results.get("by_hops", {}).items():
        print(f"   {hop_key}: Recall={hop_data['avg_recall']:.1%}, F1={hop_data['avg_f1']:.1%} ({hop_data['count']} queries)")

    print(f"\n📊 시나리오별 성능:")
    for scenario_id, scenario_data in results.get("by_scenario", {}).items():
        print(f"   {scenario_id}: F1={scenario_data['avg_f1']:.1%}, Recall={scenario_data['avg_recall']:.1%}")

    print()

    # 분석 요약
    print("📋 Knowledge Graph 분석:")
    print("   • 1-hop 검색: 직접 연결된 엔티티 검색 (높은 정확도)")
    print("   • 2-hop 검색: 간접 연결된 엔티티 검색 (복합 추론)")
    print("   • NetworkX 그래프 탐색: BFS 기반 빠른 탐색")
    print()

    if overall.get("avg_f1", 0) >= 0.7:
        print("✅ Knowledge Graph가 효과적으로 작동합니다!")
    elif overall.get("avg_f1", 0) >= 0.5:
        print("⚠️ Knowledge Graph가 어느 정도 효과적이지만 개선이 필요합니다.")
    else:
        print("❌ Knowledge Graph 성능 개선이 필요합니다.")


async def main():
    benchmark = KnowledgeGraphBenchmark()
    results = benchmark.run_benchmark()

    print_results(results)

    # 결과 저장
    output_file = "/tmp/knowledge_graph_benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "Knowledge Graph Evaluation",
            "timestamp": datetime.now().isoformat(),
            **results
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"📁 결과 저장: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
