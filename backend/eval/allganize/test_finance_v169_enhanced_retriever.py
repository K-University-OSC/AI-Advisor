# -*- coding: utf-8 -*-
"""
Advisor OSC Finance v1.6.9 테스트 (EnhancedHierarchicalRetriever)

v1.6.8 (GPT-4o Vision OCR) + EnhancedHierarchicalRetriever:
- Query Expansion: LLM 기반 쿼리 확장 + 동의어 사전
- Multi-Query Retrieval: 3개 대안 쿼리 병렬 검색
- Query-Adaptive Weights: 쿼리 특성별 Dense/Sparse 가중치 동적 조정
- RRF Hybrid Search: Dense + Sparse 순위 융합
- Fallback Search: 결과 부족 시 대안 전략
- Image Chunk Boosting: 이미지 관련 쿼리 가중치 부여

파이프라인:
- Parser: GPT-4o Vision OCR (페이지별 병렬)
- Chunking: HierarchicalChunker
- Retriever: EnhancedHierarchicalRetriever (NEW)
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

# v1.6.9 설정
VERSION = "v1.6.9"
COLLECTION_NAME = "advisor_osc_finance_v168_gpt4o_ocr"  # v1.6.8 컬렉션 재사용
PDF_DIR = Path(__file__).parent / "files" / "finance"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 병렬 처리 설정
OCR_CONCURRENCY = 5  # 페이지 OCR 병렬 수
TEST_CONCURRENCY = 5  # 테스트 케이스 병렬 수

# GPT-4o OCR 시스템 프롬프트 (v1.6.8과 동일)
OCR_SYSTEM_PROMPT = """당신은 문서 OCR 전문가입니다. 이미지에서 텍스트를 정확하게 추출하세요.

## 출력 규칙
1. 마크다운 형식으로 출력
2. 제목은 # ## ### 사용
3. 테이블은 마크다운 테이블 형식 (| col1 | col2 |)
4. 목록은 - 또는 1. 2. 3. 사용
5. 이미지/차트/그래프는 [IMAGE: 상세 설명] 형식으로 내용을 설명
   - 차트의 경우 데이터, 트렌드, 수치를 포함
   - 다이어그램의 경우 구조와 관계를 설명
   - 모든 텍스트/레이블/범례를 정확히 추출
6. 원본 레이아웃 최대한 유지
7. 모든 수치, 날짜, 기관명, 법률명 등 정확히 기재

## 출력 형식
텍스트 내용만 출력 (추가 설명이나 코멘트 없이)"""

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
# GPT-4o Vision OCR Parser (v1.6.8과 동일)
# ============================================================================

class GPT4oOCRParser:
    """GPT-4o Vision 기반 PDF 파서 (Gemini OCR 방식)"""

    def __init__(self, api_key: str, model: str = "gpt-4o", dpi: int = 150):
        self.api_key = api_key
        self.model = model
        self.dpi = dpi
        self.client = AsyncOpenAI(api_key=api_key)

    def pdf_to_images(self, pdf_path: Path) -> List[bytes]:
        """PDF를 페이지별 이미지로 변환"""
        doc = fitz.open(pdf_path)
        images = []
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)

        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            images.append(img_data)

        doc.close()
        return images

    async def ocr_page(self, image_data: bytes, page_num: int) -> str:
        """단일 페이지 OCR (GPT-4o Vision)"""
        img_base64 = base64.b64encode(image_data).decode()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": OCR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"이 문서 페이지({page_num}페이지)의 내용을 추출해주세요."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=8192,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    OCR 오류 (page {page_num}): {e}")
            return ""

    async def parse_pdf(self, pdf_path: Path) -> tuple:
        """
        PDF 파싱 (페이지별 병렬 OCR)

        Returns:
            (전체 마크다운 텍스트, 총 페이지 수)
        """
        # PDF → 이미지 변환
        images = self.pdf_to_images(pdf_path)
        total_pages = len(images)

        # 페이지별 병렬 OCR
        semaphore = asyncio.Semaphore(OCR_CONCURRENCY)

        async def ocr_with_limit(img_data: bytes, page_num: int) -> tuple:
            async with semaphore:
                text = await self.ocr_page(img_data, page_num)
                return page_num, text

        tasks = [
            ocr_with_limit(img_data, page_num)
            for page_num, img_data in enumerate(images, 1)
        ]
        results = await asyncio.gather(*tasks)

        # 페이지 순서대로 정렬 및 병합
        results.sort(key=lambda x: x[0])
        full_text = ""
        for page_num, text in results:
            if text.strip():
                full_text += f"\n\n<!-- Page {page_num} -->\n{text}"

        return full_text.strip(), total_pages


# ============================================================================
# 인덱싱 함수 (GPT-4o Vision OCR) - v1.6.8과 동일
# ============================================================================

async def index_documents():
    """문서 인덱싱 (GPT-4o Vision OCR + 병렬 처리)"""
    print("=" * 60)
    print(f"v1.6.9 인덱싱 시작 (GPT-4o Vision OCR)")
    print("=" * 60)
    print("파이프라인:")
    print(f"  - PDF → 이미지 (150 DPI)")
    print(f"  - GPT-4o Vision OCR (병렬 {OCR_CONCURRENCY}개)")
    print(f"  - 마크다운 + [IMAGE: 설명] 출력")
    print("=" * 60)

    settings = get_settings()

    # 서비스 초기화
    dense_service = OpenAIEmbeddingService(api_key=settings.openai_api_key, model="text-embedding-3-large")
    sparse_service = SparseEmbeddingService()
    embedding_service = MultimodalEmbeddingService(dense_service, sparse_service)

    # GPT-4o OCR 파서 초기화
    parser = GPT4oOCRParser(api_key=OPENAI_API_KEY)

    # 청커 초기화
    chunker = HierarchicalChunker(
        parent_chunk_size=settings.parent_chunk_size,
        child_chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=COLLECTION_NAME,
    )
    await vector_store.initialize()

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"PDF 파일: {len(pdf_files)}개\n")

    total_chunks = 0
    total_pages_processed = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
        try:
            # GPT-4o Vision OCR (페이지 병렬 처리)
            full_text, total_pages = await parser.parse_pdf(pdf_path)
            print(f"  - 페이지: {total_pages}, OCR 완료")
            total_pages_processed += total_pages

            # [IMAGE: ...] 개수 카운트
            image_count = len(re.findall(r'\[IMAGE:', full_text))
            print(f"  - 이미지 설명: {image_count}개")

            # ParsedDocument 생성
            elements = [ParsedElement(
                element_id=str(uuid.uuid4()),
                element_type=ElementType.PARAGRAPH,
                content=full_text,
                page=1,
                markdown_content=full_text,
                metadata={
                    "parser": "gpt4o-vision-ocr",
                    "source": pdf_path.name,
                    "images": image_count
                }
            )]

            parsed_doc = ParsedDocument(
                source=str(pdf_path),
                filename=pdf_path.name,
                total_pages=total_pages,
                elements=elements,
                metadata={"parser": "gpt4o-vision-ocr", "images": image_count}
            )

            # 청킹
            parent_chunks, child_chunks = chunker.chunk_document(document=parsed_doc, caption_results=[])

            # 임베딩 및 저장
            dense_embeddings, sparse_embeddings = await embedding_service.embed_chunks(
                parent_chunks=parent_chunks, child_chunks=child_chunks)

            await vector_store.add_chunks(
                parent_chunks=parent_chunks, child_chunks=child_chunks,
                dense_embeddings=dense_embeddings, sparse_embeddings=sparse_embeddings)

            total_chunks += len(child_chunks)
            print(f"  ✓ 완료 (자식 청크: {len(child_chunks)})")

        except Exception as e:
            print(f"  ✗ 오류: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n인덱싱 완료: 총 {total_chunks}개 청크, {total_pages_processed}개 페이지 처리")
    await vector_store.close()
    return total_chunks


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

            # 컨텍스트 추출
            context = retrieval_result.context

            # 메타데이터 추출 (디버깅용)
            enhancements = retrieval_result.metadata.get("enhancements_applied", [])

            answer = await generate_answer(query=question, context=context)
            judge_result = await call_judge(question, target, answer)
            verdict = judge_result.get("verdict", "X")
            is_correct = verdict == "O"
            progress_counter[0] += 1

            # 적용된 향상 기능 표시
            enhancement_str = ",".join(enhancements[:3]) if enhancements else "none"
            print(f"  [{progress_counter[0]}/{total}] idx={idx} -> {'O' if is_correct else 'X'} ({enhancement_str})")

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
                "adaptive_weights": retrieval_result.metadata.get("adaptive_weights", {}),
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
    print(f"  파서: GPT-4o Vision OCR (v1.6.8과 동일)")
    print(f"  Retriever: EnhancedHierarchicalRetriever (NEW)")
    print(f"  임베딩: OpenAI text-embedding-3-large")
    print(f"  Reranker: BGE v2-m3 (내장)")
    print(f"  LLM: GPT-5-mini (no thinking)")
    print(f"  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n핵심 변경 (vs v1.6.8):")
    print("  - HierarchicalRetriever → EnhancedHierarchicalRetriever")
    print("  - Query Expansion: LLM 기반 쿼리 확장 + 동의어 사전")
    print("  - Multi-Query: 3개 대안 쿼리 병렬 검색")
    print("  - Query-Adaptive Weights: 쿼리 특성별 가중치 조정")
    print("  - RRF Hybrid: Dense + Sparse 순위 융합")
    print("  - Fallback Search: 결과 부족 시 대안 전략")
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
            print(f"\n컬렉션 '{COLLECTION_NAME}'이 존재하지 않습니다. 인덱싱을 먼저 실행합니다...")
            await index_documents()
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        if collection_info.points_count == 0:
            print(f"\n컬렉션이 비어있습니다. 인덱싱을 실행합니다...")
            await index_documents()
        print(f"\n컬렉션 정보: {qdrant_client.get_collection(COLLECTION_NAME).points_count} points")
    except Exception as e:
        print(f"Qdrant 연결 오류: {e}")
        return 0

    # EnhancedHierarchicalRetriever 설정
    enhanced_config = EnhancedRetrievalConfig(
        # 기본 검색 설정
        top_k=10,
        rerank_top_k=30,
        expand_to_parent=True,
        rerank=True,

        # Reranker 선택 (BGE 사용)
        reranker_type="bge",

        # 향상 기능 활성화
        enable_query_expansion=True,      # 쿼리 확장
        enable_multi_query=True,          # 다중 쿼리
        enable_adaptive_search=True,      # 적응형 검색
        enable_rrf=True,                  # RRF 하이브리드
        enable_hyde=False,                # HyDE는 비활성화 (실험적)

        # 다중 쿼리 설정
        num_sub_queries=3,

        # RRF 설정
        rrf_k=60,
        rrf_dense_weight=1.5,
        rrf_sparse_weight=0.5,

        # 테이블/이미지 적응형 검색
        enable_table_adaptive=True,
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

    # 이전 버전과 비교
    print(f"\n{'=' * 60}")
    print("이전 버전 대비 변화:")
    prev_results = {
        "v1.6.6": {"overall": 68.3, "image": 40.0, "paragraph": 76.7, "table": 100.0},
        "v1.6.7": {"overall": 61.7, "image": 40.0, "paragraph": 70.0, "table": 80.0},
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
    output_file = output_dir / f"finance_{VERSION}_enhanced_retriever_{timestamp}.json"

    output_data = {
        "version": VERSION,
        "approach": "v1.6.8 (GPT-4o OCR) + EnhancedHierarchicalRetriever",
        "pipeline": {
            "parser": "GPT-4o Vision OCR",
            "retriever": "EnhancedHierarchicalRetriever",
            "enhancements": [
                "Query Expansion (LLM + 동의어 사전)",
                "Multi-Query Retrieval (3 sub-queries)",
                "Query-Adaptive Weights",
                "RRF Hybrid Search",
                "Fallback Search",
                "Image Chunk Boosting",
            ],
            "chunking": "HierarchicalChunker",
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
            "enable_hyde": enhanced_config.enable_hyde,
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
