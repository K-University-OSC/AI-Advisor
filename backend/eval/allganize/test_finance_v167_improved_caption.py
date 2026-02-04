# -*- coding: utf-8 -*-
"""
Advisor OSC Finance v1.6.7 테스트 (개선된 이미지 캡셔닝)

v1.6.6 대비 개선 사항:
1. 이미지 캡션을 페이지별로 해당 텍스트와 통합 (문서 끝 추가 X)
2. 상세한 캡셔닝 프롬프트 (유형/제목/주요내용/상세설명/키워드)
3. max_tokens 2500으로 증가
4. ParsedElement 구조로 이미지 처리

파이프라인:
- Document Parser: PyMuPDF4LLM (페이지별 파싱)
- Image Captioning: GPT-4o Vision (상세 프롬프트)
- Chunking: HierarchicalChunker
- Embedding: OpenAI text-embedding-3-large (3072 dim)
- Reranker: BGE v2-m3 (GPU, Thread-safe)
- LLM: GPT-5-mini (no thinking)
- Judge: GPT-5-mini (no thinking)

Collection: advisor_osc_finance_v167_improved
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
from dataclasses import dataclass
from typing import List, Dict, Optional

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
import pymupdf4llm
import fitz
from openai import AsyncOpenAI

from config import get_settings
from rag.embeddings import OpenAIEmbeddingService, SparseEmbeddingService, MultimodalEmbeddingService
from rag.vectorstore import QdrantVectorStore
from rag.retriever import HierarchicalRetriever, RetrievalConfig
from rag.parsers.document_parser import ParsedDocument, ParsedElement, ElementType
from rag.chunkers import HierarchicalChunker

# v1.6.7 설정
VERSION = "v1.6.7"
COLLECTION_NAME = "advisor_osc_finance_v167_improved"
PDF_DIR = Path(__file__).parent / "files" / "finance"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """당신은 문서 분석 AI 어시스턴트입니다.
제공된 컨텍스트를 기반으로 사용자의 질문에 정확하고 완전하게 답변해주세요.

**핵심 규칙:**
1. 컨텍스트의 모든 관련 정보를 빠짐없이 포함하세요
2. 수치/데이터는 단위와 함께 정확히 인용하세요
3. 질문에 여러 하위 질문이 있으면 각각 답변하세요
4. 테이블에서 조건에 맞는 모든 항목을 나열하세요
5. 구체적인 사례/예시가 컨텍스트에 있으면 반드시 포함하세요
6. [이미지 분석] 또는 [차트 분석] 섹션의 정보도 적극 활용하세요

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


# ============================================================================
# 개선된 GPT-4o Vision Image Captioner
# ============================================================================

# V7.8 스타일 상세 캡셔닝 프롬프트
CAPTION_SYSTEM_PROMPT = """당신은 문서에서 추출한 이미지와 차트를 분석하는 전문가입니다.
주어진 이미지를 분석하고 다음 형식으로 상세히 설명해주세요:

1. **유형**: 차트/그래프/표/다이어그램/플로우차트/인포그래픽 중 무엇인지
2. **제목**: 이미지에 표시된 제목 (정확히 기재)
3. **주요 내용**:
   - 모든 텍스트를 정확히 추출 (법률명, 기관명, 용어 등)
   - 구조화된 정보가 있다면 번호를 매겨 나열 (예: 4가지 요인, 3단계 등)
   - 핵심 데이터와 수치를 구체적으로 나열
   - 항목별 값과 단위를 명확히 기재
   - 화살표/연결선이 있다면 관계와 흐름 설명
4. **상세 설명**:
   - 각 항목의 세부 내용 (박스 안 텍스트 전체 추출)
   - 범례, 주석, 출처 등 부가 정보
5. **키워드**: 검색에 도움될 핵심 키워드 10-15개 (고유명사, 전문용어 포함)

중요: 이미지 내 모든 텍스트를 빠짐없이 추출하세요. 특히 법률명, 기관명, 조직명, 전문용어는 정확히 기재해야 합니다."""


class ImprovedImageCaptioner:
    """개선된 GPT-4o Vision 기반 이미지 캡셔닝"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def caption_image(self, image_bytes: bytes, context: str = None) -> str:
        """단일 이미지 캡셔닝 (상세 프롬프트)"""
        img_base64 = base64.b64encode(image_bytes).decode()

        user_content = []
        if context:
            user_content.append({
                "type": "text",
                "text": f"문서 컨텍스트: {context}\n\n위 컨텍스트를 참고하여 아래 이미지를 분석해주세요."
            })
        else:
            user_content.append({
                "type": "text",
                "text": "아래 이미지를 분석해주세요."
            })

        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_base64}", "detail": "high"}
        })

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=2500,  # v1.6.6의 1024에서 증가
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    캡셔닝 오류: {e}")
            return None

    async def extract_page_images(self, doc: fitz.Document, page_num: int) -> List[Dict]:
        """특정 페이지에서 이미지 추출 및 캡셔닝"""
        page = doc[page_num]
        images = page.get_images()
        results = []

        for img_idx, img in enumerate(images):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                # 너무 작은 이미지 스킵
                if pix.width < 50 or pix.height < 50:
                    continue

                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_bytes = pix.tobytes("png")

                # 5KB 미만 스킵
                if len(img_bytes) < 5000:
                    continue

                # 캡셔닝
                caption = await self.caption_image(img_bytes)
                if caption:
                    results.append({
                        "page": page_num + 1,
                        "image_idx": img_idx,
                        "caption": caption,
                        "width": pix.width,
                        "height": pix.height
                    })

            except Exception as e:
                pass  # 이미지 처리 실패는 무시

        return results


# ============================================================================
# 페이지별 파싱 및 이미지 통합
# ============================================================================

async def parse_pdf_with_page_captions(pdf_path: Path, captioner: ImprovedImageCaptioner) -> tuple:
    """
    PDF를 페이지별로 파싱하고 이미지 캡션을 해당 페이지 텍스트와 통합

    핵심 개선: 이미지 캡션이 해당 페이지의 텍스트 바로 다음에 배치됨
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # 페이지별 마크다운 추출
    page_texts = []
    for page_num in range(total_pages):
        try:
            # 단일 페이지만 마크다운으로 변환
            md = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_num])
            page_texts.append(md)
        except:
            page_texts.append("")

    # 페이지별 이미지 캡셔닝 (병렬 처리)
    total_captions = 0
    integrated_text = ""

    for page_num in range(total_pages):
        # 페이지 텍스트 추가
        page_text = page_texts[page_num].strip()
        if page_text:
            integrated_text += f"\n\n--- 페이지 {page_num + 1} ---\n\n{page_text}"

        # 해당 페이지의 이미지 캡셔닝
        page_captions = await captioner.extract_page_images(doc, page_num)

        # 이미지 캡션을 해당 페이지 텍스트 바로 다음에 추가
        if page_captions:
            integrated_text += f"\n\n### [페이지 {page_num + 1} 이미지 분석]\n"
            for cap in page_captions:
                integrated_text += f"\n**[이미지 {cap['image_idx'] + 1}]**\n{cap['caption']}\n"
                total_captions += 1

    doc.close()
    return integrated_text, total_pages, total_captions


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
# 인덱싱 함수 (개선된 이미지 캡셔닝)
# ============================================================================

async def index_documents():
    """문서 인덱싱 (페이지별 이미지 캡션 통합)"""
    print("=" * 60)
    print(f"v1.6.7 인덱싱 시작 (개선된 이미지 캡셔닝)")
    print("=" * 60)
    print("개선 사항:")
    print("  - 이미지 캡션을 해당 페이지 텍스트와 통합")
    print("  - 상세 캡셔닝 프롬프트 (5단계 분석)")
    print("  - max_tokens 2500")
    print("=" * 60)

    settings = get_settings()

    # 서비스 초기화
    dense_service = OpenAIEmbeddingService(api_key=settings.openai_api_key, model="text-embedding-3-large")
    sparse_service = SparseEmbeddingService()
    embedding_service = MultimodalEmbeddingService(dense_service, sparse_service)

    # 개선된 이미지 캡셔너 초기화
    captioner = ImprovedImageCaptioner(api_key=OPENAI_API_KEY)

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
    total_captions = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
        try:
            # 개선된 파싱: 페이지별 이미지 캡션 통합
            integrated_text, total_pages, num_captions = await parse_pdf_with_page_captions(
                pdf_path, captioner
            )
            print(f"  - 페이지: {total_pages}, 이미지 캡션: {num_captions}개")
            total_captions += num_captions

            # ParsedDocument 생성
            elements = [ParsedElement(
                element_id=str(uuid.uuid4()),
                element_type=ElementType.PARAGRAPH,
                content=integrated_text,
                page=1,
                markdown_content=integrated_text,
                metadata={
                    "parser": "pymupdf4llm+gpt4o_improved",
                    "source": pdf_path.name,
                    "captions": num_captions,
                    "integration": "page_level"  # 페이지 수준 통합 표시
                }
            )]

            parsed_doc = ParsedDocument(
                source=str(pdf_path),
                filename=pdf_path.name,
                total_pages=total_pages,
                elements=elements,
                metadata={"parser": "pymupdf4llm+gpt4o_improved", "captions": num_captions}
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

    print(f"\n인덱싱 완료: 총 {total_chunks}개 청크, {total_captions}개 이미지 캡션")
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
[이미지 분석] 섹션의 정보도 적극 활용해주세요.

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


async def process_single_case(item, retriever, reranker, config, semaphore, progress_counter, total):
    async with semaphore:
        idx, question, target, context_type = item["idx"], item["question"], item["target_answer"], item["context_type"]
        try:
            no_rerank_config = RetrievalConfig(
                top_k=config.rerank_top_k,
                rerank=False,
                expand_to_parent=config.expand_to_parent,
                use_hybrid=config.use_hybrid,
            )
            retrieval_result = await retriever.retrieve_with_context(query=question, config=no_rerank_config)

            if retrieval_result.child_results and reranker.is_initialized:
                documents = [r.content for r in retrieval_result.child_results]
                reranked = await reranker.rerank_async(question, documents, top_k=config.top_k)
                context = "\n\n---\n\n".join([doc for doc, score in reranked])
            else:
                context = retrieval_result.context

            answer = await generate_answer(query=question, context=context)
            judge_result = await call_judge(question, target, answer)
            verdict = judge_result.get("verdict", "X")
            is_correct = verdict == "O"
            progress_counter[0] += 1
            print(f"  [{progress_counter[0]}/{total}] idx={idx} -> {'O' if is_correct else 'X'}")
            return {"idx": idx, "context_type": context_type, "question": question, "target_answer": target,
                    "generated_answer": answer, "verdict": verdict, "reason": judge_result.get("reason", ""), "is_correct": is_correct}
        except Exception as e:
            progress_counter[0] += 1
            print(f"  [{progress_counter[0]}/{total}] idx={idx} -> 오류: {e}")
            return {"idx": idx, "context_type": context_type, "question": question, "target_answer": target,
                    "generated_answer": f"오류: {e}", "verdict": "X", "reason": "실행 오류", "is_correct": False}


async def run_test():
    print("=" * 60)
    print(f"Advisor OSC Finance {VERSION} 테스트")
    print("=" * 60)
    print(f"  컬렉션: {COLLECTION_NAME}")
    print(f"  파서: PyMuPDF4LLM + GPT-4o Vision (페이지별 통합)")
    print(f"  임베딩: OpenAI text-embedding-3-large")
    print(f"  Reranker: BGE v2-m3 (GPU, Thread-safe)")
    print(f"  LLM: GPT-5-mini (no thinking)")
    print(f"  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n개선 사항 (vs v1.6.6):")
    print("  1. 이미지 캡션을 해당 페이지 텍스트와 통합 (문서 끝 X)")
    print("  2. 상세 캡셔닝 프롬프트 (5단계 분석)")
    print("  3. max_tokens 1024 → 2500")
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
            print(f"\n컬렉션 '{COLLECTION_NAME}'이 존재하지 않습니다. 인덱싱을 먼저 실행합니다...")
            await index_documents()
        print(f"\n컬렉션 정보: {qdrant_client.get_collection(COLLECTION_NAME).points_count} points")
    except Exception as e:
        print(f"Qdrant 연결 오류: {e}")
        return 0

    vector_store = QdrantVectorStore(host=settings.qdrant_host, port=settings.qdrant_port, collection_name=COLLECTION_NAME)
    retriever = HierarchicalRetriever(vector_store=vector_store, embedding_service=embedding_service, config=RetrievalConfig())
    config = RetrievalConfig(top_k=15, rerank_top_k=50, rerank=False, expand_to_parent=True, use_hybrid=False)

    print(f"\n{'=' * 60}\n테스트 실행 중...\n{'=' * 60}")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    progress_counter = [0]
    tasks = [process_single_case(item, retriever, reranker, config, semaphore, progress_counter, total) for item in test_data]
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

    # v1.6.6과 비교
    print(f"\n{'=' * 60}")
    print("v1.6.6 대비 변화:")
    v166_results = {"overall": 68.3, "image": 40.0, "paragraph": 76.7, "table": 100.0}
    print(f"  전체: {v166_results['overall']:.1f}% → {overall_acc:.1f}% ({overall_acc - v166_results['overall']:+.1f}%)")
    for ct in ["image", "paragraph", "table"]:
        if ct in by_context:
            new_acc = by_context[ct]["correct"] / by_context[ct]["total"] * 100
            diff = new_acc - v166_results.get(ct, 0)
            print(f"  {ct}: {v166_results.get(ct, 0):.1f}% → {new_acc:.1f}% ({diff:+.1f}%)")
    print("=" * 60)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"finance_{VERSION}_improved_caption_{timestamp}.json"

    output_data = {
        "version": VERSION,
        "improvements": [
            "이미지 캡션을 해당 페이지 텍스트와 통합",
            "상세 캡셔닝 프롬프트 (5단계 분석)",
            "max_tokens 2500"
        ],
        "pipeline": {
            "parser": "PyMuPDF4LLM (페이지별)",
            "image_captioning": "GPT-4o Vision (상세 프롬프트)",
            "caption_integration": "page_level",
            "chunking": "HierarchicalChunker",
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
        "comparison_v166": {
            "overall_diff": overall_acc - v166_results["overall"],
            "image_diff": (by_context.get("image", {}).get("correct", 0) / by_context.get("image", {}).get("total", 1) * 100) - v166_results["image"] if "image" in by_context else 0
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
