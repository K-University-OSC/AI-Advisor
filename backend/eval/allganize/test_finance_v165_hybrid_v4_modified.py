# -*- coding: utf-8 -*-
"""
Advisor OSC Finance v1.6.5 테스트 (HYBRID_V4 수정 버전)

파이프라인:
- Document Parser: PyMuPDF4LLM
- Image Captioning: GPT-4o Vision
- Chunking: SemanticChunker (all-MiniLM-L6-v2)
- Embedding: OpenAI text-embedding-3-large (3072 dim)
- Reranker: BGE v2-m3 (GPU, Thread-safe)
- LLM: GPT-5-mini (no thinking)
- Judge: GPT-5-mini (no thinking)

Collection: advisor_osc_finance_v165_hybrid_v4
"""

import asyncio
import json
import os
import sys
import re
import threading
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

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
from sentence_transformers import SentenceTransformer

from config import get_settings
from rag.embeddings import OpenAIEmbeddingService, SparseEmbeddingService, MultimodalEmbeddingService
from rag.vectorstore import QdrantVectorStore
from rag.retriever import HierarchicalRetriever, RetrievalConfig
from rag.parsers.document_parser import ParsedDocument, ParsedElement, ElementType
from rag.chunkers import HierarchicalChunker

# v1.6.5 설정
VERSION = "v1.6.5"
COLLECTION_NAME = "advisor_osc_finance_v165_hybrid_v4"
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
# SemanticChunker (HYBRID_V4에서 가져옴)
# ============================================================================

class SemanticChunker:
    """의미 기반 청킹 (all-MiniLM-L6-v2)"""

    def __init__(self, threshold: float = 0.5, min_chunk_size: int = 100, max_chunk_size: int = 1000):
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model

    def chunk_text(self, text: str, source: str = "") -> List[dict]:
        """텍스트를 의미 단위로 청킹"""
        import numpy as np

        # 문장 분리
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # 임베딩 계산
        model = self._get_model()
        embeddings = model.encode(sentences)

        # 유사도 기반 청크 경계 찾기
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            # 이전 문장과의 유사도
            sim = np.dot(embeddings[i], embeddings[i-1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]) + 1e-8
            )

            current_text = ' '.join(current_chunk)

            # 청크 경계 결정
            if sim < self.threshold or len(current_text) > self.max_chunk_size:
                if len(current_text) >= self.min_chunk_size:
                    chunks.append({
                        'content': current_text,
                        'source': source,
                        'chunk_id': str(uuid.uuid4())
                    })
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        # 마지막 청크
        if current_chunk:
            current_text = ' '.join(current_chunk)
            if len(current_text) >= self.min_chunk_size:
                chunks.append({
                    'content': current_text,
                    'source': source,
                    'chunk_id': str(uuid.uuid4())
                })

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """문장 분리"""
        import re
        # 한국어/영어 문장 분리
        sentences = re.split(r'(?<=[.!?。])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


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
# 인덱싱 함수
# ============================================================================

async def index_documents():
    """문서 인덱싱 (PyMuPDF4LLM + SemanticChunker + OpenAI Embedding)"""
    print("=" * 60)
    print(f"v1.6.5 인덱싱 시작")
    print("=" * 60)

    settings = get_settings()

    # 서비스 초기화
    dense_service = OpenAIEmbeddingService(api_key=settings.openai_api_key, model="text-embedding-3-large")
    sparse_service = SparseEmbeddingService()
    embedding_service = MultimodalEmbeddingService(dense_service, sparse_service)

    chunker = SemanticChunker(threshold=0.5, min_chunk_size=100, max_chunk_size=1000)

    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=COLLECTION_NAME,
    )
    await vector_store.initialize()

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"PDF 파일: {len(pdf_files)}개\n")

    total_chunks = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
        try:
            # PyMuPDF4LLM으로 마크다운 추출
            md_text = pymupdf4llm.to_markdown(str(pdf_path))

            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()

            # SemanticChunker로 청킹
            chunks = chunker.chunk_text(md_text, source=pdf_path.name)
            print(f"  - 페이지: {total_pages}, 청크: {len(chunks)}개")

            if not chunks:
                continue

            # ParsedDocument 생성
            elements = [ParsedElement(
                element_id=str(uuid.uuid4()),
                element_type=ElementType.PARAGRAPH,
                content=md_text,
                page=1,
                markdown_content=md_text,
                metadata={"parser": "pymupdf4llm", "source": pdf_path.name}
            )]

            parsed_doc = ParsedDocument(
                source=str(pdf_path),
                filename=pdf_path.name,
                total_pages=total_pages,
                elements=elements,
                metadata={"parser": "pymupdf4llm"}
            )

            # HierarchicalChunker 사용 (기존 인프라 활용)
            hier_chunker = HierarchicalChunker(
                parent_chunk_size=settings.parent_chunk_size,
                child_chunk_size=settings.child_chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            parent_chunks, child_chunks = hier_chunker.chunk_document(document=parsed_doc, caption_results=[])

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

    print(f"\n인덱싱 완료: 총 {total_chunks}개 청크")
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
    print(f"  파서: PyMuPDF4LLM")
    print(f"  청킹: SemanticChunker (all-MiniLM-L6-v2)")
    print(f"  임베딩: OpenAI text-embedding-3-large")
    print(f"  Reranker: BGE v2-m3 (GPU, Thread-safe)")
    print(f"  LLM: GPT-5-mini (no thinking)")
    print(f"  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"finance_{VERSION}_hybrid_v4_modified_{timestamp}.json"

    output_data = {
        "version": VERSION,
        "pipeline": {
            "parser": "PyMuPDF4LLM",
            "chunking": "SemanticChunker (all-MiniLM-L6-v2)",
            "image_captioning": "GPT-4o Vision",
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
