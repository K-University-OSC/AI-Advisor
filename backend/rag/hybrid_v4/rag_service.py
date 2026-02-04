# -*- coding: utf-8 -*-
"""
HYBRID_V4 RAG Service

전체 RAG 파이프라인 통합 서비스

파이프라인:
1. PDF 파싱 (PyMuPDF + pdfplumber)
2. 청킹 (Semantic Chunking + 문단/테이블/이미지)
3. 임베딩 (text-embedding-3-small)
4. 벡터 저장 (Qdrant)
5. 검색 (Vector Search → BGE Reranker)
6. 답변 생성 (GPT-4o)

성능: 90.9% 정확도 (Azure/Upstage 미사용)
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .config import HybridV4Config
from .semantic_chunker import SemanticChunker, ChunkResult, ChunkType
from .image_captioner import ImageCaptioner, CaptionResult

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """검색 결과"""
    content: str
    score: float
    chunk_type: str = ""
    file_name: str = ""
    page: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryResult:
    """질의 결과"""
    answer: str
    sources: List[SearchResult]
    query: str
    metadata: dict = field(default_factory=dict)


class HybridV4RAGService:
    """
    HYBRID_V4 RAG 서비스

    전체 파이프라인을 관리하는 메인 클래스
    """

    def __init__(self, config: Optional[HybridV4Config] = None):
        """
        Args:
            config: HYBRID_V4 설정 (없으면 기본값 사용)
        """
        self.config = config or HybridV4Config()

        # 컴포넌트
        self.qdrant_client = None
        self.openai_client = None
        self.semantic_chunker = None
        self.image_captioner = None
        self.reranker = None

        self._initialized = False

    async def initialize(self) -> bool:
        """
        서비스 초기화

        Returns:
            초기화 성공 여부
        """
        try:
            logger.info("HYBRID_V4 RAG Service 초기화 시작...")

            # 설정 검증
            self.config.validate()

            # 1. OpenAI 클라이언트
            from openai import OpenAI, AsyncOpenAI
            self.openai_client = OpenAI(api_key=self.config.openai_api_key)
            self.async_openai_client = AsyncOpenAI(api_key=self.config.openai_api_key)
            logger.info("OpenAI 클라이언트 초기화 완료")

            # 2. Qdrant 클라이언트
            from qdrant_client import QdrantClient, AsyncQdrantClient
            self.qdrant_client = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port
            )
            self.async_qdrant_client = AsyncQdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port
            )
            logger.info(f"Qdrant 클라이언트 초기화 완료 ({self.config.qdrant_host}:{self.config.qdrant_port})")

            # 3. Semantic Chunker
            self.semantic_chunker = SemanticChunker(
                threshold=self.config.semantic_threshold,
                min_chunk_size=self.config.min_chunk_size,
                max_chunk_size=self.config.max_chunk_size
            )
            self.semantic_chunker.initialize()
            logger.info("Semantic Chunker 초기화 완료")

            # 4. Image Captioner
            self.image_captioner = ImageCaptioner(
                model=self.config.image_caption_model,
                detail=self.config.image_detail
            )
            self.image_captioner.initialize(self.openai_client)
            logger.info("Image Captioner 초기화 완료")

            # 5. BGE Reranker
            self._init_reranker()
            logger.info("BGE Reranker 초기화 완료")

            # 6. 컬렉션 확인/생성
            await self._ensure_collection()

            self._initialized = True
            logger.info("HYBRID_V4 RAG Service 초기화 완료")
            logger.info(f"  - Collection: {self.config.collection_name}")
            logger.info(f"  - Embedding: {self.config.embedding_model}")
            logger.info(f"  - Reranker: {self.config.reranker_model}")
            logger.info(f"  - LLM: {self.config.llm_model}")

            return True

        except Exception as e:
            logger.error(f"HYBRID_V4 RAG Service 초기화 실패: {e}")
            return False

    def _init_reranker(self):
        """BGE Reranker 초기화"""
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(
                self.config.reranker_model,
                max_length=self.config.reranker_max_length
            )
            logger.info(f"BGE Reranker 로드 완료: {self.config.reranker_model}")
        except Exception as e:
            logger.warning(f"BGE Reranker 초기화 실패 (검색 품질 저하 가능): {e}")
            self.reranker = None

    async def _ensure_collection(self):
        """Qdrant 컬렉션 확인/생성"""
        from qdrant_client.models import Distance, VectorParams

        collections = await self.async_qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.config.collection_name not in collection_names:
            await self.async_qdrant_client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"컬렉션 생성됨: {self.config.collection_name}")
        else:
            logger.info(f"컬렉션 존재함: {self.config.collection_name}")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ==================== 문서 인덱싱 ====================

    async def index_document(self, file_path: str) -> Dict[str, Any]:
        """
        문서 인덱싱

        Args:
            file_path: 문서 파일 경로

        Returns:
            인덱싱 결과 (청크 수, 상태 등)
        """
        if not self._initialized:
            raise RuntimeError("서비스가 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        logger.info(f"문서 인덱싱 시작: {file_path.name}")

        # 청크 수집
        all_chunks: List[ChunkResult] = []
        chunk_stats = {"text": 0, "paragraph": 0, "semantic": 0, "table": 0, "image": 0}

        # PDF 처리
        if file_path.suffix.lower() == ".pdf":
            chunks, stats = await self._process_pdf(file_path)
            all_chunks.extend(chunks)
            for k, v in stats.items():
                chunk_stats[k] = chunk_stats.get(k, 0) + v
        else:
            # 텍스트 파일 처리
            chunks = await self._process_text_file(file_path)
            all_chunks.extend(chunks)
            chunk_stats["text"] = len(chunks)

        logger.info(f"청크 생성 완료: {len(all_chunks)}개")
        logger.info(f"  - text: {chunk_stats['text']}, paragraph: {chunk_stats['paragraph']}")
        logger.info(f"  - semantic: {chunk_stats['semantic']}, table: {chunk_stats['table']}, image: {chunk_stats['image']}")

        # 임베딩 생성 및 저장
        await self._embed_and_store(all_chunks)

        return {
            "file_name": file_path.name,
            "total_chunks": len(all_chunks),
            "chunk_stats": chunk_stats,
            "status": "indexed"
        }

    async def _process_pdf(self, file_path: Path) -> tuple:
        """PDF 처리"""
        import fitz  # PyMuPDF
        import pdfplumber

        all_chunks = []
        chunk_stats = {"text": 0, "paragraph": 0, "semantic": 0, "table": 0, "image": 0}
        file_name = file_path.name

        # pdfplumber 테이블 설정
        table_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "text_keep_blank_chars": True,
            "intersection_tolerance": 3,
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
            "min_words_vertical": 3,
            "min_words_horizontal": 1
        }

        # 1. PyMuPDF로 텍스트 및 이미지 추출
        doc = fitz.open(str(file_path))
        full_text = ""
        page_texts = {}

        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            full_text += page_text + "\n"
            page_texts[page_num] = page_text

            # 이미지 추출 및 캡셔닝
            images = page.get_images(full=True)
            for img in images:
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # 작은 이미지 제외
                    if len(image_bytes) < self.config.min_image_size:
                        continue

                    # 캡셔닝
                    surrounding_text = page_text[:500] if page_text else ""
                    caption_result = await self.image_captioner.caption_image_async(
                        image_bytes, page_num + 1, surrounding_text
                    )

                    if caption_result.is_valid:
                        content = caption_result.to_chunk_content(surrounding_text)
                        all_chunks.append(ChunkResult(
                            content=content,
                            chunk_type=ChunkType.IMAGE,
                            file_name=file_name,
                            page=page_num + 1,
                            metadata={"image_type": caption_result.image_type}
                        ))
                        chunk_stats["image"] += 1

                except Exception as e:
                    logger.debug(f"이미지 처리 오류: {e}")
                    continue

        doc.close()

        # 2. pdfplumber로 테이블 추출
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_tables = page.extract_tables(table_settings)
                        for table in page_tables:
                            if table and len(table) > 1:
                                markdown = self._table_to_markdown(table)
                                if markdown and len(markdown) > 50:
                                    # 테이블 최대 길이 제한
                                    if len(markdown) > self.config.table_max_length:
                                        markdown = markdown[:self.config.table_max_length] + "\n... (표 일부 생략)"

                                    all_chunks.append(ChunkResult(
                                        content=f"[표]\n{markdown}",
                                        chunk_type=ChunkType.TABLE,
                                        file_name=file_name,
                                        page=page_num + 1
                                    ))
                                    chunk_stats["table"] += 1
                    except Exception as e:
                        logger.debug(f"테이블 추출 오류: {e}")
                        continue
        except Exception as e:
            logger.warning(f"pdfplumber 오류: {e}")

        # 3. 문단 추출 및 Semantic Chunking
        paragraphs = re.split(r'\n\s*\n', full_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 50]

        para_chunks = self.semantic_chunker.chunk_with_context(paragraphs, file_name)
        for chunk in para_chunks:
            all_chunks.append(chunk)
            if chunk.chunk_type == ChunkType.SEMANTIC:
                chunk_stats["semantic"] += 1
            else:
                chunk_stats["paragraph"] += 1

        # 4. 추가 텍스트 청킹 (고정 크기)
        for i in range(0, len(full_text), 450):
            chunk = full_text[i:i + 500]
            if chunk.strip() and len(chunk.strip()) > 100:
                all_chunks.append(ChunkResult(
                    content=chunk.strip(),
                    chunk_type=ChunkType.TEXT,
                    file_name=file_name,
                    page=0
                ))
                chunk_stats["text"] += 1

        return all_chunks, chunk_stats

    async def _process_text_file(self, file_path: Path) -> List[ChunkResult]:
        """텍스트 파일 처리"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        file_name = file_path.name
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 50]

        return self.semantic_chunker.chunk_with_context(paragraphs, file_name)

    def _table_to_markdown(self, table: list) -> str:
        """테이블을 Markdown으로 변환"""
        if not table or len(table) == 0:
            return ""

        lines = []
        if table[0]:
            header = [str(cell) if cell else "" for cell in table[0]]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for row in table[1:]:
            if row:
                cells = [str(cell) if cell else "" for cell in row]
                lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    async def _embed_and_store(self, chunks: List[ChunkResult]):
        """임베딩 생성 및 Qdrant 저장"""
        from qdrant_client.models import PointStruct

        if not chunks:
            return

        batch_size = 100
        points = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.content for c in batch]

            # 임베딩 생성
            response = self.openai_client.embeddings.create(
                model=self.config.embedding_model,
                input=texts
            )

            for j, emb in enumerate(response.data):
                chunk = batch[j]
                points.append(PointStruct(
                    id=len(points),
                    vector=emb.embedding,
                    payload={
                        "content": chunk.content,
                        "chunk_type": chunk.chunk_type.value,
                        "file_name": chunk.file_name,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata
                    }
                ))

        # Qdrant에 업로드
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await self.async_qdrant_client.upsert(
                collection_name=self.config.collection_name,
                points=batch
            )

        logger.info(f"Qdrant에 {len(points)}개 포인트 저장 완료")

    # ==================== 검색 및 답변 ====================

    async def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        검색 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        if not self._initialized:
            raise RuntimeError("서비스가 초기화되지 않았습니다.")

        top_k = top_k or self.config.rerank_top_k

        # 1단계: 벡터 검색
        query_embedding = self.openai_client.embeddings.create(
            model=self.config.embedding_model,
            input=[query]
        ).data[0].embedding

        results = await self.async_qdrant_client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding,
            limit=self.config.vector_top_k
        )

        if not results:
            return []

        # 2단계: 리랭킹
        documents = [r.payload["content"] for r in results]

        if self.reranker:
            reranked = self._rerank(query, documents, top_k)
            # 리랭킹된 순서로 결과 매핑
            reranked_results = []
            for doc in reranked:
                for r in results:
                    if r.payload["content"] == doc:
                        reranked_results.append(SearchResult(
                            content=r.payload["content"],
                            score=r.score,
                            chunk_type=r.payload.get("chunk_type", ""),
                            file_name=r.payload.get("file_name", ""),
                            page=r.payload.get("page", 0),
                            metadata=r.payload.get("metadata", {})
                        ))
                        break
            return reranked_results[:top_k]
        else:
            return [
                SearchResult(
                    content=r.payload["content"],
                    score=r.score,
                    chunk_type=r.payload.get("chunk_type", ""),
                    file_name=r.payload.get("file_name", ""),
                    page=r.payload.get("page", 0),
                    metadata=r.payload.get("metadata", {})
                )
                for r in results[:top_k]
            ]

    def _rerank(self, query: str, documents: List[str], top_k: int) -> List[str]:
        """BGE Reranker로 리랭킹"""
        if not self.reranker or not documents:
            return documents[:top_k]

        try:
            pairs = [(query, doc) for doc in documents]
            scores = self.reranker.predict(pairs)
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:top_k]]
        except Exception as e:
            logger.error(f"리랭킹 실패: {e}")
            return documents[:top_k]

    async def query(self, question: str, top_k: Optional[int] = None) -> QueryResult:
        """
        질의응답 수행

        Args:
            question: 질문
            top_k: 검색 결과 수

        Returns:
            QueryResult (답변 + 소스)
        """
        if not self._initialized:
            raise RuntimeError("서비스가 초기화되지 않았습니다.")

        # 검색
        search_results = await self.search(question, top_k)

        if not search_results:
            return QueryResult(
                answer="관련 정보를 찾을 수 없습니다.",
                sources=[],
                query=question
            )

        # 컨텍스트 구성
        context = "\n\n".join([r.content for r in search_results])

        # 답변 생성
        answer = await self._generate_answer(question, context)

        return QueryResult(
            answer=answer,
            sources=search_results,
            query=question,
            metadata={
                "model": self.config.llm_model,
                "sources_count": len(search_results)
            }
        )

    async def _generate_answer(self, question: str, context: str) -> str:
        """GPT-4o로 답변 생성"""
        try:
            response = await self.async_openai_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "주어진 컨텍스트를 바탕으로 질문에 정확하고 간결하게 답변하세요. "
                                   "이미지 관련 정보도 컨텍스트에 포함되어 있습니다. "
                                   "컨텍스트에 없는 정보는 답변하지 마세요."
                    },
                    {
                        "role": "user",
                        "content": f"컨텍스트:\n{context}\n\n질문: {question}"
                    }
                ],
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {e}"

    # ==================== 유틸리티 ====================

    async def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        if not self._initialized:
            return {"error": "서비스가 초기화되지 않았습니다."}

        try:
            info = await self.async_qdrant_client.get_collection(self.config.collection_name)
            return {
                "collection_name": self.config.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value
            }
        except Exception as e:
            return {"error": str(e)}

    async def clear_collection(self):
        """컬렉션 초기화"""
        if not self._initialized:
            raise RuntimeError("서비스가 초기화되지 않았습니다.")

        await self.async_qdrant_client.delete_collection(self.config.collection_name)
        await self._ensure_collection()
        logger.info(f"컬렉션 초기화 완료: {self.config.collection_name}")
