"""
Advisor RAG 설정 관리 모듈
모든 환경 변수와 설정을 중앙에서 관리합니다.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=True)


# =============================================================================
# API Keys 설정 (memory_service.py 호환용)
# =============================================================================
@dataclass
class APIKeysConfig:
    """API 키 설정"""
    openai: Optional[str] = os.getenv("OPENAI_API_KEY")
    google: Optional[str] = os.getenv("GOOGLE_API_KEY")
    anthropic: Optional[str] = os.getenv("CLAUDE_API_KEY")
    perplexity: Optional[str] = os.getenv("PERPLEXITY_API_KEY")
    tavily: Optional[str] = os.getenv("TAVILY_API_KEY")
    cohere: Optional[str] = os.getenv("COHERE_API_KEY")

    def get_key_for_provider(self, provider: str) -> Optional[str]:
        """프로바이더별 API 키 조회"""
        mapping = {
            "openai": self.openai,
            "google": self.google,
            "anthropic": self.anthropic,
            "perplexity": self.perplexity,
            "tavily": self.tavily,
            "cohere": self.cohere,
        }
        return mapping.get(provider)


# =============================================================================
# Provider 설정 (memory_service.py 호환용)
# =============================================================================
@dataclass
class ProvidersConfig:
    """Provider 설정 - personalization_core 호환"""
    # LLM Provider
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("LLM_MODEL", "gpt5")

    # Embedding Provider
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    # VectorDB Provider
    vectordb_provider: str = os.getenv("VECTORDB_PROVIDER", "qdrant")
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))

    # Reranker Provider
    reranker_provider: str = os.getenv("RERANKER_PROVIDER", "bge")
    bge_reranker_model: str = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    # Query Expansion
    query_expansion_model: str = os.getenv("QUERY_EXPANSION_MODEL", "gpt5-mini")


class Settings(BaseSettings):
    """Advisor RAG 애플리케이션 설정"""

    # API Keys
    upstage_api_key: str = Field(default="", env="UPSTAGE_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")

    # Qdrant Settings (advisor 전용 컬렉션)
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(
        default="advisor_memory",
        env="QDRANT_COLLECTION_NAME"
    )

    # Model Settings (Baseline V1: gpt-4o + text-embedding-3-large)
    embedding_model: str = Field(
        default="text-embedding-3-large",
        env="EMBEDDING_MODEL"
    )
    llm_model: str = Field(default="gpt-5.2", env="LLM_MODEL")
    vlm_model: str = Field(default="gpt-4o", env="VLM_MODEL")

    # Chunking Settings
    parent_chunk_size: int = Field(default=2000, env="PARENT_CHUNK_SIZE")
    child_chunk_size: int = Field(default=500, env="CHILD_CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")

    # Retrieval Settings
    top_k: int = Field(default=5)
    hybrid_alpha: float = Field(default=0.7)  # Vector search weight

    # Document/PDF Storage Settings
    pdf_base_path: str = Field(
        default="/home/aiedu/workspace/advisor/eval/allganize/pdf",
        env="PDF_BASE_PATH",
        description="PDF 문서 저장 기본 경로"
    )
    pdf_valid_domains: str = Field(
        default="finance,medical,law,commerce,public",
        env="PDF_VALID_DOMAINS",
        description="허용된 PDF 도메인 목록 (쉼표 구분)"
    )

    class Config:
        env_file = str(Path(__file__).parent.parent / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """설정 싱글톤 반환"""
    return Settings()
