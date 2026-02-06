"""
중앙 설정 관리 모듈

모든 설정을 한 곳에서 관리하여 유지보수성과 재사용성을 높입니다.
다른 서비스에서 사용 시: from config import settings
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(override=True)


# =============================================================================
# LLM 모델 설정
# =============================================================================
@dataclass
class LLMModelConfig:
    """LLM 모델 설정"""
    model_id: str           # API에서 사용하는 모델 ID
    provider: str           # openai, google, anthropic, perplexity
    display_name: str       # UI에 표시할 이름
    supports_json: bool = True  # JSON 응답 지원 여부
    max_tokens: int = 4096


# 사용 가능한 모델 정의
LLM_MODELS: Dict[str, LLMModelConfig] = {
    # OpenAI 모델
    "gpt4o": LLMModelConfig("gpt-4o", "openai", "GPT-4o"),
    "gpt4o-mini": LLMModelConfig("gpt-4o-mini", "openai", "GPT-4o Mini"),
    "gpt5": LLMModelConfig("gpt-5", "openai", "GPT-5"),
    "gpt5-mini": LLMModelConfig("gpt-5-mini", "openai", "GPT-5 Mini"),
    "o1": LLMModelConfig("o1", "openai", "O1"),
    "o1-mini": LLMModelConfig("o1-mini", "openai", "O1 Mini"),
    "o3-mini": LLMModelConfig("o3-mini", "openai", "O3 Mini"),

    # Google 모델
    "gemini-pro": LLMModelConfig("gemini-1.5-pro", "google", "Gemini 1.5 Pro"),
    "gemini-flash": LLMModelConfig("gemini-1.5-flash", "google", "Gemini 1.5 Flash"),
    "gemini-2-flash": LLMModelConfig("gemini-2.0-flash-exp", "google", "Gemini 2.0 Flash"),

    # Anthropic 모델
    "claude-sonnet": LLMModelConfig("claude-sonnet-4-20250514", "anthropic", "Claude Sonnet 4"),
    "claude-haiku": LLMModelConfig("claude-3-5-haiku-20241022", "anthropic", "Claude 3.5 Haiku"),

    # Perplexity 모델
    "sonar": LLMModelConfig("sonar", "perplexity", "Sonar"),
    "sonar-pro": LLMModelConfig("sonar-pro", "perplexity", "Sonar Pro"),
}

# 별칭 (하위 호환성)
MODEL_ALIASES: Dict[str, str] = {
    "gpt5m": "gpt5-mini",
    "gpt4m": "gpt4o-mini",
}


def get_model_config(model_key: str) -> LLMModelConfig:
    """모델 설정 조회 (별칭 지원)"""
    key = MODEL_ALIASES.get(model_key, model_key)
    if key not in LLM_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(LLM_MODELS.keys())}")
    return LLM_MODELS[key]


def get_model_id(model_key: str) -> str:
    """모델 키로 실제 API 모델 ID 조회"""
    return get_model_config(model_key).model_id


# =============================================================================
# 프로파일 서비스 설정
# =============================================================================
@dataclass
class ProfileServiceConfig:
    """프로파일 추출 서비스 설정"""
    # 프로파일 추출에 사용할 모델
    extraction_model: str = "gpt5-mini"

    # 추출 설정
    extraction_temperature: float = 0.1
    max_messages_to_analyze: int = 10

    # 신뢰도 설정
    default_confidence: float = 0.8
    confidence_boost_on_repeat: float = 0.1
    max_confidence: float = 1.0

    # 카테고리 정의
    categories: List[str] = field(default_factory=lambda: [
        "location", "food", "occupation", "hobby",
        "family", "schedule", "goal", "academic"
    ])

    # 대상 사용자 컨텍스트 (프롬프트에 반영)
    user_context: str = "대학생, 대학교 직원, 교수"


# =============================================================================
# Agent 서비스 설정
# =============================================================================
@dataclass
class AgentServiceConfig:
    """Agent 서비스 설정"""
    # Supervisor/Router 모델 (경량 모델)
    supervisor_model: str = os.getenv("AGENT_SUPERVISOR_MODEL", "gpt5-mini")

    # 기본 응답 모델
    default_response_model: str = os.getenv("AGENT_RESPONSE_MODEL", "gpt5")

    # 웹 검색 설정
    enable_web_search: bool = True
    web_search_provider: str = "tavily"

    # RAG 설정
    enable_rag: bool = True
    rag_top_k: int = 5

    # 프로파일 설정
    enable_profile_extraction: bool = True
    enable_profile_context: bool = True


# =============================================================================
# Provider 설정 (외부 서비스 추상화)
# =============================================================================
@dataclass
class ProvidersConfig:
    """
    Provider 설정

    외부 서비스(LLM, DB, 검색 등)를 추상화하여 교체 가능하게 합니다.
    환경변수로 Provider를 변경할 수 있습니다.
    """
    # LLM Provider
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")  # openai, claude, google
    llm_model: str = os.getenv("LLM_MODEL", "gpt5")

    # Embedding Provider
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    # VectorDB Provider
    vectordb_provider: str = os.getenv("VECTORDB_PROVIDER", "qdrant")  # qdrant, pinecone, chroma
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))

    # Search Provider
    search_provider: str = os.getenv("SEARCH_PROVIDER", "tavily")  # tavily, google, bing

    # Reranker Provider
    reranker_provider: str = os.getenv("RERANKER_PROVIDER", "bge")  # bge, cohere
    bge_reranker_model: str = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    # 대안 Reranker 모델: "BAAI/bge-reranker-large", "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Query Expansion 모델 설정
    query_expansion_model: str = os.getenv("QUERY_EXPANSION_MODEL", "gpt5-mini")
    # 대안: "gpt-4o-mini" (더 고품질), "gemini-flash" (빠름)


# =============================================================================
# RAG 파이프라인 설정
# =============================================================================
@dataclass
class RAGConfig:
    """RAG 파이프라인 설정"""
    # 파이프라인 모드: "2-stage", "3-stage", "vector-only", "hybrid-only"
    pipeline_mode: str = os.getenv("RAG_PIPELINE_MODE", "3-stage")

    # Reranker 설정 (하위 호환성)
    reranker_type: str = os.getenv("RERANKER_TYPE", "bge")
    bge_reranker_model: str = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    # 검색 설정
    vector_top_k: int = 20
    rerank_top_k: int = 5
    similarity_threshold: float = 0.5


# =============================================================================
# 데이터베이스 설정
# =============================================================================
@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    url: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://advisor:CHANGE_ME@localhost:5432/advisor_osc_db")
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "50"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "100"))
    pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))


# =============================================================================
# Redis 설정
# =============================================================================
@dataclass
class RedisConfig:
    """Redis 캐시 설정"""
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    pool_size: int = int(os.getenv("REDIS_POOL_SIZE", "20"))


# =============================================================================
# JWT/인증 설정
# =============================================================================
@dataclass
class AuthConfig:
    """인증 설정"""
    secret_key: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    algorithm: str = "HS256"
    expire_hours: int = 24


# =============================================================================
# 서버 설정
# =============================================================================
@dataclass
class ServerConfig:
    """서버 설정"""
    host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVER_PORT", "8600"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


# =============================================================================
# API 키 설정
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
# 통합 설정 클래스
# =============================================================================
@dataclass
class Settings:
    """전체 설정 통합"""
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api_keys: APIKeysConfig = field(default_factory=APIKeysConfig)

    # 서비스 설정
    profile: ProfileServiceConfig = field(default_factory=ProfileServiceConfig)
    agent: AgentServiceConfig = field(default_factory=AgentServiceConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    # Provider 설정
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)

    def get_model_id(self, model_key: str) -> str:
        """모델 키로 실제 API 모델 ID 조회"""
        return get_model_id(model_key)

    def get_model_config(self, model_key: str) -> LLMModelConfig:
        """모델 설정 조회"""
        return get_model_config(model_key)


# 싱글톤 설정 인스턴스
settings = Settings()


# =============================================================================
# 하위 호환성을 위한 기존 변수들
# =============================================================================
SERVER_HOST = settings.server.host
SERVER_PORT = settings.server.port
JWT_SECRET_KEY = settings.auth.secret_key
JWT_ALGORITHM = settings.auth.algorithm
JWT_EXPIRE_HOURS = settings.auth.expire_hours
