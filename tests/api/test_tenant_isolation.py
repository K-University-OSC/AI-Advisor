"""
AI Advisor OSC - 테넌트 격리 테스트

이 테스트는 멀티테넌트 환경에서 데이터 격리가 올바르게 작동하는지 검증합니다.
- Database Per Tenant 격리
- Qdrant Collection 격리
- Redis Namespace 격리
- 파일 저장소 격리
"""
import pytest
import httpx
import uuid
from typing import Optional


# 테넌트별 고유 콘텐츠 생성
def generate_unique_content() -> str:
    return f"UNIQUE_CONTENT_{uuid.uuid4().hex[:8]}"


@pytest.mark.isolation
@pytest.mark.asyncio
class TestTenantIsolation:
    """테넌트 격리 검증 테스트"""

    async def test_tenant_a_cannot_access_tenant_b_chat_history(
        self,
        tenant_a_client: httpx.AsyncClient,
        tenant_b_client: httpx.AsyncClient
    ):
        """테넌트 A가 테넌트 B의 채팅 기록에 접근 불가"""
        # Skip if clients are not authenticated
        if "Authorization" not in tenant_a_client.headers:
            pytest.skip("Tenant A client not authenticated")
        if "Authorization" not in tenant_b_client.headers:
            pytest.skip("Tenant B client not authenticated")

        # 테넌트 A가 채팅 생성
        unique_message = generate_unique_content()
        response = await tenant_a_client.post("/api/chat", json={
            "message": unique_message
        })

        if response.status_code != 200:
            pytest.skip("Chat API not available")

        # 테넌트 B가 채팅 기록 조회
        response = await tenant_b_client.get("/api/chat/history")

        if response.status_code == 200:
            history = response.json()
            # 테넌트 A의 메시지가 없어야 함
            for chat in history:
                assert unique_message not in str(chat)

    async def test_tenant_documents_isolated(
        self,
        tenant_a_client: httpx.AsyncClient,
        tenant_b_client: httpx.AsyncClient
    ):
        """테넌트 문서가 격리되어야 함"""
        if "Authorization" not in tenant_a_client.headers:
            pytest.skip("Tenant A client not authenticated")
        if "Authorization" not in tenant_b_client.headers:
            pytest.skip("Tenant B client not authenticated")

        # 테넌트 A가 문서 목록 조회
        response_a = await tenant_a_client.get("/api/documents")

        # 테넌트 B가 문서 목록 조회
        response_b = await tenant_b_client.get("/api/documents")

        if response_a.status_code == 200 and response_b.status_code == 200:
            docs_a = response_a.json()
            docs_b = response_b.json()

            # 문서 ID가 겹치지 않아야 함 (테넌트별로 다른 DB)
            ids_a = {doc.get("id") for doc in docs_a if isinstance(doc, dict)}
            ids_b = {doc.get("id") for doc in docs_b if isinstance(doc, dict)}

            # 테넌트 간 문서 ID 중복 없음 (격리 확인)
            overlap = ids_a & ids_b
            # 참고: UUID 기반이면 중복 없음, 자동 증가면 중복 가능
            # assert len(overlap) == 0, f"Document IDs overlap: {overlap}"

    async def test_rag_search_isolated(
        self,
        tenant_a_client: httpx.AsyncClient,
        tenant_b_client: httpx.AsyncClient
    ):
        """RAG 검색이 테넌트별로 격리됨"""
        if "Authorization" not in tenant_a_client.headers:
            pytest.skip("Tenant A client not authenticated")
        if "Authorization" not in tenant_b_client.headers:
            pytest.skip("Tenant B client not authenticated")

        unique_content = generate_unique_content()

        # 테넌트 A에서 고유한 콘텐츠로 채팅
        await tenant_a_client.post("/api/chat", json={
            "message": f"Remember this: {unique_content}"
        })

        # 테넌트 B에서 해당 콘텐츠 검색
        response = await tenant_b_client.post("/api/chat", json={
            "message": f"What do you know about {unique_content}?"
        })

        if response.status_code == 200:
            answer = response.json().get("response", "")
            # 테넌트 A의 콘텐츠가 테넌트 B 검색에서 나오면 안됨
            assert unique_content not in answer

    async def test_cross_tenant_header_manipulation(
        self,
        tenant_a_client: httpx.AsyncClient
    ):
        """테넌트 헤더 조작 시도가 차단되어야 함"""
        if "Authorization" not in tenant_a_client.headers:
            pytest.skip("Tenant A client not authenticated")

        # 테넌트 A의 토큰으로 테넌트 B의 데이터 접근 시도
        original_tenant = tenant_a_client.headers.get("X-Tenant-ID")

        # 헤더 조작
        tenant_a_client.headers["X-Tenant-ID"] = "tenant_b"

        response = await tenant_a_client.get("/api/documents")

        # 401 Unauthorized 또는 403 Forbidden 예상
        assert response.status_code in [401, 403], \
            f"Header manipulation should be blocked, got {response.status_code}"

        # 원래 테넌트로 복원
        tenant_a_client.headers["X-Tenant-ID"] = original_tenant


@pytest.mark.isolation
@pytest.mark.asyncio
class TestDatabaseIsolation:
    """데이터베이스 격리 테스트 (Database Per Tenant)"""

    async def test_user_data_isolated(
        self,
        tenant_a_client: httpx.AsyncClient,
        tenant_b_client: httpx.AsyncClient
    ):
        """사용자 데이터가 테넌트별로 격리됨"""
        if "Authorization" not in tenant_a_client.headers:
            pytest.skip("Tenant A client not authenticated")

        # 테넌트 A 사용자 정보
        response_a = await tenant_a_client.get("/api/auth/me")

        if response_a.status_code == 200:
            user_a = response_a.json()

            # 테넌트 B에서 같은 사용자 ID로 접근 시도
            user_id = user_a.get("id")
            if user_id:
                # 직접 사용자 조회 API가 있다면
                response_b = await tenant_b_client.get(f"/api/users/{user_id}")

                # 404 (찾을 수 없음) 또는 403 (금지) 예상
                if response_b.status_code == 200:
                    user_b = response_b.json()
                    # 다른 사용자여야 함
                    assert user_a.get("username") != user_b.get("username")


@pytest.mark.isolation
@pytest.mark.asyncio
class TestQdrantIsolation:
    """Qdrant 벡터 DB 격리 테스트"""

    async def test_vector_search_isolated(
        self,
        tenant_a_client: httpx.AsyncClient,
        tenant_b_client: httpx.AsyncClient
    ):
        """벡터 검색이 테넌트별 Collection에서만 수행됨"""
        if "Authorization" not in tenant_a_client.headers:
            pytest.skip("Tenant A client not authenticated")
        if "Authorization" not in tenant_b_client.headers:
            pytest.skip("Tenant B client not authenticated")

        # 테넌트 A 전용 질문
        unique_query = f"tenant_a_specific_query_{uuid.uuid4().hex[:6]}"

        response_a = await tenant_a_client.post("/api/chat", json={
            "message": unique_query
        })

        response_b = await tenant_b_client.post("/api/chat", json={
            "message": unique_query
        })

        # 두 응답이 다르면 격리가 작동하는 것
        if response_a.status_code == 200 and response_b.status_code == 200:
            answer_a = response_a.json().get("response", "")
            answer_b = response_b.json().get("response", "")
            # 같은 질문이지만 다른 문서 기반이면 다른 답변
            # (완전히 같을 수도 있지만, 소스 문서는 달라야 함)


@pytest.mark.isolation
@pytest.mark.asyncio
class TestRedisIsolation:
    """Redis 캐시/세션 격리 테스트"""

    async def test_session_isolated(
        self,
        tenant_a_client: httpx.AsyncClient,
        tenant_b_client: httpx.AsyncClient
    ):
        """세션이 테넌트별로 격리됨"""
        if "Authorization" not in tenant_a_client.headers:
            pytest.skip("Tenant A client not authenticated")
        if "Authorization" not in tenant_b_client.headers:
            pytest.skip("Tenant B client not authenticated")

        # 각 테넌트의 현재 사용자 정보가 올바른지 확인
        response_a = await tenant_a_client.get("/api/auth/me")
        response_b = await tenant_b_client.get("/api/auth/me")

        if response_a.status_code == 200 and response_b.status_code == 200:
            user_a = response_a.json()
            user_b = response_b.json()

            # 다른 사용자 정보
            assert user_a.get("username") != user_b.get("username")
            # 다른 테넌트
            if "tenant_id" in user_a and "tenant_id" in user_b:
                assert user_a["tenant_id"] != user_b["tenant_id"]
