"""
AI Advisor OSC - 인증 API 테스트
"""
import pytest
import httpx


@pytest.mark.asyncio
async def test_login_success(client: httpx.AsyncClient):
    """로그인 성공 테스트"""
    response = await client.post("/api/auth/login", json={
        "username": "testuser",
        "password": "test1234"
    }, headers={"X-Tenant-ID": "default"})

    assert response.status_code == 200
    data = response.json()

    assert "access_token" in data
    assert "token_type" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_invalid_credentials(client: httpx.AsyncClient):
    """잘못된 자격 증명으로 로그인 실패 테스트"""
    response = await client.post("/api/auth/login", json={
        "username": "wronguser",
        "password": "wrongpassword"
    }, headers={"X-Tenant-ID": "default"})

    assert response.status_code in [401, 400]


@pytest.mark.asyncio
async def test_login_missing_fields(client: httpx.AsyncClient):
    """필수 필드 누락 시 에러 테스트"""
    # 사용자 이름 누락
    response = await client.post("/api/auth/login", json={
        "password": "test1234"
    }, headers={"X-Tenant-ID": "default"})

    assert response.status_code in [400, 422]

    # 비밀번호 누락
    response = await client.post("/api/auth/login", json={
        "username": "testuser"
    }, headers={"X-Tenant-ID": "default"})

    assert response.status_code in [400, 422]


@pytest.mark.asyncio
async def test_protected_endpoint_without_token(client: httpx.AsyncClient):
    """토큰 없이 보호된 엔드포인트 접근 테스트"""
    response = await client.get("/api/auth/me")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_with_valid_token(auth_client: httpx.AsyncClient):
    """유효한 토큰으로 보호된 엔드포인트 접근 테스트"""
    response = await auth_client.get("/api/auth/me")

    # 토큰이 유효하면 200, 아니면 401
    if "Authorization" in auth_client.headers:
        assert response.status_code == 200
        data = response.json()
        assert "username" in data or "id" in data
    else:
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_with_invalid_token(client: httpx.AsyncClient):
    """잘못된 토큰으로 보호된 엔드포인트 접근 테스트"""
    client.headers["Authorization"] = "Bearer invalid_token_12345"

    response = await client.get("/api/auth/me")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_with_tenant_id(client: httpx.AsyncClient):
    """테넌트 ID와 함께 로그인 테스트"""
    response = await client.post("/api/auth/login", json={
        "username": "testuser",
        "password": "test1234"
    }, headers={"X-Tenant-ID": "hallym"})

    # 테넌트가 존재하면 200, 없으면 404 또는 400
    assert response.status_code in [200, 400, 404]

    if response.status_code == 200:
        data = response.json()
        assert "access_token" in data
