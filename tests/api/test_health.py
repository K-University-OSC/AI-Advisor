"""
AI Advisor OSC - 헬스체크 API 테스트
"""
import pytest
import httpx


@pytest.mark.asyncio
async def test_health_endpoint(client: httpx.AsyncClient):
    """헬스체크 엔드포인트 테스트"""
    response = await client.get("/api/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_root_endpoint(client: httpx.AsyncClient):
    """루트 엔드포인트 테스트"""
    response = await client.get("/")

    # 200 또는 리다이렉트(301, 302, 307, 308) 허용
    assert response.status_code in [200, 301, 302, 307, 308]


@pytest.mark.asyncio
async def test_api_docs_available(client: httpx.AsyncClient):
    """API 문서 (Swagger) 접근 테스트"""
    response = await client.get("/docs")

    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "redoc" in response.text.lower() or "openapi" in response.text.lower()


@pytest.mark.asyncio
async def test_openapi_schema(client: httpx.AsyncClient):
    """OpenAPI 스키마 접근 테스트"""
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    data = response.json()

    assert "openapi" in data
    assert "info" in data
    assert "paths" in data
