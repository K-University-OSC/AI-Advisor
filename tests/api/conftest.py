"""
AI Advisor OSC - API 테스트 설정 (pytest fixtures)
"""
import os
import pytest
import httpx
from typing import AsyncGenerator

# 테스트 환경 설정
BASE_URL = os.getenv("BASE_URL", "http://localhost:10311")
TEST_TIMEOUT = 30.0

# 테스트 사용자 정보
TEST_USERS = {
    "tenant_a": {
        "username": "testuser_a",
        "password": "test1234",
        "tenant_id": "tenant_a"
    },
    "tenant_b": {
        "username": "testuser_b",
        "password": "test1234",
        "tenant_id": "tenant_b"
    },
    "default": {
        "username": "testuser",
        "password": "test1234",
        "tenant_id": "default"
    }
}


@pytest.fixture
def base_url() -> str:
    """API Base URL"""
    return BASE_URL


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """기본 HTTP 클라이언트 (인증 없음)"""
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=TEST_TIMEOUT
    ) as client:
        yield client


@pytest.fixture
async def auth_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """인증된 HTTP 클라이언트 (기본 테넌트)"""
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=TEST_TIMEOUT
    ) as client:
        # 로그인
        user = TEST_USERS["default"]
        response = await client.post("/api/auth/login", json={
            "username": user["username"],
            "password": user["password"]
        }, headers={"X-Tenant-ID": user["tenant_id"]})

        if response.status_code == 200:
            token = response.json()["access_token"]
            client.headers["Authorization"] = f"Bearer {token}"
            client.headers["X-Tenant-ID"] = user["tenant_id"]

        yield client


@pytest.fixture
async def tenant_a_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """테넌트 A 인증 클라이언트"""
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=TEST_TIMEOUT
    ) as client:
        user = TEST_USERS["tenant_a"]
        response = await client.post("/api/auth/login", json={
            "username": user["username"],
            "password": user["password"]
        }, headers={"X-Tenant-ID": user["tenant_id"]})

        if response.status_code == 200:
            token = response.json()["access_token"]
            client.headers["Authorization"] = f"Bearer {token}"
            client.headers["X-Tenant-ID"] = user["tenant_id"]

        yield client


@pytest.fixture
async def tenant_b_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """테넌트 B 인증 클라이언트"""
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=TEST_TIMEOUT
    ) as client:
        user = TEST_USERS["tenant_b"]
        response = await client.post("/api/auth/login", json={
            "username": user["username"],
            "password": user["password"]
        }, headers={"X-Tenant-ID": user["tenant_id"]})

        if response.status_code == 200:
            token = response.json()["access_token"]
            client.headers["Authorization"] = f"Bearer {token}"
            client.headers["X-Tenant-ID"] = user["tenant_id"]

        yield client


def pytest_configure(config):
    """pytest 설정"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "isolation: marks tests as tenant isolation tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
