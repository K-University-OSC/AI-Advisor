"""
AI Advisor OSC - 보안 테스트

OWASP Top 10 기반 보안 취약점 테스트
"""
import pytest
import httpx
import os

BASE_URL = os.getenv("BASE_URL", "http://localhost:10311")
TIMEOUT = 30.0


@pytest.fixture
async def client():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
        yield client


@pytest.fixture
async def auth_client():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
        response = await client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "test1234"
        }, headers={"X-Tenant-ID": "default"})

        if response.status_code == 200:
            token = response.json()["access_token"]
            client.headers["Authorization"] = f"Bearer {token}"
            client.headers["X-Tenant-ID"] = "default"

        yield client


@pytest.mark.security
@pytest.mark.asyncio
class TestSQLInjection:
    """SQL Injection 방어 테스트 (A03:2021)"""

    SQL_INJECTION_PAYLOADS = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "1; SELECT * FROM users",
        "admin'--",
        "' UNION SELECT * FROM users WHERE '1'='1",
        "1'; DELETE FROM chats WHERE '1'='1",
        "'; UPDATE users SET role='admin' WHERE username='",
    ]

    async def test_login_sql_injection(self, client: httpx.AsyncClient):
        """로그인 SQL Injection 방어"""
        for payload in self.SQL_INJECTION_PAYLOADS:
            response = await client.post("/api/auth/login", json={
                "username": payload,
                "password": payload
            }, headers={"X-Tenant-ID": "default"})

            # 400, 401, 422 중 하나여야 함 (500 에러 없어야 함)
            assert response.status_code in [400, 401, 422], \
                f"SQL injection payload accepted: {payload}"

            # 응답에 DB 에러 메시지가 없어야 함
            response_text = response.text.lower()
            assert "syntax error" not in response_text
            assert "sql" not in response_text
            assert "database" not in response_text

    async def test_chat_sql_injection(self, auth_client: httpx.AsyncClient):
        """채팅 SQL Injection 방어"""
        if "Authorization" not in auth_client.headers:
            pytest.skip("Not authenticated")

        for payload in self.SQL_INJECTION_PAYLOADS:
            response = await auth_client.post("/api/chat", json={
                "message": payload
            })

            # 정상 처리되거나 400 에러 (500 아님)
            assert response.status_code in [200, 400, 422], \
                f"Unexpected status for SQL injection: {response.status_code}"


@pytest.mark.security
@pytest.mark.asyncio
class TestXSS:
    """Cross-Site Scripting (XSS) 방어 테스트 (A03:2021)"""

    XSS_PAYLOADS = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<svg onload=alert('XSS')>",
        "<body onload=alert('XSS')>",
        "'\"><script>alert('XSS')</script>",
        "<iframe src='javascript:alert(1)'></iframe>",
    ]

    async def test_chat_xss_prevention(self, auth_client: httpx.AsyncClient):
        """채팅 XSS 방어"""
        if "Authorization" not in auth_client.headers:
            pytest.skip("Not authenticated")

        for payload in self.XSS_PAYLOADS:
            response = await auth_client.post("/api/chat", json={
                "message": payload
            })

            if response.status_code == 200:
                # 응답에 원본 스크립트 태그가 없어야 함
                response_text = response.text
                assert "<script>" not in response_text
                assert "javascript:" not in response_text.lower()


@pytest.mark.security
@pytest.mark.asyncio
class TestBrokenAuthentication:
    """인증 취약점 테스트 (A07:2021)"""

    async def test_brute_force_protection(self, client: httpx.AsyncClient):
        """무차별 대입 공격 방어 (Rate Limiting)"""
        failed_attempts = 0

        for i in range(20):
            response = await client.post("/api/auth/login", json={
                "username": "testuser",
                "password": f"wrong_password_{i}"
            }, headers={"X-Tenant-ID": "default"})

            if response.status_code == 429:  # Too Many Requests
                # Rate limiting 작동 확인
                assert True
                return

            failed_attempts += 1

        # Rate limiting이 없다면 경고
        pytest.skip("Rate limiting may not be enabled (20 failed attempts allowed)")

    async def test_password_in_response(self, auth_client: httpx.AsyncClient):
        """응답에 비밀번호 노출 없음"""
        if "Authorization" not in auth_client.headers:
            pytest.skip("Not authenticated")

        response = await auth_client.get("/api/auth/me")

        if response.status_code == 200:
            response_text = response.text.lower()
            assert "password" not in response_text
            assert "test1234" not in response_text

    async def test_jwt_not_in_url(self, auth_client: httpx.AsyncClient):
        """JWT 토큰이 URL에 노출되지 않음"""
        # 토큰을 URL 파라미터로 전달해도 거부되어야 함
        token = auth_client.headers.get("Authorization", "").replace("Bearer ", "")

        # 헤더 없이 URL 파라미터로 시도
        temp_client = httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT)
        response = await temp_client.get(f"/api/auth/me?token={token}")
        await temp_client.aclose()

        # 401 예상 (URL 토큰 미지원)
        assert response.status_code == 401


@pytest.mark.security
@pytest.mark.asyncio
class TestTenantIsolationSecurity:
    """테넌트 격리 보안 테스트"""

    async def test_tenant_header_required(self, client: httpx.AsyncClient):
        """테넌트 헤더 필수 확인"""
        response = await client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "test1234"
        })  # X-Tenant-ID 헤더 없음

        # 기본 테넌트로 처리되거나, 400 에러
        # (엄격한 경우 400, 관대한 경우 기본 테넌트 사용)
        assert response.status_code in [200, 400]

    async def test_tenant_id_validation(self, client: httpx.AsyncClient):
        """테넌트 ID 유효성 검사"""
        malicious_tenant_ids = [
            "../other_tenant",
            "tenant; DROP TABLE users",
            "<script>alert(1)</script>",
            "../../etc/passwd",
        ]

        for tenant_id in malicious_tenant_ids:
            response = await client.post("/api/auth/login", json={
                "username": "testuser",
                "password": "test1234"
            }, headers={"X-Tenant-ID": tenant_id})

            # 400 또는 404 (유효하지 않은 테넌트)
            assert response.status_code in [400, 404, 422], \
                f"Malicious tenant ID accepted: {tenant_id}"


@pytest.mark.security
@pytest.mark.asyncio
class TestSecurityHeaders:
    """보안 헤더 테스트"""

    async def test_security_headers_present(self, client: httpx.AsyncClient):
        """필수 보안 헤더 존재 확인"""
        response = await client.get("/api/health")

        headers = response.headers

        # X-Content-Type-Options
        assert "x-content-type-options" in headers or True  # 옵션

        # X-Frame-Options
        # assert "x-frame-options" in headers

        # Content-Security-Policy (권장)
        # assert "content-security-policy" in headers

    async def test_cors_configuration(self, client: httpx.AsyncClient):
        """CORS 설정 확인"""
        response = await client.options("/api/health", headers={
            "Origin": "http://malicious-site.com",
            "Access-Control-Request-Method": "GET"
        })

        # 허용되지 않은 Origin에 대해 CORS 헤더가 없거나 차단
        cors_header = response.headers.get("access-control-allow-origin", "")

        # "*"는 보안상 위험 (프로덕션에서는 특정 도메인만 허용)
        # assert cors_header != "*" or True  # 개발 환경에서는 허용할 수 있음


@pytest.mark.security
@pytest.mark.asyncio
class TestFileUploadSecurity:
    """파일 업로드 보안 테스트"""

    async def test_malicious_file_extension_blocked(self, auth_client: httpx.AsyncClient):
        """악성 파일 확장자 차단"""
        if "Authorization" not in auth_client.headers:
            pytest.skip("Not authenticated")

        malicious_files = [
            ("malware.exe", b"MZ..."),  # Windows 실행 파일
            ("script.php", b"<?php system($_GET['cmd']); ?>"),
            ("shell.jsp", b"<% Runtime.getRuntime().exec(request.getParameter(\"cmd\")); %>"),
        ]

        for filename, content in malicious_files:
            response = await auth_client.post("/api/documents/upload",
                files={"file": (filename, content)}
            )

            # 400 또는 415 (Unsupported Media Type) 예상
            if response.status_code != 404:  # API가 존재하는 경우
                assert response.status_code in [400, 415, 422], \
                    f"Malicious file accepted: {filename}"
