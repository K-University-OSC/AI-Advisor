# AI Advisor RAG Chatbot 인프라 구성 문서

**프로젝트**: AI Advisor RAG Chatbot (OSC - Open Source Community)
**작성일**: 2026-01-31
**버전**: 1.1

---

## 1. 개요

본 문서는 AI Advisor RAG 챗봇 시스템의 운영 환경 배포를 위한 인프라 구성을 설명합니다.

### 1.1 목표
- 개발 서버에서 운영 서버로 **최소 작업**으로 배포
- 변경 사항 발생 시 **쉽고 빠른 재배포**
- **오토스케일링** 지원으로 부하 대응
- **프로젝트 완전 격리**: 다른 프로젝트와 네트워크/포트/볼륨 완전 분리
- **멀티 테넌트 완전 격리**: Database Per Tenant 아키텍처
- **테스트 자동화**: Playwright E2E + pytest + 성능/보안 테스트

### 1.2 격리 정책

| 격리 수준 | 대상 | 방법 |
|----------|------|------|
| **프로젝트 격리** | 다른 프로젝트 | 전용 네트워크, 전용 포트 범위, 전용 볼륨 |
| **테넌트 격리** | 테넌트 간 | Database Per Tenant, 전용 Qdrant Collection, 전용 Redis Namespace |

### 1.3 지원 배포 방식

| 방식 | 용도 | 오토스케일링 | 복잡도 |
|-----|------|-------------|-------|
| Docker Compose | 개발/소규모 | 수동 | 낮음 |
| Docker Compose + Prod | 중규모 프로덕션 | 수동 | 낮음 |
| Kubernetes (K8s) | 대규모 프로덕션 | **자동 (HPA)** | 중간 |
| GitHub Actions CI/CD | 자동 배포 | - | 낮음 |

---

## 2. 격리 아키텍처

### 2.1 프로젝트 격리 (다른 프로젝트와 완전 분리)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        서버: 220.66.157.70                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                    │
│  │   LLM Chatbot 프로젝트   │    │  Advisor OSC 프로젝트    │                    │
│  │   (포트: 10700-10704)   │    │   (포트: 10310-10314)   │                    │
│  │                         │    │                         │                    │
│  │  Network:               │    │  Network:               │                    │
│  │  llm-chatbot-network    │    │  advisor-osc-network    │  ◀── 완전 분리     │
│  │                         │    │                         │                    │
│  │  Volumes:               │    │  Volumes:               │                    │
│  │  llm-chatbot-*-data     │    │  advisor-osc-*-data     │  ◀── 완전 분리     │
│  │                         │    │                         │                    │
│  └─────────────────────────┘    └─────────────────────────┘                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Advisor OSC 전용 리소스:**

| 리소스 | 이름 | 포트 | 비고 |
|--------|------|------|------|
| **네트워크** | `advisor-osc-network` | - | 프로젝트 전용 |
| **Frontend** | `advisor-osc-frontend` | 10310 | 외부 접근 허용 |
| **Backend** | `advisor-osc-backend` | 10311 | 외부 접근 허용 |
| **PostgreSQL** | `advisor-osc-postgres` | 10312 | **localhost only** |
| **Redis** | `advisor-osc-redis` | 10313 | **localhost only** |
| **Qdrant** | `advisor-osc-qdrant` | 10314 | **localhost only** |
| **Volumes** | `advisor-osc-*-data` | - | 프로젝트 전용 |

### 2.2 테넌트 격리 (Database Per Tenant)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     Advisor OSC - 멀티테넌트 아키텍처                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         API Gateway / Load Balancer                      │   │
│  │                              (Nginx / K8s Ingress)                       │   │
│  │                                                                          │   │
│  │    GET /api/tenants/hallym/chat  →  X-Tenant-ID: hallym                 │   │
│  │    GET /api/tenants/univ_a/chat  →  X-Tenant-ID: univ_a                 │   │
│  └──────────────────────────────────────┬──────────────────────────────────┘   │
│                                         │                                       │
│                                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Backend (FastAPI + Tenant Middleware)                 │   │
│  │                                                                          │   │
│  │    TenantMiddleware:                                                     │   │
│  │    1. X-Tenant-ID 헤더에서 테넌트 식별                                    │   │
│  │    2. X-API-Key 헤더로 API 인증                                          │   │
│  │    3. 테넌트별 DB/Collection/Namespace 라우팅                             │   │
│  └──────────────────────────────────────┬──────────────────────────────────┘   │
│                                         │                                       │
│           ┌─────────────────────────────┼─────────────────────────────┐        │
│           │                             │                             │        │
│           ▼                             ▼                             ▼        │
│  ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐   │
│  │   PostgreSQL    │         │     Qdrant      │         │      Redis      │   │
│  │                 │         │                 │         │                 │   │
│  │ ┌─────────────┐ │         │ ┌─────────────┐ │         │ ┌─────────────┐ │   │
│  │ │ tenant_     │ │         │ │ hallym_     │ │         │ │ hallym:*    │ │   │
│  │ │ hallym      │ │         │ │ documents   │ │         │ │             │ │   │
│  │ │ (Database)  │ │         │ │ (Collection)│ │         │ │ (Namespace) │ │   │
│  │ └─────────────┘ │         │ └─────────────┘ │         │ └─────────────┘ │   │
│  │                 │         │                 │         │                 │   │
│  │ ┌─────────────┐ │         │ ┌─────────────┐ │         │ ┌─────────────┐ │   │
│  │ │ tenant_     │ │         │ │ univ_a_     │ │         │ │ univ_a:*    │ │   │
│  │ │ univ_a      │ │         │ │ documents   │ │         │ │             │ │   │
│  │ │ (Database)  │ │         │ │ (Collection)│ │         │ │ (Namespace) │ │   │
│  │ └─────────────┘ │         │ └─────────────┘ │         │ └─────────────┘ │   │
│  └─────────────────┘         └─────────────────┘         └─────────────────┘   │
│        ▲                            ▲                            ▲             │
│        │                            │                            │             │
│        └────────────────────────────┴────────────────────────────┘             │
│                         완전 격리 (Cross-tenant 접근 불가)                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 테넌트 격리 상세

| 리소스 | 격리 방식 | 네이밍 규칙 | 예시 |
|--------|----------|-------------|------|
| **PostgreSQL** | Database Per Tenant | `tenant_{tenant_id}` | `tenant_hallym`, `tenant_univ_a` |
| **Qdrant** | Collection Per Tenant | `{tenant_id}_documents` | `hallym_documents`, `univ_a_documents` |
| **Redis** | Key Prefix (Namespace) | `{tenant_id}:{key}` | `hallym:session:xxx`, `univ_a:cache:xxx` |
| **파일 저장소** | Directory Per Tenant | `uploads/{tenant_id}/` | `uploads/hallym/`, `uploads/univ_a/` |

---

## 3. 시스템 아키텍처

### 3.1 전체 구성도

```
                                    ┌─────────────────────────────────────┐
                                    │            Load Balancer            │
                                    │              (Nginx)                │
                                    │         - Rate Limiting             │
                                    │         - SSL Termination           │
                                    │         - Tenant Routing            │
                                    └──────────────┬──────────────────────┘
                                                   │
                         ┌─────────────────────────┼─────────────────────────┐
                         │                         │                         │
                         ▼                         ▼                         ▼
              ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
              │    Frontend      │     │    Backend #1    │     │    Backend #N    │
              │    (React)       │     │    (FastAPI)     │     │    (FastAPI)     │
              │                  │     │                  │     │                  │
              │  - Static Files  │     │  - Tenant Router │     │  - Tenant Router │
              │  - SPA Routing   │     │  - RAG Pipeline  │     │  - RAG Pipeline  │
              └──────────────────┘     │  - LLM Gateway   │     │  - LLM Gateway   │
                                       └────────┬─────────┘     └────────┬─────────┘
                                                │                        │
                         ┌──────────────────────┴────────────────────────┴───┐
                         │                   Tenant Router                   │
                         │                                                   │
              ┌──────────┴──────────┐  ┌──────────────────┐  ┌──────────────┴───────┐
              │     PostgreSQL      │  │      Qdrant      │  │        Redis         │
              │   (Central DB)      │  │                  │  │                      │
              │                     │  │  - Per-Tenant    │  │  - Per-Tenant        │
              │  - tenants table    │  │    Collections   │  │    Namespaces        │
              │  ┌───────────────┐  │  │                  │  │                      │
              │  │ tenant_hallym │  │  │  - hallym_docs   │  │  - hallym:*          │
              │  │ tenant_univ_a │  │  │  - univ_a_docs   │  │  - univ_a:*          │
              │  └───────────────┘  │  │                  │  │                      │
              └─────────────────────┘  └──────────────────┘  └──────────────────────┘
```

### 3.2 서비스 구성

| 서비스 | 이미지 | 개발 포트 | K8s 포트 | 역할 |
|-------|-------|----------|----------|-----|
| nginx | nginx:alpine | - | 80, 443 | 로드밸런서, 테넌트 라우팅 |
| frontend | advisor-osc-frontend | 10310 | 80 | React SPA |
| backend | advisor-osc-backend | 10311 | 8000 | FastAPI, Tenant Middleware |
| postgres | postgres:15-alpine | 10312 | 5432 | Central + Per-Tenant DBs |
| qdrant | qdrant/qdrant | 10314 | 6333 | Per-Tenant Collections |
| redis | redis:7-alpine | 10313 | 6379 | Per-Tenant Namespaces |

---

## 4. 파일 구조

```
advisor_osc/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── server.py
│   ├── core/
│   │   └── middleware/
│   │       └── tenant.py          # 테넌트 미들웨어 (격리 핵심)
│   ├── database/
│   │   └── tenant_manager.py      # 테넌트 DB 관리
│   ├── rag/                       # RAG 파이프라인
│   ├── routers/                   # API 라우터
│   └── tests/                     # pytest 테스트
│
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
│
├── tests/                         # E2E & Integration Tests
│   ├── e2e/
│   │   ├── playwright.config.ts   # Playwright 설정
│   │   ├── auth.spec.ts           # 인증 테스트
│   │   ├── chat.spec.ts           # 채팅 테스트
│   │   ├── document.spec.ts       # 문서 업로드 테스트
│   │   └── tenant-isolation.spec.ts  # 테넌트 격리 테스트
│   ├── api/
│   │   ├── test_health.py         # API 헬스체크
│   │   ├── test_auth.py           # 인증 API
│   │   ├── test_rag.py            # RAG API
│   │   └── test_tenant_isolation.py  # 테넌트 격리 검증
│   ├── performance/
│   │   ├── locustfile.py          # 부하 테스트
│   │   └── k6_script.js           # k6 성능 테스트
│   └── security/
│       └── security_scan.py       # OWASP 보안 스캔
│
├── k8s/                           # Kubernetes 매니페스트
│   ├── namespace.yaml
│   ├── network-policy.yaml        # 네트워크 격리 정책
│   ├── backend-deployment.yaml
│   ├── backend-hpa.yaml           # 오토스케일링
│   └── kustomization.yaml
│
├── scripts/
│   ├── deploy.sh                  # 배포 스크립트
│   ├── onboard-tenant.sh          # 테넌트 온보딩
│   └── run-tests.sh               # 테스트 실행
│
├── docker-compose.yml
├── docker-compose.prod.yml
├── docker-compose.test.yml        # 테스트 환경
│
└── docs/
    ├── INFRASTRUCTURE.md          # 본 문서
    └── TESTING.md                 # 테스트 가이드
```

---

## 5. 테스트 전략 (Playwright + 모든 방법)

### 5.1 테스트 피라미드

```
                    ┌─────────────────┐
                    │    E2E Tests    │  ◀── Playwright
                    │   (10% 커버리지) │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ Integration     │  ◀── pytest + httpx
                    │   (30% 커버리지) │
                    └────────┬────────┘
                             │
           ┌─────────────────┴─────────────────┐
           │         Unit Tests                │  ◀── pytest
           │         (60% 커버리지)             │
           └───────────────────────────────────┘
```

### 5.2 E2E 테스트 (Playwright)

```typescript
// tests/e2e/playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30000,
  retries: 2,
  workers: 4,
  reporter: [
    ['html', { outputFolder: 'test-results/html' }],
    ['json', { outputFile: 'test-results/results.json' }]
  ],
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:10310',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure'
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'mobile', use: { ...devices['iPhone 13'] } }
  ]
});
```

```typescript
// tests/e2e/auth.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test('should login successfully', async ({ page }) => {
    await page.goto('/');
    await page.fill('[data-testid="username"]', 'testuser');
    await page.fill('[data-testid="password"]', 'test1234');
    await page.click('[data-testid="login-button"]');

    await expect(page).toHaveURL('/chat');
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/');
    await page.fill('[data-testid="username"]', 'wrong');
    await page.fill('[data-testid="password"]', 'wrong');
    await page.click('[data-testid="login-button"]');

    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
  });
});

// tests/e2e/tenant-isolation.spec.ts
test.describe('Tenant Isolation', () => {
  test('tenant A cannot access tenant B data', async ({ page }) => {
    // Login as tenant A user
    await loginAsTenant(page, 'tenant_a');

    // Try to access tenant B's data via URL manipulation
    await page.goto('/api/tenants/tenant_b/chat');

    // Should be denied
    await expect(page.locator('body')).toContainText('Forbidden');
  });

  test('tenant A documents not visible to tenant B', async ({ page }) => {
    // Upload document as tenant A
    await loginAsTenant(page, 'tenant_a');
    await uploadDocument(page, 'tenant_a_doc.pdf');

    // Login as tenant B
    await logout(page);
    await loginAsTenant(page, 'tenant_b');

    // Search for tenant A's document
    await page.fill('[data-testid="chat-input"]', 'tenant_a_doc content');
    await page.click('[data-testid="send-button"]');

    // Should not find tenant A's document
    await expect(page.locator('[data-testid="response"]')).not.toContainText('tenant_a_doc');
  });
});
```

### 5.3 API 테스트 (pytest + httpx)

```python
# tests/api/test_tenant_isolation.py
import pytest
import httpx

BASE_URL = "http://localhost:10311"

@pytest.fixture
async def tenant_a_client():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # Login as tenant A
        response = await client.post("/api/auth/login", json={
            "username": "user_a", "password": "pass"
        }, headers={"X-Tenant-ID": "tenant_a"})
        token = response.json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        client.headers["X-Tenant-ID"] = "tenant_a"
        yield client

@pytest.fixture
async def tenant_b_client():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        response = await client.post("/api/auth/login", json={
            "username": "user_b", "password": "pass"
        }, headers={"X-Tenant-ID": "tenant_b"})
        token = response.json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        client.headers["X-Tenant-ID"] = "tenant_b"
        yield client

class TestTenantIsolation:
    """테넌트 격리 검증 테스트"""

    async def test_tenant_a_cannot_access_tenant_b_chat(
        self, tenant_a_client, tenant_b_client
    ):
        """테넌트 A가 테넌트 B의 채팅에 접근 불가"""
        # Tenant A creates a chat
        response = await tenant_a_client.post("/api/chat", json={
            "message": "Hello from tenant A"
        })
        chat_id = response.json()["id"]

        # Tenant B tries to access tenant A's chat
        response = await tenant_b_client.get(f"/api/chat/{chat_id}")
        assert response.status_code == 404  # Not found (isolation working)

    async def test_tenant_documents_isolated(
        self, tenant_a_client, tenant_b_client
    ):
        """테넌트 문서 격리 검증"""
        # Tenant A uploads a document
        with open("test_doc.pdf", "rb") as f:
            response = await tenant_a_client.post(
                "/api/documents/upload",
                files={"file": f}
            )
        assert response.status_code == 200

        # Tenant B should not see tenant A's documents
        response = await tenant_b_client.get("/api/documents")
        documents = response.json()
        assert len([d for d in documents if "tenant_a" in d["name"]]) == 0

    async def test_rag_search_isolated(
        self, tenant_a_client, tenant_b_client
    ):
        """RAG 검색이 테넌트별로 격리됨"""
        # Tenant A adds document with unique content
        unique_content = "UNIQUE_TENANT_A_CONTENT_12345"

        # Tenant B searches for tenant A's unique content
        response = await tenant_b_client.post("/api/chat", json={
            "message": unique_content
        })

        # Should not find tenant A's content
        assert unique_content not in response.json()["response"]
```

### 5.4 성능 테스트 (Locust + k6)

```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between

class ChatUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        # Login
        response = self.client.post("/api/auth/login", json={
            "username": "loadtest_user",
            "password": "test1234"
        }, headers={"X-Tenant-ID": "loadtest"})
        self.token = response.json()["access_token"]
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Tenant-ID": "loadtest"
        }

    @task(10)
    def chat(self):
        self.client.post("/api/chat",
            json={"message": "안녕하세요"},
            headers=self.headers
        )

    @task(3)
    def get_history(self):
        self.client.get("/api/chat/history", headers=self.headers)

    @task(1)
    def search_documents(self):
        self.client.get("/api/documents?q=test", headers=self.headers)
```

```javascript
// tests/performance/k6_script.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '1m', target: 10 },   // Ramp up
    { duration: '3m', target: 50 },   // Stay at 50 users
    { duration: '1m', target: 100 },  // Peak
    { duration: '1m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'],  // 95% requests under 2s
    http_req_failed: ['rate<0.01'],     // Less than 1% failures
  },
};

export default function () {
  const loginRes = http.post(`${__ENV.BASE_URL}/api/auth/login`,
    JSON.stringify({ username: 'testuser', password: 'test1234' }),
    { headers: { 'Content-Type': 'application/json', 'X-Tenant-ID': 'test' } }
  );

  check(loginRes, {
    'login successful': (r) => r.status === 200,
  });

  const token = loginRes.json('access_token');

  const chatRes = http.post(`${__ENV.BASE_URL}/api/chat`,
    JSON.stringify({ message: '테스트 메시지' }),
    {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
        'X-Tenant-ID': 'test'
      }
    }
  );

  check(chatRes, {
    'chat response ok': (r) => r.status === 200,
    'response time ok': (r) => r.timings.duration < 2000,
  });

  sleep(1);
}
```

### 5.5 보안 테스트

```python
# tests/security/security_scan.py
import pytest
import httpx

class TestSecurityScan:
    """OWASP Top 10 보안 테스트"""

    async def test_sql_injection(self, client):
        """SQL Injection 방어 테스트"""
        payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "1; SELECT * FROM users",
        ]
        for payload in payloads:
            response = await client.post("/api/auth/login", json={
                "username": payload,
                "password": payload
            })
            assert response.status_code in [400, 401, 422]

    async def test_xss_prevention(self, client):
        """XSS 방어 테스트"""
        payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
        ]
        for payload in payloads:
            response = await client.post("/api/chat", json={
                "message": payload
            })
            assert payload not in response.text

    async def test_rate_limiting(self, client):
        """Rate Limiting 테스트"""
        # 빠르게 100번 요청
        responses = []
        for _ in range(100):
            response = await client.get("/api/health")
            responses.append(response.status_code)

        # 429 Too Many Requests가 포함되어야 함
        assert 429 in responses

    async def test_tenant_isolation_attack(self, client):
        """테넌트 격리 공격 테스트"""
        # 다른 테넌트 ID로 접근 시도
        response = await client.get("/api/documents", headers={
            "X-Tenant-ID": "other_tenant",
            "Authorization": f"Bearer {self.token}"  # tenant_a token
        })
        assert response.status_code in [401, 403]
```

### 5.6 테스트 실행 스크립트

```bash
#!/bin/bash
# scripts/run-tests.sh

set -e

echo "=== Running All Tests ==="

# 1. Unit Tests
echo ">>> Unit Tests (pytest)"
cd backend
python -m pytest tests/unit -v --cov=. --cov-report=html
cd ..

# 2. Integration Tests
echo ">>> Integration Tests (pytest)"
python -m pytest tests/api -v

# 3. E2E Tests (Playwright)
echo ">>> E2E Tests (Playwright)"
cd tests/e2e
npx playwright install --with-deps
npx playwright test --reporter=html
cd ../..

# 4. Performance Tests (Locust)
echo ">>> Performance Tests (Locust)"
locust -f tests/performance/locustfile.py \
  --headless -u 50 -r 10 --run-time 1m \
  --host http://localhost:10311

# 5. Security Tests
echo ">>> Security Tests"
python -m pytest tests/security -v

# 6. Tenant Isolation Tests (특별 중요)
echo ">>> Tenant Isolation Tests"
python -m pytest tests/api/test_tenant_isolation.py -v

echo "=== All Tests Completed ==="
```

### 5.7 CI/CD 테스트 파이프라인

```yaml
# .github/workflows/test.yml
name: Full Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: |
          pip install -r backend/requirements.txt
          pip install pytest pytest-cov pytest-asyncio
          cd backend && pytest tests/unit -v --cov

  e2e-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
      - name: Start services
        run: docker compose -f docker-compose.test.yml up -d
      - name: Install Playwright
        run: |
          cd tests/e2e
          npm ci
          npx playwright install --with-deps
      - name: Run E2E Tests
        run: npx playwright test
      - uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: playwright-report
          path: tests/e2e/test-results/

  tenant-isolation-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Start multi-tenant environment
        run: docker compose -f docker-compose.test.yml up -d
      - name: Run Tenant Isolation Tests
        run: |
          pip install pytest httpx pytest-asyncio
          pytest tests/api/test_tenant_isolation.py -v
```

---

## 6. Docker 구성

### 6.1 docker-compose.yml (프로젝트 격리)

```yaml
# 프로젝트 완전 격리 구성
services:
  frontend:
    container_name: advisor-osc-frontend
    ports:
      - "10310:80"
    networks:
      - advisor-osc-network

  backend:
    container_name: advisor-osc-backend
    ports:
      - "10311:8000"
    networks:
      - advisor-osc-network

  postgres:
    container_name: advisor-osc-postgres
    ports:
      - "127.0.0.1:10312:5432"  # localhost only
    volumes:
      - advisor-osc-postgres-data:/var/lib/postgresql/data
    networks:
      - advisor-osc-network

  redis:
    container_name: advisor-osc-redis
    ports:
      - "127.0.0.1:10313:6379"  # localhost only
    networks:
      - advisor-osc-network

  qdrant:
    container_name: advisor-osc-qdrant
    ports:
      - "127.0.0.1:10314:6333"  # localhost only
    networks:
      - advisor-osc-network

networks:
  advisor-osc-network:
    name: advisor-osc-network
    driver: bridge

volumes:
  advisor-osc-postgres-data:
    name: advisor-osc-postgres-data
```

### 6.2 docker-compose.test.yml

```yaml
# 테스트 환경용
version: '3.8'
services:
  backend:
    environment:
      - TESTING=true
      - DATABASE_URL=postgresql+asyncpg://test:test@postgres:5432/test_db

  playwright:
    image: mcr.microsoft.com/playwright:v1.40.0-focal
    volumes:
      - ./tests/e2e:/tests
    working_dir: /tests
    command: npx playwright test
    depends_on:
      - frontend
      - backend
    networks:
      - advisor-osc-network
```

---

## 7. Kubernetes 구성

### 7.1 Network Policy (프로젝트 + 테넌트 격리)

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: advisor-osc-isolation
  namespace: advisor-osc
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              project: advisor-osc
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              project: advisor-osc
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
```

### 7.2 HPA (오토스케일링)

```yaml
# k8s/backend-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: advisor-backend-hpa
  namespace: advisor-osc
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

## 8. 결론

### 8.1 구현 완료 항목

| 항목 | 상태 | 파일 |
|-----|------|-----|
| 프로젝트 격리 (네트워크/포트/볼륨) | ✅ | `docker-compose.yml` |
| 테넌트 격리 (Database Per Tenant) | ✅ | `core/middleware/tenant.py` |
| 테넌트 격리 (Qdrant Collection) | ✅ | `rag/retriever.py` |
| 테넌트 격리 (Redis Namespace) | ✅ | `services/cache.py` |
| Backend Dockerfile | ✅ | `backend/Dockerfile` |
| Frontend Dockerfile | ✅ | `frontend/Dockerfile` |
| Docker Compose (개발) | ✅ | `docker-compose.yml` |
| Docker Compose (프로덕션) | 🔄 | `docker-compose.prod.yml` |
| Kubernetes 매니페스트 | 🔄 | `k8s/*.yaml` |
| Network Policy | 🔄 | `k8s/network-policy.yaml` |
| 오토스케일링 (HPA) | 🔄 | `k8s/backend-hpa.yaml` |
| **E2E Tests (Playwright)** | 🔄 | `tests/e2e/*.spec.ts` |
| **API Tests (pytest)** | 🔄 | `tests/api/*.py` |
| **Performance Tests** | 🔄 | `tests/performance/*` |
| **Security Tests** | 🔄 | `tests/security/*` |
| **Tenant Isolation Tests** | 🔄 | `tests/*/tenant_isolation*` |
| CI/CD 파이프라인 | 🔄 | `.github/workflows/*.yml` |

### 8.2 테스트 체크리스트

- [ ] Unit Tests 통과 (pytest)
- [ ] API Integration Tests 통과 (pytest + httpx)
- [ ] E2E Tests 통과 (Playwright - Chrome, Firefox, Safari, Mobile)
- [ ] Tenant Isolation Tests 통과 (Cross-tenant 접근 차단 확인)
- [ ] Performance Tests 통과 (p95 < 2s, 에러율 < 1%)
- [ ] Security Tests 통과 (SQL Injection, XSS, Rate Limiting)
- [ ] Load Tests 통과 (Locust/k6 - 100 concurrent users)

### 8.3 운영 서버 이전 체크리스트

- [ ] 모든 테스트 통과
- [ ] `.env.example` → `.env` 복사 및 실제 값 설정
- [ ] 포트 범위 (10310-10314) 방화벽 확인
- [ ] DB/Cache 포트 localhost only 확인
- [ ] SSL 인증서 설정 (Let's Encrypt)
- [ ] 테넌트 온보딩 테스트
- [ ] 부하 테스트 통과

---

*Generated on 2026-01-31*
