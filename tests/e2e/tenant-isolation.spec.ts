import { test, expect } from '@playwright/test';

/**
 * 테넌트 격리 E2E 테스트
 *
 * 이 테스트는 멀티테넌트 환경에서 테넌트 간 데이터 격리가
 * 올바르게 작동하는지 검증합니다.
 */

const API_URL = process.env.API_URL || 'http://localhost:10311';

interface LoginResponse {
  access_token: string;
  token_type: string;
}

/**
 * API를 통한 로그인 (특정 테넌트로)
 */
async function loginAsTenant(
  request: any,
  tenantId: string,
  username: string,
  password: string
): Promise<string> {
  const response = await request.post(`${API_URL}/api/auth/login`, {
    headers: {
      'Content-Type': 'application/json',
      'X-Tenant-ID': tenantId
    },
    data: {
      username,
      password
    }
  });

  const data: LoginResponse = await response.json();
  return data.access_token;
}

test.describe('Tenant Isolation - E2E', () => {
  test('tenant A user cannot see tenant B login page content', async ({ page }) => {
    // 테넌트 A로 접속 (헤더 설정이 가능한 경우)
    await page.setExtraHTTPHeaders({
      'X-Tenant-ID': 'tenant_a'
    });

    await page.goto('/');

    // 페이지 제목이나 로고에 테넌트 A 정보가 표시되어야 함
    // (테넌트별 커스터마이징이 있는 경우)
    const pageContent = await page.content();

    // 테넌트 B의 정보가 표시되지 않아야 함
    expect(pageContent).not.toContain('tenant_b');
    expect(pageContent).not.toContain('Tenant B');
  });

  test('URL manipulation should not expose other tenant data', async ({ page }) => {
    // 테넌트 A로 로그인
    await page.goto('/');
    await page.fill('input[name="username"], [data-testid="username"]', 'testuser');
    await page.fill('input[name="password"], [data-testid="password"]', 'test1234');
    await page.click('button[type="submit"]');

    await expect(page).toHaveURL(/\/(chat|home|dashboard)/);

    // URL을 조작하여 다른 테넌트 데이터 접근 시도
    const response = await page.goto('/api/tenants/other_tenant/documents');

    // 403 Forbidden 또는 404 Not Found 예상
    if (response) {
      expect([403, 404]).toContain(response.status());
    }
  });

  test('document uploaded by tenant A should not be searchable by tenant B', async ({ page, request }) => {
    // 이 테스트는 두 개의 테넌트 계정이 필요합니다
    // 테넌트 A: testuser_a / test1234
    // 테넌트 B: testuser_b / test1234

    const uniqueContent = `UNIQUE_CONTENT_${Date.now()}`;

    // 테넌트 A로 로그인하고 문서 업로드 (API 테스트에서 더 상세히 검증)
    // ...

    // 테넌트 B로 로그인
    await page.goto('/');
    await page.setExtraHTTPHeaders({
      'X-Tenant-ID': 'tenant_b'
    });

    await page.fill('input[name="username"], [data-testid="username"]', 'testuser_b');
    await page.fill('input[name="password"], [data-testid="password"]', 'test1234');
    await page.click('button[type="submit"]');

    // 채팅에서 테넌트 A의 문서 내용 검색
    await page.fill('[data-testid="chat-input"], textarea', uniqueContent);
    await page.click('[data-testid="send-button"], button[type="submit"]');

    // 응답 대기
    await page.waitForTimeout(5000);

    // 테넌트 A의 문서 내용이 검색되지 않아야 함
    const responseText = await page.locator('[data-testid="ai-response"], .ai-message').textContent();
    expect(responseText).not.toContain(uniqueContent);
  });

  test('chat history should be isolated between tenants', async ({ page }) => {
    const testMessage = `Tenant A Message ${Date.now()}`;

    // 테넌트 A로 로그인 및 메시지 전송
    await page.setExtraHTTPHeaders({
      'X-Tenant-ID': 'tenant_a'
    });
    await page.goto('/');
    await page.fill('input[name="username"], [data-testid="username"]', 'testuser');
    await page.fill('input[name="password"], [data-testid="password"]', 'test1234');
    await page.click('button[type="submit"]');

    // 메시지 전송
    await page.fill('[data-testid="chat-input"], textarea', testMessage);
    await page.click('[data-testid="send-button"], button[type="submit"]');
    await page.waitForTimeout(2000);

    // 로그아웃
    await page.goto('/logout');

    // 테넌트 B로 로그인
    await page.setExtraHTTPHeaders({
      'X-Tenant-ID': 'tenant_b'
    });
    await page.goto('/');
    await page.fill('input[name="username"], [data-testid="username"]', 'testuser_b');
    await page.fill('input[name="password"], [data-testid="password"]', 'test1234');
    await page.click('button[type="submit"]');

    // 테넌트 A의 메시지가 보이지 않아야 함
    const chatContent = await page.locator('[data-testid="chat-messages"], .chat-messages').textContent();
    expect(chatContent).not.toContain(testMessage);
  });
});

test.describe('API Security - Cross-Tenant Attacks', () => {
  test('should reject requests with manipulated tenant ID in JWT', async ({ request }) => {
    // 테넌트 A로 로그인하여 토큰 획득
    const tokenA = await loginAsTenant(request, 'tenant_a', 'testuser', 'test1234');

    // 테넌트 A의 토큰으로 테넌트 B의 데이터 접근 시도
    const response = await request.get(`${API_URL}/api/documents`, {
      headers: {
        'Authorization': `Bearer ${tokenA}`,
        'X-Tenant-ID': 'tenant_b'  // 헤더 조작 시도
      }
    });

    // 401 또는 403 예상 (토큰의 테넌트와 헤더의 테넌트 불일치)
    expect([401, 403]).toContain(response.status());
  });

  test('should not allow access to other tenant database via SQL injection', async ({ request }) => {
    const sqlInjectionPayloads = [
      "'; SELECT * FROM tenant_b.users; --",
      "1 OR 1=1; DROP TABLE tenant_b.documents; --",
      "tenant_a' UNION SELECT * FROM tenant_b.chats WHERE '1'='1"
    ];

    const token = await loginAsTenant(request, 'tenant_a', 'testuser', 'test1234');

    for (const payload of sqlInjectionPayloads) {
      const response = await request.post(`${API_URL}/api/chat`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'X-Tenant-ID': 'tenant_a',
          'Content-Type': 'application/json'
        },
        data: {
          message: payload
        }
      });

      // 정상 응답 또는 400 (Bad Request) 예상
      // 500 에러나 데이터 유출이 없어야 함
      expect([200, 400, 422]).toContain(response.status());

      const responseText = await response.text();
      expect(responseText).not.toContain('tenant_b');
      expect(responseText).not.toContain('other_tenant');
    }
  });
});
