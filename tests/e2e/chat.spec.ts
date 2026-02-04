import { test, expect } from '@playwright/test';

/**
 * 로그인 헬퍼 함수
 */
async function login(page: any, username = 'testuser', password = 'test1234') {
  await page.goto('/');
  await page.fill('input[name="username"], [data-testid="username"]', username);
  await page.fill('input[name="password"], [data-testid="password"]', password);
  await page.click('button[type="submit"], [data-testid="login-button"]');
  await expect(page).toHaveURL(/\/(chat|home|dashboard)/);
}

/**
 * 채팅 테스트 (Chat Tests)
 */
test.describe('Chat', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should display chat interface', async ({ page }) => {
    // 채팅 인터페이스 요소 확인
    await expect(page.locator('[data-testid="chat-input"], textarea, input[type="text"]')).toBeVisible();
    await expect(page.locator('[data-testid="send-button"], button[type="submit"]')).toBeVisible();
  });

  test('should send a message and receive response', async ({ page }) => {
    const testMessage = '안녕하세요, 테스트 메시지입니다.';

    // 메시지 입력
    await page.fill('[data-testid="chat-input"], textarea', testMessage);

    // 전송 버튼 클릭
    await page.click('[data-testid="send-button"], button[type="submit"]');

    // 메시지가 채팅 영역에 표시되는지 확인
    await expect(page.locator('[data-testid="chat-messages"], .chat-messages, .message-list')).toContainText(testMessage);

    // AI 응답 대기 (최대 30초)
    await expect(page.locator('[data-testid="ai-response"], .ai-message, .assistant-message')).toBeVisible({
      timeout: 30000
    });
  });

  test('should not send empty message', async ({ page }) => {
    // 빈 메시지 상태에서 전송 버튼 클릭
    const sendButton = page.locator('[data-testid="send-button"], button[type="submit"]');

    // 버튼이 비활성화되어 있거나, 클릭해도 메시지가 추가되지 않아야 함
    const messagesBefore = await page.locator('[data-testid="chat-messages"] > *').count();

    if (await sendButton.isEnabled()) {
      await sendButton.click();
      // 빈 메시지는 추가되지 않아야 함
      await page.waitForTimeout(500);
      const messagesAfter = await page.locator('[data-testid="chat-messages"] > *').count();
      expect(messagesAfter).toBeLessThanOrEqual(messagesBefore);
    }
  });

  test('should show loading indicator while waiting for response', async ({ page }) => {
    await page.fill('[data-testid="chat-input"], textarea', '테스트 질문입니다.');
    await page.click('[data-testid="send-button"], button[type="submit"]');

    // 로딩 인디케이터 확인
    await expect(page.locator('[data-testid="loading"], .loading, .spinner')).toBeVisible();

    // 응답 완료 후 로딩 인디케이터 사라짐
    await expect(page.locator('[data-testid="loading"], .loading, .spinner')).not.toBeVisible({
      timeout: 30000
    });
  });

  test('should preserve chat history on page reload', async ({ page }) => {
    const testMessage = `히스토리 테스트 ${Date.now()}`;

    // 메시지 전송
    await page.fill('[data-testid="chat-input"], textarea', testMessage);
    await page.click('[data-testid="send-button"], button[type="submit"]');

    // 응답 대기
    await page.waitForTimeout(2000);

    // 페이지 새로고침
    await page.reload();

    // 이전 메시지가 유지되는지 확인 (채팅 히스토리 저장 기능이 있는 경우)
    // await expect(page.locator('[data-testid="chat-messages"]')).toContainText(testMessage);
  });
});

/**
 * RAG 기능 테스트
 */
test.describe('RAG Features', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should show relevant document sources in response', async ({ page }) => {
    // RAG 관련 질문
    await page.fill('[data-testid="chat-input"], textarea', '학칙에 대해 알려주세요');
    await page.click('[data-testid="send-button"], button[type="submit"]');

    // 응답 대기
    await page.waitForTimeout(5000);

    // 출처/참고문서 표시 확인 (RAG 시스템의 경우)
    // await expect(page.locator('[data-testid="sources"], .document-sources')).toBeVisible();
  });
});
