// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * 채팅 기능 테스트
 */

// 테스트용 랜덤 사용자 생성 및 로그인 헬퍼
const loginAsTestUser = async (page) => {
    const user = {
        username: `chattest_${Date.now()}`,
        password: 'test1234',
        displayName: '채팅테스트'
    };

    await page.goto('/');

    // 회원가입 모드로 전환
    await page.click('.auth-toggle button');

    // 회원가입
    await page.fill('#username', user.username);
    await page.fill('#password', user.password);
    await page.fill('#displayName', user.displayName);
    await page.click('button[type="submit"]');

    // 채팅 화면 로딩 대기
    await expect(page.locator('.app-container')).toBeVisible({ timeout: 10000 });

    return user;
};

test.describe('채팅 화면 UI', () => {
    test.beforeEach(async ({ page }) => {
        await loginAsTestUser(page);
    });

    test('채팅 화면 기본 요소가 표시되어야 함', async ({ page }) => {
        // 사이드바 확인
        await expect(page.locator('.sidebar')).toBeVisible();

        // 새 채팅 버튼 확인
        await expect(page.locator('.new-chat-btn')).toBeVisible();

        // 입력 영역 확인
        await expect(page.locator('.input-area')).toBeVisible();
        await expect(page.locator('.input-wrapper textarea')).toBeVisible();

        // 모델 선택기 확인
        await expect(page.locator('.model-selector')).toBeVisible();

        // 환영 메시지 확인
        await expect(page.locator('.welcome-title')).toContainText('무엇을 도와드릴까요?');
    });

    test('사이드바 토글 동작', async ({ page }) => {
        // 초기 사이드바 열림 상태 확인
        await expect(page.locator('.sidebar.open')).toBeVisible();

        // 메뉴 버튼 클릭하여 사이드바 닫기
        await page.click('.sidebar-header .icon-btn');

        // 사이드바 닫힘 상태 확인
        await expect(page.locator('.sidebar.closed')).toBeVisible();

        // 다시 메뉴 버튼 클릭하여 사이드바 열기
        await page.click('.top-bar-left .icon-btn');

        // 사이드바 다시 열림 확인
        await expect(page.locator('.sidebar.open')).toBeVisible();
    });

    test('모델 선택 드롭다운 동작', async ({ page }) => {
        // 모델 선택 버튼 클릭
        await page.click('.model-selector-btn');

        // 드롭다운 표시 확인
        await expect(page.locator('.model-dropdown')).toBeVisible();

        // 모델 옵션들이 있는지 확인
        const modelOptions = page.locator('.model-option');
        await expect(modelOptions.first()).toBeVisible();

        // 다른 모델 선택
        await modelOptions.nth(1).click();

        // 드롭다운 닫힘 확인
        await expect(page.locator('.model-dropdown')).not.toBeVisible();
    });
});

test.describe('채팅 메시지 기능', () => {
    test.beforeEach(async ({ page }) => {
        await loginAsTestUser(page);
    });

    test('메시지 입력 및 전송', async ({ page }) => {
        const testMessage = '안녕하세요, 테스트 메시지입니다.';

        // 메시지 입력
        await page.fill('.input-wrapper textarea', testMessage);

        // 전송 버튼 활성화 확인
        await expect(page.locator('.send-btn')).toBeEnabled();

        // 전송 버튼 클릭
        await page.click('.send-btn');

        // 사용자 메시지가 표시되는지 확인
        await expect(page.locator('.message.user').first()).toBeVisible({ timeout: 5000 });
        await expect(page.locator('.message.user .user-text')).toContainText(testMessage);

        // AI 응답 메시지 영역 확인 (스트리밍)
        await expect(page.locator('.message.assistant')).toBeVisible({ timeout: 10000 });
    });

    test('Enter 키로 메시지 전송', async ({ page }) => {
        const testMessage = 'Enter 키 테스트';

        // 메시지 입력
        await page.fill('.input-wrapper textarea', testMessage);

        // Enter 키 입력
        await page.keyboard.press('Enter');

        // 메시지 전송 확인
        await expect(page.locator('.message.user').first()).toBeVisible({ timeout: 5000 });
    });

    test('Shift+Enter로 줄바꿈', async ({ page }) => {
        // 메시지 입력
        await page.fill('.input-wrapper textarea', '첫 번째 줄');

        // Shift+Enter로 줄바꿈
        await page.keyboard.press('Shift+Enter');
        await page.keyboard.type('두 번째 줄');

        // textarea 값 확인 (줄바꿈 포함)
        const textareaValue = await page.locator('.input-wrapper textarea').inputValue();
        expect(textareaValue).toContain('첫 번째 줄');
        expect(textareaValue).toContain('두 번째 줄');
    });

    test('빈 메시지는 전송 불가', async ({ page }) => {
        // 빈 상태에서 전송 버튼 비활성화 확인
        await expect(page.locator('.send-btn')).toBeDisabled();

        // 공백만 입력
        await page.fill('.input-wrapper textarea', '   ');

        // 여전히 전송 버튼 비활성화
        await expect(page.locator('.send-btn')).toBeDisabled();
    });
});

test.describe('세션 관리', () => {
    test.beforeEach(async ({ page }) => {
        await loginAsTestUser(page);
    });

    test('새 대화 시작', async ({ page }) => {
        // 메시지 전송하여 세션 생성
        await page.fill('.input-wrapper textarea', '세션 테스트 메시지');
        await page.click('.send-btn');
        await expect(page.locator('.message.user').first()).toBeVisible({ timeout: 5000 });

        // 새 채팅 버튼 클릭
        await page.click('.new-chat-btn');

        // 환영 메시지로 돌아가는지 확인
        await expect(page.locator('.welcome-title')).toBeVisible();

        // 이전 메시지들이 사라졌는지 확인
        await expect(page.locator('.messages-container')).not.toBeVisible();
    });

    test('대화 히스토리에 세션 표시', async ({ page }) => {
        // 메시지 전송하여 세션 생성
        await page.fill('.input-wrapper textarea', '히스토리 테스트');
        await page.click('.send-btn');

        // AI 응답 대기
        await expect(page.locator('.message.assistant')).toBeVisible({ timeout: 30000 });

        // 잠시 대기 (세션 저장)
        await page.waitForTimeout(2000);

        // 대화 히스토리에 세션이 표시되는지 확인
        await expect(page.locator('.history-item').first()).toBeVisible({ timeout: 10000 });
    });
});

test.describe('응답 생성 중 동작', () => {
    test.beforeEach(async ({ page }) => {
        await loginAsTestUser(page);
    });

    test('응답 생성 중 중지 버튼 표시', async ({ page }) => {
        // 메시지 전송
        await page.fill('.input-wrapper textarea', '긴 응답을 생성해주세요. 한국의 역사에 대해 자세히 설명해주세요.');
        await page.click('.send-btn');

        // 중지 버튼이 표시되는지 확인
        await expect(page.locator('.stop-btn')).toBeVisible({ timeout: 5000 });
    });

    test('응답 생성 중 입력 비활성화', async ({ page }) => {
        // 메시지 전송
        await page.fill('.input-wrapper textarea', '테스트 메시지');
        await page.click('.send-btn');

        // 입력 필드 비활성화 확인
        await expect(page.locator('.input-wrapper textarea')).toBeDisabled({ timeout: 5000 });
    });
});
