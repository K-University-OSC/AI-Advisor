// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * 사용자 인증 테스트 (회원가입, 로그인, 로그아웃)
 */

// 테스트용 랜덤 사용자 생성
const generateTestUser = () => ({
    username: `testuser_${Date.now()}`,
    password: 'test1234',
    displayName: '테스트사용자'
});

test.describe('사용자 인증', () => {
    test('로그인 페이지가 표시되어야 함', async ({ page }) => {
        await page.goto('/');

        // 로그인 폼 확인
        await expect(page.locator('.auth-container')).toBeVisible();
        await expect(page.locator('.auth-title')).toContainText('Multi-LLM Chatbot');
        await expect(page.locator('#username')).toBeVisible();
        await expect(page.locator('#password')).toBeVisible();
        await expect(page.locator('button[type="submit"]')).toContainText('로그인');
    });

    test('회원가입 폼으로 전환할 수 있어야 함', async ({ page }) => {
        await page.goto('/');

        // 회원가입 버튼 클릭
        await page.click('.auth-toggle button');

        // 회원가입 폼 확인
        await expect(page.locator('button[type="submit"]')).toContainText('회원가입');
        await expect(page.locator('#displayName')).toBeVisible();
    });

    test('회원가입 후 로그인 성공', async ({ page }) => {
        const user = generateTestUser();
        await page.goto('/');

        // 회원가입 모드로 전환
        await page.click('.auth-toggle button');

        // 회원가입 정보 입력
        await page.fill('#username', user.username);
        await page.fill('#password', user.password);
        await page.fill('#displayName', user.displayName);

        // 제출
        await page.click('button[type="submit"]');

        // 채팅 화면으로 이동 확인 (로그인 성공)
        await expect(page.locator('.app-container')).toBeVisible({ timeout: 10000 });
        await expect(page.locator('.welcome-title')).toContainText('무엇을 도와드릴까요?');
    });

    test('기존 사용자로 로그인 성공', async ({ page }) => {
        // 먼저 회원가입
        const user = generateTestUser();
        await page.goto('/');
        await page.click('.auth-toggle button');
        await page.fill('#username', user.username);
        await page.fill('#password', user.password);
        await page.click('button[type="submit"]');
        await expect(page.locator('.app-container')).toBeVisible({ timeout: 10000 });

        // 로그아웃
        await page.click('button[title="로그아웃"]');

        // 다시 로그인 페이지 확인
        await expect(page.locator('.auth-container')).toBeVisible();

        // 로그인
        await page.fill('#username', user.username);
        await page.fill('#password', user.password);
        await page.click('button[type="submit"]');

        // 채팅 화면 확인
        await expect(page.locator('.app-container')).toBeVisible({ timeout: 10000 });
    });

    test('잘못된 비밀번호로 로그인 실패', async ({ page }) => {
        await page.goto('/');

        await page.fill('#username', 'nonexistent_user');
        await page.fill('#password', 'wrongpassword');
        await page.click('button[type="submit"]');

        // 에러 메시지 확인
        await expect(page.locator('.auth-error')).toBeVisible({ timeout: 5000 });
    });

    test('로그아웃 기능 동작', async ({ page }) => {
        const user = generateTestUser();
        await page.goto('/');

        // 회원가입 및 로그인
        await page.click('.auth-toggle button');
        await page.fill('#username', user.username);
        await page.fill('#password', user.password);
        await page.click('button[type="submit"]');
        await expect(page.locator('.app-container')).toBeVisible({ timeout: 10000 });

        // 로그아웃 버튼 클릭
        await page.click('button[title="로그아웃"]');

        // 로그인 페이지로 돌아가는지 확인
        await expect(page.locator('.auth-container')).toBeVisible();
    });

    test('짧은 비밀번호로 회원가입 실패', async ({ page }) => {
        await page.goto('/');

        // 회원가입 모드로 전환
        await page.click('.auth-toggle button');

        // 짧은 비밀번호 입력
        await page.fill('#username', 'shortpwduser');
        await page.fill('#password', '123');  // 4자 미만

        // 제출
        await page.click('button[type="submit"]');

        // 에러 메시지 확인
        await expect(page.locator('.auth-error')).toContainText('4자 이상');
    });
});
