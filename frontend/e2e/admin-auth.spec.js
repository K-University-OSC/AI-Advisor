// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * 관리자 인증 테스트
 */

// 관리자 계정 정보
const ADMIN_CREDENTIALS = {
    username: 'admin',
    password: 'admin1234'
};

test.describe('관리자 인증', () => {
    test('관리자 로그인 페이지가 표시되어야 함', async ({ page }) => {
        await page.goto('/admin-login');

        // 관리자 로그인 폼 확인
        await expect(page.locator('.admin-login')).toBeVisible();
        await expect(page.locator('h1')).toContainText('관리자 로그인');
        await expect(page.locator('#username')).toBeVisible();
        await expect(page.locator('#password')).toBeVisible();
    });

    test('관리자 로그인 성공', async ({ page }) => {
        await page.goto('/admin-login');

        // 관리자 로그인 정보 입력
        await page.fill('#username', ADMIN_CREDENTIALS.username);
        await page.fill('#password', ADMIN_CREDENTIALS.password);

        // 로그인 버튼 클릭
        await page.click('button[type="submit"]');

        // 관리자 대시보드 화면 확인
        await expect(page.locator('.admin-dashboard, .dashboard-container')).toBeVisible({ timeout: 10000 });
    });

    test('잘못된 관리자 비밀번호로 로그인 실패', async ({ page }) => {
        await page.goto('/admin-login');

        await page.fill('#username', 'admin');
        await page.fill('#password', 'wrongpassword');
        await page.click('button[type="submit"]');

        // 에러 메시지 확인
        await expect(page.locator('.login-error')).toBeVisible({ timeout: 5000 });
    });

    test('존재하지 않는 관리자 계정으로 로그인 실패', async ({ page }) => {
        await page.goto('/admin-login');

        await page.fill('#username', 'notadmin');
        await page.fill('#password', 'somepassword');
        await page.click('button[type="submit"]');

        // 에러 메시지 확인
        await expect(page.locator('.login-error')).toBeVisible({ timeout: 5000 });
    });
});

test.describe('관리자 대시보드', () => {
    test.beforeEach(async ({ page }) => {
        // 관리자로 로그인
        await page.goto('/admin-login');
        await page.fill('#username', ADMIN_CREDENTIALS.username);
        await page.fill('#password', ADMIN_CREDENTIALS.password);
        await page.click('button[type="submit"]');
        await expect(page.locator('.admin-dashboard, .dashboard-container')).toBeVisible({ timeout: 10000 });
    });

    test('대시보드 주요 섹션이 표시되어야 함', async ({ page }) => {
        // 통계 카드 확인
        await expect(page.locator('.stat-card, .stats-grid, .dashboard-stats').first()).toBeVisible();
    });

    test('사용자 관리 탭이 동작해야 함', async ({ page }) => {
        // 사용자 관리 탭 클릭 (있는 경우)
        const userTab = page.locator('button:has-text("사용자"), [data-tab="users"]');
        if (await userTab.isVisible()) {
            await userTab.click();
            // 사용자 목록 테이블 확인
            await expect(page.locator('table, .user-list, .users-table')).toBeVisible({ timeout: 5000 });
        }
    });

    test('통계/분석 탭이 동작해야 함', async ({ page }) => {
        // 통계 탭 클릭 (있는 경우)
        const statsTab = page.locator('button:has-text("통계"), button:has-text("분석"), [data-tab="stats"]');
        if (await statsTab.isVisible()) {
            await statsTab.click();
            // 차트나 통계 컴포넌트 확인
            await expect(page.locator('.chart, .stats-content, canvas')).toBeVisible({ timeout: 5000 });
        }
    });

    test('로그아웃 기능이 동작해야 함', async ({ page }) => {
        // 로그아웃 버튼 클릭
        const logoutBtn = page.locator('button:has-text("로그아웃"), button[title="로그아웃"]');
        await logoutBtn.click();

        // 로그인 페이지로 돌아가는지 확인
        await expect(page.locator('.admin-login, .login-card')).toBeVisible({ timeout: 5000 });
    });
});
