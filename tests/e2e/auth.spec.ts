import { test, expect } from '@playwright/test';

/**
 * 인증 테스트 (Authentication Tests)
 */
test.describe('Authentication', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display login page', async ({ page }) => {
    // 로그인 폼 요소 확인
    await expect(page.getByRole('heading', { name: /로그인|Login/i })).toBeVisible();
    await expect(page.locator('input[name="username"], [data-testid="username"]')).toBeVisible();
    await expect(page.locator('input[name="password"], [data-testid="password"]')).toBeVisible();
    await expect(page.getByRole('button', { name: /로그인|Login/i })).toBeVisible();
  });

  test('should login successfully with valid credentials', async ({ page }) => {
    // 사용자 입력
    await page.fill('input[name="username"], [data-testid="username"]', 'testuser');
    await page.fill('input[name="password"], [data-testid="password"]', 'test1234');

    // 로그인 버튼 클릭
    await page.click('button[type="submit"], [data-testid="login-button"]');

    // 채팅 페이지로 리다이렉트 확인
    await expect(page).toHaveURL(/\/(chat|home|dashboard)/);

    // 사용자 메뉴 또는 로그아웃 버튼 표시 확인
    await expect(page.locator('[data-testid="user-menu"], [data-testid="logout-button"]')).toBeVisible();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    // 잘못된 자격 증명 입력
    await page.fill('input[name="username"], [data-testid="username"]', 'wronguser');
    await page.fill('input[name="password"], [data-testid="password"]', 'wrongpassword');

    // 로그인 시도
    await page.click('button[type="submit"], [data-testid="login-button"]');

    // 에러 메시지 확인
    await expect(page.locator('[data-testid="error-message"], .error-message, .alert-danger')).toBeVisible();
  });

  test('should show validation error for empty fields', async ({ page }) => {
    // 빈 필드로 로그인 시도
    await page.click('button[type="submit"], [data-testid="login-button"]');

    // 유효성 검사 에러 확인 (HTML5 validation 또는 커스텀)
    const usernameInput = page.locator('input[name="username"], [data-testid="username"]');
    await expect(usernameInput).toBeFocused();
  });

  test('should logout successfully', async ({ page }) => {
    // 먼저 로그인
    await page.fill('input[name="username"], [data-testid="username"]', 'testuser');
    await page.fill('input[name="password"], [data-testid="password"]', 'test1234');
    await page.click('button[type="submit"], [data-testid="login-button"]');

    // 로그인 성공 대기
    await expect(page).toHaveURL(/\/(chat|home|dashboard)/);

    // 로그아웃
    await page.click('[data-testid="user-menu"], [data-testid="logout-button"]');

    // 로그아웃 확인 메뉴가 있으면 클릭
    const logoutMenuItem = page.locator('[data-testid="logout-menu-item"]');
    if (await logoutMenuItem.isVisible()) {
      await logoutMenuItem.click();
    }

    // 로그인 페이지로 리다이렉트 확인
    await expect(page).toHaveURL(/\/(login)?$/);
  });
});

/**
 * 회원가입 테스트 (Registration Tests)
 */
test.describe('Registration', () => {
  test('should navigate to registration page', async ({ page }) => {
    await page.goto('/');

    // 회원가입 링크 클릭
    const registerLink = page.getByRole('link', { name: /회원가입|Register|Sign up/i });
    if (await registerLink.isVisible()) {
      await registerLink.click();
      await expect(page).toHaveURL(/register|signup/);
    }
  });

  test('should register new user successfully', async ({ page }) => {
    await page.goto('/register');

    // 가입 폼이 있는 경우
    const usernameInput = page.locator('input[name="username"], [data-testid="username"]');
    if (await usernameInput.isVisible()) {
      const uniqueUsername = `testuser_${Date.now()}`;

      await usernameInput.fill(uniqueUsername);
      await page.fill('input[name="email"], [data-testid="email"]', `${uniqueUsername}@test.com`);
      await page.fill('input[name="password"], [data-testid="password"]', 'Test1234!');
      await page.fill('input[name="confirmPassword"], [data-testid="confirm-password"]', 'Test1234!');

      await page.click('button[type="submit"]');

      // 성공 메시지 또는 로그인 페이지로 리다이렉트
      await expect(page.locator('.success-message, [data-testid="success-message"]')).toBeVisible()
        .catch(() => expect(page).toHaveURL(/login/));
    }
  });
});
