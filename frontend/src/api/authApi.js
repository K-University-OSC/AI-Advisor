/**
 * 인증 관련 API 함수 (OSC 공개 버전)
 */
import { API_BASE_URL } from '../Config';

// 토큰 저장 키
const TOKEN_KEY = 'advisor_token';
const USER_KEY = 'advisor_user';

/**
 * 토큰 저장
 */
export const saveToken = (token) => {
    localStorage.setItem(TOKEN_KEY, token);
};

/**
 * 토큰 가져오기
 */
export const getToken = () => {
    return localStorage.getItem(TOKEN_KEY);
};

/**
 * 토큰 삭제
 */
export const removeToken = () => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
};

/**
 * 사용자 정보 저장
 */
export const saveUser = (user) => {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
};

/**
 * 사용자 정보 가져오기
 */
export const getUser = () => {
    const user = localStorage.getItem(USER_KEY);
    return user ? JSON.parse(user) : null;
};

/**
 * 인증 헤더 생성
 */
export const getAuthHeader = () => {
    const token = getToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
};

/**
 * 회원가입
 */
export const signup = async (username, password, displayName = null) => {
    const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            username,
            password,
            display_name: displayName
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '회원가입에 실패했습니다');
    }

    const data = await response.json();
    saveToken(data.access_token);
    saveUser(data.user);
    return data;
};

/**
 * 로그인
 */
export const login = async (username, password) => {
    const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            username,
            password
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '로그인에 실패했습니다');
    }

    const data = await response.json();
    saveToken(data.access_token);
    saveUser(data.user);
    return data;
};

/**
 * 로그아웃
 */
export const logout = () => {
    removeToken();
};

/**
 * 현재 사용자 정보 조회
 */
export const getCurrentUser = async () => {
    const token = getToken();
    if (!token) return null;

    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
            headers: getAuthHeader()
        });

        if (!response.ok) {
            removeToken();
            return null;
        }

        return await response.json();
    } catch (error) {
        removeToken();
        return null;
    }
};

/**
 * 로그인 상태 확인
 */
export const isLoggedIn = () => {
    return !!getToken();
};
