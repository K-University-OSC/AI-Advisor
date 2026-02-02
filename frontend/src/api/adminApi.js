/**
 * 관리자 API 모듈 (OSC 공개 버전)
 * 관리자 인증 및 대시보드 API
 */
import { API_BASE_URL } from '../Config';

// ============================================================================
// 헬퍼 함수
// ============================================================================

const ADMIN_TOKEN_KEY = 'admin_token';
const ADMIN_USER_KEY = 'admin_user';

const getAdminToken = () => localStorage.getItem(ADMIN_TOKEN_KEY);

const getAdminHeaders = () => ({
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${getAdminToken()}`
});

// ============================================================================
// 관리자 인증 API
// ============================================================================

export const adminLogin = async (username, password) => {
    const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '로그인 실패');
    }

    const data = await response.json();

    // admin 역할 확인
    if (data.user.role !== 'admin') {
        throw new Error('관리자 권한이 없습니다');
    }

    localStorage.setItem(ADMIN_TOKEN_KEY, data.access_token);
    localStorage.setItem(ADMIN_USER_KEY, JSON.stringify(data.user));

    return data;
};

export const adminLogout = () => {
    localStorage.removeItem(ADMIN_TOKEN_KEY);
    localStorage.removeItem(ADMIN_USER_KEY);
};

export const getAdmin = () => {
    const admin = localStorage.getItem(ADMIN_USER_KEY);
    return admin ? JSON.parse(admin) : null;
};

export const isAdminLoggedIn = () => {
    return !!getAdminToken() && !!getAdmin();
};

// Aliases for backward compatibility
export const getTenantAdmin = getAdmin;
export const tenantAdminLogout = adminLogout;
export const isTenantAdminLoggedIn = isAdminLoggedIn;

// ============================================================================
// 관리자 대시보드 API
// ============================================================================

export const getAdminDashboard = async () => {
    const response = await fetch(`${API_BASE_URL}/api/admin/dashboard`, {
        headers: getAdminHeaders()
    });

    if (!response.ok) {
        if (response.status === 401) {
            adminLogout();
            throw new Error('세션이 만료되었습니다. 다시 로그인해주세요.');
        }
        if (response.status === 403) throw new Error('관리자 권한 필요');
        throw new Error('대시보드 로드 실패');
    }

    return response.json();
};

export const getUsagePatterns = async (days = 30) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/dashboard/usage-patterns?days=${days}`, {
        headers: getAdminHeaders()
    });

    if (!response.ok) throw new Error('사용 패턴 로드 실패');
    return response.json();
};

export const getTopUsers = async (days = 30, limit = 20) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/dashboard/top-users?days=${days}&limit=${limit}`, {
        headers: getAdminHeaders()
    });

    if (!response.ok) throw new Error('상위 사용자 로드 실패');
    return response.json();
};

export const getCosts = async (days = 30) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/dashboard/costs?days=${days}`, {
        headers: getAdminHeaders()
    });

    if (!response.ok) throw new Error('비용 분석 로드 실패');
    return response.json();
};

// Aliases for backward compatibility
export const getTenantAdminDashboard = getAdminDashboard;
export const getTenantUsagePatterns = getUsagePatterns;
export const getTenantTopUsers = getTopUsers;
export const getTenantCosts = getCosts;

// ============================================================================
// 사용자 관리 API
// ============================================================================

export const getUsers = async (page = 1, limit = 50, search = '') => {
    let url = `${API_BASE_URL}/api/admin/users?page=${page}&limit=${limit}`;
    if (search) url += `&search=${encodeURIComponent(search)}`;

    const response = await fetch(url, {
        headers: getAdminHeaders()
    });

    if (!response.ok) throw new Error('사용자 목록 로드 실패');
    return response.json();
};

export const getUserDetail = async (userId) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/users/${userId}`, {
        headers: getAdminHeaders()
    });

    if (!response.ok) throw new Error('사용자 정보 로드 실패');
    return response.json();
};

export const updateUser = async (userId, updateData) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/users/${userId}`, {
        method: 'PUT',
        headers: getAdminHeaders(),
        body: JSON.stringify(updateData)
    });

    if (!response.ok) throw new Error('사용자 정보 수정 실패');
    return response.json();
};

export const suspendUser = async (userId) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/users/${userId}/suspend`, {
        method: 'POST',
        headers: getAdminHeaders()
    });

    if (!response.ok) throw new Error('사용자 정지 실패');
    return response.json();
};

export const activateUser = async (userId) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/users/${userId}/activate`, {
        method: 'POST',
        headers: getAdminHeaders()
    });

    if (!response.ok) throw new Error('사용자 활성화 실패');
    return response.json();
};

export const createUser = async (userData) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/users`, {
        method: 'POST',
        headers: getAdminHeaders(),
        body: JSON.stringify(userData)
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '사용자 생성 실패');
    }
    return response.json();
};

export const createUsersBulk = async (users) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/users/bulk`, {
        method: 'POST',
        headers: getAdminHeaders(),
        body: JSON.stringify({ users })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '일괄 사용자 생성 실패');
    }
    return response.json();
};

export const deleteUser = async (userId) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/users/${userId}`, {
        method: 'DELETE',
        headers: getAdminHeaders()
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '사용자 삭제 실패');
    }
    return response.json();
};

// Aliases for compatibility
export const getTenantUsers = getUsers;
export const suspendTenantUser = suspendUser;
export const activateTenantUser = activateUser;
export const createTenantUser = createUser;
export const createTenantUsersBulk = createUsersBulk;
export const deleteTenantUser = deleteUser;

// ============================================================================
// 관리자 관리 API
// ============================================================================

export const getAdmins = async () => {
    const response = await fetch(`${API_BASE_URL}/api/admin/admins`, {
        headers: getAdminHeaders()
    });

    if (!response.ok) throw new Error('관리자 목록 로드 실패');
    return response.json();
};

export const createAdmin = async (adminData) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/admins`, {
        method: 'POST',
        headers: getAdminHeaders(),
        body: JSON.stringify(adminData)
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '관리자 생성 실패');
    }
    return response.json();
};

export const deleteAdmin = async (userId) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/admins/${userId}`, {
        method: 'DELETE',
        headers: getAdminHeaders()
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '관리자 삭제 실패');
    }
    return response.json();
};

// Aliases for compatibility
export const getTenantAdmins = getAdmins;
export const createTenantAdmin = createAdmin;
export const deleteTenantAdmin = deleteAdmin;

// ============================================================================
// 시스템 설정 API
// ============================================================================

export const getSystemSettings = async () => {
    const response = await fetch(`${API_BASE_URL}/api/admin/settings`, {
        headers: getAdminHeaders()
    });

    if (!response.ok) throw new Error('시스템 설정 로드 실패');
    return response.json();
};

export const updateSystemSettings = async (settings) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/settings`, {
        method: 'PUT',
        headers: getAdminHeaders(),
        body: JSON.stringify(settings)
    });

    if (!response.ok) throw new Error('시스템 설정 저장 실패');
    return response.json();
};

// ============================================================================
// 데이터 관리 API (문서)
// ============================================================================

// 데이터 통계
export const getDataStats = async () => {
    const response = await fetch(`${API_BASE_URL}/api/admin/data/stats`, {
        headers: getAdminHeaders()
    });

    if (!response.ok) {
        if (response.status === 401) throw new Error('인증 필요');
        if (response.status === 403) throw new Error('관리자 권한 필요');
        throw new Error('데이터 통계 로드 실패');
    }

    return response.json();
};

// 문서 관리
export const getDocuments = async (page = 1, limit = 50, status = null) => {
    let url = `${API_BASE_URL}/api/admin/data/documents?page=${page}&limit=${limit}`;
    if (status) url += `&status=${status}`;

    const response = await fetch(url, {
        headers: getAdminHeaders()
    });

    if (!response.ok) throw new Error('문서 목록 로드 실패');
    return response.json();
};

export const uploadDocument = async (file, autoIndex = true) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('auto_index', autoIndex);

    const response = await fetch(`${API_BASE_URL}/api/admin/data/documents/upload`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${getAdminToken()}`
        },
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '문서 업로드 실패');
    }

    return response.json();
};

export const deleteDocument = async (docId) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/data/documents/${docId}`, {
        method: 'DELETE',
        headers: getAdminHeaders()
    });

    if (!response.ok) throw new Error('문서 삭제 실패');
    return response.json();
};

export const reindexDocument = async (docId) => {
    const response = await fetch(`${API_BASE_URL}/api/admin/data/documents/${docId}/reindex`, {
        method: 'POST',
        headers: getAdminHeaders()
    });

    if (!response.ok) throw new Error('문서 재인덱싱 실패');
    return response.json();
};
