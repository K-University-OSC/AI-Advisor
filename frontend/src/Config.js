/**
 * 프론트엔드 설정
 */

// API 서버 URL 설정
// - 프로덕션: Nginx 리버스 프록시 사용 (상대 경로)
// - 개발: 동적 Hostname + 백엔드 포트 직접 접속
const getApiBaseUrl = () => {
    // 1. 환경변수로 직접 지정된 경우 (최우선, 빈 문자열도 유효)
    if (process.env.REACT_APP_API_URL !== undefined && process.env.REACT_APP_API_URL !== '') {
        return process.env.REACT_APP_API_URL;
    }

    // 2. 프로덕션 빌드: Nginx 리버스 프록시 사용 (상대 경로)
    //    - /api/* 요청은 Nginx가 백엔드로 프록시
    //    - 방화벽으로 백엔드 포트가 막혀있어도 동작
    //    - CORS 문제 없음
    //    - 주의: API 호출 시 /api 경로가 이미 포함되므로 빈 문자열 반환
    if (process.env.NODE_ENV === 'production') {
        return '';
    }

    // 3. 개발 환경: 동적 Hostname + 백엔드 포트
    const hostname = window.location.hostname;
    const port = window.location.port;
    const protocol = window.location.protocol;

    // 프론트엔드 포트 기반으로 백엔드 포트 결정
    // 개발(Dev): Frontend 8612 → Backend 8600
    // 로컬 운영(Local Prod): Frontend 8611 → Backend 8601
    let backendPort = '8600'; // 기본값

    if (port === '8611') {
        backendPort = '8601';
    } else if (port === '8612' || port === '3000') {
        backendPort = '8600';
    }

    // 주의: API 호출 시 /api 경로가 이미 포함되므로 /api 제외
    return `${protocol}//${hostname}:${backendPort}`;
};

export const API_BASE_URL = getApiBaseUrl();
