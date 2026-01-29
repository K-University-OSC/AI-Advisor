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

// 지원하는 LLM 모델 정보
export const LLM_MODELS = {
    // OpenAI GPT-5 시리즈
    'gpt52': { name: 'GPT-5.2', provider: 'OpenAI', description: '최신 GPT 모델' },
    'gpt5': { name: 'GPT-5', provider: 'OpenAI', description: '코딩 및 에이전트 작업에 최상의 모델' },
    'gpt5m': { name: 'GPT-5 mini', provider: 'OpenAI', description: '빠르고 저렴한 버전' },
    'gpt5n': { name: 'GPT-5 nano', provider: 'OpenAI', description: '가장 빠르고 저렴한 버전' },

    // Google
    'gmn30': { name: 'Gemini 3.0 Pro', provider: 'Google', description: 'Gemini 최신 모델' },
    'gmn25f': { name: 'Gemini 2.5 Flash', provider: 'Google', description: '빠른 Gemini 모델' },

    // Anthropic
    'cld45o': { name: 'Claude 4.5 Opus', provider: 'Anthropic', description: 'Claude 최고 성능 모델' },
    'cld45s': { name: 'Claude 4.5 Sonnet', provider: 'Anthropic', description: 'Claude 균형 잡힌 모델' },

    // Perplexity
    'pplx': { name: 'Perplexity Sonar', provider: 'Perplexity', description: '실시간 검색 AI' },
    'pplxp': { name: 'Perplexity Pro', provider: 'Perplexity', description: 'Perplexity Pro' }
};

// 기본 모델
export const DEFAULT_MODEL = 'gpt5m';
