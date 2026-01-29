/**
 * Axios 인스턴스 설정
 */
import axios from 'axios';
import { API_BASE_URL } from '../Config';

const axiosInstance = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
    headers: {
        'Content-Type': 'application/json',
    }
});

// 요청 인터셉터
axiosInstance.interceptors.request.use(
    (config) => {
        console.log(`[API Request] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
    },
    (error) => {
        console.error('[API Request Error]', error);
        return Promise.reject(error);
    }
);

// 응답 인터셉터
axiosInstance.interceptors.response.use(
    (response) => {
        return response;
    },
    (error) => {
        console.error('[API Response Error]', error.response?.status, error.message);
        return Promise.reject(error);
    }
);

export default axiosInstance;
