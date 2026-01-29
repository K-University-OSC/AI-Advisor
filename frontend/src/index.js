import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import './styles/App.css';
import App from './App';
import { AdminApp } from './components/Admin';

/**
 * 라우터 컴포넌트 - URL 해시 기반 라우팅 (OSC 공개 버전)
 * #/admin -> 관리자 대시보드
 * 그 외 -> 일반 사용자 앱
 */
function Router() {
    const [route, setRoute] = useState(window.location.hash);

    useEffect(() => {
        const handleHashChange = () => setRoute(window.location.hash);
        window.addEventListener('hashchange', handleHashChange);
        return () => window.removeEventListener('hashchange', handleHashChange);
    }, []);

    // 관리자 페이지
    if (route === '#/admin') {
        return <AdminApp />;
    }

    // 기본 앱 (일반 사용자)
    return <App />;
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <Router />
  </React.StrictMode>
);
