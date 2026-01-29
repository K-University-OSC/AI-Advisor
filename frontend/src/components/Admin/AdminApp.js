/**
 * 관리자 앱 (OSC 공개 버전)
 */
import React, { useState, useEffect } from 'react';
import AdminLogin from './AdminLogin';
import AdminDashboard from './AdminDashboard';
import { isAdminLoggedIn, adminLogout } from '../../api/adminApi';

function AdminApp() {
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [checking, setChecking] = useState(true);

    useEffect(() => {
        setIsLoggedIn(isAdminLoggedIn());
        setChecking(false);
    }, []);

    const handleLoginSuccess = () => {
        setIsLoggedIn(true);
    };

    const handleLogout = () => {
        adminLogout();
        setIsLoggedIn(false);
    };

    if (checking) {
        return (
            <div className="admin-dashboard loading">
                <div className="loading-spinner"></div>
                <p>확인 중...</p>
            </div>
        );
    }

    if (!isLoggedIn) {
        return <AdminLogin onLoginSuccess={handleLoginSuccess} />;
    }

    return <AdminDashboard onLogout={handleLogout} />;
}

export default AdminApp;
