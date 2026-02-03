/**
 * 관리자 대시보드 (OSC 단일 테넌트)
 * 사용자 관리, 사용량 모니터링, 비용 분석
 */
import React, { useState, useEffect } from 'react';
import {
    getTenantAdminDashboard,
    getTenantUsagePatterns,
    getTenantCosts,
    getTenantUsers,
    suspendTenantUser,
    activateTenantUser,
    getTenantAdmin,
    createTenantUsersBulk,
    deleteTenantUser,
    getTenantAdmins,
    createTenantAdmin
} from '../../api/adminApi';
import {
    LineChart, Line, BarChart, Bar, Cell,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import './AdminDashboard.css';
import * as XLSX from 'xlsx';
import DataManagement from './DataManagement';

// 모델 ID를 표시명으로 변환 (Advisor OSC는 고정 모델 사용)
const getModelDisplayName = (modelId) => {
    if (!modelId) return 'Gemini 3 Flash';
    // 환경변수 기반 고정 모델 사용시 기본값 반환
    if (modelId === 'default') return 'Gemini 3 Flash';
    return modelId;
};

// 빈 사용자 행 생성
const createEmptyUserRow = () => ({
    id: Date.now() + Math.random(),
    username: '',
    password: '',
    displayName: '',
    email: ''
});

function AdminDashboard({ onLogout }) {
    const [dashboard, setDashboard] = useState(null);
    const [patterns, setPatterns] = useState(null);
    const [costs, setCosts] = useState(null);
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('overview');
    const [searchQuery, setSearchQuery] = useState('');
    const currentUser = getTenantAdmin();

    // 사용자 등록 관련 state
    const [showRegistration, setShowRegistration] = useState(false);
    const [registrationMode, setRegistrationMode] = useState('individual');
    const [registrationRows, setRegistrationRows] = useState(() =>
        Array(5).fill(null).map(() => createEmptyUserRow())
    );
    const [registrationLoading, setRegistrationLoading] = useState(false);
    const [registrationResult, setRegistrationResult] = useState(null);
    const fileInputRef = React.useRef(null);

    // 관리자 관련 state
    const [admins, setAdmins] = useState([]);
    const [showAdminRegistration, setShowAdminRegistration] = useState(false);
    const [adminRegistrationRows, setAdminRegistrationRows] = useState(() =>
        Array(3).fill(null).map(() => createEmptyUserRow())
    );
    const [adminRegistrationLoading, setAdminRegistrationLoading] = useState(false);
    const [adminRegistrationResult, setAdminRegistrationResult] = useState(null);

    // 차트 필터 state
    const currentDate = new Date();
    const [dailyChartYear, setDailyChartYear] = useState(currentDate.getFullYear());
    const [dailyChartMonth, setDailyChartMonth] = useState(currentDate.getMonth() + 1);
    const [monthlyChartYear, setMonthlyChartYear] = useState(currentDate.getFullYear());

    // 년도 옵션 생성 (현재 년도 - 2년부터 현재 년도까지)
    const yearOptions = Array.from({ length: 3 }, (_, i) => currentDate.getFullYear() - 2 + i);
    const monthOptions = Array.from({ length: 12 }, (_, i) => i + 1);

    useEffect(() => {
        loadDashboardData();
    }, []);

    const loadDashboardData = async () => {
        setLoading(true);
        setError(null);
        try {
            const [dashData, patternsData, costsData, usersData, adminsData] = await Promise.all([
                getTenantAdminDashboard(),
                getTenantUsagePatterns(30),
                getTenantCosts(30),
                getTenantUsers(1, 50),
                getTenantAdmins()
            ]);
            setDashboard(dashData);
            setPatterns(patternsData);
            setCosts(costsData);
            setUsers(usersData.users || []);
            setAdmins(adminsData.admins || []);
        } catch (err) {
            setError(err.message);
            if (err.message.includes('인증') || err.message.includes('권한')) {
                // 권한 없으면 메인으로 돌아가기
            }
        } finally {
            setLoading(false);
        }
    };

    const handleUserAction = async (userId, action) => {
        try {
            if (action === 'suspend') {
                await suspendTenantUser(userId);
            } else {
                await activateTenantUser(userId);
            }
            const usersData = await getTenantUsers(1, 50);
            setUsers(usersData.users || []);
        } catch (err) {
            alert(err.message);
        }
    };

    const handleDeleteUser = async (userId, username) => {
        if (!window.confirm(`정말 "${username}" 사용자를 삭제하시겠습니까?\n\n이 작업은 되돌릴 수 없으며, 사용자의 모든 데이터(세션, 메시지 등)가 삭제됩니다.`)) {
            return;
        }
        try {
            await deleteTenantUser(userId);
            alert('사용자가 삭제되었습니다.');
            const usersData = await getTenantUsers(1, 50);
            setUsers(usersData.users || []);
        } catch (err) {
            alert(err.message);
        }
    };

    const searchUsers = async () => {
        try {
            const usersData = await getTenantUsers(1, 50, searchQuery);
            setUsers(usersData.users || []);
        } catch (err) {
            alert(err.message);
        }
    };

    // 사용자 등록 함수들
    const handleRegistrationRowChange = (rowId, field, value) => {
        setRegistrationRows(rows =>
            rows.map(row => row.id === rowId ? { ...row, [field]: value } : row)
        );
    };

    const addRegistrationRow = () => {
        setRegistrationRows(rows => [...rows, createEmptyUserRow()]);
    };

    const removeRegistrationRow = (rowId) => {
        if (registrationRows.length <= 1) return;
        setRegistrationRows(rows => rows.filter(row => row.id !== rowId));
    };

    const resetRegistrationForm = () => {
        setRegistrationRows(Array(5).fill(null).map(() => createEmptyUserRow()));
        setRegistrationResult(null);
    };

    const handleIndividualSubmit = async () => {
        const validRows = registrationRows.filter(row => row.username && row.password);
        if (validRows.length === 0) {
            alert('사용자명과 비밀번호를 입력해주세요');
            return;
        }

        setRegistrationLoading(true);
        setRegistrationResult(null);

        try {
            const users = validRows.map(row => ({
                username: row.username,
                password: row.password,
                display_name: row.displayName || null,
                email: row.email || null
            }));

            const result = await createTenantUsersBulk(users);
            setRegistrationResult(result);

            if (result.created && result.created.length > 0) {
                resetRegistrationForm();
                const usersData = await getTenantUsers(1, 50);
                setUsers(usersData.users || []);
            }
        } catch (err) {
            setRegistrationResult({ errors: [{ error: err.message }] });
        } finally {
            setRegistrationLoading(false);
        }
    };

    const downloadTemplate = () => {
        const headers = ['username', 'password', 'display_name', 'email'];
        const sampleData = [
            ['user1', 'password123', '홍길동', 'user1@example.com'],
            ['user2', 'password456', '김철수', 'user2@example.com']
        ];

        const ws = XLSX.utils.aoa_to_sheet([headers, ...sampleData]);
        ws['!cols'] = [{ wch: 15 }, { wch: 15 }, { wch: 15 }, { wch: 30 }];
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, '사용자 등록');
        XLSX.writeFile(wb, 'user_registration_template.xlsx');
    };

    const handleBulkUpload = async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setRegistrationLoading(true);
        setRegistrationResult(null);

        try {
            let rows = [];
            const fileName = file.name.toLowerCase();

            if (fileName.endsWith('.xlsx') || fileName.endsWith('.xls')) {
                const arrayBuffer = await file.arrayBuffer();
                const workbook = XLSX.read(arrayBuffer, { type: 'array' });
                const sheetName = workbook.SheetNames[0];
                const worksheet = workbook.Sheets[sheetName];
                rows = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
            } else {
                const text = await file.text();
                rows = text.split('\n').filter(line => line.trim()).map(line => line.split(',').map(v => v.trim()));
            }

            if (rows.length < 2) {
                throw new Error('파일에 데이터가 없습니다');
            }

            const headers = rows[0].map(h => String(h || '').trim().toLowerCase());
            const usernameIdx = headers.findIndex(h => h === 'username' || h === '사용자명');
            const passwordIdx = headers.findIndex(h => h === 'password' || h === '비밀번호');
            const displayNameIdx = headers.findIndex(h => h === 'display_name' || h === '이름');
            const emailIdx = headers.findIndex(h => h === 'email' || h === '이메일');

            if (usernameIdx === -1 || passwordIdx === -1) {
                throw new Error('username과 password 컬럼이 필요합니다');
            }

            const users = [];
            for (let i = 1; i < rows.length; i++) {
                const values = rows[i].map(v => String(v || '').trim());
                if (values[usernameIdx] && values[passwordIdx]) {
                    users.push({
                        username: values[usernameIdx],
                        password: values[passwordIdx],
                        display_name: displayNameIdx >= 0 ? values[displayNameIdx] : null,
                        email: emailIdx >= 0 ? values[emailIdx] : null
                    });
                }
            }

            if (users.length === 0) {
                throw new Error('유효한 사용자 데이터가 없습니다');
            }

            const result = await createTenantUsersBulk(users);
            setRegistrationResult(result);

            if (result.created && result.created.length > 0) {
                const usersData = await getTenantUsers(1, 50);
                setUsers(usersData.users || []);
            }
        } catch (err) {
            setRegistrationResult({ errors: [{ error: err.message }] });
        } finally {
            setRegistrationLoading(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    // 관리자 등록 함수들
    const handleAdminRowChange = (rowId, field, value) => {
        setAdminRegistrationRows(rows =>
            rows.map(row => row.id === rowId ? { ...row, [field]: value } : row)
        );
    };

    const addAdminRow = () => {
        setAdminRegistrationRows(rows => [...rows, createEmptyUserRow()]);
    };

    const removeAdminRow = (rowId) => {
        if (adminRegistrationRows.length <= 1) return;
        setAdminRegistrationRows(rows => rows.filter(row => row.id !== rowId));
    };

    const resetAdminForm = () => {
        setAdminRegistrationRows(Array(3).fill(null).map(() => createEmptyUserRow()));
        setAdminRegistrationResult(null);
    };

    const handleAdminSubmit = async () => {
        const validRows = adminRegistrationRows.filter(row => row.username && row.password);
        if (validRows.length === 0) {
            alert('사용자명과 비밀번호를 입력해주세요');
            return;
        }

        setAdminRegistrationLoading(true);
        setAdminRegistrationResult(null);

        const results = { created: [], errors: [] };

        for (const row of validRows) {
            try {
                const result = await createTenantAdmin({
                    username: row.username,
                    password: row.password,
                    display_name: row.displayName || null,
                    email: row.email || null
                });
                results.created.push({ username: row.username, user_id: result.user_id });
            } catch (err) {
                results.errors.push({ username: row.username, error: err.message });
            }
        }

        results.message = `${results.created.length}명 생성 완료, ${results.errors.length}명 실패`;
        setAdminRegistrationResult(results);

        if (results.created.length > 0) {
            resetAdminForm();
            const adminsData = await getTenantAdmins();
            setAdmins(adminsData.admins || []);
        }

        setAdminRegistrationLoading(false);
    };

    const handleDeleteAdmin = async (userId, username) => {
        if (!window.confirm(`정말 "${username}" 관리자를 삭제하시겠습니까?\n\n관리자 권한이 제거되고 사용자 계정도 함께 삭제됩니다.`)) {
            return;
        }
        try {
            await deleteTenantUser(userId);
            alert('관리자가 삭제되었습니다.');
            const adminsData = await getTenantAdmins();
            setAdmins(adminsData.admins || []);
        } catch (err) {
            alert(err.message);
        }
    };

    const formatNumber = (num) => {
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return num?.toString() || '0';
    };

    const formatCurrency = (amount) => {
        const val = amount || 0;
        if (val === 0) return '$0.00';
        if (val < 0.01) return '$' + val.toFixed(4);
        if (val < 1) return '$' + val.toFixed(3);
        return '$' + val.toFixed(2);
    };

    const formatDate = (dateStr) => {
        if (!dateStr) return '-';
        return new Date(dateStr).toLocaleDateString('ko-KR');
    };

    const formatDateTime = (dateStr) => {
        if (!dateStr) return '-';
        return new Date(dateStr).toLocaleString('ko-KR');
    };

    if (loading) {
        return (
            <div className="admin-dashboard loading">
                <div className="loading-spinner"></div>
                <p>대시보드 로딩 중...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="admin-dashboard error">
                <h2>오류 발생</h2>
                <p>{error}</p>
                <button onClick={loadDashboardData} className="btn btn-primary">다시 시도</button>
                <button onClick={onLogout} className="btn btn-secondary">로그아웃</button>
            </div>
        );
    }

    return (
        <div className="admin-dashboard tenant-admin">
            {/* 헤더 */}
            <header className="dashboard-header">
                <div className="header-left">
                    <h1>관리자 대시보드</h1>
                </div>
                <div className="header-right">
                    <span className="user-info">{currentUser?.display_name || currentUser?.username}</span>
                    <button onClick={loadDashboardData} className="btn btn-secondary">
                        새로고침
                    </button>
                    <button onClick={onLogout} className="btn btn-danger">
                        로그아웃
                    </button>
                </div>
            </header>

            {/* 탭 네비게이션 */}
            <nav className="dashboard-tabs">
                <button
                    className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
                    onClick={() => setActiveTab('overview')}
                >
                    개요
                </button>
                <button
                    className={`tab ${activeTab === 'users' ? 'active' : ''}`}
                    onClick={() => setActiveTab('users')}
                >
                    사용자 관리
                </button>
                <button
                    className={`tab ${activeTab === 'data' ? 'active' : ''}`}
                    onClick={() => setActiveTab('data')}
                >
                    데이터 관리
                </button>
            </nav>

            {/* 메인 컨텐츠 */}
            <main className="dashboard-content">
                {activeTab === 'overview' && dashboard && (
                    <div className="overview-section">
                        {/* 주요 지표 카드 - DAU, WAU, MAU, 월 예상 비용 */}
                        <div className="stats-cards">
                            <div className="stat-card users">
                                <div className="stat-icon">👤</div>
                                <div className="stat-info">
                                    <h3>일 사용자</h3>
                                    <div className="stat-value">
                                        {dashboard.daily_trend?.length > 0
                                            ? dashboard.daily_trend[dashboard.daily_trend.length - 1]?.users || 0
                                            : 0}
                                    </div>
                                    <div className="stat-detail">DAU (중복 제외)</div>
                                </div>
                            </div>

                            <div className="stat-card users">
                                <div className="stat-icon">👥</div>
                                <div className="stat-info">
                                    <h3>주 사용자</h3>
                                    <div className="stat-value">
                                        {dashboard.daily_trend?.length > 0
                                            ? dashboard.daily_trend.slice(-7).reduce((sum, d) => sum + (d.users || 0), 0)
                                            : 0}
                                    </div>
                                    <div className="stat-detail">WAU (7일 DAU 합계)</div>
                                </div>
                            </div>

                            <div className="stat-card users">
                                <div className="stat-icon">👨‍👩‍👧‍👦</div>
                                <div className="stat-info">
                                    <h3>월 사용자</h3>
                                    <div className="stat-value">
                                        {dashboard.daily_trend?.length > 0
                                            ? dashboard.daily_trend.reduce((sum, d) => sum + (d.users || 0), 0)
                                            : 0}
                                    </div>
                                    <div className="stat-detail">MAU (30일 DAU 합계)</div>
                                </div>
                            </div>

                            <div className="stat-card cost">
                                <div className="stat-icon">💰</div>
                                <div className="stat-info">
                                    <h3>월 예상 비용</h3>
                                    <div className="stat-value">{formatCurrency(costs?.estimated_cost_usd || 0)}</div>
                                    <div className="stat-detail">30일 기준</div>
                                </div>
                            </div>
                        </div>

                        {/* 일 사용자수 추이 - Recharts */}
                        {dashboard.daily_trend && dashboard.daily_trend.length > 0 && (
                            <div className="section-card">
                                <div className="chart-header">
                                    <h3>일 사용자수 추이</h3>
                                    <div className="chart-filters">
                                        <select
                                            value={dailyChartYear}
                                            onChange={(e) => setDailyChartYear(Number(e.target.value))}
                                            className="chart-select"
                                        >
                                            {yearOptions.map(year => (
                                                <option key={year} value={year}>{year}년</option>
                                            ))}
                                        </select>
                                        <select
                                            value={dailyChartMonth}
                                            onChange={(e) => setDailyChartMonth(Number(e.target.value))}
                                            className="chart-select"
                                        >
                                            {monthOptions.map(month => (
                                                <option key={month} value={month}>{month}월</option>
                                            ))}
                                        </select>
                                    </div>
                                </div>
                                <div className="recharts-wrapper">
                                    <ResponsiveContainer width="100%" height={280}>
                                        <LineChart data={dashboard.daily_trend
                                            .filter(d => {
                                                const date = new Date(d.date);
                                                return date.getFullYear() === dailyChartYear &&
                                                       (date.getMonth() + 1) === dailyChartMonth;
                                            })
                                            .map(d => ({
                                                date: d.date.slice(8),
                                                사용자수: d.users || 0
                                            }))}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                            <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="#6b7280" />
                                            <YAxis tick={{ fontSize: 11 }} stroke="#6b7280" />
                                            <Tooltip
                                                contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                                                labelStyle={{ fontWeight: 'bold' }}
                                            />
                                            <Legend />
                                            <Line type="monotone" dataKey="사용자수" stroke="#5e35b1" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 6 }} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}

                        {/* 월 사용자수 추이 - Recharts */}
                        {dashboard.monthly_user_trend && dashboard.monthly_user_trend.length > 0 && (
                            <div className="section-card">
                                <div className="chart-header">
                                    <h3>월 사용자수 추이</h3>
                                    <div className="chart-filters">
                                        <select
                                            value={monthlyChartYear}
                                            onChange={(e) => setMonthlyChartYear(Number(e.target.value))}
                                            className="chart-select"
                                        >
                                            {yearOptions.map(year => (
                                                <option key={year} value={year}>{year}년</option>
                                            ))}
                                        </select>
                                    </div>
                                </div>
                                <div className="recharts-wrapper">
                                    <ResponsiveContainer width="100%" height={280}>
                                        <LineChart data={dashboard.monthly_user_trend
                                            .filter(d => {
                                                const year = parseInt(d.month.slice(0, 4), 10);
                                                return year === monthlyChartYear;
                                            })
                                            .map(d => ({
                                                월: d.month.slice(5) + '월',
                                                사용자수: d.users || 0
                                            }))}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                            <XAxis dataKey="월" tick={{ fontSize: 11 }} stroke="#6b7280" />
                                            <YAxis tick={{ fontSize: 11 }} stroke="#6b7280" />
                                            <Tooltip
                                                contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                                                labelStyle={{ fontWeight: 'bold' }}
                                            />
                                            <Legend />
                                            <Line type="monotone" dataKey="사용자수" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 6 }} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}

                        {/* 월별 비용 추이 (12개월) - Recharts BarChart */}
                        {costs?.monthly_costs && costs.monthly_costs.length > 0 && (
                            <div className="section-card">
                                <h3>월별 비용 추이 (12개월)</h3>
                                <div className="recharts-wrapper">
                                    <ResponsiveContainer width="100%" height={280}>
                                        <BarChart data={costs.monthly_costs.map(d => ({
                                            월: d.month.slice(2),
                                            비용: d.cost_usd || 0
                                        }))}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                            <XAxis dataKey="월" tick={{ fontSize: 11 }} stroke="#6b7280" />
                                            <YAxis
                                                tick={{ fontSize: 11 }}
                                                stroke="#6b7280"
                                                tickFormatter={(value) => `$${value.toFixed(3)}`}
                                            />
                                            <Tooltip
                                                formatter={(value) => [`$${Number(value).toFixed(4)}`, '비용']}
                                                contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                                                labelStyle={{ fontWeight: 'bold' }}
                                            />
                                            <Legend />
                                            <Bar dataKey="비용" fill="#5e35b1" radius={[4, 4, 0, 0]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                                <div className="chart-stats-inline">
                                    <span>최고: {formatCurrency(Math.max(...(costs.monthly_costs?.map(d => d.cost_usd) || [0])))}</span>
                                    <span>평균: {formatCurrency((costs.monthly_costs?.reduce((sum, d) => sum + (d.cost_usd || 0), 0) || 0) / Math.max(costs.monthly_costs?.length || 1, 1))}</span>
                                </div>
                            </div>
                        )}

                        {/* 요일별 분포 - Recharts Bar */}
                        {patterns?.weekday_distribution && patterns.weekday_distribution.length > 0 && (
                            <div className="section-card">
                                <h3>요일별 분포</h3>
                                <div className="recharts-wrapper">
                                    <ResponsiveContainer width="100%" height={280}>
                                        <BarChart data={['월', '화', '수', '목', '금', '토', '일'].map((day, idx) => ({
                                            요일: day,
                                            메시지: patterns.weekday_distribution[idx] || 0,
                                            fill: idx >= 5 ? '#f59e0b' : '#3b82f6'
                                        }))}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                            <XAxis dataKey="요일" tick={{ fontSize: 12 }} stroke="#6b7280" />
                                            <YAxis tick={{ fontSize: 11 }} stroke="#6b7280" />
                                            <Tooltip
                                                contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                                                labelStyle={{ fontWeight: 'bold' }}
                                            />
                                            <Legend />
                                            <Bar dataKey="메시지" radius={[4, 4, 0, 0]}>
                                                {['월', '화', '수', '목', '금', '토', '일'].map((day, idx) => (
                                                    <Cell key={day} fill={idx >= 5 ? '#f59e0b' : '#3b82f6'} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}

                    </div>
                )}

                {activeTab === 'data' && (
                    <DataManagement onLogout={onLogout} embedded={true} />
                )}

                {activeTab === 'users' && (
                    <div className="users-management-section">
                        {/* 사용자 등록 영역 */}
                        <div className="section-card user-registration-section">
                            <div className="registration-header">
                                <h3>사용자 등록</h3>
                                <div className="registration-actions">
                                    <button
                                        className={`btn ${showAdminRegistration ? 'btn-primary' : 'btn-outline'}`}
                                        onClick={() => {
                                            setShowAdminRegistration(true);
                                            setShowRegistration(false);
                                            setAdminRegistrationResult(null);
                                        }}
                                    >
                                        관리자 등록
                                    </button>
                                    <button
                                        className={`btn ${showRegistration && registrationMode === 'individual' ? 'btn-primary' : 'btn-outline'}`}
                                        onClick={() => {
                                            setShowRegistration(true);
                                            setShowAdminRegistration(false);
                                            setRegistrationMode('individual');
                                            setRegistrationResult(null);
                                        }}
                                    >
                                        개별 등록
                                    </button>
                                    <button
                                        className="btn btn-outline"
                                        onClick={() => fileInputRef.current?.click()}
                                    >
                                        일괄 등록
                                    </button>
                                    <button
                                        className="btn btn-outline"
                                        onClick={downloadTemplate}
                                    >
                                        양식 다운로드
                                    </button>
                                    <input
                                        type="file"
                                        ref={fileInputRef}
                                        style={{ display: 'none' }}
                                        accept=".csv,.xlsx,.xls"
                                        onChange={handleBulkUpload}
                                    />
                                </div>
                            </div>

                            {/* 등록 결과 메시지 */}
                            {registrationResult && (
                                <div className={`registration-result ${registrationResult.created?.length > 0 ? 'success' : 'error'}`}>
                                    {registrationResult.message && <p>{registrationResult.message}</p>}
                                    {registrationResult.errors?.length > 0 && (
                                        <ul className="error-list">
                                            {registrationResult.errors.map((err, idx) => (
                                                <li key={idx}>{err.username ? `${err.username}: ` : ''}{err.error}</li>
                                            ))}
                                        </ul>
                                    )}
                                </div>
                            )}

                            {/* 개별 등록 폼 */}
                            {showRegistration && registrationMode === 'individual' && (
                                <div className="individual-registration-form">
                                    <table className="registration-table">
                                        <thead>
                                            <tr>
                                                <th>사용자명 *</th>
                                                <th>비밀번호 *</th>
                                                <th>이름</th>
                                                <th>이메일</th>
                                                <th></th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {registrationRows.map((row) => (
                                                <tr key={row.id}>
                                                    <td>
                                                        <input
                                                            type="text"
                                                            value={row.username}
                                                            onChange={(e) => handleRegistrationRowChange(row.id, 'username', e.target.value)}
                                                            placeholder="username"
                                                        />
                                                    </td>
                                                    <td>
                                                        <input
                                                            type="password"
                                                            value={row.password}
                                                            onChange={(e) => handleRegistrationRowChange(row.id, 'password', e.target.value)}
                                                            placeholder="password"
                                                        />
                                                    </td>
                                                    <td>
                                                        <input
                                                            type="text"
                                                            value={row.displayName}
                                                            onChange={(e) => handleRegistrationRowChange(row.id, 'displayName', e.target.value)}
                                                            placeholder="홍길동"
                                                        />
                                                    </td>
                                                    <td>
                                                        <input
                                                            type="email"
                                                            value={row.email}
                                                            onChange={(e) => handleRegistrationRowChange(row.id, 'email', e.target.value)}
                                                            placeholder="user@example.com"
                                                        />
                                                    </td>
                                                    <td>
                                                        <button
                                                            className="btn-icon delete"
                                                            onClick={() => removeRegistrationRow(row.id)}
                                                            disabled={registrationRows.length <= 1}
                                                            title="삭제"
                                                        >
                                                            🗑️
                                                        </button>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>

                                    <div className="registration-form-actions">
                                        <button className="btn btn-outline" onClick={addRegistrationRow}>
                                            + 항목 추가
                                        </button>
                                        <div className="form-buttons">
                                            <button
                                                className="btn btn-outline"
                                                onClick={() => {
                                                    setShowRegistration(false);
                                                    resetRegistrationForm();
                                                }}
                                            >
                                                취소
                                            </button>
                                            <button
                                                className="btn btn-primary"
                                                onClick={handleIndividualSubmit}
                                                disabled={registrationLoading}
                                            >
                                                {registrationLoading ? '처리 중...' : '등록'}
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* 관리자 등록 폼 */}
                            {showAdminRegistration && (
                                <div className="individual-registration-form admin-registration-form">
                                    {adminRegistrationResult && (
                                        <div className={`registration-result ${adminRegistrationResult.created?.length > 0 ? 'success' : 'error'}`}>
                                            {adminRegistrationResult.message && <p>{adminRegistrationResult.message}</p>}
                                            {adminRegistrationResult.errors?.length > 0 && (
                                                <ul className="error-list">
                                                    {adminRegistrationResult.errors.map((err, idx) => (
                                                        <li key={idx}>{err.username ? `${err.username}: ` : ''}{err.error}</li>
                                                    ))}
                                                </ul>
                                            )}
                                        </div>
                                    )}
                                    <table className="registration-table">
                                        <thead>
                                            <tr>
                                                <th>관리자명 *</th>
                                                <th>비밀번호 *</th>
                                                <th>이름</th>
                                                <th>이메일</th>
                                                <th></th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {adminRegistrationRows.map((row) => (
                                                <tr key={row.id}>
                                                    <td>
                                                        <input
                                                            type="text"
                                                            value={row.username}
                                                            onChange={(e) => handleAdminRowChange(row.id, 'username', e.target.value)}
                                                            placeholder="admin_username"
                                                        />
                                                    </td>
                                                    <td>
                                                        <input
                                                            type="password"
                                                            value={row.password}
                                                            onChange={(e) => handleAdminRowChange(row.id, 'password', e.target.value)}
                                                            placeholder="password"
                                                        />
                                                    </td>
                                                    <td>
                                                        <input
                                                            type="text"
                                                            value={row.displayName}
                                                            onChange={(e) => handleAdminRowChange(row.id, 'displayName', e.target.value)}
                                                            placeholder="관리자 이름"
                                                        />
                                                    </td>
                                                    <td>
                                                        <input
                                                            type="email"
                                                            value={row.email}
                                                            onChange={(e) => handleAdminRowChange(row.id, 'email', e.target.value)}
                                                            placeholder="admin@example.com"
                                                        />
                                                    </td>
                                                    <td>
                                                        <button
                                                            className="btn-icon delete"
                                                            onClick={() => removeAdminRow(row.id)}
                                                            disabled={adminRegistrationRows.length <= 1}
                                                            title="삭제"
                                                        >
                                                            🗑️
                                                        </button>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>

                                    <div className="registration-form-actions">
                                        <button className="btn btn-outline" onClick={addAdminRow}>
                                            + 항목 추가
                                        </button>
                                        <div className="form-buttons">
                                            <button
                                                className="btn btn-outline"
                                                onClick={() => {
                                                    setShowAdminRegistration(false);
                                                    resetAdminForm();
                                                }}
                                            >
                                                취소
                                            </button>
                                            <button
                                                className="btn btn-primary"
                                                onClick={handleAdminSubmit}
                                                disabled={adminRegistrationLoading}
                                            >
                                                {adminRegistrationLoading ? '처리 중...' : '관리자 등록'}
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* 검색 */}
                        <div className="search-bar">
                            <input
                                type="text"
                                placeholder="사용자 검색..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && searchUsers()}
                            />
                            <button onClick={searchUsers} className="btn btn-primary">검색</button>
                        </div>

                        {/* 사용자 목록 */}
                        <div className="section-card">
                            <h3>사용자 목록</h3>
                            <table className="data-table users-table">
                                <thead>
                                    <tr>
                                        <th>사용자명</th>
                                        <th>이름</th>
                                        <th>상태</th>
                                        <th>가입일</th>
                                        <th>마지막 로그인</th>
                                        <th>이번 달 비용</th>
                                        <th>작업</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {users.map(user => (
                                        <tr key={user.id} className={!user.is_active ? 'inactive' : ''}>
                                            <td><code>{user.username}</code></td>
                                            <td>{user.display_name || '-'}</td>
                                            <td>
                                                <span className={`status-badge ${user.is_active ? 'active' : 'suspended'}`}>
                                                    {user.is_active ? '활성' : '정지'}
                                                </span>
                                            </td>
                                            <td>{formatDate(user.created_at)}</td>
                                            <td>{formatDateTime(user.last_login)}</td>
                                            <td>{formatCurrency(user.monthly_cost_usd)}</td>
                                            <td className="action-buttons">
                                                {user.is_active ? (
                                                    <button
                                                        className="btn btn-sm btn-warning"
                                                        onClick={() => handleUserAction(user.id, 'suspend')}
                                                        disabled={user.id === currentUser?.id}
                                                    >
                                                        정지
                                                    </button>
                                                ) : (
                                                    <button
                                                        className="btn btn-sm btn-success"
                                                        onClick={() => handleUserAction(user.id, 'activate')}
                                                    >
                                                        활성화
                                                    </button>
                                                )}
                                                <button
                                                    className="btn btn-sm btn-danger"
                                                    onClick={() => handleDeleteUser(user.id, user.username)}
                                                    disabled={user.id === currentUser?.id}
                                                    title="사용자 삭제"
                                                >
                                                    삭제
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {/* 관리자 목록 */}
                        <div className="section-card">
                            <h3>관리자 목록</h3>
                            <table className="data-table users-table">
                                <thead>
                                    <tr>
                                        <th>관리자명</th>
                                        <th>이름</th>
                                        <th>이메일</th>
                                        <th>권한</th>
                                        <th>등록일</th>
                                        <th>작업</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {admins.map(admin => (
                                        <tr key={admin.user_id}>
                                            <td><code>{admin.username}</code></td>
                                            <td>{admin.display_name || '-'}</td>
                                            <td>{admin.email || '-'}</td>
                                            <td>
                                                <span className="status-badge admin">
                                                    {admin.role === 'admin' ? '관리자' : '모더레이터'}
                                                </span>
                                            </td>
                                            <td>{formatDate(admin.created_at)}</td>
                                            <td className="action-buttons">
                                                <button
                                                    className="btn btn-sm btn-danger"
                                                    onClick={() => handleDeleteAdmin(admin.user_id, admin.username)}
                                                    disabled={admin.user_id === currentUser?.id}
                                                    title="관리자 삭제"
                                                >
                                                    삭제
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                    {admins.length === 0 && (
                                        <tr>
                                            <td colSpan="6" style={{ textAlign: 'center', color: '#6b7280' }}>
                                                등록된 관리자가 없습니다
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}

export default AdminDashboard;
