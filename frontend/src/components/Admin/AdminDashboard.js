/**
 * 관리자 대시보드 (OSC 공개 버전)
 * 사용자 관리, 사용량 모니터링, 비용 분석
 */
import React, { useState, useEffect } from 'react';
import {
    getAdminDashboard,
    getUsagePatterns,
    getCosts,
    getUsers,
    suspendUser,
    activateUser,
    getAdmin
} from '../../api/adminApi';
import './AdminDashboard.css';

function AdminDashboard({ onLogout }) {
    const [dashboard, setDashboard] = useState(null);
    const [patterns, setPatterns] = useState(null);
    const [costs, setCosts] = useState(null);
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('overview');
    const [searchQuery, setSearchQuery] = useState('');
    const currentUser = getAdmin();

    useEffect(() => {
        loadDashboardData();
    }, []);

    const loadDashboardData = async () => {
        setLoading(true);
        setError(null);
        try {
            const [dashData, patternsData, costsData, usersData] = await Promise.all([
                getAdminDashboard(),
                getUsagePatterns(30),
                getCosts(30),
                getUsers(1, 50)
            ]);
            setDashboard(dashData);
            setPatterns(patternsData);
            setCosts(costsData);
            setUsers(usersData.users || []);
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
                await suspendUser(userId);
            } else {
                await activateUser(userId);
            }
            // 사용자 목록 새로고침
            const usersData = await getUsers(1, 50);
            setUsers(usersData.users || []);
        } catch (err) {
            alert(err.message);
        }
    };

    const searchUsers = async () => {
        try {
            const usersData = await getUsers(1, 50, searchQuery);
            setUsers(usersData.users || []);
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
        return '$' + (amount || 0).toFixed(2);
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
        <div className="admin-dashboard">
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
                    className={`tab ${activeTab === 'patterns' ? 'active' : ''}`}
                    onClick={() => setActiveTab('patterns')}
                >
                    사용 패턴
                </button>
                <button
                    className={`tab ${activeTab === 'costs' ? 'active' : ''}`}
                    onClick={() => setActiveTab('costs')}
                >
                    비용 분석
                </button>
            </nav>

            {/* 메인 컨텐츠 */}
            <main className="dashboard-content">
                {activeTab === 'overview' && dashboard && (
                    <div className="overview-section">
                        {/* 주요 지표 카드 */}
                        <div className="stats-cards">
                            <div className="stat-card users">
                                <div className="stat-icon">👥</div>
                                <div className="stat-info">
                                    <h3>사용자</h3>
                                    <div className="stat-value">{dashboard.users?.total || 0}</div>
                                    <div className="stat-detail">
                                        활성: {dashboard.users?.active || 0} |
                                        신규(7일): {dashboard.users?.new_this_week || 0}
                                    </div>
                                </div>
                            </div>

                            <div className="stat-card sessions">
                                <div className="stat-icon">💬</div>
                                <div className="stat-info">
                                    <h3>세션</h3>
                                    <div className="stat-value">{formatNumber(dashboard.usage?.total_sessions)}</div>
                                    <div className="stat-detail">
                                        오늘: {dashboard.usage?.sessions_today || 0}
                                    </div>
                                </div>
                            </div>

                            <div className="stat-card messages">
                                <div className="stat-icon">📝</div>
                                <div className="stat-info">
                                    <h3>메시지</h3>
                                    <div className="stat-value">{formatNumber(dashboard.usage?.total_messages)}</div>
                                    <div className="stat-detail">
                                        오늘: {dashboard.usage?.messages_today || 0}
                                    </div>
                                </div>
                            </div>

                            <div className="stat-card tokens">
                                <div className="stat-icon">🔤</div>
                                <div className="stat-info">
                                    <h3>토큰</h3>
                                    <div className="stat-value">{formatNumber(dashboard.usage?.total_tokens)}</div>
                                    <div className="stat-detail">
                                        오늘: {formatNumber(dashboard.usage?.tokens_today)}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* 활성 사용자 현황 */}
                        <div className="section-card">
                            <h3>사용자 활동 현황</h3>
                            <div className="user-activity-stats">
                                <div className="activity-item">
                                    <span className="label">7일 내 활성</span>
                                    <span className="value">{dashboard.users?.active_7_days || 0}명</span>
                                </div>
                                <div className="activity-item">
                                    <span className="label">30일 내 활성</span>
                                    <span className="value">{dashboard.users?.active_30_days || 0}명</span>
                                </div>
                                <div className="activity-item">
                                    <span className="label">관리자</span>
                                    <span className="value">{dashboard.users?.admin_count || 0}명</span>
                                </div>
                            </div>
                        </div>

                        {/* 최근 활동 사용자 */}
                        <div className="section-card">
                            <h3>최근 활동 사용자 (7일)</h3>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th>사용자</th>
                                        <th>세션</th>
                                        <th>메시지</th>
                                        <th>마지막 로그인</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {dashboard.recent_activity?.map((user, idx) => (
                                        <tr key={user.user_id}>
                                            <td>
                                                <strong>{user.display_name || user.username}</strong>
                                            </td>
                                            <td>{user.session_count}</td>
                                            <td>{user.message_count}</td>
                                            <td>{formatDateTime(user.last_login)}</td>
                                        </tr>
                                    ))}
                                    {(!dashboard.recent_activity || dashboard.recent_activity.length === 0) && (
                                        <tr>
                                            <td colSpan="4" className="no-data">최근 활동 데이터 없음</td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>

                        {/* 모델별 사용량 */}
                        <div className="section-card">
                            <h3>모델별 사용량 (30일)</h3>
                            <div className="model-usage-list">
                                {dashboard.model_usage?.map((model, idx) => (
                                    <div key={model.model} className="model-usage-item">
                                        <div className="model-name">{model.model}</div>
                                        <div className="model-bar">
                                            <div
                                                className="bar-fill"
                                                style={{
                                                    width: `${Math.min((model.count / Math.max(...dashboard.model_usage.map(m => m.count))) * 100, 100)}%`
                                                }}
                                            />
                                        </div>
                                        <div className="model-stats">
                                            {formatNumber(model.count)} 호출 | {formatNumber(model.tokens)} 토큰
                                        </div>
                                    </div>
                                ))}
                                {(!dashboard.model_usage || dashboard.model_usage.length === 0) && (
                                    <p className="no-data">모델 사용 데이터 없음</p>
                                )}
                            </div>
                        </div>

                        {/* 일별 추이 */}
                        {dashboard.daily_trend && dashboard.daily_trend.length > 0 && (
                            <div className="section-card">
                                <h3>일별 사용량 추이 (14일)</h3>
                                <div className="chart-placeholder">
                                    <div className="mini-chart">
                                        {dashboard.daily_trend.map((day, idx) => (
                                            <div
                                                key={day.date}
                                                className="chart-bar"
                                                style={{
                                                    height: `${Math.min((day.messages / Math.max(...dashboard.daily_trend.map(d => d.messages || 1))) * 100, 100)}%`
                                                }}
                                                title={`${day.date}: ${day.messages} 메시지, ${day.users} 사용자`}
                                            />
                                        ))}
                                    </div>
                                    <div className="chart-labels">
                                        {dashboard.daily_trend.filter((_, i) => i % 3 === 0).map(day => (
                                            <span key={day.date}>{day.date.slice(5)}</span>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {activeTab === 'users' && (
                    <div className="users-management-section">
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
                                        <th>ID</th>
                                        <th>사용자명</th>
                                        <th>이름</th>
                                        <th>역할</th>
                                        <th>상태</th>
                                        <th>가입일</th>
                                        <th>마지막 로그인</th>
                                        <th>작업</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {users.map(user => (
                                        <tr key={user.id} className={!user.is_active ? 'inactive' : ''}>
                                            <td>{user.id}</td>
                                            <td><code>{user.username}</code></td>
                                            <td>{user.display_name || '-'}</td>
                                            <td>
                                                <span className={`role-badge ${user.role}`}>
                                                    {user.role}
                                                </span>
                                            </td>
                                            <td>
                                                <span className={`status-badge ${user.is_active ? 'active' : 'suspended'}`}>
                                                    {user.is_active ? '활성' : '정지'}
                                                </span>
                                            </td>
                                            <td>{formatDate(user.created_at)}</td>
                                            <td>{formatDateTime(user.last_login)}</td>
                                            <td>
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
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {activeTab === 'patterns' && patterns && (
                    <div className="patterns-section">
                        {/* 시간대별 분포 */}
                        <div className="section-card">
                            <h3>시간대별 사용량</h3>
                            <div className="hourly-chart">
                                {patterns.hourly_distribution?.map((count, hour) => (
                                    <div key={hour} className="hour-bar-container">
                                        <div
                                            className="hour-bar"
                                            style={{
                                                height: `${Math.min((count / Math.max(...patterns.hourly_distribution.filter(c => c > 0), 1)) * 100, 100) || 0}%`
                                            }}
                                            title={`${hour}시: ${count}건`}
                                        />
                                        <span className="hour-label">{hour}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* 요일별 분포 */}
                        <div className="section-card">
                            <h3>요일별 사용량</h3>
                            <div className="weekday-chart">
                                {['월', '화', '수', '목', '금', '토', '일'].map((day, idx) => (
                                    <div key={day} className="weekday-bar-container">
                                        <div
                                            className="weekday-bar"
                                            style={{
                                                height: `${Math.min((patterns.weekday_distribution?.[idx] / Math.max(...patterns.weekday_distribution.filter(c => c > 0), 1)) * 100, 100) || 0}%`
                                            }}
                                            title={`${day}: ${patterns.weekday_distribution?.[idx] || 0}건`}
                                        />
                                        <span className="weekday-label">{day}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* 사용자별 성향 분석 */}
                        <div className="section-card">
                            <h3>사용자별 성향 분석</h3>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th>사용자</th>
                                        <th>세션</th>
                                        <th>메시지</th>
                                        <th>선호 모델</th>
                                        <th>평균 사용 시간</th>
                                        <th>참여도</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {patterns.user_patterns?.map((user, idx) => (
                                        <tr key={user.user_id}>
                                            <td>
                                                <strong>{user.display_name || user.username}</strong>
                                            </td>
                                            <td>{user.session_count}</td>
                                            <td>{user.message_count}</td>
                                            <td>{user.preferred_model || '-'}</td>
                                            <td>{Math.round(user.avg_usage_hour)}시</td>
                                            <td>
                                                <span className={`engagement-badge ${user.engagement}`}>
                                                    {user.engagement === 'high' ? '높음' :
                                                     user.engagement === 'medium' ? '보통' : '낮음'}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {/* 모델 선호도 */}
                        <div className="section-card">
                            <h3>모델 선호도</h3>
                            <div className="model-preferences">
                                {patterns.model_preferences?.map((model, idx) => (
                                    <div key={model.model} className="model-pref-item">
                                        <div className="model-info">
                                            <span className="model-name">{model.model}</span>
                                            <span className="model-users">{model.unique_users}명 사용</span>
                                        </div>
                                        <div className="model-stats">
                                            <span>{formatNumber(model.usage_count)} 호출</span>
                                            <span>{formatNumber(model.total_tokens)} 토큰</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* 통계 요약 */}
                        <div className="stats-summary">
                            <div className="summary-item">
                                <span className="label">평균 세션당 메시지</span>
                                <span className="value">{patterns.avg_session_length || 0}</span>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'costs' && costs && (
                    <div className="costs-section">
                        {/* 비용 요약 */}
                        <div className="cost-summary">
                            <div className="cost-total">
                                <h3>총 예상 비용 (30일)</h3>
                                <div className="cost-value">{formatCurrency(costs.estimated_cost_usd)}</div>
                                <div className="cost-tokens">
                                    <span>입력: {formatNumber(costs.total_input_tokens || 0)}</span>
                                    <span> | </span>
                                    <span>출력: {formatNumber(costs.total_output_tokens || 0)}</span>
                                    <span> | </span>
                                    <span>총: {formatNumber(costs.total_tokens || 0)} 토큰</span>
                                </div>
                            </div>
                            <div className="cost-note">
                                <small>* 비용은 모델별 공식 가격 기준으로 계산됩니다 (로컬 모델은 무료)</small>
                            </div>
                        </div>

                        {/* 모델별 비용 */}
                        <div className="section-card">
                            <h3>모델별 비용</h3>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th>모델</th>
                                        <th>메시지</th>
                                        <th>입력 토큰</th>
                                        <th>출력 토큰</th>
                                        <th>비용</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {costs.by_model?.map(model => (
                                        <tr key={model.model} className={model.is_free ? 'free-model' : ''}>
                                            <td>
                                                <strong>{model.display_name || model.model}</strong>
                                                {model.is_free && <span className="free-badge">무료</span>}
                                            </td>
                                            <td>{formatNumber(model.message_count)}</td>
                                            <td>{formatNumber(model.input_tokens || 0)}</td>
                                            <td>{formatNumber(model.output_tokens || 0)}</td>
                                            <td>{formatCurrency(model.cost_usd)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {/* 사용자별 비용 TOP 10 */}
                        <div className="section-card">
                            <h3>사용자별 비용 TOP 10</h3>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th>사용자</th>
                                        <th>메시지</th>
                                        <th>입력 토큰</th>
                                        <th>출력 토큰</th>
                                        <th>예상 비용</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {costs.by_user?.map(user => (
                                        <tr key={user.user_id}>
                                            <td>
                                                <strong>{user.display_name || user.username}</strong>
                                                {user.models_used?.length > 0 && (
                                                    <small className="models-used">
                                                        {user.models_used.slice(0, 2).join(', ')}
                                                        {user.models_used.length > 2 && ` 외 ${user.models_used.length - 2}개`}
                                                    </small>
                                                )}
                                            </td>
                                            <td>{user.message_count}</td>
                                            <td>{formatNumber(user.input_tokens || 0)}</td>
                                            <td>{formatNumber(user.output_tokens || 0)}</td>
                                            <td>{formatCurrency(user.estimated_cost_usd)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {/* 일별 비용 추이 */}
                        <div className="section-card">
                            <h3>일별 비용 추이</h3>
                            <div className="daily-costs">
                                {costs.daily_costs?.slice(-14).map(day => (
                                    <div key={day.date} className="daily-cost-item">
                                        <span className="date">{day.date.slice(5)}</span>
                                        <span className="cost">{formatCurrency(day.cost_usd)}</span>
                                        <span className="tokens">{formatNumber(day.total_tokens || 0)} 토큰</span>
                                        <span className="messages">{day.messages}건</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}

export default AdminDashboard;
