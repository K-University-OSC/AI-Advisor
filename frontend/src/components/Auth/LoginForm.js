/**
 * 로그인/회원가입 폼 컴포넌트 (OSC 공개 버전)
 */
import React, { useState } from 'react';
import { login, signup } from '../../api/authApi';
import './Auth.css';

function LoginForm({ onLoginSuccess }) {
    const [isLogin, setIsLogin] = useState(true);
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [displayName, setDisplayName] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            if (isLogin) {
                const data = await login(username, password);
                onLoginSuccess(data.user);
            } else {
                if (password.length < 4) {
                    setError('비밀번호는 4자 이상이어야 합니다');
                    setIsLoading(false);
                    return;
                }
                const data = await signup(username, password, displayName || username);
                onLoginSuccess(data.user);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const toggleMode = () => {
        setIsLogin(!isLogin);
        setError('');
    };

    return (
        <div className="auth-container">
            <div className="auth-box">
                <h1 className="auth-title">Advisor OSC</h1>
                <p className="auth-subtitle">
                    {isLogin ? '로그인하여 대화를 시작하세요' : '새 계정을 만드세요'}
                </p>

                <form onSubmit={handleSubmit} className="auth-form">
                    <div className="form-group">
                        <label htmlFor="username">사용자 ID</label>
                        <input
                            id="username"
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            placeholder="아이디를 입력하세요"
                            required
                            minLength={3}
                            disabled={isLoading}
                        />
                    </div>

                    <div className="form-group">
                        <label htmlFor="password">비밀번호</label>
                        <input
                            id="password"
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="비밀번호를 입력하세요"
                            required
                            minLength={4}
                            disabled={isLoading}
                        />
                    </div>

                    {!isLogin && (
                        <div className="form-group">
                            <label htmlFor="displayName">표시 이름 (선택)</label>
                            <input
                                id="displayName"
                                type="text"
                                value={displayName}
                                onChange={(e) => setDisplayName(e.target.value)}
                                placeholder="채팅에 표시될 이름"
                                disabled={isLoading}
                            />
                        </div>
                    )}

                    {error && <div className="auth-error">{error}</div>}

                    <button
                        type="submit"
                        className="auth-button"
                        disabled={isLoading}
                    >
                        {isLoading ? '처리 중...' : (isLogin ? '로그인' : '회원가입')}
                    </button>
                </form>

                <div className="auth-toggle">
                    {isLogin ? (
                        <p>계정이 없으신가요? <button onClick={toggleMode}>회원가입</button></p>
                    ) : (
                        <p>이미 계정이 있으신가요? <button onClick={toggleMode}>로그인</button></p>
                    )}
                </div>
            </div>
        </div>
    );
}

export default LoginForm;
