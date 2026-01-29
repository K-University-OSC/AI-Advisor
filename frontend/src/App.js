/**
 * Multi-LLM Chatbot - Gemini 스타일 다크 테마 UI (OSC 공개 버전)
 */
import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';

// Lucide Icons
import {
    Send, Plus, MessageSquare, Menu, Settings, HelpCircle, History,
    User, Copy, ThumbsUp, ThumbsDown, RotateCcw, MoreVertical,
    Paperclip, X, LogOut, Check, ChevronDown, Trash2, Edit3, Square,
    Search, Brain, BookOpen, Globe, PenTool
} from 'lucide-react';

import { API_BASE_URL, LLM_MODELS, DEFAULT_MODEL } from './Config';
import { getToken, getUser, logout, getAuthHeader } from './api/authApi';
import LoginForm from './components/Auth/LoginForm';
import './styles/App.css';
import AiProfile from './assets/hallym.svg';

function App() {
    // 인증 상태
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [currentUser, setCurrentUser] = useState(null);
    const [isCheckingAuth, setIsCheckingAuth] = useState(true);

    // 채팅 상태
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState(null);
    const [sessions, setSessions] = useState([]);

    // UI 상태
    const [isSidebarOpen, setSidebarOpen] = useState(true);
    const [showModelDropdown, setShowModelDropdown] = useState(false);

    // 파일 업로드 상태
    const [attachments, setAttachments] = useState([]);
    const [isUploading, setIsUploading] = useState(false);
    const [isDragOver, setIsDragOver] = useState(false);

    // Agent 상태 (Agent는 기본으로 항상 활성화)
    const [agentStatus, setAgentStatus] = useState(null);  // "검색 중...", "문서 분석 중..." 등

    // 다이얼로그 상태
    const [deleteDialog, setDeleteDialog] = useState({ open: false, sessionId: null });
    const [renameDialog, setRenameDialog] = useState({ open: false, sessionId: null, currentTitle: '' });
    const [newSessionTitle, setNewSessionTitle] = useState('');

    // 세션 메뉴 상태
    const [sessionMenuId, setSessionMenuId] = useState(null);
    const [hoveredSession, setHoveredSession] = useState(null);

    // 메시지 피드백/복사 상태
    const [messageFeedback, setMessageFeedback] = useState({});
    const [messageCopied, setMessageCopied] = useState({});

    // Refs
    const bottomRef = useRef(null);
    const textareaRef = useRef(null);
    const fileInputRef = useRef(null);
    const abortControllerRef = useRef(null);
    const modelDropdownRef = useRef(null);
    const activeSessionIdRef = useRef(null);  // 현재 보고 있는 세션 ID
    // 백그라운드 스트리밍 버퍼: { [sessionId]: { content, isStreaming, messages, abortController } }
    const streamingBuffersRef = useRef({});

    // 스크롤 자동 이동
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    // 모델 드롭다운 외부 클릭 감지
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (modelDropdownRef.current && !modelDropdownRef.current.contains(e.target)) {
                setShowModelDropdown(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // 인증 체크
    useEffect(() => {
        const token = getToken();
        const user = getUser();
        if (token && user) {
            setIsAuthenticated(true);
            setCurrentUser(user);
        }
        setIsCheckingAuth(false);
    }, []);

    // 세션 목록 불러오기
    useEffect(() => {
        if (isAuthenticated) {
            fetchSessions();
        }
    }, [isAuthenticated]);

    const fetchSessions = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/chat/sessions`, {
                headers: getAuthHeader()
            });
            if (response.ok) {
                const data = await response.json();
                setSessions(data.sessions || []);
            }
        } catch (error) {
            console.error('세션 목록 불러오기 실패:', error);
        }
    };

    const loadSession = async (sessId) => {
        const prevActiveSession = activeSessionIdRef.current;
        console.log('[loadSession] 세션 전환:', prevActiveSession, '->', sessId);

        // 현재 세션 ID 업데이트 (스트리밍 중인 세션은 abort하지 않음, 백그라운드에서 계속)
        activeSessionIdRef.current = sessId;

        // 해당 세션에 진행 중인 스트리밍 버퍼가 있는지 확인
        const buffer = streamingBuffersRef.current[sessId];

        if (buffer && buffer.isStreaming) {
            // 스트리밍 중인 세션으로 복귀: 버퍼 내용 즉시 표시
            console.log('[loadSession] 스트리밍 중인 세션 복귀, 버퍼 내용 표시:', buffer.content?.length);
            setSessionId(sessId);
            setMessages(buffer.messages);
            setIsLoading(true);
            setAgentStatus(buffer.agentStatus || null);
            setMessageFeedback({});
            setMessageCopied({});
            // 해당 세션의 abortController 복원
            abortControllerRef.current = buffer.abortController || null;
        } else {
            // 일반 세션 로드 (DB에서)
            setIsLoading(false);
            setAgentStatus(null);

            try {
                const response = await fetch(`${API_BASE_URL}/api/chat/sessions/${sessId}`, {
                    headers: getAuthHeader()
                });
                if (response.ok) {
                    const data = await response.json();
                    setSessionId(sessId);
                    setMessages(data.messages || []);
                    setMessageFeedback({});
                    setMessageCopied({});
                }
            } catch (error) {
                console.error('세션 불러오기 실패:', error);
            }
        }
    };

    const handleDeleteSession = async () => {
        const sessId = deleteDialog.sessionId;
        setDeleteDialog({ open: false, sessionId: null });

        try {
            const response = await fetch(`${API_BASE_URL}/api/chat/sessions/${sessId}`, {
                method: 'DELETE',
                headers: getAuthHeader()
            });
            if (response.ok) {
                if (sessionId === sessId) {
                    startNewConversation();
                }
                fetchSessions();
            }
        } catch (error) {
            console.error('삭제 실패:', error);
        }
    };

    const handleRenameSession = async () => {
        const sessId = renameDialog.sessionId;
        const newTitle = newSessionTitle.trim();
        setRenameDialog({ open: false, sessionId: null, currentTitle: '' });
        setNewSessionTitle('');

        if (!newTitle) return;

        try {
            const response = await fetch(`${API_BASE_URL}/api/chat/sessions/${sessId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify({ title: newTitle })
            });
            if (response.ok) {
                fetchSessions();
            }
        } catch (error) {
            console.error('이름 변경 실패:', error);
        }
    };

    const startNewConversation = () => {
        // 백그라운드 스트리밍은 계속 진행 (abort하지 않음)
        // abortControllerRef는 현재 보고 있는 세션용이므로 null로만 설정
        // (실제 abort는 버퍼에 저장된 abortController로 할 수 있음)

        activeSessionIdRef.current = null;  // 새 대화는 null
        abortControllerRef.current = null;  // 현재 세션의 참조만 해제
        setSessionId(null);
        setMessages([]);
        setAttachments([]);
        setInputValue('');
        setIsLoading(false);
        setAgentStatus(null);
        setMessageFeedback({});
        setMessageCopied({});
    };

    const handleLogout = () => {
        logout();
        setIsAuthenticated(false);
        setCurrentUser(null);
        startNewConversation();
    };

    const handleLoginSuccess = (user) => {
        setIsAuthenticated(true);
        setCurrentUser(user);
    };

    // 파일 업로드 처리
    const handleFileUpload = async (files) => {
        if (!files || files.length === 0) return;

        setIsUploading(true);
        const newAttachments = [];
        let uploadSessionId = null;

        for (const file of files) {
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch(`${API_BASE_URL}/api/chat/upload`, {
                    method: 'POST',
                    headers: getAuthHeader(),
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    // 서버에서 반환하는 id 사용 (attachment_id가 아닌 id)
                    const serverId = data.id || data.attachment_id;
                    // 고유 ID 생성: 서버 ID + 타임스탬프 + 랜덤값
                    const uniqueId = `${serverId || 'local'}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                    newAttachments.push({
                        id: uniqueId,
                        serverId: serverId,
                        original_filename: file.name,
                        file_type: file.type,
                        preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : null
                    });
                    // 서버가 반환한 session_id 저장 (새 대화에서 파일 업로드 시 사용)
                    if (data.session_id && !uploadSessionId) {
                        uploadSessionId = data.session_id;
                    }
                    console.log('Upload success:', { serverId, uniqueId, sessionId: data.session_id, data });
                }
            } catch (error) {
                console.error(`${file.name} 업로드 실패:`, error);
            }
        }

        setAttachments(prev => [...prev, ...newAttachments]);
        // 새 대화(sessionId가 없을 때)에서 파일 업로드 시, 서버가 생성한 세션 사용
        if (!sessionId && uploadSessionId) {
            setSessionId(uploadSessionId);
            console.log('Set session from upload:', uploadSessionId);
        }
        setIsUploading(false);
    };

    const removeAttachment = (attachmentId) => {
        console.log('Removing attachment:', attachmentId);
        console.log('Current attachments:', attachments);
        setAttachments(prev => {
            const filtered = prev.filter(a => a.id !== attachmentId);
            console.log('After filter:', filtered);
            return filtered;
        });
    };

    // 드래그 앤 드롭 핸들러
    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragOver(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragOver(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragOver(false);

        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            handleFileUpload(files);
        }
    };

    // 클립보드 붙여넣기 핸들러 (캡쳐 이미지)
    const handlePaste = (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;

        const files = [];
        for (const item of items) {
            if (item.type.startsWith('image/')) {
                const file = item.getAsFile();
                if (file) {
                    // 파일명 생성 (캡쳐 이미지는 이름이 없으므로)
                    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                    const newFile = new File([file], `screenshot-${timestamp}.png`, { type: file.type });
                    files.push(newFile);
                }
            }
        }

        if (files.length > 0) {
            e.preventDefault();
            handleFileUpload(files);
        }
    };

    // 메시지 전송 (백그라운드 스트리밍 버퍼링 지원)
    const handleSendMessage = async () => {
        if ((!inputValue.trim() && attachments.length === 0) || isLoading) return;

        const userMessage = {
            role: 'user',
            content: inputValue.trim(),
            attachments: attachments.map(a => ({
                id: a.id,
                original_filename: a.original_filename,
                file_type: a.file_type,
                preview: a.preview
            }))
        };

        const assistantMessage = {
            role: 'assistant',
            content: '',
            model: selectedModel,
            isStreaming: true
        };

        // 초기 메시지 상태
        const initialMessages = [...messages, userMessage, assistantMessage];

        setMessages(initialMessages);
        setInputValue('');
        const currentAttachments = [...attachments];
        setAttachments([]);
        setIsLoading(true);

        // 텍스트 영역 높이 리셋
        if (textareaRef.current) {
            textareaRef.current.style.height = '24px';
        }

        // 현재 세션 ID 저장 (스트리밍 중 세션 ID 변경될 수 있음)
        // 새 대화(sessionId=null)는 'new' 키로 관리
        let streamingSessionId = sessionId || 'new';
        activeSessionIdRef.current = streamingSessionId;

        try {
            const abortController = new AbortController();
            abortControllerRef.current = abortController;

            const response = await fetch(`${API_BASE_URL}/api/chat/send`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify({
                    message: userMessage.content,
                    model: selectedModel,
                    session_id: sessionId,
                    attachment_ids: currentAttachments.map(a => a.serverId).filter(id => id != null),
                    agent_mode: true
                }),
                signal: abortController.signal
            });

            if (!response.ok) throw new Error('전송 실패');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullContent = '';
            let currentMessages = [...initialMessages];
            let currentAgentStatus = null;

            // 버퍼 초기화 (streamingSessionId는 이미 'new' 또는 실제 세션 ID)
            streamingBuffersRef.current[streamingSessionId] = {
                isStreaming: true,
                content: '',
                messages: currentMessages,
                agentStatus: null,
                abortController: abortController
            };

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));

                            // 새 세션 ID 받으면 버퍼 키 업데이트
                            if (data.session_id && data.session_id !== streamingSessionId) {
                                const oldKey = streamingSessionId;
                                const newKey = data.session_id;
                                console.log('[streaming] 세션 ID 변경:', {
                                    old: oldKey,
                                    new: newKey,
                                    activeSession: activeSessionIdRef.current
                                });

                                // 기존 버퍼 삭제하고 새 세션 ID로 이동
                                const oldBuffer = streamingBuffersRef.current[oldKey];
                                if (oldBuffer) {
                                    delete streamingBuffersRef.current[oldKey];
                                    streamingBuffersRef.current[newKey] = oldBuffer;
                                }

                                // activeSessionIdRef도 함께 업데이트 (새 대화였던 경우)
                                if (activeSessionIdRef.current === oldKey) {
                                    activeSessionIdRef.current = newKey;
                                    setSessionId(newKey);
                                }

                                streamingSessionId = newKey;

                                // 새 세션이 생성된 경우 즉시 세션 목록 갱신
                                if (oldKey === 'new') {
                                    console.log('[streaming] 새 세션 생성됨, 세션 목록 갱신');
                                    fetchSessions();
                                }
                            }

                            // Agent 상태 메시지
                            if (data.status) {
                                currentAgentStatus = data.status;
                                streamingBuffersRef.current[streamingSessionId].agentStatus = currentAgentStatus;

                                // 현재 보고 있는 세션이면 UI 업데이트
                                if (activeSessionIdRef.current === streamingSessionId) {
                                    setAgentStatus(currentAgentStatus);
                                }
                            }

                            // 컨텐츠
                            if (data.content) {
                                currentAgentStatus = null;
                                fullContent += data.content;

                                // 메시지 업데이트
                                currentMessages = [...currentMessages];
                                currentMessages[currentMessages.length - 1] = {
                                    ...currentMessages[currentMessages.length - 1],
                                    content: fullContent
                                };

                                // 버퍼 업데이트
                                const buffer = streamingBuffersRef.current[streamingSessionId];
                                if (buffer) {
                                    buffer.content = fullContent;
                                    buffer.messages = currentMessages;
                                    buffer.agentStatus = null;
                                }

                                // 현재 보고 있는 세션이면 UI 업데이트
                                const isActiveSession = activeSessionIdRef.current === streamingSessionId;
                                console.log('[streaming] 컨텐츠 수신:', {
                                    activeSession: activeSessionIdRef.current,
                                    streamingSession: streamingSessionId,
                                    isActive: isActiveSession,
                                    contentLen: fullContent.length
                                });
                                if (isActiveSession) {
                                    setMessages(currentMessages);
                                    setAgentStatus(null);
                                }
                            }

                            // 완료
                            if (data.done) {
                                currentMessages = [...currentMessages];
                                currentMessages[currentMessages.length - 1] = {
                                    ...currentMessages[currentMessages.length - 1],
                                    isStreaming: false,
                                    messageId: data.message_id,
                                    toolUsed: data.tool_used
                                };

                                // 버퍼에서 스트리밍 완료 표시 후 삭제
                                delete streamingBuffersRef.current[streamingSessionId];

                                // 현재 보고 있는 세션이면 UI 업데이트
                                if (activeSessionIdRef.current === streamingSessionId) {
                                    setMessages(currentMessages);
                                    setIsLoading(false);
                                    setAgentStatus(null);
                                }

                                // 세션 목록 갱신
                                fetchSessions();
                            }
                        } catch (e) {}
                    }
                }
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                const errorMessages = [...messages, userMessage, {
                    role: 'assistant',
                    content: '오류가 발생했습니다. 다시 시도해주세요.',
                    isError: true
                }];

                // 버퍼에서 제거
                delete streamingBuffersRef.current[streamingSessionId];

                // 현재 보고 있는 세션이면 UI 업데이트
                if (activeSessionIdRef.current === streamingSessionId) {
                    setMessages(errorMessages);
                    setIsLoading(false);
                    setAgentStatus(null);
                }
            }
        } finally {
            abortControllerRef.current = null;
        }
    };

    const handleStopGeneration = () => {
        const currentSession = activeSessionIdRef.current;

        // 현재 세션의 abortController 사용 (버퍼 또는 ref에서)
        const buffer = currentSession ? streamingBuffersRef.current[currentSession] : null;
        const controller = buffer?.abortController || abortControllerRef.current;

        if (controller) {
            controller.abort();
        }

        // 버퍼에서 제거
        if (currentSession && streamingBuffersRef.current[currentSession]) {
            delete streamingBuffersRef.current[currentSession];
        }

        abortControllerRef.current = null;
        setIsLoading(false);
        setAgentStatus(null);
        // 스트리밍 중인 메시지의 isStreaming 상태를 false로 변경
        setMessages(prev => prev.map(msg =>
            msg.isStreaming ? { ...msg, isStreaming: false } : msg
        ));
    };

    // 메시지 피드백 핸들러
    const handleMessageFeedback = async (msgIndex, feedbackType) => {
        const newFeedback = messageFeedback[msgIndex] === feedbackType ? null : feedbackType;
        setMessageFeedback(prev => ({ ...prev, [msgIndex]: newFeedback }));

        const message = messages[msgIndex];
        if (!message || !message.messageId) return;

        try {
            await fetch(`${API_BASE_URL}/api/chat/messages/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify({
                    message_id: message.messageId,
                    feedback: newFeedback || ''
                })
            });
        } catch (err) {
            console.error('Feedback API error:', err);
        }
    };

    // 메시지 복사 핸들러
    const handleCopyMessage = async (msgIndex, content) => {
        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(content);
            } else {
                const textArea = document.createElement('textarea');
                textArea.value = content;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
            }
            setMessageCopied(prev => ({ ...prev, [msgIndex]: true }));
            setTimeout(() => {
                setMessageCopied(prev => ({ ...prev, [msgIndex]: false }));
            }, 2000);
        } catch (err) {
            console.error('Copy failed:', err);
        }
    };

    // 세션 그룹화
    const groupSessionsByDate = (sessions) => {
        const groups = { today: [], week: [], older: [] };
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

        sessions.forEach(session => {
            const sessionDate = new Date(session.updated_at);
            if (sessionDate >= today) {
                groups.today.push(session);
            } else if (sessionDate >= weekAgo) {
                groups.week.push(session);
            } else {
                groups.older.push(session);
            }
        });

        return groups;
    };

    // 로딩 중
    if (isCheckingAuth) {
        return (
            <div className="loading-screen">
                <div className="loading-spinner"></div>
            </div>
        );
    }

    // 로그인 필요
    if (!isAuthenticated) {
        return <LoginForm onLoginSuccess={handleLoginSuccess} />;
    }

    const groupedSessions = groupSessionsByDate(sessions);

    // 스트리밍 중 불완전한 마크다운 처리 함수
    const preprocessStreamingMarkdown = (content) => {
        if (!content) return '';

        // 코드블록 패턴 검사 (``` 로 시작하는 블록)
        const codeBlockPattern = /```/g;
        const matches = content.match(codeBlockPattern);

        // 백틱이 홀수 개면 (열린 코드블록이 닫히지 않음)
        if (matches && matches.length % 2 !== 0) {
            // 마지막으로 열린 코드블록 찾기
            const lastOpenIndex = content.lastIndexOf('```');
            const afterLastOpen = content.substring(lastOpenIndex + 3);

            // 언어 지정 후 줄바꿈이 있는지 확인
            const hasNewlineAfterLang = afterLastOpen.includes('\n');

            if (hasNewlineAfterLang) {
                // 코드 내용이 있으면 임시로 코드블록 닫기
                return content + '\n```';
            } else {
                // 아직 언어 지정 중이면 (```python 타이핑 중)
                // 코드블록을 그대로 유지하되 임시로 닫기
                return content + '\n```';
            }
        }

        return content;
    };

    // 마크다운 코드 블록 컴포넌트
    const CodeBlockComponent = ({ node, inline, className, children, ...props }) => {
        const [copied, setCopied] = useState(false);
        const match = /language-(\w+)/.exec(className || '');
        const language = match ? match[1] : '';
        const codeString = String(children).replace(/\n$/, '');

        const handleCopy = async () => {
            try {
                await navigator.clipboard.writeText(codeString);
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
            } catch (err) {
                console.error('Copy failed:', err);
            }
        };

        // 인라인 코드이거나, 언어 지정 없이 한 줄짜리 짧은 코드는 인라인으로 표시
        const hasNewLine = codeString.includes('\n');
        const isShortCode = !hasNewLine && codeString.length < 100;
        const shouldBeInline = inline || (!language && isShortCode);

        if (shouldBeInline) {
            return <code className="inline-code" {...props}>{children}</code>;
        }

        // 블록 코드 (언어 지정 또는 여러 줄)
        return (
            <div className="code-block-wrapper">
                <div className="code-block-header">
                    <span>{language || 'code'}</span>
                    <button onClick={handleCopy} className="code-copy-btn">
                        {copied ? <Check size={14} /> : <Copy size={14} />}
                    </button>
                </div>
                <SyntaxHighlighter
                    style={oneLight}
                    language={language || 'text'}
                    PreTag="div"
                    customStyle={{ margin: 0, padding: '16px', background: '#F8F9FA', borderRadius: '0 0 8px 8px' }}
                    {...props}
                >
                    {codeString}
                </SyntaxHighlighter>
            </div>
        );
    };

    return (
        <div className="app-container">
            {/* 사이드바 */}
            <div className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
                {/* 메뉴 토글 버튼 */}
                <div className="sidebar-header">
                    <button className="icon-btn" onClick={() => setSidebarOpen(!isSidebarOpen)}>
                        <Menu size={24} />
                    </button>
                </div>

                {/* 새 채팅 버튼 */}
                <button className="new-chat-btn" onClick={startNewConversation}>
                    <Plus size={24} />
                    {isSidebarOpen && <span>New Chat</span>}
                </button>

                {/* 대화 히스토리 */}
                {isSidebarOpen && (
                    <div className="chat-history">
                        {groupedSessions.today.length > 0 && (
                            <div className="history-group">
                                <p className="history-label">오늘</p>
                                {groupedSessions.today.map(session => (
                                    <div
                                        key={session.id}
                                        className={`history-item ${sessionId === session.id ? 'active' : ''}`}
                                        onClick={() => loadSession(session.id)}
                                        onMouseEnter={() => setHoveredSession(session.id)}
                                        onMouseLeave={() => setHoveredSession(null)}
                                    >
                                        <MessageSquare size={18} />
                                        <span className="history-title">{session.title || '새 대화'}</span>
                                        {(hoveredSession === session.id || sessionMenuId === session.id) && (
                                            <button
                                                className="history-menu-btn"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    setSessionMenuId(sessionMenuId === session.id ? null : session.id);
                                                }}
                                            >
                                                <MoreVertical size={16} />
                                            </button>
                                        )}
                                        {sessionMenuId === session.id && (
                                            <div className="session-menu">
                                                <button onClick={(e) => {
                                                    e.stopPropagation();
                                                    setRenameDialog({ open: true, sessionId: session.id, currentTitle: session.title });
                                                    setNewSessionTitle(session.title || '');
                                                    setSessionMenuId(null);
                                                }}>
                                                    <Edit3 size={14} /> 이름 변경
                                                </button>
                                                <button onClick={(e) => {
                                                    e.stopPropagation();
                                                    setDeleteDialog({ open: true, sessionId: session.id });
                                                    setSessionMenuId(null);
                                                }} className="danger">
                                                    <Trash2 size={14} /> 삭제
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}

                        {groupedSessions.week.length > 0 && (
                            <div className="history-group">
                                <p className="history-label">지난 7일</p>
                                {groupedSessions.week.map(session => (
                                    <div
                                        key={session.id}
                                        className={`history-item ${sessionId === session.id ? 'active' : ''}`}
                                        onClick={() => loadSession(session.id)}
                                        onMouseEnter={() => setHoveredSession(session.id)}
                                        onMouseLeave={() => setHoveredSession(null)}
                                    >
                                        <MessageSquare size={18} />
                                        <span className="history-title">{session.title || '새 대화'}</span>
                                        {(hoveredSession === session.id || sessionMenuId === session.id) && (
                                            <button
                                                className="history-menu-btn"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    setSessionMenuId(sessionMenuId === session.id ? null : session.id);
                                                }}
                                            >
                                                <MoreVertical size={16} />
                                            </button>
                                        )}
                                        {sessionMenuId === session.id && (
                                            <div className="session-menu">
                                                <button onClick={(e) => {
                                                    e.stopPropagation();
                                                    setRenameDialog({ open: true, sessionId: session.id, currentTitle: session.title });
                                                    setNewSessionTitle(session.title || '');
                                                    setSessionMenuId(null);
                                                }}>
                                                    <Edit3 size={14} /> 이름 변경
                                                </button>
                                                <button onClick={(e) => {
                                                    e.stopPropagation();
                                                    setDeleteDialog({ open: true, sessionId: session.id });
                                                    setSessionMenuId(null);
                                                }} className="danger">
                                                    <Trash2 size={14} /> 삭제
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}

                        {groupedSessions.older.length > 0 && (
                            <div className="history-group">
                                <p className="history-label">이전</p>
                                {groupedSessions.older.map(session => (
                                    <div
                                        key={session.id}
                                        className={`history-item ${sessionId === session.id ? 'active' : ''}`}
                                        onClick={() => loadSession(session.id)}
                                        onMouseEnter={() => setHoveredSession(session.id)}
                                        onMouseLeave={() => setHoveredSession(null)}
                                    >
                                        <MessageSquare size={18} />
                                        <span className="history-title">{session.title || '새 대화'}</span>
                                        {(hoveredSession === session.id || sessionMenuId === session.id) && (
                                            <button
                                                className="history-menu-btn"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    setSessionMenuId(sessionMenuId === session.id ? null : session.id);
                                                }}
                                            >
                                                <MoreVertical size={16} />
                                            </button>
                                        )}
                                        {sessionMenuId === session.id && (
                                            <div className="session-menu">
                                                <button onClick={(e) => {
                                                    e.stopPropagation();
                                                    setRenameDialog({ open: true, sessionId: session.id, currentTitle: session.title });
                                                    setNewSessionTitle(session.title || '');
                                                    setSessionMenuId(null);
                                                }}>
                                                    <Edit3 size={14} /> 이름 변경
                                                </button>
                                                <button onClick={(e) => {
                                                    e.stopPropagation();
                                                    setDeleteDialog({ open: true, sessionId: session.id });
                                                    setSessionMenuId(null);
                                                }} className="danger">
                                                    <Trash2 size={14} /> 삭제
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}

{/* 하단 메뉴 - 숨김 처리
                <div className="sidebar-footer">
                    <div className="footer-item">
                        <HelpCircle size={20} />
                        {isSidebarOpen && <span>Help</span>}
                    </div>
                    <div className="footer-item">
                        <History size={20} />
                        {isSidebarOpen && <span>Activity</span>}
                    </div>
                    <div className="footer-item">
                        <Settings size={20} />
                        {isSidebarOpen && <span>Settings</span>}
                    </div>
                </div>
                */}
            </div>

            {/* 메인 콘텐츠 */}
            <div className="main-content">
                {/* 상단 바 */}
                <div className="top-bar">
                    <div className="top-bar-left">
                        {!isSidebarOpen && (
                            <button className="icon-btn" onClick={() => setSidebarOpen(true)}>
                                <Menu size={24} />
                            </button>
                        )}
                        <span className="app-title">Multi-LLM Chatbot</span>
                    </div>

                    <div className="top-bar-right">
                        <span className="user-name">{currentUser?.display_name || currentUser?.username}</span>
                        <button className="icon-btn" onClick={handleLogout} title="로그아웃">
                            <LogOut size={20} />
                        </button>
                    </div>
                </div>

                {/* 채팅 영역 */}
                <div className="chat-area">
                    {messages.length === 0 ? (
                        <div className="welcome-section">
                            <h1 className="welcome-title">무엇을 도와드릴까요?</h1>
                            <p className="welcome-subtitle">궁금한 것을 자유롭게 물어보세요!</p>
                        </div>
                    ) : (
                        <div className="messages-container">
                            {messages.map((msg, idx) => (
                                <div key={idx} className={`message ${msg.role}`}>
                                    <div className="message-content-wrapper">
                                        {/* 아바타 */}
                                        <div className="avatar">
                                            {msg.role === 'user' ? (
                                                <User size={20} />
                                            ) : (
                                                <img
                                                    src={AiProfile}
                                                    alt="AI"
                                                />
                                            )}
                                        </div>

                                        <div className="message-body">
                                            {/* 모델명 표시 (AI 응답) */}
                                            {msg.role === 'assistant' && msg.model && LLM_MODELS[msg.model] && (
                                                <span className="model-badge">{LLM_MODELS[msg.model].name}</span>
                                            )}

                                            {/* 첨부파일 */}
                                            {msg.attachments && msg.attachments.length > 0 && (
                                                <div className="message-attachments">
                                                    {msg.attachments.map((att, i) => (
                                                        <div key={i} className="attachment-chip">
                                                            <Paperclip size={14} />
                                                            <span>{att.original_filename}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}

                                            {/* 메시지 내용 */}
                                            {msg.role === 'user' ? (
                                                <div className="user-text">{msg.content}</div>
                                            ) : (
                                                <div className="markdown-content">
                                                    <ReactMarkdown
                                                        remarkPlugins={[[remarkGfm, { singleTilde: false }], remarkMath]}
                                                        rehypePlugins={[rehypeKatex]}
                                                        components={{
                                                            code: CodeBlockComponent
                                                        }}
                                                    >
                                                        {msg.isStreaming
                                                            ? preprocessStreamingMarkdown(msg.content || '')
                                                            : (msg.content || '')}
                                                    </ReactMarkdown>
                                                    {msg.isStreaming && !msg.content && (
                                                        <div className={`thinking-indicator ${
                                                            agentStatus?.includes('분석') ? 'step-analyzing' :
                                                            agentStatus?.includes('대화') ? 'step-memory' :
                                                            (agentStatus?.includes('검색') || agentStatus?.includes('인터넷')) ? 'step-searching' :
                                                            agentStatus?.includes('생성') ? 'step-generating' : ''
                                                        }`}>
                                                            {agentStatus ? (
                                                                <div className="agent-status-container">
                                                                    <div className="agent-status-content">
                                                                        {agentStatus.includes('분석') && <Brain size={16} className="agent-status-icon" />}
                                                                        {agentStatus.includes('대화') && <BookOpen size={16} className="agent-status-icon" />}
                                                                        {(agentStatus.includes('검색') || agentStatus.includes('인터넷')) && <Globe size={16} className="agent-status-icon" />}
                                                                        {agentStatus.includes('생성') && <PenTool size={16} className="agent-status-icon" />}
                                                                        <span className="agent-status-text">
                                                                            {agentStatus.replace('...', '')}
                                                                            <span className="status-dots">
                                                                                <span className="status-dot"></span>
                                                                                <span className="status-dot"></span>
                                                                                <span className="status-dot"></span>
                                                                            </span>
                                                                        </span>
                                                                    </div>
                                                                    <div className="agent-progress-bar">
                                                                        <div className="progress-fill" style={{
                                                                            width: agentStatus.includes('분석') ? '25%' :
                                                                                   agentStatus.includes('대화') ? '40%' :
                                                                                   (agentStatus.includes('검색') || agentStatus.includes('인터넷')) ? '60%' :
                                                                                   agentStatus.includes('생성') ? '85%' : '50%'
                                                                        }}></div>
                                                                    </div>
                                                                </div>
                                                            ) : (
                                                                <>
                                                                    <span>Thinking</span>
                                                                    <div className="dots">
                                                                        <span className="dot"></span>
                                                                        <span className="dot"></span>
                                                                        <span className="dot"></span>
                                                                    </div>
                                                                </>
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            )}

                                            {/* AI 응답 액션 버튼 */}
                                            {msg.role === 'assistant' && !msg.isStreaming && msg.content && (
                                                <div className="message-actions">
                                                    <button
                                                        className={messageFeedback[idx] === 'like' ? 'active' : ''}
                                                        onClick={() => handleMessageFeedback(idx, 'like')}
                                                        title="마음에 들어요"
                                                    >
                                                        <ThumbsUp size={16} />
                                                    </button>
                                                    <button
                                                        className={messageFeedback[idx] === 'dislike' ? 'active' : ''}
                                                        onClick={() => handleMessageFeedback(idx, 'dislike')}
                                                        title="마음에 들지 않아요"
                                                    >
                                                        <ThumbsDown size={16} />
                                                    </button>
                                                    <button
                                                        className={messageCopied[idx] ? 'copied' : ''}
                                                        onClick={() => handleCopyMessage(idx, msg.content)}
                                                        title="복사"
                                                    >
                                                        {messageCopied[idx] ? <Check size={16} /> : <Copy size={16} />}
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                            <div ref={bottomRef}></div>
                        </div>
                    )}
                </div>

                {/* 입력 영역 */}
                <div className="input-area">
                    {/* 첨부파일 미리보기 */}
                    {attachments.length > 0 && (
                        <div className="attachments-preview">
                            {attachments.map(att => (
                                <div key={att.id} className="attachment-item">
                                    {att.preview ? (
                                        <img src={att.preview} alt={att.original_filename} />
                                    ) : (
                                        <Paperclip size={20} />
                                    )}
                                    <span>{att.original_filename}</span>
                                    <button onClick={(e) => {
                                        e.stopPropagation();
                                        removeAttachment(att.id);
                                    }}>
                                        <X size={14} />
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}

                    <div
                        className={`input-wrapper ${isDragOver ? 'drag-over' : ''}`}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                    >
                        <div className="input-row">
                            {/* 모델 선택 */}
                            <div className="model-selector" ref={modelDropdownRef}>
                                <button
                                    className="model-selector-btn"
                                    onClick={() => setShowModelDropdown(!showModelDropdown)}
                                >
                                    {LLM_MODELS[selectedModel]?.name || selectedModel}
                                    <ChevronDown size={16} />
                                </button>
                                {showModelDropdown && (
                                    <div className="model-dropdown">
                                        {Object.entries(LLM_MODELS).map(([key, model]) => (
                                            <div
                                                key={key}
                                                className={`model-option ${selectedModel === key ? 'selected' : ''}`}
                                                onClick={() => {
                                                    setSelectedModel(key);
                                                    setShowModelDropdown(false);
                                                }}
                                            >
                                                <span className="model-name">{model.name}</span>
                                                <span className="model-provider">{model.provider}</span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {/* 파일 첨부 버튼 */}
                            <input
                                type="file"
                                multiple
                                hidden
                                ref={fileInputRef}
                                onChange={(e) => handleFileUpload(Array.from(e.target.files))}
                            />
                            <button
                                className="icon-btn attach-btn"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isLoading || isUploading}
                                title="파일 첨부"
                            >
                                <Paperclip size={22} />
                            </button>

                            {/* 텍스트 입력 */}
                            <textarea
                                ref={textareaRef}
                                value={inputValue}
                                onChange={(e) => {
                                    setInputValue(e.target.value);
                                    e.target.style.height = 'auto';
                                    e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
                                }}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && !e.shiftKey) {
                                        e.preventDefault();
                                        handleSendMessage();
                                    }
                                }}
                                onPaste={handlePaste}
                                placeholder="메시지를 입력하세요..."
                                disabled={isLoading}
                                rows={1}
                            />

                            {/* 전송/중지 버튼 */}
                            {isLoading ? (
                                <button className="icon-btn stop-btn" onClick={handleStopGeneration} title="중지">
                                    <Square size={20} />
                                </button>
                            ) : (
                                <button
                                    className="icon-btn send-btn"
                                    onClick={handleSendMessage}
                                    title="전송"
                                    disabled={!inputValue.trim() && attachments.length === 0}
                                >
                                    <Send size={20} />
                                </button>
                            )}
                        </div>
                    </div>

                    <div className="input-disclaimer">
                        AI는 부정확한 정보를 생성할 수 있습니다. 중요한 정보는 확인하세요.
                    </div>
                </div>
            </div>

            {/* 삭제 확인 다이얼로그 */}
            {deleteDialog.open && (
                <div className="dialog-overlay" onClick={() => setDeleteDialog({ open: false, sessionId: null })}>
                    <div className="dialog" onClick={e => e.stopPropagation()}>
                        <h3>대화 삭제</h3>
                        <p>이 대화를 삭제하시겠습니까? 삭제된 대화는 복구할 수 없습니다.</p>
                        <div className="dialog-actions">
                            <button className="btn-secondary" onClick={() => setDeleteDialog({ open: false, sessionId: null })}>
                                취소
                            </button>
                            <button className="btn-danger" onClick={handleDeleteSession}>
                                삭제
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* 이름 변경 다이얼로그 */}
            {renameDialog.open && (
                <div className="dialog-overlay" onClick={() => setRenameDialog({ open: false, sessionId: null, currentTitle: '' })}>
                    <div className="dialog" onClick={e => e.stopPropagation()}>
                        <h3>대화 이름 변경</h3>
                        <input
                            type="text"
                            value={newSessionTitle}
                            onChange={(e) => setNewSessionTitle(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleRenameSession()}
                            placeholder="새 이름 입력"
                            autoFocus
                        />
                        <div className="dialog-actions">
                            <button className="btn-secondary" onClick={() => setRenameDialog({ open: false, sessionId: null, currentTitle: '' })}>
                                취소
                            </button>
                            <button className="btn-primary" onClick={handleRenameSession}>
                                변경
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default App;
