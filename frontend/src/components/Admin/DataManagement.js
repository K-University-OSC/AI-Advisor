/**
 * 데이터 관리 대시보드 (OSC 버전)
 * 문서 관리 기능
 */
import React, { useState, useEffect, useRef } from 'react';
import {
    FileText, Upload, Trash2, RefreshCw,
    X, AlertCircle, CheckCircle, Clock,
    Database, HardDrive, Layers, LogOut
} from 'lucide-react';
import {
    getDataStats, getDocuments, uploadDocument, deleteDocument, reindexDocument,
    adminLogout, getAdmin
} from '../../api/adminApi';
import './DataManagement.css';

function DataManagement({ onLogout, embedded = false }) {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const admin = getAdmin();

    // 문서 상태
    const [documents, setDocuments] = useState([]);
    const [docUploading, setDocUploading] = useState(false);
    const docInputRef = useRef(null);

    // 삭제 확인 다이얼로그
    const [deleteConfirm, setDeleteConfirm] = useState({ show: false, id: '', name: '' });

    // 드래그앤드롭 상태
    const [docDragActive, setDocDragActive] = useState(false);

    useEffect(() => {
        loadData();
        loadDocuments();
    }, []);

    // 문서 상태 자동 갱신 (processing 상태가 있을 때만)
    useEffect(() => {
        const hasProcessing = documents.some(doc => doc.status === 'pending' || doc.status === 'processing');
        if (hasProcessing) {
            const interval = setInterval(() => {
                loadDocuments();
            }, 3000); // 3초마다 갱신
            return () => clearInterval(interval);
        }
    }, [documents]);

    const loadData = async () => {
        try {
            setLoading(true);
            const statsData = await getDataStats();
            setStats(statsData);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const loadDocuments = async () => {
        try {
            const data = await getDocuments();
            setDocuments(data);
        } catch (err) {
            console.error('문서 로드 실패:', err);
        }
    };

    // 문서 업로드 (파일 선택 또는 드래그앤드롭)
    const handleDocUpload = async (e) => {
        const files = Array.from(e.target.files);
        if (files.length === 0) return;
        await uploadDocFiles(files);
    };

    // 문서 파일 업로드 처리 (공통)
    const uploadDocFiles = async (files) => {
        setDocUploading(true);
        try {
            for (const file of files) {
                await uploadDocument(file, true);
            }
            loadDocuments();
            loadData();
        } catch (err) {
            setError(err.message);
        } finally {
            setDocUploading(false);
            if (docInputRef.current) docInputRef.current.value = '';
        }
    };

    // 문서 드래그앤드롭 핸들러
    const handleDocDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDocDragActive(true);
        } else if (e.type === 'dragleave') {
            setDocDragActive(false);
        }
    };

    const handleDocDrop = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDocDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const files = Array.from(e.dataTransfer.files);
            const allowedExts = ['.pdf', '.doc', '.docx', '.txt', '.pptx', '.xlsx', '.csv', '.hwp'];
            const validFiles = files.filter(f => allowedExts.some(ext => f.name.toLowerCase().endsWith(ext)));
            if (validFiles.length > 0) {
                await uploadDocFiles(validFiles);
            } else {
                setError('지원하지 않는 파일 형식입니다. (PDF, DOC, DOCX, TXT, PPTX, XLSX, CSV, HWP)');
            }
        }
    };

    // 문서 삭제
    const handleDocDelete = async () => {
        try {
            await deleteDocument(deleteConfirm.id);
            loadDocuments();
            loadData();
        } catch (err) {
            setError(err.message);
        }
        setDeleteConfirm({ show: false, id: '', name: '' });
    };

    // 문서 재인덱싱
    const handleDocReindex = async (docId) => {
        try {
            await reindexDocument(docId);
            loadDocuments();
        } catch (err) {
            setError(err.message);
        }
    };

    // 로그아웃
    const handleLogout = () => {
        adminLogout();
        onLogout();
    };

    // 상태 배지 렌더링
    const renderStatusBadge = (status) => {
        const statusMap = {
            'pending': { icon: Clock, color: 'yellow', text: '대기' },
            'processing': { icon: RefreshCw, color: 'blue', text: '인덱싱중', spin: true },
            'indexed': { icon: CheckCircle, color: 'green', text: '완료' },
            'error': { icon: AlertCircle, color: 'red', text: '오류' },
            'uploaded': { icon: Clock, color: 'gray', text: '업로드됨' }
        };
        const { icon: Icon, color, text, spin } = statusMap[status] || { icon: Clock, color: 'gray', text: status };
        return (
            <span className={`status-badge ${color} ${spin ? 'spin' : ''}`}>
                <Icon size={12} className={spin ? 'spinning' : ''} />
                {text}
            </span>
        );
    };

    // 파일 크기 포맷
    const formatSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
    };

    // 날짜 포맷
    const formatDate = (dateStr) => {
        if (!dateStr) return '-';
        return new Date(dateStr).toLocaleString('ko-KR');
    };

    if (loading) {
        return (
            <div className="data-management loading">
                <div className="loading-spinner"></div>
                <p>로딩 중...</p>
            </div>
        );
    }

    return (
        <div className={`data-management ${embedded ? 'embedded' : ''}`}>
            {/* 헤더 - embedded 모드에서는 숨김 */}
            {!embedded && (
                <header className="dm-header">
                    <div className="dm-header-left">
                        <Database size={24} />
                        <h1>데이터 관리</h1>
                    </div>
                    <div className="dm-header-right">
                        <span className="admin-name">{admin?.display_name || admin?.username}</span>
                        <button className="btn-icon" onClick={handleLogout} title="로그아웃">
                            <LogOut size={20} />
                        </button>
                    </div>
                </header>
            )}

            {/* 에러 메시지 */}
            {error && (
                <div className="error-banner">
                    <AlertCircle size={16} />
                    <span>{error}</span>
                    <button onClick={() => setError('')}><X size={14} /></button>
                </div>
            )}

            {/* 통계 카드 */}
            <div className="stats-grid">
                <div className="stat-card">
                    <FileText size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{stats?.total_documents || 0}</span>
                        <span className="stat-label">전체 문서</span>
                    </div>
                    <span className="stat-sub">{stats?.indexed_documents || 0} 인덱싱됨</span>
                </div>
                <div className="stat-card">
                    <Layers size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{stats?.total_chunks || 0}</span>
                        <span className="stat-label">총 청크</span>
                    </div>
                    <span className="stat-sub">벡터 임베딩</span>
                </div>
                <div className="stat-card">
                    <HardDrive size={24} />
                    <div className="stat-info">
                        <span className="stat-value">{stats?.storage_used_mb?.toFixed(1) || 0} MB</span>
                        <span className="stat-label">스토리지</span>
                    </div>
                    <span className="stat-sub">사용 용량</span>
                </div>
            </div>

            {/* 문서 관리 영역 */}
            <div className="tab-content">
                <div className="content-section">
                    <div className="section-header">
                        <h2>문서 목록</h2>
                        <div className="section-actions">
                            <input
                                type="file"
                                ref={docInputRef}
                                onChange={handleDocUpload}
                                multiple
                                accept=".pdf,.doc,.docx,.txt,.pptx,.xlsx,.csv,.hwp"
                                hidden
                            />
                            <button
                                className="btn btn-primary"
                                onClick={() => docInputRef.current?.click()}
                                disabled={docUploading}
                            >
                                <Upload size={16} />
                                {docUploading ? '업로드 중...' : '문서 업로드'}
                            </button>
                            <button className="btn btn-secondary" onClick={loadDocuments}>
                                <RefreshCw size={16} />
                                새로고침
                            </button>
                        </div>
                    </div>

                    {/* 드래그앤드롭 영역 */}
                    <div
                        className={`drop-zone ${docDragActive ? 'active' : ''} ${docUploading ? 'uploading' : ''}`}
                        onDragEnter={handleDocDrag}
                        onDragLeave={handleDocDrag}
                        onDragOver={handleDocDrag}
                        onDrop={handleDocDrop}
                        onClick={() => !docUploading && docInputRef.current?.click()}
                    >
                        <Upload size={32} />
                        <p>{docUploading ? '업로드 중...' : '파일을 여기에 드래그하거나 클릭하여 업로드'}</p>
                        <span className="drop-zone-hint">PDF, DOC, DOCX, TXT, PPTX, XLSX, CSV, HWP</span>
                    </div>

                    {documents.length === 0 ? (
                        <div className="empty-state">
                            <FileText size={48} />
                            <p>업로드된 문서가 없습니다</p>
                        </div>
                    ) : (
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>파일명</th>
                                    <th>크기</th>
                                    <th>상태</th>
                                    <th>업로드일</th>
                                    <th>청크</th>
                                    <th>액션</th>
                                </tr>
                            </thead>
                            <tbody>
                                {documents.map(doc => (
                                    <tr key={doc.id}>
                                        <td className="file-name">
                                            <FileText size={16} />
                                            {doc.original_filename}
                                        </td>
                                        <td>{formatSize(doc.file_size)}</td>
                                        <td>{renderStatusBadge(doc.status)}</td>
                                        <td>{formatDate(doc.uploaded_at)}</td>
                                        <td>{doc.chunk_count || 0}</td>
                                        <td className="actions">
                                            <button
                                                className="btn-icon"
                                                onClick={() => handleDocReindex(doc.id)}
                                                title="재인덱싱"
                                            >
                                                <RefreshCw size={14} />
                                            </button>
                                            <button
                                                className="btn-icon danger"
                                                onClick={() => setDeleteConfirm({ show: true, id: doc.id, name: doc.original_filename })}
                                                title="삭제"
                                            >
                                                <Trash2 size={14} />
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
            </div>

            {/* 삭제 확인 모달 */}
            {deleteConfirm.show && (
                <div className="modal-overlay" onClick={() => setDeleteConfirm({ show: false, id: '', name: '' })}>
                    <div className="modal confirm-modal" onClick={e => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3>삭제 확인</h3>
                        </div>
                        <p>
                            <strong>{deleteConfirm.name}</strong>을(를) 삭제하시겠습니까?
                            <br />
                            <span className="warning-text">이 작업은 되돌릴 수 없습니다.</span>
                        </p>
                        <div className="modal-actions">
                            <button
                                className="btn btn-secondary"
                                onClick={() => setDeleteConfirm({ show: false, id: '', name: '' })}
                            >
                                취소
                            </button>
                            <button
                                className="btn btn-danger"
                                onClick={handleDocDelete}
                            >
                                삭제
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default DataManagement;
