# database 패키지 (단일 테넌트 버전)
from .engine import engine, async_session_factory, get_db, close_db

# Legacy 함수 re-export (하위 호환성)
from .legacy import (
    init_db,
    create_user,
    get_user_by_username,
    get_user_by_id,
    create_session,
    get_session,
    get_user_sessions,
    update_session_title,
    delete_session,
    add_message,
    get_session_messages,
    get_user_preferences,
    update_user_preferences,
    add_attachment,
    get_attachment,
    get_session_attachments,
    get_message_attachments,
    update_attachment_message_id,
    delete_attachment,
    Base
)

__all__ = [
    # Engine
    'engine',
    'async_session_factory',
    'get_db',
    'close_db',
    # Legacy functions
    'init_db',
    'create_user',
    'get_user_by_username',
    'get_user_by_id',
    'create_session',
    'get_session',
    'get_user_sessions',
    'update_session_title',
    'delete_session',
    'add_message',
    'get_session_messages',
    'get_user_preferences',
    'update_user_preferences',
    'add_attachment',
    'get_attachment',
    'get_session_attachments',
    'get_message_attachments',
    'update_attachment_message_id',
    'delete_attachment',
    'Base'
]
