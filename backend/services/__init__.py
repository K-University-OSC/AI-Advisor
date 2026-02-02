# -*- coding: utf-8 -*-
"""
Services Module

개인화 관련 서비스는 personalization_core 패키지에서 가져옵니다.
로컬 확장만 이 디렉토리에 유지됩니다.
"""

import logging

# 기본값 설정
PersonalizationEnhancer = None
HierarchicalMemoryService = None
ProfileService = None
get_profile_service = None
SearchEnhancements = None
_PERSONALIZATION_AVAILABLE = False

# personalization_core에서 가져오는 공통 서비스 (옵셔널)
try:
    from personalization_core import (
        PersonalizationEnhancer,
        HierarchicalMemoryService,
        ProfileService,
        get_profile_service,
    )
    _PERSONALIZATION_AVAILABLE = True
except (ImportError, AttributeError) as e:
    logging.debug(f"personalization_core base imports not available: {e}")

# 로컬 확장 (advisor 전용)
from .mh_rag_service import (
    MHRAGService,
    get_mh_rag_service,
    needs_mh_rag_search,
)

__all__ = [
    # from personalization_core
    "PersonalizationEnhancer",
    "HierarchicalMemoryService",
    "ProfileService",
    "get_profile_service",
    "SearchEnhancements",
    "_PERSONALIZATION_AVAILABLE",
    # local extensions
    "MHRAGService",
    "get_mh_rag_service",
    "needs_mh_rag_search",
]
