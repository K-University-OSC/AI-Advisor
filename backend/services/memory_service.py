# -*- coding: utf-8 -*-
"""
Memory Service - personalization_core에서 가져옴

이 파일은 personalization_core.memory_service를 래핑합니다.
개인화 기능 개선 시 personalization_core에서 직접 수정합니다.
"""

# 기본값
HierarchicalMemoryService = None
get_memory_service = None
MemoryType = None
MemoryEntry = None

try:
    from personalization_core.memory_service import *
    from personalization_core.memory_service import (
        MemoryService as HierarchicalMemoryService,
        get_memory_service,
    )
    # MemoryType, MemoryEntry는 선택적
    try:
        from personalization_core.memory_service import MemoryType, MemoryEntry
    except ImportError:
        pass
except (ImportError, AttributeError) as e:
    import logging
    logging.warning(f"MemoryService not available: {e}")

__all__ = [
    "HierarchicalMemoryService",
    "get_memory_service",
    "MemoryType",
    "MemoryEntry",
]
