# -*- coding: utf-8 -*-
"""
Agent Service - personalization_core에서 가져옴

이 파일은 personalization_core.agent_service를 래핑합니다.
개인화 기능 개선 시 personalization_core에서 직접 수정합니다.
"""

# 기본값
get_agent_service = None

try:
    from personalization_core.agent_service import *
    from personalization_core.agent_service import (
        get_agent_service,
    )
except (ImportError, AttributeError) as e:
    import logging
    logging.debug(f"AgentService not available: {e}")

__all__ = [
    "get_agent_service",
]
