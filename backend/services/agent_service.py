# -*- coding: utf-8 -*-
"""
Agent Service - personalization_core에서 가져옴

이 파일은 personalization_core.agent_service를 래핑합니다.
개인화 기능 개선 시 llm_chatbot에서 수정 후:
    cd /home/aiedu/workspace/personalization_core
    ./sync_from_llm_chatbot.sh
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
