# -*- coding: utf-8 -*-
"""
Profile Service - personalization_core에서 가져옴

이 파일은 personalization_core.profile_service를 래핑합니다.
개인화 기능 개선 시 llm_chatbot에서 수정 후:
    cd /home/aiedu/workspace/personalization_core
    ./sync_from_llm_chatbot.sh
"""

# 기본값
ProfileService = None
get_profile_service = None

try:
    from personalization_core.profile_service import *
    from personalization_core.profile_service import (
        ProfileService,
        get_profile_service,
    )
except (ImportError, AttributeError) as e:
    import logging
    logging.debug(f"ProfileService not available: {e}")

__all__ = [
    "ProfileService",
    "get_profile_service",
]
