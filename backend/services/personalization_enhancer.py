# -*- coding: utf-8 -*-
"""
Personalization Enhancer - personalization_core에서 가져옴

이 파일은 personalization_core.personalization_enhancer를 래핑합니다.
개인화 기능 개선 시 llm_chatbot에서 수정 후:
    cd /home/aiedu/workspace/personalization_core
    ./sync_from_llm_chatbot.sh
"""

# 기본값
PersonalizationEnhancer = None
get_personalization_enhancer = None

try:
    from personalization_core.personalization_enhancer import *
    from personalization_core.personalization_enhancer import (
        PersonalizationEnhancer,
        get_personalization_enhancer,
    )
except (ImportError, AttributeError) as e:
    import logging
    logging.debug(f"PersonalizationEnhancer not available: {e}")

__all__ = [
    "PersonalizationEnhancer",
    "get_personalization_enhancer",
]
