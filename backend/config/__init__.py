# -*- coding: utf-8 -*-
"""
Config 패키지 - personalization_core 호환
memory_service.py가 config.settings.providers와 api_keys를 사용하므로
settings 객체에 해당 속성들을 추가
"""

from .settings import Settings, get_settings, ProvidersConfig, APIKeysConfig


class SettingsWithProviders:
    """memory_service.py 호환을 위한 Settings 래퍼"""

    def __init__(self):
        self._settings = get_settings()
        self.providers = ProvidersConfig()
        self.api_keys = APIKeysConfig()

    def __getattr__(self, name):
        return getattr(self._settings, name)


# 싱글톤 settings 인스턴스
settings = SettingsWithProviders()

__all__ = ["Settings", "get_settings", "ProvidersConfig", "APIKeysConfig", "settings"]
