# config_loader.py
from .config import load_config, Config

class ConfigManager:
    _instance: Config = None

    @classmethod
    def get(cls) -> Config:
        if cls._instance is None:
            cls._instance = load_config()
        return cls._instance
