"""
Configuration loader and settings validation for AirSpark.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from app.utils.paths import get_project_root, resolve_path


class Settings:
    """Singleton-style settings holder for AirSpark."""
    _instance: Optional["Settings"] = None

    def __init__(self, config_path: Optional[str] = "configs/config.yaml"):
        self.config_path = resolve_path(config_path) if config_path else get_project_root() / "configs" / "config.yaml"
        self._raw_config: Dict[str, Any] = self._load_yaml(self.config_path)
        
        # Load AQI Breakpoints config
        bp_path = get_project_root() / "configs" / "aqi_breakpoints.yaml"
        self.aqi_config: Dict[str, Any] = self._load_yaml(bp_path) if bp_path.exists() else {}

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def get_settings(cls, config_path: Optional[str] = None) -> "Settings":
        if cls._instance is None or config_path is not None:
            cls._instance = cls(config_path)
        return cls._instance

    @property
    def app_name(self) -> str:
        return self._raw_config.get("app", {}).get("name", "AirSpark")

    @property
    def spark_config(self) -> Dict[str, Any]:
        return self._raw_config.get("spark", {
            "app_name": "AirSpark-BDA-Engine",
            "master": "local[*]",
            "driver_memory": "4g",
            "executor_memory": "2g",
            "shuffle_partitions": 8,
            "default_parallelism": 8,
            "adaptive_execution_enabled": True,
            "log_level": "WARN",
        })

    @property
    def paths(self) -> Dict[str, Any]:
        raw_paths = self._raw_config.get("paths", {})
        return {k: str(resolve_path(v)) for k, v in raw_paths.items()}

    @property
    def aqi_standard(self) -> str:
        return self._raw_config.get("aqi", {}).get("standard", "US_EPA")

    @property
    def primary_pollutants(self) -> List[str]:
        return self._raw_config.get("aqi", {}).get("primary_pollutants", ["pm2_5", "pm10", "no2", "so2", "co", "o3"])

    @property
    def data_quality_config(self) -> Dict[str, Any]:
        return self._raw_config.get("data_quality", {})

    @property
    def streaming_config(self) -> Dict[str, Any]:
        return self._raw_config.get("streaming", {})

    @property
    def benchmark_config(self) -> Dict[str, Any]:
        return self._raw_config.get("benchmark", {})


def get_settings() -> Settings:
    """Global accessor for settings."""
    return Settings.get_settings()
