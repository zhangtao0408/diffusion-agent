"""Application configuration via environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "DA_"}

    llm_provider: Literal["openai", "anthropic", "local"] = "openai"
    llm_model: str = "gpt-4o"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    log_level: str = "INFO"
    work_dir: Path = Path("./workspace")
    npu_ssh_host: Optional[str] = None
    npu_conda_env: Optional[str] = None

    def get_work_dir(self) -> Path:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return self.work_dir.resolve()


def load_settings() -> Settings:
    return Settings()
