"""Loads config/models.yaml: which AI model does which job, prices, and run limits."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "models.yaml"


class ProviderCfg(BaseModel):
    base_url: str
    api_key_env: str


class RoleCfg(BaseModel):
    model: str | None = None
    models: list[str] = Field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4000
    web_search: bool = False

    @model_validator(mode="after")
    def _sync(self):
        if self.model is None and self.models:
            self.model = self.models[0]
        if not self.models and self.model:
            self.models = [self.model]
        if not self.model:
            raise ValueError("each role needs 'model' or 'models'")
        return self


class RunCfg(BaseModel):
    concurrency: int = 16
    max_retries: int = 3
    budget_usd: float = 10.0
    intent_weights: dict[str, float] = Field(default_factory=lambda: {"definitely": 0.75, "probably": 0.25})


class Config(BaseModel):
    providers: dict[str, ProviderCfg]
    default_provider: str = "openrouter"
    roles: dict[str, RoleCfg]
    prices: dict[str, list[float]] = Field(default_factory=dict)
    run: RunCfg = Field(default_factory=RunCfg)

    def role(self, name: str) -> RoleCfg:
        if name not in self.roles:
            raise KeyError(f"Role '{name}' missing from config 'roles' section")
        return self.roles[name]

    def all_models(self) -> list[str]:
        seen: list[str] = []
        for role in self.roles.values():
            for m in role.models:
                if m not in seen:
                    seen.append(m)
        return seen


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        return Config.model_validate(yaml.safe_load(fh))
