"""Loads config/models.yaml: which AI model does which job, prices, and run limits."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    concurrency: int = Field(default=16, ge=1)
    max_retries: int = Field(default=3, ge=0)
    budget_usd: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    web_search_cost_usd: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    web_search_max_uses: int = Field(default=2, ge=1, le=5)
    web_search_max_results: int = Field(default=5, ge=1, le=10)
    intent_weights: dict[str, float] = Field(default_factory=lambda: {"definitely": 0.75, "probably": 0.25})


class ScrapingCfg(BaseModel):
    enabled: bool = False
    max_pages_per_scan: int = Field(default=6, ge=1, le=20)
    concurrency: int = Field(default=2, ge=1, le=4)
    timeout_seconds: float = Field(default=15, ge=1, le=60, allow_inf_nan=False)
    max_chars_per_page: int = Field(default=8000, ge=500, le=20000)


class GuardrailCfg(BaseModel):
    advantage_review_mode: Literal['enforce', 'observe'] = 'enforce'
    model: str = "typesafe/jev-1.13"
    # Starting policy, not a threshold calibrated against real customer studies.
    max_risk_probability: float = Field(default=0.2, ge=0, lt=1, allow_inf_nan=False)
    max_stage_retries: int = Field(default=1, ge=0, le=2)
    retry_min_probability: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    self_heal_enabled: bool = True
    repairable_min_probability: float = Field(default=0.25, gt=0, le=1, allow_inf_nan=False)
    advisory_max_probability: float = Field(default=0.35, ge=0, le=1, allow_inf_nan=False)
    batch_size: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def _thresholds(self):
        if self.retry_min_probability <= self.max_risk_probability:
            raise ValueError("retry_min_probability must exceed max_risk_probability")
        return self


class Config(BaseModel):
    providers: dict[str, ProviderCfg]
    default_provider: str = "openrouter"
    roles: dict[str, RoleCfg]
    prices: dict[str, list[float]] = Field(default_factory=dict)
    reasoning_enabled: dict[str, bool] = Field(default_factory=dict)
    run: RunCfg = Field(default_factory=RunCfg)
    guardrails: GuardrailCfg = Field(default_factory=GuardrailCfg)
    scraping: ScrapingCfg = Field(default_factory=ScrapingCfg)

    @model_validator(mode="after")
    def _validate_prices(self):
        import math
        for model, price in self.prices.items():
            if len(price) != 2 or any(not math.isfinite(p) or p < 0 for p in price):
                raise ValueError(f"prices.{model} must contain two finite nonnegative rates")
        return self

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
        if self.guardrails.model not in seen:
            seen.append(self.guardrails.model)
        return seen


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        return Config.model_validate(yaml.safe_load(fh))
