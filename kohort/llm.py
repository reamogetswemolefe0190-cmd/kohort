"""One gateway for every AI call: routing, concurrency, retries, JSON parsing,
cost tracking and a hard budget stop. In mock mode no network calls are made."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .config import Config

if TYPE_CHECKING:
    from .mock import Mocker

T = TypeVar("T", bound=BaseModel)


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


@dataclass
class CostTracker:
    budget_usd: float
    calls: int = 0
    failures: int = 0
    by_model: dict[str, dict] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(m["cost_usd"] for m in self.by_model.values())

    def add(self, model: str, input_tokens: int, output_tokens: int, cost: float) -> None:
        self.calls += 1
        row = self.by_model.setdefault(
            model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        )
        row["calls"] += 1
        row["input_tokens"] += input_tokens
        row["output_tokens"] += output_tokens
        row["cost_usd"] += cost

    def check(self) -> None:
        if self.total > self.budget_usd:
            raise BudgetExceeded(
                f"Spend ${self.total:.2f} exceeded budget ${self.budget_usd:.2f}. "
                "Raise run.budget_usd in config/models.yaml if this is expected."
            )

    def summary(self) -> dict:
        return {
            "total_usd": round(self.total, 4),
            "budget_usd": self.budget_usd,
            "calls": self.calls,
            "failed_tasks": self.failures,
            "by_model": {
                k: {**v, "cost_usd": round(v["cost_usd"], 4)} for k, v in sorted(self.by_model.items())
            },
        }


def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model reply (tolerates ```json fences and chatter)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in reply")
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("reply JSON is not an object")
    return data


class LLM:
    def __init__(self, cfg: Config, tracker: CostTracker, mocker: "Mocker | None" = None):
        self.cfg = cfg
        self.tracker = tracker
        self.mocker = mocker
        self._sem = asyncio.Semaphore(cfg.run.concurrency)
        self._clients: dict[str, AsyncOpenAI] = {}

    @property
    def is_mock(self) -> bool:
        return self.mocker is not None

    # ------------------------------------------------------------------ routing

    def _split(self, model_ref: str) -> tuple[str, str]:
        if "::" in model_ref:
            provider, model = model_ref.split("::", 1)
            return provider, model
        return self.cfg.default_provider, model_ref

    def _client(self, provider: str) -> AsyncOpenAI:
        if provider not in self._clients:
            if provider not in self.cfg.providers:
                raise LLMError(f"Provider '{provider}' is not defined in config/models.yaml")
            p = self.cfg.providers[provider]
            key = os.environ.get(p.api_key_env, "").strip()
            if not key:
                raise LLMError(
                    f"Missing API key: set {p.api_key_env} in your .env file "
                    "(copy .env.example to .env), or run with --mock."
                )
            self._clients[provider] = AsyncOpenAI(
                base_url=p.base_url,
                api_key=key,
                timeout=180,
                max_retries=0,  # we retry ourselves
                default_headers={"X-Title": "Kohort"},
            )
        return self._clients[provider]

    def _price_cost(self, model: str, in_t: int, out_t: int) -> float:
        price = self.cfg.prices.get(model.removesuffix(":online"))
        if not price:
            return 0.0
        return in_t * price[0] / 1e6 + out_t * price[1] / 1e6

    # ------------------------------------------------------------------ calls

    async def _call(self, model_ref: str, messages: list[dict], temperature: float,
                    max_tokens: int, web_search: bool) -> str:
        provider, model = self._split(model_ref)
        if web_search and provider == "openrouter" and not model.endswith(":online"):
            model += ":online"
        kwargs: dict = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if provider == "openrouter":
            kwargs["extra_body"] = {"usage": {"include": True}}

        client = self._client(provider)
        async with self._sem:
            if provider == "google":
                # Google Free Tier allows 15 RPM (1 request every 4 seconds)
                await asyncio.sleep(4.1)
            resp = await client.chat.completions.create(**kwargs)

        usage = resp.usage
        in_t = getattr(usage, "prompt_tokens", 0) or 0
        out_t = getattr(usage, "completion_tokens", 0) or 0
        reported = (getattr(usage, "model_extra", None) or {}).get("cost") if usage else None
        cost = float(reported) if reported is not None else self._price_cost(model, in_t, out_t)
        self.tracker.add(model_ref, in_t, out_t, cost)

        if not resp.choices or not resp.choices[0].message.content:
            raise ValueError("empty reply")
        return resp.choices[0].message.content

    async def decide(self, model_ref: str, state: str, questions: list[dict], web_search: bool = False) -> dict:
        """Call a System One decision model (e.g. Jev) via OpenRouter."""
        provider, model = self._split(model_ref)
        if provider != "openrouter":
            raise LLMError("Decision models currently only supported via OpenRouter.")
            
        p = self.cfg.providers[provider]
        key = os.environ.get(p.api_key_env, "").strip()
        
        import httpx
        async with self._sem:
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = p.base_url.removesuffix('/chat/completions').removesuffix('/') + "/systemone"
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {key}", "X-Title": "Kohort-Critic"},
                    json={
                        "model": model,
                        "state": state,
                        "questions": questions
                    }
                )
                if resp.status_code != 200:
                    raise LLMError(f"System One API failed: {resp.status_code} {resp.text}")
                
                # Jev doesn't return output tokens, so cost tracking is just input length roughly
                self.tracker.add(model_ref, len(state) // 4, 0, 0.0) 
                return resp.json()

    async def structured(
        self,
        *,
        task: str,
        schema: type[T],
        model: str,
        system: str,
        user: str | None = None,
        messages: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        web_search: bool = False,
        mock_ctx: dict | None = None,
    ) -> tuple[T, str]:
        """Ask a model for JSON matching `schema`. Returns (parsed object, raw text)."""
        if self.mocker is not None:
            data = self.mocker.respond(task, mock_ctx or {})
            self.tracker.add(f"mock:{model}", 0, 0, 0.0)
            return schema.model_validate(data), json.dumps(data, ensure_ascii=False)

        convo = [{"role": "system", "content": system}]
        convo += messages if messages is not None else [{"role": "user", "content": user or ""}]

        last_err: Exception | None = None
        repaired = False
        for attempt in range(self.cfg.run.max_retries + 1):
            self.tracker.check()
            raw = None
            try:
                raw = await self._call(model, convo, temperature, max_tokens, web_search)
                return schema.model_validate(extract_json(raw)), raw
            except (BudgetExceeded, LLMError):
                raise
            except (ValueError, ValidationError) as exc:
                last_err = exc
                if raw and not repaired:
                    # Show the model its bad reply and the error, once.
                    repaired = True
                    convo = convo + [
                        {"role": "assistant", "content": raw[:6000]},
                        {"role": "user", "content": (
                            f"That reply could not be parsed: {str(exc)[:500]}\n"
                            "Reply again with ONLY the corrected JSON object."
                        )},
                    ]
                    continue
            except Exception as exc:  # network / rate limit / provider errors
                last_err = exc
            await asyncio.sleep(min(2 ** attempt + random.random(), 20))

        self.tracker.failures += 1
        raise LLMError(f"{task} failed with {model} after retries: {last_err}")
