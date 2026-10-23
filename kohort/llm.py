"""One gateway for every AI call: routing, concurrency, retries, JSON parsing,
cost tracking and a hard budget stop. In mock mode no network calls are made."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from urllib.parse import urlsplit
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


async def gather_cancel_on_error(*coroutines):
    """Drain sibling tasks on failure before persisting costs or closing clients."""
    tasks = [asyncio.create_task(coro) for coro in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


@dataclass
class CostTracker:
    budget_usd: float
    calls: int = 0
    failures: int = 0
    by_model: dict[str, dict] = field(default_factory=dict)
    reserved_usd: float = 0.0
    uncertain_calls: int = 0

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
        if self.total + self.reserved_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"Spend/reservations ${self.total + self.reserved_usd:.4f} reached budget ${self.budget_usd:.2f}. "
                "Raise run.budget_usd in config/models.yaml if this is expected."
            )

    def reserve(self, amount: float) -> None:
        self.check()
        if self.total + self.reserved_usd + amount > self.budget_usd:
            raise BudgetExceeded("Not enough budget for this request and outstanding reservations.")
        self.reserved_usd += amount

    def settle(self, reservation: float, model: str, input_tokens: int, output_tokens: int, cost: float) -> None:
        self.reserved_usd = max(0.0, self.reserved_usd - reservation)
        self.add(model, input_tokens, output_tokens, cost)
        if self.total > self.budget_usd:
            raise BudgetExceeded("Provider-reported charges exceeded the configured budget; stopping.")

    def summary(self) -> dict:
        return {
            "total_usd": round(self.total, 8),
            "budget_usd": self.budget_usd,
            "calls": self.calls,
            "failed_tasks": self.failures,
            "reserved_or_uncertain_usd": round(self.reserved_usd, 6),
            "uncertain_calls": self.uncertain_calls,
            "by_model": {
                k: {**v, "cost_usd": round(v["cost_usd"], 8)} for k, v in sorted(self.by_model.items())
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
        self.stage_feedback: str = ""
        self._search_sources = ContextVar("search_sources", default=None)
        self.search_log: list[dict] = []

    @property
    def last_search_sources(self) -> list[dict]:
        return self._search_sources.get() or []

    @property
    def is_mock(self) -> bool:
        return self.mocker is not None

    async def aclose(self) -> None:
        await asyncio.gather(*(client.close() for client in self._clients.values()))

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
        if price is None:
            raise LLMError(f"Missing prices for {model}; configure input/output USD per million tokens before live calls.")
        return in_t * price[0] / 1e6 + out_t * price[1] / 1e6

    def _reservation(self, model: str, payload: dict, max_tokens: int, *, questions: int = 1,
                     web_search: bool = False) -> float:
        # Bytes bound token count conservatively; reserve overhead and all output tokens.
        in_t = (len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 1024) * questions
        if web_search:
            # Exa excerpts capped at 2,000 characters/result; UTF-8 uses at most 4 bytes/character.
            in_t += self.cfg.run.web_search_max_uses * self.cfg.run.web_search_max_results * 9000
        cost = self._price_cost(model, in_t, max_tokens)
        if web_search:
            cost += self.cfg.run.web_search_cost_usd * self.cfg.run.web_search_max_uses
        elif model.endswith(":online") or model.endswith("-online"):
            cost += self.cfg.run.web_search_cost_usd
        return cost

    def is_decision_model(self, model_ref: str) -> bool:
        _, model = self._split(model_ref)
        return model.startswith(("typesafe/jev-", "~typesafe/jev-")) and model != "typesafe/jev-router"

    # ------------------------------------------------------------------ calls

    async def _call(self, model_ref: str, messages: list[dict], temperature: float,
                    max_tokens: int, web_search: bool) -> str:
        provider, model = self._split(model_ref)
        self._search_sources.set([])
        if web_search and provider != "openrouter":
            raise LLMError("Grounded web research currently requires the OpenRouter provider.")
        kwargs: dict = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if provider == "openrouter":
            kwargs["extra_body"] = {"usage": {"include": True}}
            if model in self.cfg.reasoning_enabled:
                kwargs["extra_body"]["reasoning"] = {"enabled": self.cfg.reasoning_enabled[model]}
        if web_search:
            kwargs["extra_body"].update({
                "tools": [{"type": "openrouter:web_search", "parameters": {
                    "engine": "exa", "mode": "auto", "max_characters": 2000,
                    "max_results": self.cfg.run.web_search_max_results,
                    "max_total_results": self.cfg.run.web_search_max_results * self.cfg.run.web_search_max_uses,
                    "max_uses": self.cfg.run.web_search_max_uses,
                }}], "max_tool_calls": self.cfg.run.web_search_max_uses,
            })

        client = self._client(provider)
        async with self._sem:
            reservation = self._reservation(model, {"messages": messages}, max_tokens, web_search=web_search)
            self.tracker.reserve(reservation)
            try:
                if provider == "google":
                    await asyncio.sleep(4.1)
                resp = await client.chat.completions.create(**kwargs)
                usage = resp.usage
                in_t = getattr(usage, "prompt_tokens", 0) or 0
                out_t = getattr(usage, "completion_tokens", 0) or 0
                reported = (getattr(usage, "model_extra", None) or {}).get("cost") if usage else None
                if reported is not None:
                    cost = float(reported)
                elif usage is None or getattr(usage, "prompt_tokens", None) is None or getattr(usage, "completion_tokens", None) is None:
                    cost = reservation
                else:
                    cost = self._price_cost(model, in_t, out_t)
                    if web_search:
                        cost += self.cfg.run.web_search_cost_usd * self.cfg.run.web_search_max_uses
                    elif model.endswith(":online") or model.endswith("-online"):
                        cost += self.cfg.run.web_search_cost_usd
                if not math.isfinite(cost) or cost < 0:
                    raise LLMError("Invalid cost returned by provider.")
            except BaseException:
                # A timeout/cancellation can still be billed. Keep its reservation.
                self.tracker.uncertain_calls += 1
                raise
            self.tracker.settle(reservation, model_ref, in_t, out_t, cost)

        if not resp.choices:
            raise ValueError("empty reply: no choices returned")
        message = resp.choices[0].message
        if web_search:
            sources = []
            for annotation in getattr(message, "annotations", None) or []:
                annotation = annotation.model_dump() if hasattr(annotation, "model_dump") else annotation
                if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
                    continue
                citation = annotation.get("url_citation", {})
                url = citation.get("url", "")
                parsed = urlsplit(url)
                if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username:
                    continue
                if url not in {s["url"] for s in sources}:
                    sources.append({"url": url, "title": citation.get("title", url),
                                    "content": str(citation.get("content") or "")[:2000]})
            self._search_sources.set(sources)
            self.search_log.append({"model": model_ref, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                    "sources": sources,
                                    "finish_reason": getattr(resp.choices[0], "finish_reason", None),
                                    "input_tokens": in_t, "output_tokens": out_t,
                                    "empty_content": not bool(message.content)})
        if not message.content:
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", None)
            raise ValueError(f"empty reply: finish_reason={getattr(resp.choices[0], 'finish_reason', None)}, "
                             f"output_tokens={out_t}, reasoning_tokens={reasoning_tokens}")
        return message.content

    async def decide(self, model_ref: str, state: str | dict, questions: dict[str, dict]) -> dict:
        """Retry explicit transient HTTP failures; every attempt reserves its own budget."""
        for attempt in range(min(3, self.cfg.run.max_retries + 1)):
            try:
                return await self._decide_once(model_ref, state, questions)
            except LLMError as exc:
                match = re.fullmatch(r"Jev Decisions API failed: HTTP (\d+)\.", str(exc))
                transient = match and (int(match[1]) == 429 or 500 <= int(match[1]) <= 599)
                if not transient or attempt == min(3, self.cfg.run.max_retries + 1) - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def _decide_once(self, model_ref: str, state: str | dict, questions: dict[str, dict]) -> dict:
        """Call OpenRouter's typed Decisions API; never use the chat router for judgment."""
        if not questions or any(q.get("type") != "noul" for q in questions.values()):
            raise LLMError("Kohort decision checks require a nonempty map of noul questions.")
        if not self.is_decision_model(model_ref):
            raise LLMError("Use a Jev decision model such as typesafe/jev-1.13, not typesafe/jev-router.")
        if self.is_mock:
            self.tracker.add(f"mock:{model_ref}", 0, 0, 0.0)
            return {"model": model_ref, "answers": {
                name: {"type": "noul", "noul": 1.0} for name in questions
            }, "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}}
        provider, model = self._split(model_ref)
        if provider != "openrouter":
            raise LLMError("Decision models currently only supported via OpenRouter.")
            
        if provider not in self.cfg.providers:
            raise LLMError(f"Provider '{provider}' is not configured.")
        p = self.cfg.providers[provider]
        key = os.environ.get(p.api_key_env, "").strip()
        if not key:
            raise LLMError(f"Missing API key: set {p.api_key_env} or run with --mock.")
        payload = {"model": model, "state": state, "questions": questions}
        import httpx
        async with self._sem:
            reservation = self._reservation(model, payload, 0, questions=len(questions))
            self.tracker.reserve(reservation)
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    url = p.base_url.rstrip("/").removesuffix("/v1") + "/alpha/decisions"
                    resp = await client.post(url, headers={"Authorization": f"Bearer {key}",
                                                          "X-Title": "Kohort-Guardrails"}, json=payload)
                    if resp.status_code != 200:
                        raise LLMError(f"Jev Decisions API failed: HTTP {resp.status_code}.")
                    data = resp.json()
                usage = data["usage"]
                cost = float(usage["cost"])
                in_t, out_t = int(usage["input_tokens"]), int(usage.get("output_tokens", 0))
                if not math.isfinite(cost) or cost < 0 or in_t < 0 or out_t < 0:
                    raise ValueError("invalid usage")
            except BaseException as exc:
                self.tracker.uncertain_calls += 1
                self.tracker.failures += 1
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit, LLMError)):
                    raise
                raise LLMError(f"Jev request failed ({type(exc).__name__}); no decision accepted.") from exc
            self.tracker.settle(reservation, model_ref, in_t, out_t, cost)
            try:
                for name, question in questions.items():
                    answer = data["answers"][name]
                    if answer["type"] != question["type"] or question["type"] != "noul":
                        raise ValueError("unexpected answer type")
                    prob = answer["noul"]
                    if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not math.isfinite(prob) or not 0 <= prob <= 1:
                        raise ValueError("invalid probability")
            except (KeyError, TypeError, ValueError) as exc:
                self.tracker.failures += 1
                raise LLMError("Invalid Jev answer; no decision accepted.") from exc
            return data

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

        if self.stage_feedback:
            system += ("\n\nQUALITY REVIEW FEEDBACK FOR THIS STAGE\n" + self.stage_feedback +
                       "\nCorrect the flagged weaknesses honestly. Do not hide uncertainty, invent "
                       "supporting evidence, or change fixed traits or arithmetic to pass a check.")
        convo = [{"role": "system", "content": system}]
        convo += messages if messages is not None else [{"role": "user", "content": user or ""}]

        last_err: Exception | None = None
        repaired = False
        empty_replies = 0
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
                if str(exc).startswith("empty reply"):
                    empty_replies += 1
                    if empty_replies >= 2:
                        break
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
