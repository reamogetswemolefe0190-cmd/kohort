"""Stage 3: interview every persona about the concept.

Two-step interview, like a real concept test:
  1. Concept WITHOUT price -> reactions + Van Westendorp price thresholds
     (so their price expectations aren't anchored by the client's price).
  2. Price revealed -> value for money + purchase intent at that price.
"""

from __future__ import annotations

import json
from typing import Callable

from ..config import Config
from ..llm import LLM, LLMError, gather_cancel_on_error
from ..sampling import assign_variant
from ..schemas import Brief, HookVariant, InterviewPart1, InterviewPart2, InterviewResponse, Persona

SYSTEM = """You are role-playing a real person taking part in a paid market research interview.

WHO YOU ARE
Name: {name}, {age}, {gender}, {occupation}
{bio}
Household: {household}. Lives in: {location}, {region}. Income level: {income}.
What you use today: {current_solution}
Values: {values}. Frustrations: {frustrations}.
Attitude to new products: {adopter_type}. Price sensitivity: {price_sensitivity}. Skepticism: {skepticism}.{lens_block}

HOW TO ANSWER
- Stay fully in character. Answer as this person really would, in their own voice - not as an AI assistant.
- Be honest, not polite. Real people are often indifferent, confused, or negative about new products, and that is
  valuable data. Do not praise things this person would not care about.
- Strict consistency rules:
  1. Your 4 price thresholds MUST strictly increase (`price_too_cheap < price_bargain < price_getting_expensive < price_too_expensive`) and match your income level and price sensitivity.
  2. Your `purchase_intent`, `value_for_money`, `first_reaction`, and `quote` MUST match your `appeal`/`believability` ratings, your `adopter_type`, and your `current_solution` (e.g. if you are content with your current setup, a `laggard`, or rate appeal/believability 1-2, your `purchase_intent` must be 1 or 2, never 4-5).
  3. If a price is revealed and it exceeds your `price_getting_expensive` or `price_too_expensive`, your `value_for_money` and `purchase_intent` must drop accordingly.
- Keep answers short and natural, like spoken answers.
- Reply with ONLY a JSON object, no other text."""

PART1 = """The interviewer shows you this product concept:

---
{concept}
---

Answer these questions as yourself.
Ratings use 1-5 (1 = very low, 5 = very high).
For the four price questions (`price_too_cheap`, `price_bargain`, `price_getting_expensive`, `price_too_expensive`), give plain numbers in {currency}{unit_hint} (no currency symbols or text; must strictly increase):
- `price_too_cheap`: price at which it is so cheap you'd doubt the quality
- `price_bargain`: price at which it is a bargain - great value
- `price_getting_expensive`: price at which it starts to feel expensive, but you'd still consider it
- `price_too_expensive`: price at which it is too expensive to consider at all

JSON FORMAT
{{
  "first_reaction": "your honest gut reaction in 1-2 sentences",
  "appeal": 1-5,
  "uniqueness": 1-5,
  "believability": 1-5,
  "likes": ["0-3 things you genuinely like, in your words"],
  "concerns": ["0-3 concerns or doubts, in your words"],
  "price_too_cheap": 0,
  "price_bargain": 0,
  "price_getting_expensive": 0,
  "price_too_expensive": 0,
  "client_questions": [{client_questions}]{no_price_fields}
}}"""

NO_PRICE_FIELDS = """,
  "purchase_intent": 1-5,
  "would_buy_if": "what would have to be true for you to buy it",
  "quote": "one sentence you might say to a friend about it\""""

PART2 = """Now the interviewer tells you the actual price: {price}.

JSON FORMAT
{{
  "price_reaction": "your honest reaction to that price in 1 sentence",
  "value_for_money": 1-5,
  "purchase_intent": "1 = definitely would not buy, 2 = probably not, 3 = might or might not, 4 = probably would, 5 = definitely would",
  "would_buy_if": "what would have to be true for you to buy it (or buy it sooner)",
  "quote": "one sentence you might say to a friend about it"
}}"""


def _system(brief: Brief, p: Persona) -> str:
    t = p.traits
    lens_block = ""
    if p.category_lens:
        habits = "; ".join(p.category_lens.get("buying_habits") or [])
        anchors = "; ".join(p.category_lens.get("price_anchors") or [])
        alts = ", ".join(p.category_lens.get("status_quo_alternatives") or [])
        frusts = "; ".join(p.category_lens.get("real_frustrations") or [])
        lines = ["\n\nCATEGORY & BUYING CONTEXT"]
        if habits:
            lines.append(f"- Category habits: {habits}")
        if anchors:
            lines.append(f"- Typical price anchors: {anchors}")
        if alts:
            lines.append(f"- Known alternatives in market: {alts}")
        if frusts:
            lines.append(f"- Category frustrations: {frusts}")
        if len(lines) > 1:
            lens_block = "\n".join(lines)
    return SYSTEM.format(
        name=p.name, age=t.age, gender=t.gender, occupation=p.occupation, bio=p.bio, household=t.household,
        location=t.location, region=brief.region, income=t.income, current_solution=p.current_solution or "n/a",
        values=", ".join(p.values) or "n/a", frustrations=", ".join(p.frustrations) or "n/a",
        adopter_type=t.adopter_type, price_sensitivity=t.price_sensitivity, skepticism=t.skepticism,
        lens_block=lens_block,
    )


def _part1_prompt(brief: Brief, variant: HookVariant | None = None) -> str:
    cq = ", ".join(json.dumps({"question": q, "answer": "your answer"}) for q in brief.questions)
    return PART1.format(
        concept=brief.concept_board(include_price=False, variant=variant),
        currency=brief.currency,
        unit_hint=f" ({brief.price_unit})" if brief.price_unit else "",
        client_questions=cq,
        no_price_fields=NO_PRICE_FIELDS if brief.price is None else "",
    )


def validate_interview_item(resp: InterviewResponse, brief: Brief) -> list[str]:
    """Tier-1 deterministic check for single-interview defects before Jev sees the batch."""
    issues: list[str] = []
    prices = [resp.price_too_cheap, resp.price_bargain, resp.price_getting_expensive, resp.price_too_expensive]
    if any(p is None or p <= 0 for p in prices):
        issues.append("All four Van Westendorp price thresholds must be positive numbers (> 0) in strict ascending order")
    elif not (prices[0] < prices[1] < prices[2] < prices[3]):
        issues.append("Price thresholds must strictly increase: price_too_cheap < price_bargain < price_getting_expensive < price_too_expensive")
    if resp.appeal <= 2 and resp.purchase_intent >= 4:
        issues.append(f"Contradictory ratings: appeal={resp.appeal} is low but purchase_intent={resp.purchase_intent} is high")
    elif brief.price is None and resp.appeal >= 4 and resp.purchase_intent == 1:
        issues.append(f"Contradictory unpriced ratings: appeal={resp.appeal} is high but purchase_intent={resp.purchase_intent} is 1")
    combined = f"{resp.first_reaction} {resp.quote}".lower()
    if "as an ai" in combined or "language model" in combined:
        issues.append("Stay in persona character; never refer to yourself as an AI or language model")
    return issues


async def _run_interview_parts(
    llm: LLM, brief: Brief, p: Persona, common: dict, user1: str, ctx: dict, variant: HookVariant | None = None
) -> InterviewResponse:
    part1, raw1 = await llm.structured(task="interview1", schema=InterviewPart1, user=user1, mock_ctx=ctx, **common)
    data = part1.model_dump()
    if brief.price is not None:
        messages = [
            {"role": "user", "content": user1},
            {"role": "assistant", "content": raw1},
            {"role": "user", "content": PART2.format(price=brief.price_label())},
        ]
        part2, _ = await llm.structured(task="interview2", schema=InterviewPart2, messages=messages,
                                        mock_ctx={**ctx, "part1": data}, **common)
        data.update(part2.model_dump())
    elif data.get("purchase_intent") is None:
        data["purchase_intent"] = 3
    return InterviewResponse(persona_id=p.id, model=p.model, variant_id=variant.id if variant else None, **data)


async def _one(llm: LLM, cfg: Config, brief: Brief, p: Persona, seed: int = 42) -> InterviewResponse:
    role = cfg.role("personas")
    variant = assign_variant(p.id, brief.variants, seed=seed)
    system = _system(brief, p)
    user1 = _part1_prompt(brief, variant=variant)
    common = dict(model=p.model, system=system, temperature=role.temperature, max_tokens=role.max_tokens)
    ctx = {
        "persona_id": p.id,
        "traits": p.traits.model_dump(),
        "variant": variant.model_dump() if variant else None,
    }

    resp = await _run_interview_parts(llm, brief, p, common, user1, ctx, variant=variant)
    if not llm.is_mock and cfg.guardrails.self_heal_enabled:
        issues = validate_interview_item(resp, brief)
        if issues:
            repair_user1 = user1 + "\n\nSELF-HEAL CORRECTION REQUIRED:\n- " + "\n- ".join(issues)
            resp = await _run_interview_parts(llm, brief, p, common, repair_user1, ctx, variant=variant)
    return resp


async def run(
    llm: LLM, cfg: Config, brief: Brief, personas: list[Persona], on_done: Callable[[], None], seed: int = 42
) -> tuple[list[InterviewResponse], list[str]]:
    async def task(p: Persona):
        try:
            return await _one(llm, cfg, brief, p, seed=seed)
        except LLMError as exc:
            return exc
        finally:
            on_done()

    results = await gather_cancel_on_error(*(task(p) for p in personas))
    rows = [r for r in results if isinstance(r, InterviewResponse)]
    errors = [str(r) for r in results if not isinstance(r, InterviewResponse)]
    return rows, errors
