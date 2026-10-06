"""Stage 3: interview every persona about the concept.

Two-step interview, like a real concept test:
  1. Concept WITHOUT price -> reactions + Van Westendorp price thresholds
     (so their price expectations aren't anchored by the client's price).
  2. Price revealed -> value for money + purchase intent at that price.
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable

from ..config import Config
from ..llm import LLM, LLMError
from ..schemas import Brief, InterviewPart1, InterviewPart2, InterviewResponse, Persona

SYSTEM = """You are role-playing a real person taking part in a paid market research interview.

WHO YOU ARE
Name: {name}, {age}, {gender}, {occupation}
{bio}
Household: {household}. Lives in: {location}, {region}. Income level: {income}.
What you use today: {current_solution}
Values: {values}. Frustrations: {frustrations}.
Attitude to new products: {adopter_type}. Price sensitivity: {price_sensitivity}. Skepticism: {skepticism}.

HOW TO ANSWER
- Stay fully in character. Answer as this person really would, in their own voice - not as an AI assistant.
- Be honest, not polite. Real people are often indifferent, confused, or negative about new products, and that is
  valuable data. Do not praise things this person would not care about.
- People overstate how likely they are to buy. Factor in your real budget, habits, and what you use today.
- Keep answers short and natural, like spoken answers.
- Reply with ONLY a JSON object, no other text."""

PART1 = """The interviewer shows you this product concept:

---
{concept}
---

Answer these questions as yourself.
Ratings use 1-5 (1 = very low, 5 = very high).
For the four price questions, give amounts in {currency}{unit_hint}. Think about what YOU would feel.

JSON FORMAT
{{
  "first_reaction": "your honest gut reaction in 1-2 sentences",
  "appeal": 1-5,
  "uniqueness": 1-5,
  "believability": 1-5,
  "likes": ["0-3 things you genuinely like, in your words"],
  "concerns": ["0-3 concerns or doubts, in your words"],
  "price_too_cheap": "price at which it is so cheap you'd doubt the quality",
  "price_bargain": "price at which it is a bargain - great value",
  "price_getting_expensive": "price at which it starts to feel expensive, but you'd still consider it",
  "price_too_expensive": "price at which it is too expensive to consider at all",
  "client_questions": [{client_questions}]{no_price_fields}
}}"""

NO_PRICE_FIELDS = """,
  "purchase_intent": "1 = definitely would not buy, 2 = probably not, 3 = might or might not, 4 = probably would, 5 = definitely would",
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
    return SYSTEM.format(
        name=p.name, age=t.age, gender=t.gender, occupation=p.occupation, bio=p.bio, household=t.household,
        location=t.location, region=brief.region, income=t.income, current_solution=p.current_solution or "n/a",
        values=", ".join(p.values) or "n/a", frustrations=", ".join(p.frustrations) or "n/a",
        adopter_type=t.adopter_type, price_sensitivity=t.price_sensitivity, skepticism=t.skepticism,
    )


def _part1_prompt(brief: Brief) -> str:
    cq = ", ".join(json.dumps({"question": q, "answer": "your answer"}) for q in brief.questions)
    return PART1.format(
        concept=brief.concept_board(include_price=False),
        currency=brief.currency,
        unit_hint=f" ({brief.price_unit})" if brief.price_unit else "",
        client_questions=cq,
        no_price_fields=NO_PRICE_FIELDS if brief.price is None else "",
    )


async def _one(llm: LLM, cfg: Config, brief: Brief, p: Persona) -> InterviewResponse:
    role = cfg.role("personas")
    system = _system(brief, p)
    user1 = _part1_prompt(brief)
    common = dict(model=p.model, system=system, temperature=role.temperature, max_tokens=role.max_tokens)
    ctx = {"persona_id": p.id, "traits": p.traits.model_dump()}

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

    return InterviewResponse(persona_id=p.id, model=p.model, **data)


async def run(llm: LLM, cfg: Config, brief: Brief, personas: list[Persona],
              on_done: Callable[[], None]) -> tuple[list[InterviewResponse], list[str]]:
    async def task(p: Persona):
        try:
            return await _one(llm, cfg, brief, p)
        except LLMError as exc:
            return exc
        finally:
            on_done()

    results = await asyncio.gather(*(task(p) for p in personas))
    rows = [r for r in results if isinstance(r, InterviewResponse)]
    errors = [str(r) for r in results if not isinstance(r, InterviewResponse)]
    return rows, errors
