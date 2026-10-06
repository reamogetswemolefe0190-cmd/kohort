"""Stage 2: build the synthetic customer panel.

Traits are sampled in code (see sampling.py); the LLM writes a believable human
on top. Personas never see the product at this stage, to avoid priming.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from ..config import Config
from ..llm import LLM, LLMError
from ..schemas import Brief, MarketResearch, Persona, PersonaProfile, PersonaTraits

SYSTEM = """You create realistic consumer personas for market research simulations.
Make each person specific, ordinary and believable - a real individual, not a marketing archetype.
Most real people are busy, have budget limits and settled habits, and are not looking for new products.
Avoid stereotypes and avoid making everyone articulate, affluent or eco-conscious unless the traits say so.
Reply with ONLY a JSON object, no other text."""

USER = """Create one persona living in {region}.

FIXED TRAITS (must be respected exactly)
- Segment: {segment} - {segment_desc}
- Age: {age}
- Gender: {gender}
- Income level: {income}
- Household: {household}
- Location type: {location}
- Attitude to new products: {adopter_type}
- Price sensitivity: {price_sensitivity}
- General skepticism toward brands and advertising: {skepticism}

CONTEXT
This person will later be interviewed about products in the category "{category}".
Describe how they currently handle their needs in that category (brands, habits, where they buy), but do NOT
mention any specific new product.

JSON FORMAT
{{
  "name": "full name typical for the region",
  "occupation": "...",
  "bio": "4-6 sentences: daily life, priorities, money habits, personality",
  "current_solution": "what they use/do today in this category and how satisfied they are",
  "values": ["3 things that matter to them"],
  "frustrations": ["2-3 everyday frustrations"]
}}"""


async def _one(llm: LLM, cfg: Config, brief: Brief, segs: dict, pid: str, traits: PersonaTraits,
               model: str) -> Persona:
    role = cfg.role("personas")
    seg = segs.get(traits.segment)
    prompt = USER.format(
        region=brief.region, segment=traits.segment, segment_desc=seg.description if seg else "",
        category=brief.category, **traits.model_dump(exclude={"segment"}),
    )
    profile, _ = await llm.structured(
        task="persona", schema=PersonaProfile, model=model, system=SYSTEM, user=prompt,
        temperature=role.temperature, max_tokens=role.max_tokens,
        mock_ctx={"persona_id": pid, "traits": traits.model_dump()},
    )
    return Persona(id=pid, model=model, traits=traits, **profile.model_dump())


async def run(llm: LLM, cfg: Config, brief: Brief, research: MarketResearch,
              sampled: list[tuple[PersonaTraits, str]], on_done: Callable[[], None]) -> tuple[list[Persona], list[str]]:
    segs = {s.name: s for s in research.segments}

    async def task(i: int, traits: PersonaTraits, model: str):
        try:
            return await _one(llm, cfg, brief, segs, f"P{i + 1:03d}", traits, model)
        except LLMError as exc:
            return exc
        finally:
            on_done()

    results = await asyncio.gather(*(task(i, t, m) for i, (t, m) in enumerate(sampled)))
    personas = [r for r in results if isinstance(r, Persona)]
    errors = [str(r) for r in results if not isinstance(r, Persona)]
    return personas, errors
