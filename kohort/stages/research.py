"""Stage 1: desk research - market context, competitors and customer segments."""

from __future__ import annotations

from rich.console import Console

from ..config import Config
from ..llm import LLM, LLMError
from ..schemas import Brief, MarketResearch

SYSTEM = """You are a senior market research analyst preparing the groundwork for a concept test.
Be factual, specific and realistic. If you are not sure about a fact, say so in "caveats" rather than inventing it.
Reply with ONLY a JSON object, no other text."""

USER = """CLIENT BRIEF
Client: {client}
Product: {product_name}
Category: {category}
One-liner: {one_liner}
Description: {description}
Key features: {features}
Intended price: {price}
Target market (client's view): {target_market}
Region: {region}
Known competitors: {competitors}

TASK
1. Summarise the market context for this category in this region (size signals, dynamics, buyer behaviour).
2. List the most relevant competitors and alternatives, INCLUDING the status quo (what people do today instead).
3. Define 3-5 realistic customer segments for the concept test. Include segments that may be lukewarm or
   resistant, not just the enthusiastic target group. "share" = approximate share of the addressable market.
4. Note the typical price range for comparable offerings.

JSON FORMAT
{{
  "summary": "3-6 sentence market overview",
  "trends": ["..."],
  "competitors": [{{"name": "...", "offering": "...", "price_point": "...", "strengths": ["..."], "weaknesses": ["..."]}}],
  "typical_price_range": "...",
  "segments": [{{
      "name": "short segment name",
      "description": "who they are and how they buy",
      "share": 0.3,
      "age_min": 25, "age_max": 45,
      "income_levels": ["one or more of: low, lower-middle, middle, upper-middle, high"],
      "key_needs": ["..."],
      "likely_objections": ["..."]
  }}],
  "sources": ["URLs or named sources you relied on"],
  "caveats": ["what is uncertain or may be outdated"]
}}"""


def _prompt(brief: Brief) -> str:
    return USER.format(
        client=brief.client,
        product_name=brief.product_name,
        category=brief.category,
        one_liner=brief.one_liner,
        description=brief.description,
        features="; ".join(brief.key_features) or "n/a",
        price=brief.price_label(),
        target_market=brief.target_market,
        region=brief.region,
        competitors=", ".join(brief.competitors) or "none given",
    )


async def run(llm: LLM, cfg: Config, brief: Brief, console: Console) -> MarketResearch:
    role = cfg.role("research")
    kwargs = dict(task="research", schema=MarketResearch, model=role.model, system=SYSTEM, user=_prompt(brief),
                  temperature=role.temperature, max_tokens=role.max_tokens, mock_ctx={})
    try:
        research, _ = await llm.structured(web_search=role.web_search, **kwargs)
        research.web_search_used = role.web_search and not llm.is_mock
    except LLMError as exc:
        if not role.web_search:
            raise
        console.print(f"  [yellow]Web search failed ({exc}); retrying without it.[/]")
        research, _ = await llm.structured(web_search=False, **kwargs)
        research.caveats.append("Live web search was unavailable; research is based on model knowledge only.")
    return research
