"""Stage 4: synthesis - theme coding, independent critique, report narrative.

Numbers are computed in analysis.py and handed to the models; the models
interpret but never calculate.
"""

from __future__ import annotations

import asyncio
import json

from ..analysis import theme_stats
from ..config import Config
from ..llm import LLM
from ..schemas import (Brief, CriticReview, CriticIssue, InterviewResponse, MarketResearch, Persona, ReportNarrative,
                       ThemeCoding)

# --------------------------------------------------------------------------- themes

THEME_SYSTEM = """You are a qualitative market research analyst coding open-ended survey answers.
Create a compact codebook of distinct, mutually exclusive themes and assign every answer to exactly one theme.
Theme labels must be short (2-5 words) and specific (e.g. "Worried about cancellation", not "Negative").
Reply with ONLY a JSON object, no other text."""

THEME_USER = """Product: {product} - {one_liner}
Question type: {kind_desc}

Code the {n} answers below into {k_min}-{k_max} themes. Use an "Other" theme only for genuine outliers.

ANSWERS (index: text)
{items}

JSON FORMAT
{{
  "themes": [{{"id": "T1", "label": "short label", "description": "one sentence"}}],
  "assignments": {{"0": "T1", "1": "T3", "...": "every index must appear exactly once"}}
}}"""

KINDS = {
    "likes": "things respondents LIKED about the concept",
    "concerns": "CONCERNS or doubts respondents raised about the concept",
    "triggers": "what would need to be true for respondents to BUY (conversion triggers)",
}


def _coding_items(rows: list[InterviewResponse], kind: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for r in rows:
        texts = {"likes": r.likes, "concerns": r.concerns, "triggers": [r.would_buy_if]}[kind]
        items += [(r.persona_id, t.strip()) for t in texts if t and t.strip()]
    return items


async def _code(llm: LLM, cfg: Config, brief: Brief, kind: str, rows: list[InterviewResponse],
                personas: list[Persona]) -> list[dict]:
    items = _coding_items(rows, kind)
    if not items:
        return []
    role = cfg.role("analyst")
    coding, _ = await llm.structured(
        task="themes", schema=ThemeCoding, model=role.model, system=THEME_SYSTEM,
        user=THEME_USER.format(
            product=brief.product_name, one_liner=brief.one_liner, kind_desc=KINDS[kind], n=len(items),
            k_min=4, k_max=8, items="\n".join(f"{i}: {t}" for i, (_, t) in enumerate(items)),
        ),
        temperature=role.temperature, max_tokens=role.max_tokens,
        mock_ctx={"items": [t for _, t in items]},
    )
    return theme_stats(items, coding.assignments, [t.model_dump() for t in coding.themes], personas, len(rows))


async def code_themes(llm: LLM, cfg: Config, brief: Brief, rows: list[InterviewResponse],
                      personas: list[Persona]) -> dict[str, list[dict]]:
    kinds = list(KINDS)
    results = await asyncio.gather(*(_code(llm, cfg, brief, k, rows, personas) for k in kinds))
    return dict(zip(kinds, results))


def _compact_themes(themes: dict[str, list[dict]], top: int = 6) -> dict:
    return {k: [{"label": t["label"], "share_of_personas": t["share"], "by_segment": t["by_segment"],
                 "examples": t["examples"][:2]} for t in v[:top]] for k, v in themes.items()}


# --------------------------------------------------------------------------- critic

CRITIC_SYSTEM = """You are a skeptical research director reviewing a synthetic (AI-simulated) concept test before
it goes to a client. Your job is to protect the client from overconfident conclusions.
Look for: AI positivity bias, uniform or stereotyped personas, small subgroups, contradictions between numbers and
themes, weak market research, and anything the client might over-interpret.
Reply with ONLY a JSON object, no other text."""

CRITIC_USER = """CONCEPT: {product} - {one_liner} (price: {price})

QUANTITATIVE RESULTS (computed, reliable arithmetic)
{quant}

THEMES (share = fraction of personas mentioning it)
{themes}

AUTOMATIC DIAGNOSTIC FLAGS
{flags}

RESEARCH CAVEATS
{caveats}

JSON FORMAT
{{
  "issues": [{{"severity": "low|medium|high", "issue": "...", "recommendation": "what to do about it"}}],
  "confidence": "low|medium|high",
  "confidence_rationale": "1-3 sentences on how much weight the client should put on these findings"
}}"""


async def critique(llm: LLM, cfg: Config, brief: Brief, quant: dict, themes: dict,
                   research: MarketResearch) -> CriticReview:
    role = cfg.role("critic")
    if role.model.startswith(("typesafe/", "liquid/", "cloudflare/")):
        # System One Harness (Jev, Liquid, etc.)
        state_text = f"PRODUCT: {brief.product_name}\n\nQUANT:\n{json.dumps(quant['diagnostics'])}\n\nCAVEATS:\n{research.caveats}"
        resp = await llm.decide(
            model_ref=role.model,
            state=state_text,
            questions=[
                {
                    "name": "confidence",
                    "type": "choice",
                    "choices": ["low", "medium", "high"],
                    "prompt": "Given the diagnostic flags, rate the statistical confidence of this data."
                },
                {
                    "name": "has_fatal_flaw",
                    "type": "nou1",
                    "prompt": "Is there a severe hallucination, positivity bias, or fatal data flaw?"
                }
            ]
        )
        
        # Parse decision probabilities
        conf = "medium"
        if "confidence" in resp and "choice" in resp["confidence"]:
            conf = resp["confidence"]["choice"]
            
        fatal_prob = 0.0
        if "has_fatal_flaw" in resp and "probability" in resp["has_fatal_flaw"]:
            fatal_prob = resp["has_fatal_flaw"]["probability"]
            
        issues = []
        if fatal_prob > 0.5:
            issues.append(CriticIssue(severity="high", issue=f"Jev System One Harness detected fatal data flaw ({fatal_prob:.0%} probability)", recommendation="Halt report or discard synthetic data."))
            
        return CriticReview(
            confidence=conf,
            confidence_rationale=f"System One decision harness calculated a {fatal_prob:.0%} probability of fatal data flaws.",
            issues=issues
        )
        
    # Standard generative critic
    review, _ = await llm.structured(
        task="critic", schema=CriticReview, model=role.model, system=CRITIC_SYSTEM,
        user=CRITIC_USER.format(
            product=brief.product_name, one_liner=brief.one_liner, price=brief.price_label(),
            quant=json.dumps({k: quant[k] for k in ("n", "metrics", "trial_estimate", "by", "pricing")}, indent=1),
            themes=json.dumps(_compact_themes(themes), indent=1, ensure_ascii=False),
            flags="\n".join(f"- {f}" for f in flags) or "none",
            caveats="\n".join(f"- {c}" for c in research.caveats) or "none",
        ),
        temperature=role.temperature, max_tokens=role.max_tokens,
        mock_ctx={"flags": flags},
    )
    return review


# --------------------------------------------------------------------------- report narrative

REPORT_SYSTEM = """You are a senior strategy consultant at Kohort writing the narrative of a concept test report
for a business client. Write clearly and concretely for a busy decision-maker. Be candid: a useful "no" beats a
flattering "yes".
RULES
- Only use numbers that appear in the data provided. Never invent statistics.
- Treat this as directional evidence from a simulated market, not a forecast.
- Take the reviewer's critique seriously and reflect it in your confidence and wording.
Reply with ONLY a JSON object, no other text."""

REPORT_USER = """BRIEF
{brief}

MARKET RESEARCH SUMMARY
{research}

RESULTS (n = {n} synthetic personas)
{quant}

THEMES (share = fraction of personas mentioning it)
{themes}

PERSONA ANSWERS TO THE CLIENT'S QUESTIONS (sample)
{client_answers}

INDEPENDENT REVIEWER CRITIQUE (confidence: {confidence})
{critique}

VERDICT GUIDE
GO = strong appeal and calibrated trial in at least one sizeable segment, no fatal barrier.
ITERATE = real interest but fixable barriers (price, messaging, features, trust).
NO-GO = weak interest across segments or a fundamental barrier.

JSON FORMAT
{{
  "verdict": "GO | ITERATE | NO-GO",
  "verdict_rationale": "2-3 sentences",
  "executive_summary": "one tight paragraph (5-7 sentences)",
  "key_findings": ["5-7 specific findings, citing numbers from the data"],
  "segment_insights": [{{"segment": "name", "insight": "1-2 sentences"}}],
  "recommendations": [{{"priority": 1, "action": "concrete next step", "rationale": "why, tied to evidence"}}],
  "client_question_answers": [{{"question": "client's question", "answer": "synthesised answer"}}],
  "risks": ["key risks or unknowns"]
}}"""


def _client_answer_sample(brief: Brief, rows: list[InterviewResponse], personas: list[Persona],
                          per_q: int = 30) -> dict:
    pmap = {p.id: p for p in personas}
    out: dict[str, list[str]] = {q: [] for q in brief.questions}
    for r in rows:
        for qa in r.client_questions:
            for q in brief.questions:
                if qa.question.strip().lower()[:40] == q.strip().lower()[:40] and len(out[q]) < per_q:
                    out[q].append(f"[{pmap[r.persona_id].traits.segment}, intent {r.purchase_intent}] {qa.answer}")
    return out


async def write_narrative(llm: LLM, cfg: Config, brief: Brief, research: MarketResearch, quant: dict,
                          themes: dict, critic: CriticReview, rows: list[InterviewResponse],
                          personas: list[Persona]) -> ReportNarrative:
    role = cfg.role("report")
    compact = _compact_themes(themes)
    narrative, _ = await llm.structured(
        task="report", schema=ReportNarrative, model=role.model, system=REPORT_SYSTEM,
        user=REPORT_USER.format(
            brief=brief.model_dump_json(indent=1),
            research=json.dumps({"summary": research.summary, "trends": research.trends,
                                 "typical_price_range": research.typical_price_range,
                                 "competitors": [c.name for c in research.competitors]}, ensure_ascii=False),
            n=quant["n"],
            quant=json.dumps({k: quant[k] for k in ("metrics", "trial_estimate", "by", "pricing")}, indent=1),
            themes=json.dumps(compact, indent=1, ensure_ascii=False),
            client_answers=json.dumps(_client_answer_sample(brief, rows, personas), indent=1, ensure_ascii=False),
            confidence=critic.confidence,
            critique=json.dumps(critic.model_dump(), indent=1, ensure_ascii=False),
        ),
        temperature=role.temperature, max_tokens=role.max_tokens,
        mock_ctx={"quant": quant,
                  "top_likes": [t["label"] for t in themes.get("likes", [])],
                  "top_concerns": [t["label"] for t in themes.get("concerns", [])]},
    )
    return narrative
