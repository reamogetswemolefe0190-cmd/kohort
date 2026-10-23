"""Stage 4: synthesis - theme coding, independent critique, report narrative.

Numbers are computed in analysis.py and handed to the models; the models
interpret but never calculate.
"""

from __future__ import annotations

import json

from ..analysis import theme_stats
from ..config import Config
from ..llm import LLM, LLMError, gather_cancel_on_error
from ..schemas import (Brief, CriticReview, InterviewResponse, MarketResearch, Persona, ReportNarrative,
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
                personas: list[Persona], audit: dict | None = None) -> list[dict]:
    items = _coding_items(rows, kind)
    if not items:
        if audit is not None:
            audit[kind] = []
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
    theme_map = {t.id: t for t in coding.themes}
    if (len(theme_map) != len(coding.themes) or set(coding.assignments) != {str(i) for i in range(len(items))}
            or any(tid not in theme_map for tid in coding.assignments.values())):
        raise LLMError(f"Invalid {kind} theme coding: every answer must map to exactly one defined theme.")
    if audit is not None:
        audit[kind] = [{"persona_id": pid, "text": text,
                        "theme": theme_map[coding.assignments[str(i)]].model_dump()}
                       for i, (pid, text) in enumerate(items)]
    return theme_stats(items, coding.assignments, [t.model_dump() for t in coding.themes], personas, len(rows))


async def code_themes(llm: LLM, cfg: Config, brief: Brief, rows: list[InterviewResponse],
                      personas: list[Persona], audit: dict | None = None) -> dict[str, list[dict]]:
    kinds = list(KINDS)
    results = await gather_cancel_on_error(*(_code(llm, cfg, brief, k, rows, personas, audit) for k in kinds))
    return dict(zip(kinds, results))


def _compact_themes(themes: dict[str, list[dict]], top: int = 6) -> dict:
    return {k: [{"label": t["label"], "share_of_personas": t["share"], "by_segment": t["by_segment"],
                 "examples": t["examples"][:2]} for t in v[:top]] for k, v in themes.items()}


# --------------------------------------------------------------------------- critic

CRITIC_SYSTEM = """You are a skeptical research director reviewing a synthetic (AI-simulated) concept test before
it goes to a client. Your job is to protect the client from overconfident conclusions.
Look for: AI positivity bias, small subgroups, diagnostic flags, research caveats, and anything the client might over-interpret.
SEVERITY RULES (critical):
- Use "low" or "medium" severity for all methodological caveats, small sample/subgroup sizes, synthetic simulation limits, unverified client brief claims, or wide price dispersion. These will be printed prominently in the report's methodology section.
- Reserve "high" severity ONLY if the computed quantitative data is corrupted/missing or contradicts itself mathematically (since "high" aborts report generation).
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
  "issues": [{{"severity": "low|medium", "issue": "...", "recommendation": "what to do about it"}}],
  "confidence": "low|medium|high",
  "confidence_rationale": "1-3 sentences on how much weight the client should put on these findings"
}}"""


async def critique(llm: LLM, cfg: Config, brief: Brief, quant: dict, themes: dict,
                   research: MarketResearch) -> CriticReview:
    role = cfg.role("critic")
    flags = quant["diagnostics"]["flags"]
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
- Only use numbers that appear in the data provided. Never invent external statistics, market sizes, or unverified facts.
- Explicitly state that findings come from an AI-simulated synthetic panel (n personas) and are directional hypotheses to test with real customers, not a sales forecast or proof of willingness to pay.
- Reflect all research caveats and the independent reviewer's critique in your executive summary, segment insights, and risks.
- Frame all action_plan recommendations and price points as structured experiments/targets for real customer validation.
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
  "action_plan": {{
    "what_to_do_next": "clear recommendation tied to the client's decision",
    "why_this_option": "evidence, alternatives and main uncertainty",
    "what_to_change": "specific offer, message or targeting changes",
    "how_to_implement": "usable messaging, interview questions or test plan",
    "how_to_measure": "measurable outcome and decision criteria",
    "what_happens_next": "review results and identify the next decision"
  }},
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
                                 "competitors": [c.name for c in research.competitors],
                                 "caveats": research.caveats}, ensure_ascii=False),
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
