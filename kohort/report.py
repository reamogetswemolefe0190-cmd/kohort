"""Renders the client report (Markdown + styled HTML).

Tables and figures come straight from the computed analysis; the narrative
comes from the report model.
"""

from __future__ import annotations

import html
import re
from datetime import date

import markdown as md_lib

from .schemas import Brief, CriticReview, InterviewResponse, MarketResearch, Persona, ReportNarrative

INTENT_LABELS = {"5": "Definitely would buy", "4": "Probably would", "3": "Might or might not",
                 "2": "Probably not", "1": "Definitely not"}
METRIC_LABELS = {"purchase_intent": "Purchase intent", "appeal": "Appeal", "uniqueness": "Uniqueness",
                 "believability": "Believability", "value_for_money": "Value for money (at stated price)"}
VERDICT_COLORS = {"GO": "#0f9d58", "ITERATE": "#f4a300", "NO-GO": "#d93025"}


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def _bar(x: float, width: int = 20) -> str:
    return "█" * round(x * width) + "░" * (width - round(x * width))


def _esc(s: str) -> str:
    return str(s).replace("|", "/").replace("\n", " ").strip()


def _money(brief: Brief, x: float | None) -> str:
    return "-" if x is None else f"{x:,.2f} {brief.currency}"


def _pick_quotes(rows: list[InterviewResponse], personas: list[Persona], k: int = 8) -> list[tuple]:
    pmap = {p.id: p for p in personas}
    by_intent: dict[int, list[InterviewResponse]] = {}
    for r in sorted(rows, key=lambda r: r.persona_id):
        if r.quote.strip():
            by_intent.setdefault(r.purchase_intent, []).append(r)
    picked: list[InterviewResponse] = []
    used_segments: set[tuple] = set()
    # Round-robin across intent levels (high to low), preferring new segments.
    while len(picked) < k and any(by_intent.values()):
        for level in sorted(by_intent, reverse=True):
            pool = by_intent[level]
            if not pool or len(picked) >= k:
                continue
            idx = next((i for i, r in enumerate(pool)
                        if (level, pmap[r.persona_id].traits.segment) not in used_segments), 0)
            r = pool.pop(idx)
            used_segments.add((level, pmap[r.persona_id].traits.segment))
            picked.append(r)
    return [(r, pmap[r.persona_id]) for r in picked]


def _theme_table(themes: list[dict], n_show: int = 8) -> list[str]:
    if not themes:
        return ["_No responses in this category._", ""]
    lines = ["| Theme | Personas | | Example |", "|---|---:|---|---|"]
    for t in themes[:n_show]:
        ex = f"“{_esc(t['examples'][0])}”" if t["examples"] else ""
        lines.append(f"| **{_esc(t['label'])}** | {_pct(t['share'])} | `{_bar(t['share'], 12)}` | {ex} |")
    return lines + [""]


def _group_table(title: str, groups: dict, order: list[str] | None = None, small_n: int = 10) -> list[str]:
    keys = order or sorted(groups, key=lambda g: groups[g]["trial_estimate"], reverse=True)
    lines = [f"| {title} | n | Mean intent | Top-2-box | Calibrated trial | Mean appeal |",
             "|---|---:|---:|---:|---:|---:|"]
    has_small = False
    for k in keys:
        if k not in groups:
            continue
        g = groups[k]
        mark = " †" if g["n"] < small_n else ""
        has_small = has_small or bool(mark)
        lines.append(f"| {_esc(k)}{mark} | {g['n']} | {g['intent_mean']:.2f} | {_pct(g['intent_top2'])} | "
                     f"**{_pct(g['trial_estimate'])}** | {g['appeal_mean']:.2f} |")
    lines.append("")
    if has_small:
        lines += [f"<sub>† Fewer than {small_n} personas: treat as indicative only.</sub>", ""]
    return lines


def render_markdown(*, brief: Brief, research: MarketResearch, quant: dict, themes: dict,
                    critic: CriticReview, narrative: ReportNarrative, rows: list[InterviewResponse],
                    personas: list[Persona], run_id: str, mock: bool, cost: dict) -> str:
    m = quant["metrics"]
    w = quant["intent_weights"]
    L: list[str] = []
    add = L.append

    add(f"# Concept Test Report: {brief.product_name}")
    add("")
    add(f"**Prepared for:** {brief.client} &nbsp;·&nbsp; **Date:** {date.today():%d %B %Y} &nbsp;·&nbsp; "
        f"**Run:** `{run_id}` &nbsp;·&nbsp; **Panel:** {quant['n']} synthetic consumers, {brief.region}")
    add("")
    if mock:
        add("> [!WARNING]")
        add("> **MOCK RUN:** all responses below are simulated placeholders produced without AI. Use this only to "
            "review the report format. Do not share or act on these findings.")
        add("")

    add("> [!IMPORTANT]")
    add("> **Evidence Status: Synthetic Hypothesis**")
    add("> *This report was generated using a simulated market of AI personas. It has identified the strongest product positioning, primary consumer objections, and a hypothesized optimal price point. **Next Step:** We recommend taking the winning concept from this report and running a live, low-cost market validation (such as a landing page waitlist or targeted ad campaign) before initiating physical manufacturing.*")
    add("")

    # 1. Verdict
    add("## 1. Verdict")
    add("")
    add(f"<div class='verdict' style='--c:{VERDICT_COLORS[narrative.verdict]}'><span>{narrative.verdict}</span>"
        f"<small>Confidence: {critic.confidence.upper()}</small></div>")
    add("")
    add(narrative.verdict_rationale)
    add("")

    # 2. Executive summary
    add("## 2. Executive Summary")
    add("")
    add(narrative.executive_summary)
    add("")

    # 3. Headline metrics
    add("## 3. Headline Metrics")
    add("")
    add("| Metric | Mean (1-5) | Top-2-box (4-5) | Bottom-2-box (1-2) |")
    add("|---|---:|---:|---:|")
    for key, label in METRIC_LABELS.items():
        b = m.get(key, {})
        if b.get("n"):
            add(f"| {label} | {b['mean']:.2f} | {_pct(b['top2'])} | {_pct(b['bottom2'])} |")
    add("")
    add(f"**Calibrated trial estimate: {_pct(quant['trial_estimate'])}**: stated intent discounted toward "
        f"realistic behaviour ({w.get('definitely', 0.75):.0%} of \"definitely\" + "
        f"{w.get('probably', 0.25):.0%} of \"probably\").")
    add("")
    add("**Purchase intent distribution**")
    add("")
    add("| Answer | Share | |")
    add("|---|---:|---|")
    for k in ["5", "4", "3", "2", "1"]:
        share = m["purchase_intent"]["dist"].get(k, 0)
        add(f"| {INTENT_LABELS[k]} | {_pct(share)} | `{_bar(share)}` |")
    add("")

    # 4. Key findings
    add("## 4. Key Findings")
    add("")
    L.extend(f"{i}. {f}" for i, f in enumerate(narrative.key_findings, 1))
    add("")

    # 5. Segments
    add("## 5. Segment Results")
    add("")
    L.extend(_group_table("Segment", quant["by"]["segment"]))
    for s in narrative.segment_insights:
        add(f"- **{s.segment}:** {s.insight}")
    add("")

    # 6-8. Themes
    add("## 6. What People Liked")
    add("")
    L.extend(_theme_table(themes.get("likes", [])))
    add("## 7. Main Concerns and Barriers")
    add("")
    L.extend(_theme_table(themes.get("concerns", [])))
    add("## 8. What Would Make Them Buy")
    add("")
    L.extend(_theme_table(themes.get("triggers", [])))

    # 9. Pricing
    add("## 9. Pricing")
    add("")
    vw = quant["pricing"]["van_westendorp"]
    acc = quant["pricing"]["acceptance"]
    unit = f" {brief.price_unit}" if brief.price_unit else ""
    if vw:
        add(f"Price thresholds were collected **before** the price was revealed (Van Westendorp method). "
            f"Prices in {brief.currency}{unit}.")
        add("")
        add("| Measure | Price |")
        add("|---|---:|")
        add(f"| Point of marginal cheapness (lower bound of acceptable range) | {_money(brief, vw['point_of_marginal_cheapness'])} |")
        add(f"| Optimal price point | **{_money(brief, vw['optimal_price_point'])}** |")
        add(f"| Indifference price point | {_money(brief, vw['indifference_price_point'])} |")
        add(f"| Point of marginal expensiveness (upper bound) | {_money(brief, vw['point_of_marginal_expensiveness'])} |")
        add(f"| Median \"too expensive\" | {_money(brief, vw['median_too_expensive'])} |")
        add("")
    if acc:
        add(f"**How the proposed price of {brief.price_label()} lands** (vs. each persona's own thresholds):")
        add("")
        add("| Perception | Share | |")
        add("|---|---:|---|")
        for key, label in [("too_cheap", "Suspiciously cheap"), ("bargain_or_fair", "Bargain / fair"),
                           ("expensive_but_considered", "Expensive, but would consider"),
                           ("too_expensive", "Too expensive to consider")]:
            add(f"| {label} | {_pct(acc[key])} | `{_bar(acc[key])}` |")
        add("")
    if not vw and not acc:
        add("_Not enough price data collected._")
        add("")

    # 10. Adoption & demographics
    add("## 10. Who Responds Best")
    add("")
    L.extend(_group_table("Adopter type", quant["by"]["adopter_type"],
                          ["innovator", "early adopter", "early majority", "late majority", "laggard"]))
    L.extend(_group_table("Age", quant["by"]["age_band"], ["18-29", "30-44", "45-59", "60+"]))
    L.extend(_group_table("Income", quant["by"]["income"], ["low", "lower-middle", "middle", "upper-middle", "high"]))

    # 11. Client questions
    if narrative.client_question_answers:
        add("## 11. Answers to Your Questions")
        add("")
        for qa in narrative.client_question_answers:
            add(f"**{qa.question}**")
            add("")
            add(qa.answer)
            add("")

    # 12. Voices
    add("## 12. Voices from the Panel")
    add("")
    for r, p in _pick_quotes(rows, personas):
        t = p.traits
        add(f"> “{r.quote.strip()}”")
        add(f"> <br/>*{p.name}, {t.age}, {p.occupation}, {t.segment} · intent {r.purchase_intent}/5*")
        add("")

    # 13. Decision Action Plan
    add("## 13. Action Plan")
    add("")
    add("### 1. What should I do next?")
    add(narrative.action_plan.what_to_do_next)
    add("")
    add("### 2. Why this option?")
    add(narrative.action_plan.why_this_option)
    add("")
    add("### 3. What should I change?")
    add(narrative.action_plan.what_to_change)
    add("")
    add("### 4. How do I implement it?")
    add(narrative.action_plan.how_to_implement)
    add("")
    add("### 5. How will I know it worked?")
    add(narrative.action_plan.how_to_measure)
    add("")
    add("### 6. What happens afterward?")
    add(narrative.action_plan.what_happens_next)
    add("")

    # 14. Risks
    if narrative.risks:
        add("## 14. Risks and Unknowns")
        add("")
        L.extend(f"- {r}" for r in narrative.risks)
        add("")

    # 15. Market context
    add("## 15. Market Context")
    add("")
    add(research.summary)
    add("")
    if research.trends:
        add("**Trends:** " + "; ".join(research.trends))
        add("")
    if research.typical_price_range:
        add(f"**Typical price range:** {research.typical_price_range}")
        add("")
    if research.competitors:
        add("| Competitor / alternative | Offering | Price | Strengths | Weaknesses |")
        add("|---|---|---|---|---|")
        for c in research.competitors:
            add(f"| **{_esc(c.name)}** | {_esc(c.offering)} | {_esc(c.price_point)} | "
                f"{_esc('; '.join(c.strengths))} | {_esc('; '.join(c.weaknesses))} |")
        add("")

    # 16. Methodology
    add("## 16. Methodology, Quality Review and Limitations")
    add("")
    models = sorted({p.model for p in personas})
    add(f"- **Panel:** {quant['n']} synthetic consumers across {len(research.segments)} segments. Demographics "
        f"and attitudes (age, income, household, adopter type, price sensitivity, skepticism) were sampled from "
        f"explicit distributions; an AI model then wrote each individual's background.")
    add(f"- **Model diversity:** personas were split across {len(models)} AI model families "
        f"({', '.join(models)}) to reduce single-model bias.")
    add("- **Interview design:** each persona first reacted to the concept *without* the price (including Van "
        "Westendorp price thresholds), then saw the price and gave value-for-money and purchase intent.")
    add("- **Analysis:** all numbers were computed directly from responses; open answers were coded into themes "
        "by an analyst model; an independent reviewer model critiqued the results before the report was written.")
    add(f"- **Market research:** {'live web search' if research.web_search_used else 'model knowledge (no live web search)'}.")
    add("")
    d = quant["diagnostics"]
    add(f"**Panel diagnostics:** intent std dev {d['intent_std']}, most common answer share "
        f"{_pct(d['modal_share'])}, duplicate quotes {_pct(d['duplicate_quote_ratio'])}, model intent spread "
        f"{d['model_intent_spread']}.")
    add("")
    if critic.issues:
        add(f"**Independent review (confidence: {critic.confidence}).** {critic.confidence_rationale}")
        add("")
        add("| Severity | Issue | Recommendation |")
        add("|---|---|---|")
        for i in sorted(critic.issues, key=lambda i: ["high", "medium", "low"].index(i.severity)):
            add(f"| {i.severity.upper()} | {_esc(i.issue)} | {_esc(i.recommendation)} |")
        add("")
    if research.caveats:
        add("**Research caveats:** " + "; ".join(research.caveats))
        add("")
    add("> [!IMPORTANT]")
    add("> This report is based on a **simulated market** of AI-generated consumers. It is designed to surface "
        "reactions, objections and relative differences quickly and cheaply. It is directional decision support, "
        "not a forecast of real-world sales. Validate key decisions with real customers.")
    add("")
    add(f"<sub>Kohort · run {run_id} · {cost.get('calls', 0)} AI calls · "
        f"compute cost ${cost.get('total_usd', 0):.2f}</sub>")
    return "\n".join(L)


CSS = """
:root{--ink:#16181d;--muted:#5f6672;--line:#e6e8ec;--bg:#fff;--accent:#4b3df5}
*{box-sizing:border-box}
body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);
background:#f6f7f9;margin:0;line-height:1.6}
main{max-width:960px;margin:40px auto;background:var(--bg);padding:56px 64px;border-radius:16px;
box-shadow:0 2px 24px rgba(20,20,40,.06)}
.brand{font-weight:800;letter-spacing:.18em;font-size:13px;color:var(--accent);text-transform:uppercase}
h1{font-size:34px;line-height:1.2;margin:.3em 0 .4em}
h2{font-size:21px;margin-top:2.4em;padding-top:1em;border-top:1px solid var(--line)}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:14px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:#fafbfc;font-weight:600;color:var(--muted)}
code{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--accent);background:none}
blockquote{margin:1em 0;padding:.6em 1.1em;border-left:3px solid var(--accent);background:#f7f6ff;
border-radius:0 8px 8px 0}
blockquote p{margin:.2em 0}
.verdict{display:inline-flex;align-items:center;gap:16px;border:2px solid var(--c);border-radius:12px;
padding:10px 20px}
.verdict span{font-size:28px;font-weight:800;color:var(--c);letter-spacing:.04em}
.verdict small{color:var(--muted);font-weight:600}
.alert{border-radius:10px;padding:12px 16px;margin:1em 0}
.alert-warning{background:#fff4e5;border:1px solid #ffd59a}
.alert-important{background:#eef1ff;border:1px solid #c9d0ff}
sub{color:var(--muted)}
.bar{display:inline-block;width:140px;height:10px;background:#eceaff;border-radius:6px;overflow:hidden;
vertical-align:middle}
.bar i{display:block;height:100%;background:var(--accent);border-radius:6px}
@media print{body{background:#fff}main{box-shadow:none;margin:0;padding:24px}h2{break-after:avoid}
table,blockquote{break-inside:avoid}}
"""


def render_html(markdown_text: str, title: str) -> str:
    # Convert GitHub-style alerts into styled divs.
    out_lines, i, lines = [], 0, markdown_text.split("\n")
    while i < len(lines):
        line = lines[i]
        if line.strip() in ("> [!WARNING]", "> [!IMPORTANT]"):
            kind = "warning" if "WARNING" in line else "important"
            body = []
            i += 1
            while i < len(lines) and lines[i].startswith(">"):
                body.append(lines[i][1:].strip())
                i += 1
            out_lines.append(f"<div class='alert alert-{kind}' markdown='1'>\n\n{' '.join(body)}\n\n</div>")
            continue
        out_lines.append(line)
        i += 1
    body_html = md_lib.markdown("\n".join(out_lines), extensions=["tables", "md_in_html", "sane_lists"])

    def _css_bar(m: re.Match) -> str:
        full, empty = len(m.group(1)), len(m.group(2))
        pct = 100 * full / (full + empty) if full + empty else 0
        return f"<span class='bar'><i style='width:{pct:.0f}%'></i></span>"

    body_html = re.sub(r"<code>(█*)(░*)</code>", _css_bar, body_html)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><main><div class='brand'>Kohort</div>{body_html}</main></body></html>")
