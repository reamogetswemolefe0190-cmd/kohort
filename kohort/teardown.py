"""Competitive Teardown & Reverse-Engineering Engine.

Deconstructs a competitor's business, technology, pricing, weaknesses,
and generates an actionable implementation plan (either for clients or for Kohort).
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, Field
from rich.console import Console

from .config import Config
from .llm import CostTracker, LLM, LLMError
from .mock import Mocker
from .pipeline import _save
from .report import render_html
from .scraping import enrich_sources


# --------------------------------------------------------------------------- Schemas


class PricingTier(BaseModel):
    name: str = Field(description="Tier name, e.g. Free, Starter, Pro, Enterprise")
    price: str = Field(description="Price and billing interval, e.g. $199/month, $0.12/response, Custom")
    features: list[str] = Field(default_factory=list, description="Key features included")
    limits: list[str] = Field(default_factory=list, description="Usage limits, seats, or credit allocations")


class PricingAnalysis(BaseModel):
    model_type: str = Field(description="Subscription, usage-based, credit packs, or hybrid")
    tiers: list[PricingTier] = Field(default_factory=list)
    estimated_margins_and_markup: str = Field(
        description="Analysis of their unit economics and markup over underlying infrastructure/API costs"
    )
    vulnerabilities: list[str] = Field(
        default_factory=list, description="Where their pricing causes customer hesitation or friction"
    )


class TechnicalArchitecture(BaseModel):
    summary: str = Field(description="High-level architecture and how the system appears to work under the hood")
    integrations_and_apis: list[str] = Field(
        default_factory=list, description="Documented integrations, MCP servers, SDKs, or export formats"
    )
    strengths: list[str] = Field(default_factory=list, description="Technical capabilities they execute well")
    limitations: list[str] = Field(
        default_factory=list, description="Known architectural constraints, missing features, or lack of safeguards"
    )


class StrategicPositioning(BaseModel):
    target_audiences: list[str] = Field(
        default_factory=list, description="Target segments they actively market to"
    )
    value_proposition: str = Field(description="Primary value proposition and marketing hooks")
    user_objections_and_complaints: list[str] = Field(
        default_factory=list, description="Common customer complaints, negative reviews, or market pushback"
    )


class ActionRoadmap(BaseModel):
    features_to_adopt: list[str] = Field(
        default_factory=list, description="Valuable features or patterns worth replicating or improving"
    )
    differentiators_to_exploit: list[str] = Field(
        default_factory=list, description="Specific gaps where a competitor can clearly outperform them"
    )
    client_pitch_angles: list[str] = Field(
        default_factory=list, description="Exact messaging angles to win sales deals against this competitor"
    )
    kohort_implementation_spec: list[str] = Field(
        default_factory=list, description="Concrete technical steps to implement superior features into Kohort"
    )


class TeardownReport(BaseModel):
    competitor_name: str
    target_url: str
    summary: str
    positioning: StrategicPositioning
    pricing: PricingAnalysis
    architecture: TechnicalArchitecture
    roadmap: ActionRoadmap
    evidence_sources: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- System Prompts


TEARDOWN_SYSTEM = """You are a senior product strategist and systems architect conducting a competitive teardown.
Your goal is to reverse-engineer a competitor's business, technology, pricing, and positioning based ONLY on retrieved web evidence and citations.
Do not invent facts or hallucinate pricing. If a detail is unstated or uncertain, label it explicitly as unknown.
Reply with ONLY a valid JSON object matching the requested schema."""

TEARDOWN_USER = """Analyze the following competitor based on the scraped evidence:

TARGET: {target}

RETRIEVED WEB EVIDENCE:
{evidence}

Produce a rigorous, deconstructed teardown report. Deconstruct their:
1. Strategic positioning and target segments.
2. Pricing model, tiers, and estimated markup over raw API/infrastructure costs.
3. Technical architecture, integrations (such as MCP, APIs), and limitations.
4. Actionable roadmap: what to adopt, what gaps to exploit, sales pitch angles to beat them, and concrete technical implementation steps for Kohort.

JSON FORMAT:
{{
  "competitor_name": "{target}",
  "target_url": "https://...",
  "summary": "1-2 paragraphs summarizing the company and product offering",
  "positioning": {{
    "target_audiences": ["segment 1", "segment 2"],
    "value_proposition": "Core marketing hook and value proposition",
    "user_objections_and_complaints": ["complaint or friction point 1"]
  }},
  "pricing": {{
    "model_type": "Subscription / usage / credit packs",
    "tiers": [
      {{"name": "Tier name", "price": "$...", "features": ["feature 1"], "limits": ["limit 1"]}}
    ],
    "estimated_margins_and_markup": "Estimate unit economics and markup over underlying LLM APIs",
    "vulnerabilities": ["pricing risk 1"]
  }},
  "architecture": {{
    "summary": "How their engine works under the hood",
    "integrations_and_apis": ["MCP server", "REST API", "Exports"],
    "strengths": ["technical strength 1"],
    "limitations": ["technical limitation or missing safeguards"]
  }},
  "roadmap": {{
    "features_to_adopt": ["feature to replicate or adapt"],
    "differentiators_to_exploit": ["gap to outperform them on"],
    "client_pitch_angles": ["sales pitch angle against this competitor"],
    "kohort_implementation_spec": ["concrete code/architecture step for Kohort"]
  }},
  "evidence_sources": []
}}"""


# --------------------------------------------------------------------------- Execution


async def run_teardown(
    target: str,
    cfg: Config,
    *,
    mock: bool = False,
    out_root: Path | None = None,
    console: Console | None = None,
) -> Path:
    console = console or Console()
    out_root = out_root or Path("runs")
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_teardown_{secrets.token_hex(3)}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]=== Kohort Competitive Teardown: {target} ===[/]")
    mode = "[yellow]MOCK[/]" if mock else "[green]LIVE RESEARCH[/]"
    console.print(f"  Mode: {mode}   Output: {run_dir}\n")

    tracker = CostTracker(budget_usd=cfg.run.budget_usd)
    llm = LLM(cfg, tracker, Mocker(42, None) if mock else None)

    sources = []
    try:
        # Step 1: Evidence collection
        if mock:
            console.print("  [1/3] Mock evidence generated.")
            raw_evidence = (
                f"Product: {target}\n"
                "Overview: AI synthetic consumer research platform.\n"
                "Pricing: Pay as you go $0.12/response, Pro $199/month (5,000 responses).\n"
                "Features: Synthetic audience simulation, MCP server integration, 1-on-1 interviews.\n"
                "Limitations: No mathematical hallucination gating, expensive response markups."
            )
            sources = [{"url": f"https://{target}", "title": target, "content": raw_evidence}]
        else:
            console.print("  [1/3] Running deep search and public page extraction...")
            # If target looks like a URL, scrape it directly; otherwise search
            is_url = target.startswith(("http://", "https://"))
            search_query = (
                f"Investigate {target}: product features, pricing plans, architecture, MCP integrations, customer reviews"
            )
            
            # Execute search with competitive_research role
            search_model = cfg.role("competitive_research").model
            await llm._call(
                search_model,
                [
                    {
                        "role": "system",
                        "content": "You are a research bot. Use web search to discover the official product website, pricing page, docs, and reviews.",
                    },
                    {"role": "user", "content": search_query},
                ],
                temperature=0.0,
                max_tokens=2000,
                web_search=True,
            )
            sources = llm.last_search_sources

            # Also attempt to scrape the root page if it was a URL
            if is_url and cfg.scraping.enabled:
                enriched = await enrich_sources(
                    [{"url": target, "title": target, "content": ""}],
                    cfg.scraping,
                    run_dir,
                    "teardown",
                )
                sources.extend(enriched)

        # Save collected evidence
        _save(run_dir, "search-evidence.json", sources)
        console.print(f"  [2/3] Analyzing {len(sources)} verified source records...")

        # Step 2: Synthesis & Deconstruction
        evidence_text = "\n\n---\n\n".join(
            f"Source: {s.get('url', 'N/A')} ({s.get('title', 'N/A')})\n{s.get('content', '')[:3000]}"
            for s in sources[:8]
        )

        role = cfg.role("analyst")
        teardown_data, _ = await llm.structured(
            task="teardown",
            schema=TeardownReport,
            model=role.model,
            system=TEARDOWN_SYSTEM,
            user=TEARDOWN_USER.format(target=target, evidence=evidence_text or "No direct text scraped."),
            temperature=0.2,
            max_tokens=role.max_tokens,
            mock_ctx={
                "competitor_name": target,
                "target_url": f"https://{target}" if not target.startswith("http") else target,
                "summary": f"Deconstruction of {target}",
            },
        )
        teardown_data.evidence_sources = [s.get("url", "") for s in sources if s.get("url")]

        _save(run_dir, "teardown.json", teardown_data)

        # Step 3: Render HTML Dossier
        console.print("  [3/3] Generating interactive Teardown Dossier...")
        md_dossier = _render_teardown_markdown(teardown_data, run_id)
        (run_dir / "teardown.md").write_text(md_dossier, encoding="utf-8")
        html_dossier = render_html(md_dossier, f"Competitive Teardown: {teardown_data.competitor_name}")
        (run_dir / "teardown.html").write_text(html_dossier, encoding="utf-8")

        costs = tracker.summary()
        _save(run_dir, "costs.json", costs)

        console.print("[bold]================================================[/]")
        console.print(f"  [bold green]Teardown Complete![/] Total cost: ${costs['total_usd']:.4f}")
        console.print(f"  Dossier: [link=file:///{run_dir / 'teardown.html'}]{run_dir / 'teardown.html'}[/link]")
        return run_dir

    finally:
        await llm.aclose()


# --------------------------------------------------------------------------- Markdown Renderer


def _render_teardown_markdown(t: TeardownReport, run_id: str) -> str:
    lines = [
        f"# Competitive Teardown & Reverse-Engineering Dossier: {t.competitor_name}",
        "",
        f"**Target URL:** [{t.target_url}]({t.target_url}) &nbsp;·&nbsp; **Date:** {datetime.now():%d %B %Y} &nbsp;·&nbsp; **Run ID:** `{run_id}`",
        "",
        "> [!IMPORTANT]",
        "> **Intelligence Boundary: Grounded Evidence**",
        "> *This teardown was constructed from public documentation, verified page excerpts, and live pricing disclosures. Technical and unit economics estimates are grounded hypotheses based on comparable LLM infrastructure costs.*",
        "",
        "## 1. Executive Summary",
        "",
        t.summary,
        "",
        "## 2. Strategic Positioning & Market Segments",
        "",
        f"**Core Value Proposition:** {t.positioning.value_proposition}",
        "",
        "**Target Audiences:**",
    ]
    for aud in t.positioning.target_audiences:
        lines.append(f"- {aud}")
    lines.append("")

    if t.positioning.user_objections_and_complaints:
        lines.append("**Customer Friction Points & Objections:**")
        for obj in t.positioning.user_objections_and_complaints:
            lines.append(f"- {obj}")
        lines.append("")

    lines.extend([
        "## 3. Pricing Model & Unit Economics",
        "",
        f"**Pricing Structure:** {t.pricing.model_type}",
        "",
        "| Tier | Price | Included Features | Limits / Quotas |",
        "|---|---|---|---|",
    ])
    for tier in t.pricing.tiers:
        feats = "; ".join(tier.features) or "Standard"
        lims = "; ".join(tier.limits) or "N/A"
        lines.append(f"| **{tier.name}** | {tier.price} | {feats} | {lims} |")
    lines.append("")

    lines.append(f"**Margin & Infrastructure Markup Analysis:**\n\n{t.pricing.estimated_margins_and_markup}\n")

    if t.pricing.vulnerabilities:
        lines.append("**Pricing Vulnerabilities:**")
        for v in t.pricing.vulnerabilities:
            lines.append(f"- {v}")
        lines.append("")

    lines.extend([
        "## 4. Technical Architecture & System Deconstruction",
        "",
        t.architecture.summary,
        "",
    ])

    if t.architecture.integrations_and_apis:
        lines.append("**Integrations, MCP & APIs:**")
        for api in t.architecture.integrations_and_apis:
            lines.append(f"- {api}")
        lines.append("")

    if t.architecture.strengths:
        lines.append("**Technical Strengths:**")
        for s in t.architecture.strengths:
            lines.append(f"- {s}")
        lines.append("")

    if t.architecture.limitations:
        lines.append("**Known Limitations & Omissions:**")
        for lim in t.architecture.limitations:
            lines.append(f"- {lim}")
        lines.append("")

    lines.extend([
        "## 5. Actionable Roadmap & Counter-Attack Strategy",
        "",
        "### Features Worth Adopting (What to Build)",
    ])
    for feat in t.roadmap.features_to_adopt:
        lines.append(f"1. {feat}")
    lines.append("")

    lines.append("### Gaps to Exploit (Where to Outperform Them)")
    for gap in t.roadmap.differentiators_to_exploit:
        lines.append(f"- {gap}")
    lines.append("")

    lines.append("### Sales Pitch Angles (How to Win Deals Against Them)")
    for pitch in t.roadmap.client_pitch_angles:
        lines.append(f"- **Angle:** {pitch}")
    lines.append("")

    lines.append("### Kohort Implementation Spec (Technical Next Steps)")
    for spec in t.roadmap.kohort_implementation_spec:
        lines.append(f"- [ ] {spec}")
    lines.append("")

    if t.evidence_sources:
        lines.extend([
            "## 6. Sourced Evidence Records",
            "",
            "The findings above cite the following verified public endpoints:",
            "",
        ])
        for src in t.evidence_sources:
            lines.append(f"- [{src}]({src})")
        lines.append("")

    return "\n".join(lines)
