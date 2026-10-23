"""Runs a full Concept Test: brief -> research -> personas -> interviews -> analysis -> report."""

from __future__ import annotations

import json
import random
import re
import secrets
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from . import analysis
from .config import Config
from .guardrails import GuardrailBlocked, compute_integrity_summary, review_report
from .llm import LLM, CostTracker
from .mock import Mocker
from .report import render_html, render_markdown
from .resilience import audit_tier1_invariants
from .sampling import sample_traits
from .schemas import Brief
from .supervisor import PipelineSupervisor
from .stages import interviews, personas as persona_stage, research as research_stage, synthesis

MIN_SUCCESS_RATE = 0.6


def load_brief(path: str | Path) -> Brief:
    with open(path, encoding="utf-8") as fh:
        return Brief.model_validate(yaml.safe_load(fh))


def _save(run_dir: Path, name: str, obj) -> None:
    def conv(o):
        if isinstance(o, BaseModel):
            return o.model_dump()
        if isinstance(o, list):
            return [conv(x) for x in o]
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        return o
    (run_dir / name).write_text(json.dumps(conv(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def _progress(console: Console) -> Progress:
    return Progress(TextColumn("  {task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                    console=console, transient=False)


async def run_concept_test(brief_path: str | Path, cfg: Config, *, mock: bool, n_personas: int,
                           seed: int | None, out_root: Path, console: Console | None = None,
                           audience: str | Path | None = None) -> Path:
    if console is None:
        import io
        console = Console(file=io.StringIO())
    from .audiences import load_panel
    audience_panel = load_panel(audience) if audience else None
    if audience_panel is not None and audience_panel.mock != mock:
        raise ValueError("Mock and live audience panels cannot be mixed. Match --mock to the saved audience panel.")
    brief = load_brief(brief_path)
    seed = seed if seed is not None else random.randrange(1_000_000)
    slug = re.sub(r"[^a-z0-9]+", "-", brief.product_name.lower()).strip("-")[:40]
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{slug}{'_mock' if mock else ''}_{secrets.token_hex(3)}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _save(run_dir, "brief.json", brief)

    tracker = CostTracker(budget_usd=cfg.run.budget_usd)
    llm = LLM(cfg, tracker, Mocker(seed, brief) if mock else None)
    run_meta = {"seed": seed, "mock": mock, "requested_personas": n_personas, "config": cfg.model_dump()}
    if audience_panel:
        run_meta["audience_reused"] = audience_panel.spec.name
    _save(run_dir, "run.json", run_meta)
    try:
        return await _execute(brief, cfg, mock, n_personas, seed, run_dir, run_id, console, tracker, llm,
                              audience_panel=audience_panel, out_root=out_root)
    finally:
        _save(run_dir, "costs.json", tracker.summary())
        await llm.aclose()


async def _execute(brief: Brief, cfg: Config, mock: bool, n_personas: int, seed: int,
                   run_dir: Path, run_id: str, console: Console, tracker: CostTracker, llm: LLM,
                   audience_panel=None, out_root: Path | None = None) -> Path:
    from .audiences import default_audiences_dir, save_panel_from_run
    from .config import PROJECT_ROOT

    rng = random.Random(seed)
    supervisor = PipelineSupervisor(llm, cfg, run_dir, console, _save)
    batch_size = cfg.guardrails.batch_size

    mode = "[yellow]MOCK (free, simulated answers)[/]" if mock else "[green]LIVE[/]"
    console.rule(f"[bold]Kohort Concept Test: {brief.product_name}")
    console.print(f"  Mode: {mode}   Personas: {n_personas}   Seed: {seed}\n  Output: {run_dir}\n")

    console.print("[bold]0/6 Brief review[/]")
    async def produce_brief():
        return {"brief": brief}
    await supervisor.run("brief", produce_brief, lambda out: [out], retry_allowed=False)

    # 1. Research (Sense-Maker + Scraper)
    console.print("[bold]1/6 Market research & buyer discovery[/]")
    llm.run_dir = run_dir
    async def produce_research():
        result = await research_stage.run(llm, cfg, brief, console)
        _save(run_dir, "research.json", result)
        if result.buyer_dossier is not None:
            _save(run_dir, "buyer-dossier.json", result.buyer_dossier)
        out = {"research": result}
        if getattr(llm, "last_scraped_sources", None):
            out["scraped_sources"] = llm.last_scraped_sources
        return out
    output, _ = await supervisor.run("research", produce_research,
                                     lambda out: [{"brief": brief, **out}])
    research = output["research"]
    console.print(f"  {len(research.segments)} segments: " + ", ".join(
        f"{s.name} ({s.share:.0%})" for s in research.segments))

    # 2. Personas (Two-Layer Architecture: Base Persona + Scraped Category Lens)
    if audience_panel is not None:
        console.print(
            f"[bold]2/6 Reusing base audience panel ({audience_panel.spec.title}) + injecting current brief lens[/]"
        )
        if audience_panel.spec.segments:
            dossier = research.buyer_dossier
            synced_segs = []
            for s in audience_panel.spec.segments:
                habits = s.buying_habits or (dossier.buying_habits[:3] if dossier else [])
                anchors = s.price_anchors or (dossier.price_anchors[:2] if dossier else [])
                synced_segs.append(s.model_copy(update={"buying_habits": habits, "price_anchors": anchors}))
            research.segments = synced_segs
            _save(run_dir, "research.json", research)
        selected_personas = persona_stage.apply_category_lens(
            audience_panel.personas[:n_personas], research, category=brief.category
        )
        async def produce_personas():
            _save(run_dir, "personas.json", selected_personas)
            return {"personas": selected_personas, "errors": [], "requested": len(selected_personas)}
        def persona_evidence(out):
            return [{"brief": brief,
                     "segments": research.segments,
                     "personas": [p.model_dump(exclude={"category_lens", "trait_provenance"})
                                  for p in out["personas"][i:i + batch_size]],
                     "coverage": {"requested": out["requested"], "completed": len(out["personas"]),
                                  "failed": 0}}
                    for i in range(0, len(out["personas"]), batch_size)]
        output, _ = await supervisor.run("personas", produce_personas, persona_evidence)
        panel = output["personas"]
    else:
        console.print("[bold]2/6 Building tailored persona panel[/]")
        sampled = sample_traits(research.segments, n_personas, rng, cfg.role("personas").models)
        expected_traits = {f"P{i + 1:03d}": item for i, item in enumerate(sampled)}
        async def produce_personas():
            with _progress(console) as prog:
                t = prog.add_task("personas", total=len(sampled))
                panel, errors = await persona_stage.run(llm, cfg, brief, research, sampled, lambda: prog.advance(t))
            _save(run_dir, "personas.json", panel)
            _check(len(panel), len(sampled), errors, "persona generation", console)
            for p in panel:
                expected = expected_traits.get(p.id)
                if expected is None or p.traits != expected[0] or p.model != expected[1]:
                    raise RuntimeError(f"Persona {p.id} changed fixed traits or model assignment.")
            if len({p.id for p in panel}) != len(panel):
                raise RuntimeError("Duplicate persona IDs.")
            return {"personas": panel, "errors": errors, "requested": len(sampled)}
        def persona_evidence(out):
            return [{"brief": brief,
                     "segments": research.segments,
                     "personas": [p.model_dump(exclude={"category_lens", "trait_provenance"})
                                  for p in out["personas"][i:i + batch_size]],
                     "coverage": {"requested": out["requested"], "completed": len(out["personas"]),
                                  "failed": len(out["errors"])}}
                    for i in range(0, len(out["personas"]), batch_size)]
        output, _ = await supervisor.run("personas", produce_personas, persona_evidence)
        panel = output["personas"]
        vault_dir = (
            default_audiences_dir()
            if (not mock) and out_root is not None and out_root.resolve() in (
                (PROJECT_ROOT / "runs").resolve(),
                (PROJECT_ROOT / "outputs").resolve(),
            )
            else (run_dir / "audiences")
        )
        _, vault_path = save_panel_from_run(
            brief, research, panel, mock=mock, seed=seed, run_id=run_id, cost_usd=tracker.total, out_dir=vault_dir
        )
        console.print(f"  Archived reusable base panel to: {vault_path}")

    # 3. Interviews
    console.print("[bold]3/6 Interviewing personas[/]")
    async def produce_interviews():
        with _progress(console) as prog:
            t = prog.add_task("interviews", total=len(panel))
            rows, errors = await interviews.run(llm, cfg, brief, panel, lambda: prog.advance(t), seed=seed)
        _save(run_dir, "responses.json", rows)
        _check(len(rows), len(panel), errors, "interviews", console)
        pmap = {p.id: p for p in panel}
        if len({r.persona_id for r in rows}) != len(rows):
            raise RuntimeError("Duplicate interview respondent IDs.")
        if any(r.persona_id not in pmap or r.model != pmap[r.persona_id].model for r in rows):
            raise RuntimeError("Interview response does not match an approved persona/model.")
        return {"responses": rows, "errors": errors, "requested": len(panel)}
    def interview_evidence(out):
        pmap = {p.id: p.model_dump(exclude={"trait_provenance"}) for p in panel}
        return [{"brief": brief, "responses": out["responses"][i:i + batch_size],
                 "personas": [pmap[r.persona_id] for r in out["responses"][i:i + batch_size]],
                 "coverage": {"requested": out["requested"], "completed": len(out["responses"]),
                              "failed": len(out["errors"])}}
                for i in range(0, len(out["responses"]), batch_size)]
    output, _ = await supervisor.run("interviews", produce_interviews, interview_evidence)
    rows = output["responses"]
    answered = {r.persona_id for r in rows}
    panel = [p for p in panel if p.id in answered]

    # 4. Analysis
    console.print("[bold]4/6 Analysing results[/]")
    async def produce_analysis():
        quant = analysis.summarize(brief, panel, rows, cfg.run.intent_weights)
        tier1 = audit_tier1_invariants(quant, rows)
        if not tier1.passed:
            raise RuntimeError(f"Tier-1 deterministic invariant audit failed: {'; '.join(tier1.violations)}")
        coding = {}
        themes = await synthesis.code_themes(llm, cfg, brief, rows, panel, audit=coding)
        _save(run_dir, "analysis.json", quant)
        _save(run_dir, "themes.json", themes)
        _save(run_dir, "theme-coding.json", coding)
        return {"quantitative_results": quant, "themes": themes, "coding": coding}
    def analysis_evidence(out):
        items = [{"kind": kind, **item} for kind, coded in out["coding"].items() for item in coded]
        return [{"brief": brief, "quantitative_results": out["quantitative_results"],
                 "themes": out["themes"], "coded_answers": items[i:i + batch_size]}
                for i in range(0, max(1, len(items)), batch_size)]
    output, _ = await supervisor.run("analysis", produce_analysis, analysis_evidence)
    quant, themes = output["quantitative_results"], output["themes"]
    console.print(f"  Calibrated trial estimate: [bold]{quant['trial_estimate']:.0%}[/]  "
                  f"(mean intent {quant['metrics']['purchase_intent']['mean']})")
    for flag in quant["diagnostics"]["flags"]:
        console.print(f"  [yellow]! {flag}[/]")

    # 5. Critic
    console.print("[bold]5/6 Quality review[/]")
    async def produce_critic():
        critic = await synthesis.critique(llm, cfg, brief, quant, themes, research)
        _save(run_dir, "critic.json", critic)
        return {"critic": critic}
    output, _ = await supervisor.run("critic", produce_critic,
                                     lambda out: [{"brief": brief, "research": research,
                                                    "quantitative_results": quant, "themes": themes, **out}])
    critic = output["critic"]
    console.print(f"  Confidence: {critic.confidence}  ({len(critic.issues)} issues raised)")

    # 6. Report
    console.print("[bold]6/6 Writing report draft[/]")
    client_ans = synthesis._client_answer_sample(brief, rows, panel)
    async def produce_report():
        narrative = await synthesis.write_narrative(llm, cfg, brief, research, quant, themes, critic, rows, panel)
        _save(run_dir, "narrative.json", narrative)
        return {"narrative": narrative}
    async def report_review(out, allow_retry):
        return await review_report(llm, cfg, brief, research, quant, themes, critic, out["narrative"],
                                   allow_retry=allow_retry, client_answers=client_ans)
    output, guardrails = await supervisor.run("report", produce_report,
                                             lambda out: [{"brief": brief, "research": research,
                                                            "quantitative_results": quant, "themes": themes,
                                                            "critic": critic, "client_answers": client_ans, **out}],
                                             reviewer=report_review, stop_on_hold=False)
    narrative = output["narrative"]
    guardrails["pipeline_checkpoints"] = [
        {key: checkpoint[key] for key in ("stage", "attempt", "batch", "status", "action")}
        for checkpoint in supervisor.checkpoints
    ]
    guardrails["integrity"] = compute_integrity_summary(supervisor.checkpoints, guardrails)
    _save(run_dir, "guardrails.json", guardrails)
    held = guardrails["status"] in ("held", "error")
    report_name = "draft-report" if held else "report"
    cost = tracker.summary()
    md = render_markdown(brief=brief, research=research, quant=quant, themes=themes, critic=critic,
                         narrative=narrative, rows=rows, personas=panel, run_id=run_id, mock=mock, cost=cost,
                         guardrails=guardrails)
    (run_dir / f"{report_name}.md").write_text(md, encoding="utf-8")
    (run_dir / f"{report_name}.html").write_text(render_html(md, f"Kohort: {brief.product_name}"), encoding="utf-8")
    _save(run_dir, "costs.json", cost)
    if held:
        raise GuardrailBlocked(f"Report held for review: {'; '.join(guardrails['reasons'])} "
                               f"Draft and guardrails.json saved in {run_dir}")

    console.rule()
    console.print(f"  Verdict: [bold]{narrative.verdict}[/]   Confidence: {critic.confidence}   "
                  f"Cost: ${cost['total_usd']:.2f} ({cost['calls']} AI calls)")
    console.print(f"  Jev guardrail: {guardrails['status']}")
    console.print(f"  Report: [link=file:///{(run_dir / 'report.html').as_posix()}]{run_dir / 'report.html'}[/link]")
    return run_dir


def _check(ok: int, total: int, errors: list[str], stage: str, console: Console) -> None:
    if errors:
        console.print(f"  [yellow]{len(errors)} of {total} failed in {stage}; first error: {errors[0][:300]}[/]")
    if total == 0 or ok / total < MIN_SUCCESS_RATE:
        raise RuntimeError(f"Too many failures in {stage} ({ok}/{total} succeeded). Aborting.")
