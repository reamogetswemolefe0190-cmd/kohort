"""Runs a full Concept Test: brief -> research -> personas -> interviews -> analysis -> report."""

from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from . import analysis
from .config import Config
from .llm import LLM, CostTracker
from .mock import Mocker
from .report import render_html, render_markdown
from .sampling import sample_traits
from .schemas import Brief
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
                           seed: int | None, out_root: Path, console: Console) -> Path:
    brief = load_brief(brief_path)
    seed = seed if seed is not None else random.randrange(1_000_000)
    slug = re.sub(r"[^a-z0-9]+", "-", brief.product_name.lower()).strip("-")[:40]
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{slug}{'_mock' if mock else ''}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _save(run_dir, "brief.json", brief)

    tracker = CostTracker(budget_usd=cfg.run.budget_usd)
    llm = LLM(cfg, tracker, Mocker(seed, brief) if mock else None)
    rng = random.Random(seed)

    mode = "[yellow]MOCK (free, simulated answers)[/]" if mock else "[green]LIVE[/]"
    console.rule(f"[bold]Kohort Concept Test: {brief.product_name}")
    console.print(f"  Mode: {mode}   Personas: {n_personas}   Seed: {seed}\n  Output: {run_dir}\n")

    # 1. Research
    console.print("[bold]1/6 Market research[/]")
    research = await research_stage.run(llm, cfg, brief, console)
    _save(run_dir, "research.json", research)
    console.print(f"  {len(research.segments)} segments: " + ", ".join(
        f"{s.name} ({s.share:.0%})" for s in research.segments))

    # 2. Personas
    console.print("[bold]2/6 Building persona panel[/]")
    sampled = sample_traits(research.segments, n_personas, rng, cfg.role("personas").models)
    with _progress(console) as prog:
        t = prog.add_task("personas", total=len(sampled))
        panel, errors = await persona_stage.run(llm, cfg, brief, research, sampled, lambda: prog.advance(t))
    _save(run_dir, "personas.json", panel)
    _check(len(panel), len(sampled), errors, "persona generation", console)

    # 3. Interviews
    console.print("[bold]3/6 Interviewing personas[/]")
    with _progress(console) as prog:
        t = prog.add_task("interviews", total=len(panel))
        rows, errors = await interviews.run(llm, cfg, brief, panel, lambda: prog.advance(t))
    _save(run_dir, "responses.json", rows)
    _check(len(rows), len(panel), errors, "interviews", console)
    answered = {r.persona_id for r in rows}
    panel = [p for p in panel if p.id in answered]

    # 4. Analysis
    console.print("[bold]4/6 Analysing results[/]")
    quant = analysis.summarize(brief, panel, rows, cfg.run.intent_weights)
    themes = await synthesis.code_themes(llm, cfg, brief, rows, panel)
    _save(run_dir, "analysis.json", quant)
    _save(run_dir, "themes.json", themes)
    console.print(f"  Calibrated trial estimate: [bold]{quant['trial_estimate']:.0%}[/]  "
                  f"(mean intent {quant['metrics']['purchase_intent']['mean']})")
    for flag in quant["diagnostics"]["flags"]:
        console.print(f"  [yellow]! {flag}[/]")

    # 5. Critic
    console.print("[bold]5/6 Independent quality review[/]")
    critic = await synthesis.critique(llm, cfg, brief, quant, themes, research)
    _save(run_dir, "critic.json", critic)
    console.print(f"  Confidence: {critic.confidence}  ({len(critic.issues)} issues raised)")

    # 6. Report
    console.print("[bold]6/6 Writing report[/]")
    narrative = await synthesis.write_narrative(llm, cfg, brief, research, quant, themes, critic, rows, panel)
    _save(run_dir, "narrative.json", narrative)
    cost = tracker.summary()
    md = render_markdown(brief=brief, research=research, quant=quant, themes=themes, critic=critic,
                         narrative=narrative, rows=rows, personas=panel, run_id=run_id, mock=mock, cost=cost)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    (run_dir / "report.html").write_text(render_html(md, f"Kohort: {brief.product_name}"), encoding="utf-8")
    _save(run_dir, "costs.json", cost)

    console.rule()
    console.print(f"  Verdict: [bold]{narrative.verdict}[/]   Confidence: {critic.confidence}   "
                  f"Cost: ${cost['total_usd']:.2f} ({cost['calls']} AI calls)")
    console.print(f"  Report: [link=file:///{(run_dir / 'report.html').as_posix()}]{run_dir / 'report.html'}[/link]")
    return run_dir


def _check(ok: int, total: int, errors: list[str], stage: str, console: Console) -> None:
    if errors:
        console.print(f"  [yellow]{len(errors)} of {total} failed in {stage}; first error: {errors[0][:300]}[/]")
    if total == 0 or ok / total < MIN_SUCCESS_RATE:
        raise RuntimeError(f"Too many failures in {stage} ({ok}/{total} succeeded). Aborting.")
