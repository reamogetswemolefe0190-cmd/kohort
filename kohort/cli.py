"""Command line: `python -m kohort run briefs/example_loopa.yaml --mock`."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from .config import PROJECT_ROOT, load_config
from .llm import LLM, CostTracker, LLMError
from .pipeline import run_concept_test
from .advantage import run_advantage


class _Ping(BaseModel):
    ok: bool


async def _check(cfg, console: Console) -> int:
    """Send a tiny request to every configured model to verify keys and model IDs."""
    llm = LLM(cfg, CostTracker(budget_usd=1.0))
    models = cfg.all_models()
    console.print(f"Checking {len(models)} models...\n")

    async def ping(model: str):
        try:
            if llm.is_decision_model(model):
                await llm.decide(model, {"purpose": "Connectivity check"}, {
                    "connectivity": {"type": "noul", "instructions": "Is this a connectivity check?"}
                })
                return model, True, ""
            await llm.structured(task="ping", schema=_Ping, model=model, system="Reply with ONLY JSON.",
                                 user='Reply with {"ok": true}', temperature=0, max_tokens=50)
            return model, True, ""
        except Exception as exc:  # report every failure, don't stop
            return model, False, str(exc)[:160]

    try:
        results = await asyncio.gather(*(ping(m) for m in models))
    finally:
        await llm.aclose()
    table = Table("Model", "Status", "Detail")
    for model, ok, detail in results:
        table.add_row(model, "[green]OK[/]" if ok else "[red]FAILED[/]", detail)
    console.print(table)
    console.print(f"Cost of check: ${llm.tracker.total:.4f}")
    return 0 if all(ok for _, ok, _ in results) else 1


def main(argv: list[str] | None = None) -> int:
    # Redirected Windows terminals can use cp1252, which cannot encode Rich's rules.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(prog="kohort", description="Kohort: AI swarm concept testing")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a concept test on a brief")
    run.add_argument("brief", help="Path to a brief YAML file")
    run.add_argument("--mock", action="store_true", help="Simulated answers: free, no API key needed")
    run.add_argument("--personas", type=int, default=200, help="Panel size (default 200)")
    run.add_argument("--seed", type=int, default=None, help="Random seed for a reproducible panel")
    run.add_argument("--config", default=None, help="Model config (default config/models.yaml)")
    run.add_argument("--out", default=str(PROJECT_ROOT / "runs"), help="Output folder")
    run.add_argument("--debug", action="store_true", help="Show full error tracebacks")

    chk = sub.add_parser("check", help="Verify your API key and that every configured model responds")
    chk.add_argument("--config", default=None)

    advantage = sub.add_parser("advantage", help="Find my competitive advantage: competitors, tools and positioning")
    advantage.add_argument("brief", help="Path to a product brief YAML file")
    advantage.add_argument("--mock", action="store_true", help="Offline format demo; no search or API costs")
    advantage.add_argument("--seed", type=int, default=42)
    advantage.add_argument("--config", default=None)
    advantage.add_argument("--out", default=str(PROJECT_ROOT / "runs"))
    advantage.add_argument("--debug", action="store_true")

    evaluation = sub.add_parser('evaluate-review', help='Review saved evidence; no new searches or client run')
    evaluation.add_argument('run_dir')
    evaluation.add_argument('--config', default=None)
    evaluation.add_argument('--budget', type=float, default=.25)
    evaluation.add_argument('--limit', type=int, choices=range(1, 4), default=3)
    evaluation.add_argument('--item', help='Exact saved review item name; optional targeted evaluation')
    evaluation.add_argument('--stage', choices=('competitive_research', 'positioning'), default='competitive_research')
    evaluation.add_argument('--out', default=str(PROJECT_ROOT / 'runs'))
    evaluation.add_argument('--debug', action='store_true')

    recovery = sub.add_parser('recover-advantage', help='Release a saved advisory after a passing positioning evaluation; no API calls')
    recovery.add_argument('evaluation_dir')
    recovery.add_argument('--config', default=None)
    recovery.add_argument('--out', default=str(PROJECT_ROOT / 'runs'))
    recovery.add_argument('--debug', action='store_true')

    swarm = sub.add_parser('swarm', help='Compare a released advisory with synthetic buyers; reuses saved research')
    swarm.add_argument('advisory_dir')
    swarm.add_argument('--personas', type=int, default=200)
    swarm.add_argument('--seed', type=int, default=42)
    swarm.add_argument('--mock', action='store_true')
    swarm.add_argument('--audience-from', help='Reuse profiles from a saved protocol-4 swarm for the same client and buyer context')
    swarm.add_argument('--budget', type=float, default=None)
    swarm.add_argument('--config', default=None)
    swarm.add_argument('--out', default=str(PROJECT_ROOT / 'runs'))
    swarm.add_argument('--debug', action='store_true')

    scrape = sub.add_parser('scrape', help='Fetch public source pages with Scrapling; no AI calls')
    scrape.add_argument('urls', nargs='+')
    scrape.add_argument('--config', default=None)
    scrape.add_argument('--out', default=str(PROJECT_ROOT / 'runs'))
    scrape.add_argument('--debug', action='store_true')

    run.add_argument("--audience", default=None, help="Saved audience panel name or path (e.g. sa-tech-founders)")

    mcp = sub.add_parser("mcp", help="Run Kohort as a Model Context Protocol (MCP) server over stdio")
    mcp.add_argument("--config", default=None)

    aud = sub.add_parser("audience", help="Manage and build persistent synthetic audience panels")
    aud.add_argument("--config", default=None)
    aud_sub = aud.add_subparsers(dest="aud_cmd", required=True)

    aud_list = aud_sub.add_parser("list", help="List all saved panels and templates")

    aud_show = aud_sub.add_parser("show", help="Show details and segment breakdown of an audience")
    aud_show.add_argument("name", help="Audience panel name or template")

    aud_build = aud_sub.add_parser("build", help="Build and save a persistent audience panel")
    aud_build.add_argument("spec", help="Audience template name or YAML path")
    aud_build.add_argument("--personas", type=int, default=None, help="Sample size (default from spec)")
    aud_build.add_argument("--mock", action="store_true", help="Build offline for free")
    aud_build.add_argument("--seed", type=int, default=42)
    aud_build.add_argument("--out", default=None, help="Custom output directory for audience JSON")

    teardown = sub.add_parser('teardown', help='Deconstruct a competitor: pricing, architecture, vulnerabilities and implementation plan')
    teardown.add_argument('target', help='Competitor name or website URL (e.g. getminds.ai)')
    teardown.add_argument('--mock', action='store_true', help='Offline format demo; no search or API costs')
    teardown.add_argument('--config', default=None)
    teardown.add_argument('--out', default=str(PROJECT_ROOT / 'runs'))
    teardown.add_argument('--debug', action='store_true')

    args = parser.parse_args(argv)
    console = Console()
    cfg = load_config(getattr(args, "config", None))

    try:
        if args.cmd == 'scrape':
            import json
            import secrets
            from .scraping import enrich_sources
            root = Path(args.out) / ('scrape-' + secrets.token_hex(4))
            root.mkdir(parents=True)
            settings = cfg.scraping.model_copy(update={'enabled': True})
            sources = asyncio.run(enrich_sources([{'url': url, 'title': url, 'content': ''} for url in args.urls], settings, root, 'manual'))
            (root / 'sources.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding='utf-8')
            for source in sources:
                record = source['scraping']
                console.print(f"  {record['status']}: {source['url']}", markup=False)
                if record.get('reason'):
                    console.print('    ' + record['reason'], markup=False)
            console.print(f'  Evidence saved: {root / "sources.json"}')
            return 0 if any(s['scraping']['status'] == 'fetched' for s in sources) else 1
        if args.cmd == 'swarm':
            from .swarm import run_swarm
            import math
            if not 5 <= args.personas <= 1000:
                parser.error('--personas must be between 5 and 1000')
            if args.budget is not None:
                if not math.isfinite(args.budget) or args.budget <= 0:
                    parser.error('--budget must be a positive finite amount')
                cfg.run.budget_usd = args.budget
            asyncio.run(run_swarm(args.advisory_dir, cfg, n_personas=args.personas, seed=args.seed,
                                 mock=args.mock, out_root=Path(args.out), console=console, audience_from=args.audience_from))
            return 0
        if args.cmd == 'teardown':
            from .teardown import run_teardown
            asyncio.run(run_teardown(args.target, cfg, mock=args.mock, out_root=Path(args.out), console=console))
            return 0
        if args.cmd == 'recover-advantage':
            from .advisory_recovery import recover_advisory
            result = recover_advisory(args.evaluation_dir, args.out)
            console.print(f'  Recovered advisory: {result}')
            return 0
        if args.cmd == 'evaluate-review':
            from .review_evaluation import evaluate_saved_run
            if not 0 < args.budget <= 1:
                parser.error('--budget must be greater than zero and no more than $1')
            asyncio.run(evaluate_saved_run(Path(args.run_dir), cfg, Path(args.out), console,
                                          budget=args.budget, limit=args.limit, item=args.item, stage=args.stage))
            return 0
        if args.cmd == 'mcp':
            from .mcp_server import run_server
            asyncio.run(run_server())
            return 0
        if args.cmd == 'audience':
            from .audiences import build_panel, list_panels, load_audience_template, load_panel, save_panel
            if args.aud_cmd == 'list':
                panels = list_panels()
                table = Table("Name", "Title", "Region", "Category", "Size", "Status")
                for p in panels:
                    status_col = "[green]Saved[/]" if p["status"] == "saved" else "[cyan]Template[/]"
                    table.add_row(p["name"], p["title"], p["region"], p["category"], str(p["sample_size"]), status_col)
                console.print(table)
                return 0
            if args.aud_cmd == 'show':
                try:
                    panel = load_panel(args.name)
                    spec = panel.spec
                    is_saved = True
                except Exception:
                    spec = load_audience_template(args.name)
                    is_saved = False
                console.print(f"\n[bold]{spec.title}[/] ({spec.name})")
                console.print(f"  Region: {spec.region}   Category: {spec.category}")
                console.print(f"  Target Market: {spec.target_market}")
                status_desc = f"Saved Panel ({panel.sample_size} personas)" if is_saved else "Template Spec"
                console.print(f"  Status: {status_desc}")
                console.print("\n[bold]Segments:[/]")
                for seg in spec.segments:
                    console.print(f"  • [bold]{seg.name}[/] ({seg.share:.0%}): {seg.description}")
                return 0
            if args.aud_cmd == 'build':
                spec = load_audience_template(args.spec)
                out_dir = Path(args.out) if args.out else None
                console.print(f"Building audience panel: [bold]{spec.title}[/] (mock={args.mock})...")
                panel = asyncio.run(build_panel(spec, cfg, mock=args.mock, sample_size=args.personas, seed=args.seed, console=console))
                saved_path = save_panel(panel, out_dir=out_dir)
                console.print(f"  [green]Success![/] Saved {panel.sample_size} personas to {saved_path}")
                return 0
        if args.cmd == "check":
            return asyncio.run(_check(cfg, console))
        if args.cmd == "advantage":
            asyncio.run(run_advantage(args.brief, cfg, mock=args.mock, seed=args.seed,
                                       out_root=Path(args.out), console=console))
            return 0
        if args.personas < 5:
            parser.error("--personas must be at least 5")
        asyncio.run(run_concept_test(args.brief, cfg, mock=args.mock, n_personas=args.personas, seed=args.seed,
                                     out_root=Path(args.out), console=console, audience=args.audience))
        return 0
    except LLMError as exc:
        console.print(f"[red]AI error:[/] {exc}")
    except KeyboardInterrupt:
        console.print("[yellow]Cancelled.[/]")
    except Exception as exc:  # top-level: show a clean message
        console.print(f"[red]Error:[/] {exc}")
        if getattr(args, "debug", False):
            raise
    return 1
