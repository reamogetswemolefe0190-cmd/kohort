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


class _Ping(BaseModel):
    ok: bool


async def _check(cfg, console: Console) -> int:
    """Send a tiny request to every configured model to verify keys and model IDs."""
    llm = LLM(cfg, CostTracker(budget_usd=1.0))
    models = cfg.all_models()
    console.print(f"Checking {len(models)} models...\n")

    async def ping(model: str):
        try:
            await llm.structured(task="ping", schema=_Ping, model=model, system="Reply with ONLY JSON.",
                                 user='Reply with {"ok": true}', temperature=0, max_tokens=50)
            return model, True, ""
        except Exception as exc:  # report every failure, don't stop
            return model, False, str(exc)[:160]

    results = await asyncio.gather(*(ping(m) for m in models))
    table = Table("Model", "Status", "Detail")
    for model, ok, detail in results:
        table.add_row(model, "[green]OK[/]" if ok else "[red]FAILED[/]", detail)
    console.print(table)
    console.print(f"Cost of check: ${llm.tracker.total:.4f}")
    return 0 if all(ok for _, ok, _ in results) else 1


def main(argv: list[str] | None = None) -> int:
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

    args = parser.parse_args(argv)
    console = Console()
    cfg = load_config(args.config)

    try:
        if args.cmd == "check":
            return asyncio.run(_check(cfg, console))
        if args.personas < 5:
            parser.error("--personas must be at least 5")
        asyncio.run(run_concept_test(args.brief, cfg, mock=args.mock, n_personas=args.personas, seed=args.seed,
                                     out_root=Path(args.out), console=console))
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
