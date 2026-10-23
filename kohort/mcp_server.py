"""Model Context Protocol (MCP) Server for Kohort.

Exposes Kohort's market research swarm, competitive teardown, and persistent
audience tools over standard I/O (stdio) via JSON-RPC 2.0.

Compatible with Claude Desktop, Cursor, and any MCP-compliant client.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from rich.console import Console

from .config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from .schemas import Brief

MCP_VERSION = "2024-11-05"
SERVER_NAME = "kohort"
SERVER_VERSION = "0.2.0"

# All terminal diagnostics must go to stderr; stdout is reserved strictly for JSON-RPC frames.
err_console = Console(file=sys.stderr, highlight=False)


TOOLS: list[dict[str, Any]] = [
    {
        "name": "concept_test",
        "description": (
            "Run an AI swarm concept test on a product brief, interviewing synthetic customer personas "
            "and calculating Net Interest Score (NIS) and Van Westendorp pricing sensitivity curves."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "brief_path": {
                    "type": "string",
                    "description": "Path to a product brief YAML file (e.g. 'briefs/example_loopa.yaml').",
                },
                "personas": {
                    "type": "integer",
                    "description": "Panel size (default: 50 synthetic personas).",
                    "default": 50,
                },
                "mock": {
                    "type": "boolean",
                    "description": "If True, runs simulated answers offline for free with zero API costs (default: True).",
                    "default": True,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducible panels (default: 42).",
                    "default": 42,
                },
                "audience": {
                    "type": "string",
                    "description": "Optional saved audience panel name or path (e.g. 'sa-tech-founders') to reuse pre-built profiles.",
                },
            },
            "required": ["brief_path"],
        },
    },
    {
        "name": "competitive_teardown",
        "description": (
            "Deconstruct a competitor: scrape public pages, analyze pricing structure & gross margins, "
            "reverse-engineer architecture & vulnerabilities, and generate an implementation & exploitation blueprint."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Competitor name or URL (e.g. 'getminds.ai' or 'Loopa').",
                },
                "mock": {
                    "type": "boolean",
                    "description": "If True, runs format demo offline with zero API or search costs (default: True).",
                    "default": True,
                },
            },
            "required": ["target"],
        },
    },
    {
        "name": "find_advantage",
        "description": (
            "Analyze a product brief against market alternatives to identify competitive positioning opportunities, "
            "differentiators, customer trade-offs, and proposed landing page messaging."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "brief_path": {
                    "type": "string",
                    "description": "Path to a product brief YAML file.",
                },
                "mock": {
                    "type": "boolean",
                    "description": "If True, runs format demo offline (default: True).",
                    "default": True,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility (default: 42).",
                    "default": 42,
                },
            },
            "required": ["brief_path"],
        },
    },
    {
        "name": "build_audience",
        "description": (
            "Build and save a persistent, reusable panel of synthetic customer personas with controlled "
            "demographic, psychographic, and segment distributions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "spec": {
                    "type": "string",
                    "description": "Audience template name (e.g. 'sa-tech-founders', 'sa-mobile-gamers') or path to YAML spec.",
                },
                "personas": {
                    "type": "integer",
                    "description": "Panel sample size (default: 50).",
                    "default": 50,
                },
                "mock": {
                    "type": "boolean",
                    "description": "If True, simulates persona generation offline (default: True).",
                    "default": True,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed (default: 42).",
                    "default": 42,
                },
            },
            "required": ["spec"],
        },
    },
    {
        "name": "list_audiences",
        "description": (
            "List all available persistent saved audience panels and templates in the Kohort panel registry."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "explore_report",
        "description": (
            "Inspect the results of a past Kohort concept test, teardown, or swarm run: Net Interest Score, "
            "Van Westendorp pricing, top objections, and HTML dossier location."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_dir": {
                    "type": "string",
                    "description": "Path to the completed run directory (e.g. 'runs/20261014-213625_teardown_740600').",
                },
            },
            "required": ["run_dir"],
        },
    },
]


async def _run_concept_test_tool(args: dict[str, Any]) -> str:
    from .pipeline import run_concept_test

    brief_path = Path(args["brief_path"])
    if not brief_path.is_file():
        brief_path = PROJECT_ROOT / args["brief_path"]
    if not brief_path.is_file():
        raise FileNotFoundError(f"Brief file not found: {args['brief_path']}")

    n_personas = int(args.get("personas", 50))
    mock = bool(args.get("mock", True))
    seed = int(args.get("seed", 42))
    audience = args.get("audience")

    cfg = load_config()
    out_root = PROJECT_ROOT / "runs"
    run_dir = await run_concept_test(
        brief_path,
        cfg,
        mock=mock,
        n_personas=n_personas,
        seed=seed,
        out_root=out_root,
        console=err_console,
        audience=audience,
    )

    analysis_path = run_dir / "analysis.json"
    narrative_path = run_dir / "narrative.json"
    analysis_data = {}
    narrative_data = {}
    if analysis_path.is_file():
        analysis_data = json.loads(analysis_path.read_text(encoding="utf-8"))
    if narrative_path.is_file():
        narrative_data = json.loads(narrative_path.read_text(encoding="utf-8"))

    html_path = run_dir / "report.html"
    return json.dumps({
        "status": "success",
        "run_dir": str(run_dir),
        "report_html": str(html_path) if html_path.exists() else None,
        "mode": "MOCK" if mock else "LIVE",
        "personas_interviewed": n_personas,
        "trial_estimate": analysis_data.get("trial_estimate"),
        "verdict": narrative_data.get("verdict"),
        "van_westendorp": analysis_data.get("pricing", {}).get("van_westendorp") or analysis_data.get("van_westendorp"),
        "key_findings": narrative_data.get("key_findings", [])[:4],
        "action_plan": narrative_data.get("action_plan"),
    }, indent=2)


async def _run_teardown_tool(args: dict[str, Any]) -> str:
    from .teardown import run_teardown

    target = args["target"]
    mock = bool(args.get("mock", True))
    cfg = load_config()
    out_root = PROJECT_ROOT / "runs"

    run_dir = await run_teardown(
        target,
        cfg,
        mock=mock,
        out_root=out_root,
        console=err_console,
    )

    teardown_json_path = run_dir / "teardown.json"
    data = {}
    if teardown_json_path.is_file():
        data = json.loads(teardown_json_path.read_text(encoding="utf-8"))

    html_path = run_dir / "teardown.html"
    return json.dumps({
        "status": "success",
        "run_dir": str(run_dir),
        "dossier_html": str(html_path) if html_path.exists() else None,
        "target": data.get("competitor_name", target),
        "pricing_summary": data.get("pricing", {}).get("estimated_margins_and_markup"),
        "vulnerabilities": data.get("pricing", {}).get("vulnerabilities", []),
        "features_to_adopt": data.get("roadmap", {}).get("features_to_adopt", []),
        "differentiators_to_exploit": data.get("roadmap", {}).get("differentiators_to_exploit", []),
    }, indent=2)


async def _run_find_advantage_tool(args: dict[str, Any]) -> str:
    from .advantage import run_advantage

    brief_path = Path(args["brief_path"])
    if not brief_path.is_file():
        brief_path = PROJECT_ROOT / args["brief_path"]
    if not brief_path.is_file():
        raise FileNotFoundError(f"Brief file not found: {args['brief_path']}")

    mock = bool(args.get("mock", True))
    seed = int(args.get("seed", 42))
    cfg = load_config()
    out_root = PROJECT_ROOT / "runs"

    run_dir = await run_advantage(
        brief_path,
        cfg,
        mock=mock,
        seed=seed,
        out_root=out_root,
        console=err_console,
    )

    plan_path = run_dir / "plan.json"
    plan_data = {}
    if plan_path.is_file():
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))

    return json.dumps({
        "status": "success",
        "run_dir": str(run_dir),
        "positioning_summary": plan_data.get("summary"),
        "opportunities_count": len(plan_data.get("opportunities", [])),
        "opportunities": plan_data.get("opportunities", [])[:3],
    }, indent=2)


async def _run_build_audience_tool(args: dict[str, Any]) -> str:
    from .audiences import build_panel, load_audience_template, save_panel

    spec_input = args["spec"]
    spec = load_audience_template(spec_input)
    n_personas = int(args.get("personas", spec.sample_size))
    mock = bool(args.get("mock", True))
    seed = int(args.get("seed", 42))

    cfg = load_config()
    panel = await build_panel(spec, cfg, mock=mock, sample_size=n_personas, seed=seed, console=err_console)
    saved_path = save_panel(panel)

    return json.dumps({
        "status": "success",
        "name": panel.spec.name,
        "title": panel.spec.title,
        "region": panel.spec.region,
        "sample_size": panel.sample_size,
        "segments": [s.name for s in panel.spec.segments],
        "saved_path": str(saved_path),
        "mode": "MOCK" if mock else "LIVE",
    }, indent=2)


async def _run_list_audiences_tool(args: dict[str, Any]) -> str:
    from .audiences import list_panels

    panels = list_panels()
    return json.dumps({
        "status": "success",
        "count": len(panels),
        "audiences": panels,
    }, indent=2)


async def _run_explore_report_tool(args: dict[str, Any]) -> str:
    run_path = Path(args["run_dir"])
    if not run_path.is_dir():
        run_path = PROJECT_ROOT / args["run_dir"]
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {args['run_dir']}")

    results: dict[str, Any] = {"run_dir": str(run_path)}

    # Check for concept test artifacts
    analysis_file = run_path / "analysis.json"
    brief_file = run_path / "brief.json"
    narrative_file = run_path / "narrative.json"
    if analysis_file.is_file():
        an = json.loads(analysis_file.read_text(encoding="utf-8"))
        results["type"] = "concept_test"
        results["trial_estimate"] = an.get("trial_estimate")
        results["van_westendorp"] = an.get("pricing", {}).get("van_westendorp") or an.get("van_westendorp")
        if brief_file.is_file():
            bf = json.loads(brief_file.read_text(encoding="utf-8"))
            results["product_name"] = bf.get("product_name")
        if narrative_file.is_file():
            nf = json.loads(narrative_file.read_text(encoding="utf-8"))
            results["verdict"] = nf.get("verdict")
            results["key_findings"] = nf.get("key_findings", [])[:4]
            results["action_plan"] = nf.get("action_plan")
            results["recommendations"] = nf.get("recommendations") or nf.get("key_findings", [])[:4]

    # Check for teardown.json
    teardown_file = run_path / "teardown.json"
    if teardown_file.is_file():
        td = json.loads(teardown_file.read_text(encoding="utf-8"))
        results["type"] = "competitive_teardown"
        results["target"] = td.get("competitor_name")
        results["pricing"] = td.get("pricing", {}).get("estimated_margins_and_markup")
        results["features_to_adopt"] = td.get("roadmap", {}).get("features_to_adopt", [])

    # Check for plan.json (advantage)
    plan_file = run_path / "plan.json"
    if plan_file.is_file():
        pl = json.loads(plan_file.read_text(encoding="utf-8"))
        results["type"] = "advantage"
        results["summary"] = pl.get("summary")
        results["opportunities"] = pl.get("opportunities", [])[:3]

    # Collect available files
    results["files"] = [f.name for f in run_path.iterdir() if f.is_file()]
    return json.dumps(results, indent=2)


TOOL_HANDLERS = {
    "concept_test": _run_concept_test_tool,
    "competitive_teardown": _run_teardown_tool,
    "find_advantage": _run_find_advantage_tool,
    "build_audience": _run_build_audience_tool,
    "list_audiences": _run_list_audiences_tool,
    "explore_report": _run_explore_report_tool,
}


async def handle_rpc_request(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    req_id = request.get("id")

    # Notifications have no id and need no response
    if method == "notifications/initialized":
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": MCP_VERSION,
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
            },
        }

    if method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {},
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOLS,
            },
        }

    if method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True,
                },
            }

        try:
            output_text = await handler(tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": output_text}],
                    "isError": False,
                },
            }
        except Exception as exc:
            err_console.print(f"[red]Error executing tool {tool_name}:[/] {exc}")
            err_console.print(traceback.format_exc())
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error executing tool '{tool_name}': {exc}"}],
                    "isError": True,
                },
            }

    # Unsupported method
    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }
    return None


async def run_server() -> None:
    """Run the MCP server reading JSON-RPC messages from stdin and replying to stdout."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    err_console.print(f"[bold green]Kohort MCP Server v{SERVER_VERSION}[/] started on stdio.")
    err_console.print("Ready to accept JSON-RPC requests from Claude Desktop, Cursor, etc.")

    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            err_console.print(f"[yellow]Invalid JSON received:[/] {line[:80]}")
            continue

        resp = await handle_rpc_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def main() -> int:
    try:
        asyncio.run(run_server())
        return 0
    except (KeyboardInterrupt, BrokenPipeError):
        return 0
    except Exception as exc:
        err_console.print(f"[red]Fatal server error:[/] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
