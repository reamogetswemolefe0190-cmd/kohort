# Kohort Model Context Protocol (MCP) Server

Kohort provides a first-class [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that allows LLMs inside **Claude Desktop**, **Cursor**, **ChatGPT Desktop**, and **VS Code** to directly invoke synthetic buyer swarms, competitive reverse-engineering teardowns, and audience generation tools.

---

## 1. Quick Setup

The MCP server runs over standard I/O (`stdio`) via JSON-RPC 2.0 with **zero external dependencies**.

### Starting via CLI
```bash
python -m kohort mcp
```

### Claude Desktop Configuration
Add Kohort to your `claude_desktop_config.json`:

* **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
* **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "kohort": {
      "command": "python",
      "args": [
        "-m",
        "kohort",
        "mcp"
      ],
      "cwd": "C:/Users/User/.gemini/antigravity/scratch/kohort"
    }
  }
}
```

> **Tip:** If using a virtual environment, specify the full path to `python.exe` in the venv (e.g. `C:/Users/User/.gemini/antigravity/scratch/kohort/.venv-check/Scripts/python.exe`).

### Cursor Configuration
Add to `.cursor/mcp.json` or through Cursor Settings → Features → MCP:
```json
{
  "mcpServers": {
    "kohort": {
      "command": "python",
      "args": ["-m", "kohort", "mcp"]
    }
  }
}
```

---

## 2. Available MCP Tools

| Tool | Description | Inputs |
|---|---|---|
| `concept_test` | Run an AI swarm concept test on a product brief, interviewing synthetic customer personas and computing Net Interest Score & Van Westendorp pricing sensitivity curves. | `brief_path` (required), `personas` (default 50), `mock` (default true), `seed` (default 42), `audience` (optional) |
| `competitive_teardown` | Reverse-engineer a competitor: scrape public pages, analyze pricing, detect architecture & vulnerabilities, and generate an actionable exploitation blueprint. | `target` (required, URL or company name), `mock` (default true) |
| `find_advantage` | Scan market alternatives for a product brief to find differentiators, customer trade-offs, and high-converting positioning copy. | `brief_path` (required), `mock` (default true), `seed` (default 42) |
| `build_audience` | Build and save a persistent, reusable panel of synthetic customer personas with controlled demographic distributions. | `spec` (required, template name or YAML path), `personas` (default 50), `mock` (default true), `seed` (default 42) |
| `list_audiences` | List all saved audience panels and templates available in the Kohort panel registry. | *(none)* |
| `explore_report` | Inspect the results, metrics, and generated HTML dossier from a completed concept test, teardown, or swarm run. | `run_dir` (required) |

---

## 3. Example Natural Language Prompts in Claude / Cursor

Once connected, you can ask your AI assistant:
- *"Run a competitive teardown of getminds.ai and tell me their pricing markup and vulnerabilities."*
- *"Run a concept test on briefs/example_loopa.yaml using our saved South African tech founders audience."*
- *"Show me all available synthetic audiences in Kohort."*
- *"Build a new audience of 50 South African mobile gamers for testing casual entertainment products."*
- *"Inspect the report in runs/20261015-050402_loopa_mock_fe1c02 and tell me the optimal price."*
