# Kohort: AI Swarm Concept Testing

Kohort tests a client's product idea on a **simulated market**. A swarm of AI agents researches the market, builds a varied panel of synthetic consumers, interviews each one, analyses the results, has an independent AI reviewer critique them, and writes a client-ready report.

```
brief.yaml → research → persona panel → 2-step interviews → analysis → critic → report.html
```

## Quick start

```powershell
# 1. Free test run (no API key, simulated answers)
.\.venv\Scripts\python.exe -m kohort run briefs\example_loopa.yaml --mock --personas 60

# 2. Add your API key
copy .env.example .env      # then paste your OpenRouter key into .env

# 3. Check the key and every configured model (costs a fraction of a cent)
.\.venv\Scripts\python.exe -m kohort check

# 4. Small live run, then a full run
.\.venv\Scripts\python.exe -m kohort run briefs\example_loopa.yaml --personas 20
.\.venv\Scripts\python.exe -m kohort run briefs\example_loopa.yaml --personas 200
```

Each run creates a folder in `runs/` containing `report.html` (for the client), `report.md` (to edit), and the raw data from every stage (`personas.json`, `responses.json`, `analysis.json`, and so on) for auditing.

## New client test
1. Copy `briefs/example_loopa.yaml` to `briefs/<client>.yaml` and fill it in.
2. Run `python -m kohort run briefs/<client>.yaml`.
3. Review `report.md`, edit if needed, and send `report.html` (print to PDF from the browser).

## How rigor is built in
| Safeguard | Where |
|---|---|
| Demographics and attitudes sampled from explicit distributions (incl. Rogers' adopter curve), not invented by AI | `kohort/sampling.py` |
| Personas built *without* seeing the product (no priming) | `kohort/stages/personas.py` |
| Several AI model families in the panel (less single-model bias) | `config/models.yaml` |
| Anti-positivity interview instructions; price shown only *after* price thresholds are asked | `kohort/stages/interviews.py` |
| All numbers computed in code, never by the AI | `kohort/analysis.py` |
| Calibrated trial estimate (discounts stated intent) | `config/models.yaml` → `intent_weights` |
| Automatic flags: uniformity, herding, positivity, duplicate quotes, model sensitivity | `kohort/analysis.py` |
| Independent critic model reviews before the report is written | `kohort/stages/synthesis.py` |
| Hard spending cap per run | `config/models.yaml` → `run.budget_usd` |

## Changing models
Edit `config/models.yaml`. Each role (research, personas, analyst, critic, report) lists its model(s). Run `python -m kohort check` after any change.
