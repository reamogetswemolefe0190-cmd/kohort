# Kohort

> A hybrid AI market-research engine that tests product concepts against controlled synthetic consumer panels using asynchronous LLM orchestration, deterministic analytics and independent AI evaluation.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![AsyncIO](https://img.shields.io/badge/Concurrency-asyncio-orange)](https://docs.python.org/3/library/asyncio.html)
[![Pydantic](https://img.shields.io/badge/Validation-Pydantic-red)](https://docs.pydantic.dev/)
[![OpenRouter](https://img.shields.io/badge/LLM_Routing-OpenRouter-purple)](https://openrouter.ai/)
[![AI](https://img.shields.io/badge/Architecture-Multi--Model-black)](#architecture)

**Developer:** Reamogetswe Molefe  
**Status:** Active Development — 2026

---

## Overview

**Kohort** is a Python-based synthetic market-research system designed to help test product concepts before committing significant time or capital to development.

The system takes a structured client brief and runs it through a multi-stage AI research pipeline:

```text
Client Brief
    ↓
Live Market Research
    ↓
Controlled Persona Sampling
    ↓
Synthetic Persona Generation
    ↓
Two-Stage Consumer Interviews
    ↓
Deterministic Statistical Analysis
    ↓
Qualitative Theme Analysis
    ↓
Independent AI Critic
    ↓
Client-Ready Report
```

A typical run can evaluate a concept against a panel of up to **200 synthetic consumers** while preserving structured outputs, reproducibility, cost controls and quality checks.

The core design principle is simple:

> **LLMs generate qualitative evidence. Python computes the quantitative results.**

This reduces the risk of allowing language models to invent statistics or perform unreliable arithmetic.

---

## Why Kohort Exists

Traditional concept testing can require:

- participant recruitment
- scheduling
- survey design
- interviews
- qualitative coding
- statistical analysis
- reporting

Kohort explores whether parts of this process can be accelerated using synthetic consumer simulations while maintaining clear safeguards around uncertainty and model bias.

The system is intended as **directional decision support**, not as a replacement for real-world customer research.

---

## Architecture

```text
                               KOHORT
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   Client Brief   │
                         │      YAML        │
                         └────────┬─────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │     Market Research    │
                     │  Perplexity / Online   │
                     │      Retrieval         │
                     └───────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Deterministic Sampling  │
                    │       Python Logic      │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Synthetic Population  │
                    │ Demographics • Attitudes│
                    │ Adoption • Skepticism   │
                    └───────────┬─────────────┘
                                │
                                ▼
                  ┌──────────────────────────────┐
                  │   Concurrent LLM Workers     │
                  │       DeepSeek Models        │
                  │  AsyncIO + Semaphore Control │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Two-Stage Interviews    │
                    │                         │
                    │ 1. Concept reaction     │
                    │ 2. Price reveal         │
                    └───────────┬─────────────┘
                                │
                                ▼
                     ┌────────────────────────┐
                     │ Python Analysis Engine │
                     │                        │
                     │ Pricing                │
                     │ Purchase Intent        │
                     │ Segmentation           │
                     │ Diagnostics            │
                     └──────────┬─────────────┘
                                │
                   ┌────────────┴─────────────┐
                   │                          │
                   ▼                          ▼
        ┌─────────────────────┐    ┌─────────────────────┐
        │ Qualitative Coding  │    │ System One Critic   │
        │      DeepSeek       │    │        Jev          │
        └──────────┬──────────┘    └──────────┬──────────┘
                   │                          │
                   └────────────┬─────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │ Evidence-Constrained    │
                    │ Report Generation       │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   HTML / Markdown       │
                    │      Client Report      │
                    └─────────────────────────┘
```

---

## Pipeline

Kohort currently runs six major stages.

### 1. Market Research

The research stage gathers market context for the product category and region.

It identifies:

- market dynamics
- consumer behaviour
- competitors
- current alternatives
- typical price ranges
- realistic customer segments
- research caveats

The research model can use live web-enabled retrieval where configured.

---

### 2. Controlled Persona Sampling

Kohort does not simply ask an LLM to "invent 200 customers."

Consumer traits are first generated deterministically in Python.

These include:

- age
- gender
- income level
- household structure
- location type
- adopter profile
- price sensitivity
- skepticism

For example, technology-adoption behaviour follows explicit distributions based on the Rogers diffusion-of-innovation model:

```text
Innovators          2.5%
Early adopters     13.5%
Early majority     34.0%
Late majority      34.0%
Laggards           16.0%
```

Income levels also influence price sensitivity, while adopter type influences skepticism.

The LLM is then used to construct believable human context around those fixed traits.

This reduces demographic drift and the tendency for LLMs to generate overly affluent, enthusiastic or homogeneous synthetic users.

---

## Persona Generation Without Product Priming

Synthetic personas are generated **before they see the product being tested**.

At this stage, the model only knows:

- the consumer's demographic profile
- behavioural characteristics
- the general product category
- current habits
- existing solutions

The tested concept itself is hidden.

This is intended to reduce product priming and excessive positivity.

---

## Two-Stage Interviews

Each synthetic participant is interviewed in two stages.

### Stage One — Concept Without Price

The participant sees the product concept but not the intended price.

The system collects:

- first reaction
- appeal
- uniqueness
- believability
- likes
- concerns
- willingness-to-pay thresholds
- custom client questions

The participant also provides four pricing thresholds:

```text
Too cheap
Bargain
Getting expensive
Too expensive
```

### Stage Two — Price Reveal

Only after those thresholds are collected is the actual price revealed.

The participant then evaluates:

- price reaction
- value for money
- purchase intent
- purchase conditions
- final qualitative response

This structure reduces anchoring by preventing the client's intended price from influencing the initial willingness-to-pay estimates.

---

## Deterministic Analysis

One of Kohort's key design decisions is that **numerical results are not calculated by an LLM**.

All quantitative analysis is performed directly in Python.

This includes:

- mean scores
- Top-2-box intent
- Bottom-2-box intent
- rating distributions
- purchase-intent calibration
- segment analysis
- age-band analysis
- income analysis
- model comparison
- pricing analysis
- quality diagnostics

The generative models receive these computed results afterwards and are responsible only for interpretation.

---

## Purchase Intent Calibration

Stated purchase intent is deliberately discounted because consumers often overstate their willingness to buy.

Kohort currently uses configurable intent weights such as:

```yaml
intent_weights:
  definitely: 0.75
  probably: 0.25
```

The resulting trial estimate is calculated in Python instead of being generated by a model.

---

## Van Westendorp Pricing

Kohort includes a deterministic implementation of the **Van Westendorp Price Sensitivity Meter**.

Using each participant's four pricing thresholds, the system estimates:

- Point of Marginal Cheapness
- Optimal Price Point
- Indifference Price Point
- Point of Marginal Expensiveness

It also calculates median pricing thresholds and evaluates how the client's proposed price sits within the simulated population's expectations.

---

## Automatic Quality Diagnostics

Synthetic research introduces unique risks, so Kohort includes automatic diagnostics for suspicious patterns.

The analysis engine checks for:

### Low response diversity

Detects unusually low variation in purchase-intent scores.

### Herding

Flags situations where too many personas return the same score.

### Positivity bias

Flags unusually high Top-2-box purchase intent.

### Duplicate responses

Measures duplicate or highly repetitive respondent quotes.

### Model sensitivity

Compares results between different model families where multiple models are used.

### Invalid pricing sequences

Detects cases where synthetic respondents return inconsistent pricing thresholds.

These checks are calculated in code before final report generation.

---

## Hybrid System One / Generative Architecture

Kohort uses two different classes of AI reasoning.

### Generative Models

Generative models are used for:

- research
- persona construction
- interviews
- qualitative coding
- interpretation
- report writing

### System One Decision Model

An independent decision model can act as a critic of the generated dataset.

The current configuration includes a **Jev System One decision model** through OpenRouter.

Instead of producing another long generative answer, the critic evaluates structured questions such as:

```text
How confident should we be in this dataset?

Is there evidence of a severe data flaw,
positivity bias or unreliable synthetic behaviour?
```

The system receives structured decisions and probabilities which are incorporated into the research-quality assessment.

This creates a hybrid architecture where generative models create and interpret evidence while a separate decision-model layer evaluates risk.

---

## Multi-Model Orchestration

Model roles are configured independently in:

```text
config/models.yaml
```

Example architecture:

```yaml
roles:

  research:
    model: perplexity/llama-3.1-sonar-large-128k-online

  personas:
    models:
      - deepseek/deepseek-v4.1-flash

  analyst:
    model: deepseek/deepseek-v4.1-flash

  critic:
    model: typesafe/jev-router

  report:
    model: deepseek/deepseek-v4.1-flash
```

This makes it possible to swap model providers or models without changing the main application logic.

---

## Asynchronous LLM Execution

Kohort uses Python `asyncio` for concurrent AI workloads.

Persona creation and interviews are run concurrently rather than sequentially.

Concurrency is controlled using an `asyncio.Semaphore`:

```text
Panel Size
     │
     ▼
Async Task Queue
     │
     ▼
Concurrency Semaphore
     │
     ├── Worker
     ├── Worker
     ├── Worker
     ├── ...
     └── Worker
```

The default configuration currently allows:

```yaml
run:
  concurrency: 20
```

This enables large synthetic panels while preventing uncontrolled bursts against model providers.

---

## LLM Gateway

All model calls pass through a unified LLM service.

The gateway handles:

- provider routing
- asynchronous execution
- model selection
- retries
- structured JSON parsing
- Pydantic validation
- malformed-output repair
- web-enabled requests
- usage tracking
- token accounting
- estimated cost calculation
- hard spending limits

This keeps model-provider logic separate from the rest of the application.

---

## Structured Outputs with Pydantic

AI responses are validated using Pydantic schemas.

Instead of trusting arbitrary model output, each stage expects a defined data structure.

For example:

```text
MarketResearch
Persona
InterviewResponse
ThemeCoding
CriticReview
ReportNarrative
```

If an LLM produces invalid JSON or violates a schema, Kohort can provide the validation error back to the model and request a corrected response.

This improves pipeline reliability and reduces downstream parsing failures.

---

## Retry and Repair Logic

LLM calls may fail due to:

- network errors
- rate limits
- malformed JSON
- provider errors
- schema mismatches

Kohort includes automatic retry logic with exponential backoff.

When a model produces malformed structured output, the system can send the original response and parsing error back to the model with instructions to repair the JSON.

The number of retries is configurable.

---

## Cost Tracking

Each AI request is tracked.

The system records:

- model
- number of calls
- input tokens
- output tokens
- estimated cost
- failed tasks

A run produces a cost summary such as:

```json
{
  "total_usd": 0.00,
  "budget_usd": 1.00,
  "calls": 0,
  "failed_tasks": 0,
  "by_model": {}
}
```

---

## Hard Spending Limit

Kohort includes a per-run budget limit.

Example:

```yaml
run:
  budget_usd: 1.0
```

Before additional LLM requests are sent, the cost tracker checks whether the configured budget has been exceeded.

If the limit is breached, the run raises a `BudgetExceeded` exception.

This helps prevent uncontrolled API spending during large synthetic-panel runs.

---

## Failure Handling

Large asynchronous AI workflows inevitably experience occasional failures.

Kohort therefore tracks successful and failed tasks independently.

For persona-generation and interview stages, the pipeline requires a minimum successful completion rate.

The current threshold is:

```text
60%
```

If too many tasks fail, the system aborts rather than producing a report from an unreliable partial dataset.

---

## Qualitative Theme Coding

Open-ended responses are grouped into themes by an analysis model.

Current categories include:

- likes
- concerns
- conversion triggers

The LLM builds a compact codebook and assigns responses to themes.

Python then calculates:

- number of personas mentioning each theme
- proportion of the panel mentioning the theme
- segment breakdown
- representative examples

This keeps theme interpretation generative while keeping frequency calculations deterministic.

---

## Evidence-Constrained Reporting

The final reporting model is explicitly instructed not to invent statistics.

It receives:

- deterministic quantitative results
- market research
- coded qualitative themes
- client-question responses
- automatic diagnostic warnings
- critic findings

It then produces:

- GO / ITERATE / NO-GO verdict
- executive summary
- key findings
- segment insights
- recommendations
- client-question answers
- risks

The generated report treats synthetic research as directional evidence rather than a guaranteed market forecast.

---

## Output

Every run creates its own output directory.

Example:

```text
runs/
└── 20261006-120000_product-name/
    │
    ├── brief.json
    ├── research.json
    ├── personas.json
    ├── responses.json
    ├── analysis.json
    ├── themes.json
    ├── critic.json
    ├── narrative.json
    ├── costs.json
    ├── report.md
    └── report.html
```

This makes each research run auditable.

A user can inspect the raw outputs behind the final report rather than receiving only an AI-generated summary.

---

# Technology Stack

### Core

- Python
- asyncio
- Pydantic
- YAML
- JSON

### AI / LLM

- OpenRouter
- OpenAI-compatible APIs
- Google AI-compatible API
- Perplexity
- DeepSeek
- Jev System One

### Networking

- AsyncOpenAI
- HTTPX

### Reporting

- Markdown
- HTML

### CLI

- argparse
- Rich

---

## Project Structure

```text
kohort/
│
├── audiences/                 # Pre-computed and cached demographic panels
│   ├── templates/             # Audience distribution specifications
│   └── *.json                 # Compiled persona panel records
│
├── briefs/                    # Concept testing specifications
│   ├── example_loopa.yaml     # Standard reference brief
│   └── novasite.yaml          # Multi-variant Hook Shootout brief
│
├── collateral/                # Visual collateral & motion showcase
│   └── kohort_motion_showcase.html # Interactive 5-scene timeline showcase
│
├── config/
│   └── models.yaml            # Multi-model routing, pricing, & guardrails
│
├── deliverables/              # Client reports, executive briefs, & audited decks
│
├── docs/                      # Technical references & integration guides
│   ├── audience-builder.md
│   └── mcp-server.md
│
├── kohort/
│   ├── stages/
│   │   ├── research.py        # Autonomous Perplexity market retrieval
│   │   ├── personas.py        # Unprimed synthetic persona synthesis
│   │   ├── interviews.py      # Two-stage concept & price reveal interviews
│   │   └── synthesis.py       # Deterministic rollup & qualitative theme coding
│   │
│   ├── advantage.py           # Competitive positioning & blue-ocean extraction
│   ├── analysis.py            # Van Westendorp, NIS, and Wilson CI shootout math
│   ├── audiences.py           # Persistent panel compilation & sampling
│   ├── binding.py             # Deterministic Jinja2 deliverable binding
│   ├── cli.py                 # Command-line interface orchestration
│   ├── config.py              # Configuration loading & model routing
│   ├── evidence_review.py     # Deterministic citation & claim provenance verification
│   ├── guardrails.py          # Supervisor routing & pipeline gating
│   ├── llm.py                 # Async OpenRouter gateway & budget enforcement
│   ├── mcp_server.py          # Model Context Protocol stdio server
│   ├── mock.py                # Zero-cost offline deterministic simulation
│   ├── pipeline.py            # End-to-end multi-agent pipeline orchestrator
│   ├── report.py              # Self-contained HTML & Markdown report generator
│   ├── resilience.py          # 3-Tier audit suite & deliverable reconciliation
│   ├── sampling.py            # Demographics, Rogers adoption, and hash variant allocator
│   ├── schemas.py             # Pydantic v2 schemas for all pipeline artifacts
│   ├── scraping.py            # SSRF-hardened competitor content scraper
│   ├── supervisor.py          # Jev read-only audit boundary
│   ├── swarm.py               # Multi-agent swarm consensus engine
│   └── teardown.py            # Automated competitor deconstruction
│
├── tests/                     # 113 automated unit, integration & resilience tests
├── requirements.txt           # Production dependencies
└── README.md
```

---

# Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/reamogetswemolefe0190-cmd/kohort.git
cd kohort
```

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure API Keys

Create a `.env` file in the project root.

Example:

```env
OPENROUTER_API_KEY=
GEMINI_API_KEY=
```

Never commit API keys to GitHub.

---

# Running Kohort

## Free Mock Run

Kohort includes a mock mode that performs a full pipeline run without calling external AI services.

```bash
python -m kohort run briefs/example_loopa.yaml --mock --personas 60
```

This is useful for:

- development
- testing
- UI/report inspection
- debugging pipeline logic

---

## Small Live Run

Before running a large panel:

```bash
python -m kohort run briefs/example_loopa.yaml --personas 20
```

---

## Full Synthetic Panel

```bash
python -m kohort run briefs/example_loopa.yaml --personas 200
```

---

## Reproducible Runs

A fixed random seed can be supplied:

```bash
python -m kohort run briefs/example_loopa.yaml --personas 200 --seed 42
```

This makes the deterministic persona sampling reproducible.

---

---

## Hook Shootout (A/B Testing)

Test multiple positioning angles or value propositions against isolated, hash-allocated synthetic panels with Wilson 95% confidence intervals and pairwise significance tests:

```bash
# Run a multi-variant shootout (e.g. 3 variants across 200 personas)
python -m kohort run briefs/novasite.yaml --personas 200 --mock
```

---

## Competitive Teardown

Reverse-engineer a competitor or benchmark tool directly from their URL:

```bash
python -m kohort teardown getminds.ai
python -m kohort teardown getminds.ai --mock
```

---

## Persistent Audience Builder

Create and reuse named demographic & psychographic panels across multiple concept tests without incurring re-sampling costs:

```bash
python -m kohort audience list
python -m kohort audience show sa-tech-founders
python -m kohort audience build sa-tech-founders --personas 50 --mock
python -m kohort run briefs/example_loopa.yaml --audience sa-tech-founders --mock
```

---

## Model Context Protocol (MCP) Server

Connect Claude Desktop or Cursor directly to Kohort's market-research toolset:

```bash
python -m kohort mcp
```

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

## 3-Tier Resilience Suite & Audit Reconciliation

Verify mathematical invariants, claim citations, and deliverable numbers:

```bash
# Run the full test suite (113 passing tests)
pytest tests/
# or
python -m unittest discover -s tests
```

---

## Interactive Agentic Motion Showcase

Open [`collateral/kohort_motion_showcase.html`](collateral/kohort_motion_showcase.html) directly in any modern browser for a zero-dependency, 5-scene timeline visualizing:
1. Client Brief Ingestion
2. Swarm Persona Dynamics
3. Hook Shootout Progression
4. Van Westendorp Pricing Curves & Optimal Price Point
5. 3-Tier Resilience Suite Verdict

---

## Check Model Configuration

Kohort can verify that all configured AI models are reachable:

```bash
python -m kohort check
```

The command sends a small structured request to each configured model and reports:

```text
Model
Status
Error details
Check cost
```

---

# Creating a New Concept Test

Copy the example brief:

```text
briefs/example_loopa.yaml
```

Create:

```text
briefs/my_product.yaml
```

Then define information such as:

```yaml
client: Example Company

product_name: Example Product

category: Consumer Technology

one_liner: A short explanation of the concept.

description: |
  Longer product description.

key_features:
  - Feature one
  - Feature two

region: South Africa

target_market: Young professionals

price: 499
currency: R

questions:
  - What is the strongest reason you would consider this product?
  - What would stop you from buying?
```

Run the test:

```bash
python -m kohort run briefs/my_product.yaml --personas 200
```

---

# Research Safeguards

Kohort intentionally includes several safeguards against common synthetic-research failure modes.

| Risk | Safeguard |
|---|---|
| LLM invents demographics | Traits sampled deterministically in Python |
| Personas are overly enthusiastic | Explicit skepticism and adopter distributions |
| Product primes personas | Personas created before seeing concept |
| Price anchors willingness-to-pay | Price revealed only after initial pricing questions |
| LLM invents statistics | All quantitative results computed in Python |
| Responses become homogeneous | Uniformity and herding diagnostics |
| AI positivity bias | Automatic Top-2-box warning |
| Same model creates systematic bias | Model-sensitivity diagnostics |
| Synthetic outputs are overtrusted | Independent critic stage |
| Provider output is malformed | Pydantic validation + repair |
| API calls become expensive | Token tracking + hard budget limit |
| Too many tasks fail | Minimum-success threshold abort |

---

# Limitations

Kohort is experimental synthetic market-research software.

Synthetic respondents are **not real consumers**.

The system may reproduce:

- model biases
- inaccurate assumptions
- outdated market knowledge
- demographic stereotypes
- provider-specific behaviour
- unrealistic stated purchase intent

For that reason, Kohort reports should be treated as:

> **Directional evidence for hypothesis generation and concept iteration — not guaranteed predictions of real consumer behaviour.**

Important product decisions should still be validated using real customers and real market data.

---

# Current Development Priorities

Current areas of improvement include:

- stronger multi-model persona diversity
- additional model-evaluation techniques
- improved synthetic-to-real calibration
- automated benchmark datasets
- better statistical uncertainty estimates
- stronger quality gates
- improved report visualisation
- reusable research templates
- provider fallback routing
- expanded automated testing
- web application interface

---

# About the Developer

## Reamogetswe Molefe

I am a Mechanical Engineering student at the **University of Johannesburg** with a strong interest in AI engineering, software development, SaaS and product systems.

I build software independently alongside my engineering studies.

Kohort was developed as an exploration of how multiple classes of AI systems can be orchestrated while keeping deterministic logic responsible for calculations and quality checks.

Current areas of interest include:

```text
Python
AI / LLM systems
Async programming
LLM orchestration
Structured outputs
AI evaluation
Synthetic data
Decision models
REST APIs
Full-stack development
SaaS
Product engineering
```

I also develop **CreatorCashFlow**, a full-stack SaaS platform for creator agencies, brands and creators.

GitHub:

https://github.com/reamogetswemolefe0190-cmd

---

# Project Status

**Active Development — Experimental**

Kohort is currently a research and product-engineering project and should not be interpreted as a validated substitute for traditional market research.

---

<p align="center">
  <strong>Kohort</strong><br>
  Synthetic research. Deterministic analysis. Better questions before expensive decisions.
</p>
