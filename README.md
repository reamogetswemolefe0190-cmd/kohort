# Kohort

**An AI-assisted market research engine for testing product concepts before committing time and capital.**

Kohort combines synthetic consumer interviews, live market research, asynchronous LLM orchestration and deterministic Python analytics to produce structured, evidence-aware concept evaluations.

**Developer:** Reamogetswe Molefe  
**Status:** Active development  
**Built with:** Python · asyncio · Pydantic · OpenRouter · LLM APIs

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![AsyncIO](https://img.shields.io/badge/Concurrency-asyncio-orange)](https://docs.python.org/3/library/asyncio.html)
[![Pydantic](https://img.shields.io/badge/Validation-Pydantic-red)](https://docs.pydantic.dev/)
[![OpenRouter](https://img.shields.io/badge/LLM_Routing-OpenRouter-purple)](https://openrouter.ai/)

---

## The problem

Testing a business idea properly can involve recruiting participants, conducting interviews, analysing responses, researching competitors and evaluating pricing.

That process takes time and money particularly for early-stage founders who need to assess an idea before investing heavily in it.

**Kohort explores how AI can accelerate the early concept-evaluation process.**

A user provides a structured product brief. Kohort uses that brief to coordinate a research and simulation pipeline, analyse the resulting responses and generate a report highlighting potential opportunities, concerns and areas requiring further validation.

It is designed to support early decisions, not to replace real customer research.

## How it works

```text
       PRODUCT / BUSINESS BRIEF
                  |
                  v
          MARKET RESEARCH
                  |
                  v
       CONTROLLED POPULATION
              SAMPLING
                  |
                  v
       SYNTHETIC PERSONA
             GENERATION
                  |
                  v
       CONCURRENT AI INTERVIEWS
                  |
                  v
       DETERMINISTIC ANALYSIS
                  |
             +----+----+
             |         |
             v         v
       THEME CODING   QUALITY
                      REVIEW
             |         |
             +----+----+
                  |
                  v
         EVIDENCE-AWARE
          REPORT OUTPUT
```

The central engineering principle is:

> **LLMs generate and interpret qualitative material. Python calculates the quantitative results.**

This separation keeps numerical analysis out of free-form model responses and makes the results easier to inspect.

## Core capabilities

### 1. Structured research briefs

Concept tests are defined using YAML briefs and validated data structures.

A brief can describe the product, category, target region, features and questions that the research should investigate.

### 2. Controlled synthetic audiences

Kohort uses Python logic to construct demographic and behavioural profiles before generating their individual narratives.

Audience characteristics can include:

- Demographic attributes
- Income and price sensitivity
- Adoption profiles
- Existing habits and alternatives
- Behavioural characteristics

The sampling process uses configured distributions, including adopter categories based on the Rogers diffusion-of-innovation framework.

The LLM generates contextual persona descriptions around those predefined attributes.

### 3. Two-stage interviews

The interview process separates initial concept reactions from price evaluation.

**Stage one — Concept evaluation**

Participants encounter the product concept without being shown the proposed selling price. The system collects reactions, perceived value, concerns and willingness-to-pay thresholds.

**Stage two — Price reveal**

The proposed price is introduced after the initial responses. Participants then evaluate the price and provide purchase-intent responses.

This structure is intended to reduce price anchoring during the initial concept evaluation.

### 4. Deterministic quantitative analysis

Python performs the numerical calculations rather than asking an LLM to calculate or invent results.

The analysis includes:

- Response distributions and summary statistics
- Purchase-intent metrics
- Audience-segment comparisons
- Pricing analysis
- Response-quality diagnostics

Kohort also implements the Van Westendorp Price Sensitivity Meter using the four pricing thresholds collected during interviews.

### 5. Qualitative analysis

LLMs help organise open-ended responses into themes, including positive reactions, concerns and potential conversion triggers.

Python calculates theme frequencies and proportions from the coded responses.

This combines model-assisted interpretation with deterministic aggregation.

### 6. Quality checks and independent review

Synthetic research can produce misleading patterns, including repetitive answers, excessive positivity and low response diversity.

Kohort includes checks intended to identify these issues, alongside structured review of the resulting research.

A separate decision-model role can also evaluate research quality and flag potential weaknesses.

### 7. Evidence-aware reporting

The reporting stage combines research outputs, quantitative analysis, qualitative themes and quality-review findings.

Reports can include:

- Executive summaries
- Key findings
- Audience-segment insights
- Pricing observations
- Recommendations
- Research limitations and warnings
- GO / ITERATE / NO-GO decision guidance

The report is intended to support a decision, not to present a simulated result as a proven market outcome.

---

## Engineering architecture

Kohort is built around a Python orchestration layer that coordinates the stages of the research process.

| Component | Responsibility |
|---|---|
| Pipeline orchestrator | Coordinates the research workflow |
| Async LLM gateway | Routes model requests and manages execution |
| Pydantic schemas | Validates structured model outputs |
| Sampling engine | Creates controlled synthetic audience profiles |
| Interview stages | Coordinates concept and price-reveal interviews |
| Analysis engine | Calculates quantitative metrics |
| Evidence review | Checks claims and supporting evidence |
| Reporting engine | Produces Markdown and HTML reports |

### Asynchronous execution

Kohort uses Python `asyncio` to execute independent model tasks concurrently.

An `asyncio.Semaphore` controls the number of simultaneous requests, helping manage provider limits and resource consumption.

Example configuration:

```yaml
run:
  concurrency: 20
```

### Multi-model orchestration

Model assignments are configured separately in:

```text
config/models.yaml
```

Different model roles can be assigned to research, persona generation, analysis, review and report generation.

This makes it possible to change model providers without redesigning the entire pipeline.

### Structured output validation

Pydantic schemas define the expected structure of pipeline outputs.

This allows the application to validate model responses before passing them into subsequent stages.

### Cost and failure controls

The LLM gateway tracks model usage, token consumption, estimated costs and failed requests.

The pipeline also includes retry handling, output validation and configurable spending limits.

These controls are important because a large panel can involve many individual model requests.

---

## Technology stack

| Area | Technologies |
|---|---|
| Core language | Python |
| Asynchronous execution | asyncio |
| Data validation | Pydantic |
| Configuration | YAML, JSON |
| AI orchestration | OpenRouter, model APIs |
| Research retrieval | Perplexity-enabled research |
| HTTP / API | HTTPX, AsyncOpenAI |
| Reporting | Markdown, HTML |
| CLI | argparse, Rich |
| Testing | pytest, unittest |

## Running the project

### Requirements

- Python 3.10+
- API credentials for live model execution
- Git

### 1. Clone the repository

```bash
git clone https://github.com/reamogetswemolefe0190-cmd/kohort.git
cd kohort
```

### 2. Create a virtual environment

**Windows**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API credentials

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=
GEMINI_API_KEY=
```

Keep API credentials private. Do not commit your `.env` file.

### 5. Run a mock evaluation

Kohort includes a mock mode for running the pipeline without making live AI requests.

```bash
python -m kohort run briefs/example_loopa.yaml --mock --personas 60
```

### 6. Run a live evaluation

```bash
python -m kohort run briefs/example_loopa.yaml --personas 20
```

A larger synthetic panel can be requested with:

```bash
python -m kohort run briefs/example_loopa.yaml --personas 200
```

For reproducible sampling, provide a seed:

```bash
python -m kohort run briefs/example_loopa.yaml --personas 200 --seed 42
```

### Run tests

```bash
pytest tests/
```

---

## Output structure

Each evaluation generates a separate run directory containing intermediate research artifacts and the final report.

```text
runs/
└── <timestamp>_<product>/
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

Keeping intermediate outputs makes it possible to inspect how the final report was produced rather than relying exclusively on its conclusions.

## Important limitations

Kohort uses synthetic participants generated by language models. They are not real customers, and their responses should not be interpreted as representative survey data.

The outputs may be affected by:

- Model biases and training data
- Prompt design and persona construction
- Sampling assumptions
- Model-to-model variation
- Limited or incomplete market evidence

Synthetic purchase intent is not proof of actual purchasing behaviour. Pricing estimates are exploratory and should be validated with real customers.

**Kohort is an early-stage decision-support tool, not a substitute for primary market research.**

## About the developer

I'm Reamogetswe Molefe, a mechanical engineering student and software developer interested in applied AI, full-stack engineering and building systems that solve practical business problems.

Kohort is an exploration of how to combine model orchestration, deterministic computation, structured data and evidence review into a practical research workflow.

The project is actively being developed and tested.

---

**Repository:** [github.com/reamogetswemolefe0190-cmd/kohort](https://github.com/reamogetswemolefe0190-cmd/kohort)
