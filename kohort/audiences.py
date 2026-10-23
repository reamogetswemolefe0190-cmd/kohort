"""Load reusable product-blind profiles from a saved protocol-4 swarm or persistent audience panel.

Only audience inputs are imported. Past answers and chosen offers are never read.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field
from rich.console import Console

from .config import PROJECT_ROOT, Config
from .llm import CostTracker, LLM
from .mock import Mocker
from .sampling import sample_traits
from .schemas import Brief, IncomeBand, MarketResearch, Persona, Segment

if TYPE_CHECKING:
    pass


def load_saved_audience(directory, brief, *, count, mock, models):
    from .swarm import BuyerProfile, PanelDesign, validate_audience_scope

    root = Path(directory).resolve()
    hashes = {}

    def read(name):
        raw = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode('utf-8'))

    run = read('run.json')
    if run.get('swarm_protocol') != 4 or type(run.get('mock')) is not bool:
        raise ValueError('Audience reuse requires a protocol-4 swarm with a recorded mock flag.')
    if run['mock'] != mock:
        raise ValueError('Mock and live audiences cannot be mixed. Match --mock to the saved audience.')
    old_brief = read('source-advisory.json')['brief']
    context = {key: getattr(brief, key) for key in ('region', 'category', 'target_market')}
    if old_brief.get('client') != brief.client or any(old_brief.get(k) != v for k, v in context.items()):
        raise ValueError('Saved audience belongs to a different client or buyer context. Build a new audience.')
    design = PanelDesign.model_validate(read('panel-design.json'))
    validate_audience_scope(design, context)
    limitations = read('panel-limitations.json')
    issues = limitations.get('issues')
    if not isinstance(issues, list) or any(not isinstance(x, str) for x in issues):
        raise ValueError('Saved audience review limitations are invalid.')
    assignments = read('assignments.json')
    people = read('personas.json')
    if len(assignments) != count or len(people) != count or run.get('requested_personas') != count:
        raise ValueError('Reuse requires every saved profile and the original panel size; set --personas to that size.')
    people_by_id = {p['persona_id']: p for p in people}
    assignment_ids = [a['persona_id'] for a in assignments]
    if (len(people_by_id) != count or len(set(assignment_ids)) != count
            or set(assignment_ids) != set(people_by_id)):
        raise ValueError('Saved audience has duplicate or mismatched profile IDs.')
    segments = {f'S{i}': s.model_dump() for i, s in enumerate(design.segments, 1)}
    if {a['segment_id'] for a in assignments} != set(segments):
        raise ValueError('Saved audience does not cover every scenario in its design.')
    clean_assignments, profiles = [], {}
    for assignment in assignments:
        pid = assignment['persona_id']
        if not isinstance(pid, str) or not pid:
            raise ValueError('Saved audience profile IDs must be nonempty strings.')
        person = people_by_id[pid]
        segment = segments[assignment['segment_id']]
        for key in ('segment_id', 'segment', 'audience_group', 'traits', 'model'):
            if person.get(key) != assignment.get(key):
                raise ValueError(f'Saved persona {pid} disagrees with its assignment: {key}.')
        if assignment['segment'] != segment or assignment['audience_group'] != segment['audience_group']:
            raise ValueError(f'Saved persona {pid} disagrees with the panel design.')
        traits = assignment['traits']
        if (not isinstance(traits, dict) or set(traits) != {'price_sensitivity', 'switching_reluctance'}
                or any(value not in ('low', 'medium', 'high') for value in traits.values())):
            raise ValueError(f'Saved persona {pid} has unsupported fixed traits.')
        if assignment['model'] not in models:
            raise ValueError(f'Saved audience model {assignment["model"]} is not in the current personas configuration.')
        # Deliberate allowlist: previous option order, responses and extra metadata
        # must not become inputs to this comparison.
        clean_assignments.append({k: assignment[k] for k in
                                  ('persona_id', 'segment_id', 'segment', 'audience_group', 'traits', 'model')})
        profiles[pid] = BuyerProfile.model_validate(person['profile'])
    provenance = {'version': 1, 'source_run': str(root), 'source_files_sha256': hashes,
                  'profiles_reused': count, 'prior_answers_imported': False,
                  'scope': 'Same client and exact buyer context; synthetic profiles, not a validated population.'}
    return design, clean_assignments, profiles, list(issues), provenance


# ---------------------------------------------------------------------------
# Persistent Audience Panel Layer (Track B)
# ---------------------------------------------------------------------------

class SegmentSpec(BaseModel):
    name: str = Field(description="Segment name")
    description: str = Field(default="", description="Segment description")
    share: float = Field(default=1.0, ge=0.0)
    age_min: int = Field(default=18, ge=18, le=90)
    age_max: int = Field(default=75, ge=18, le=90)
    income_levels: list[IncomeBand] = Field(default_factory=lambda: ["lower-middle", "middle", "upper-middle"])
    key_needs: list[str] = Field(default_factory=list)
    likely_objections: list[str] = Field(default_factory=list)
    buying_habits: list[str] = Field(default_factory=list)
    price_anchors: list[str] = Field(default_factory=list)

    def to_segment(self) -> Segment:
        return Segment(
            name=self.name,
            description=self.description,
            share=self.share,
            age_min=self.age_min,
            age_max=self.age_max,
            income_levels=self.income_levels,
            key_needs=self.key_needs,
            likely_objections=self.likely_objections,
            buying_habits=self.buying_habits,
            price_anchors=self.price_anchors,
        )


class AudienceSpec(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9_-]+$", description="Slug identifier, e.g. sa-tech-founders")
    title: str = Field(description="Display title, e.g. South African Tech Founders")
    region: str = Field(description="Geographic region, e.g. South Africa")
    category: str = Field(description="Category or industry, e.g. B2B Software")
    target_market: str = Field(description="Target market description")
    description: str = Field(default="", description="Audience context and research scope")
    sample_size: int = Field(default=50, ge=5, le=1000)
    criteria: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    segments: list[SegmentSpec] = Field(min_length=1)


class AudiencePanel(BaseModel):
    spec: AudienceSpec
    sample_size: int
    mock: bool
    seed: int
    created_at: str
    personas: list[Persona]
    provenance: dict = Field(default_factory=dict)


def default_audiences_dir() -> Path:
    return PROJECT_ROOT / "audiences"


def default_templates_dir() -> Path:
    return PROJECT_ROOT / "audiences" / "templates"


def load_audience_template(name_or_path: str | Path) -> AudienceSpec:
    """Load an AudienceSpec from a yaml file path or template name."""
    p = Path(name_or_path)
    if p.is_file():
        with open(p, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return AudienceSpec.model_validate(data)

    template_p = default_templates_dir() / f"{name_or_path}.yaml"
    if not template_p.is_file():
        template_p = default_templates_dir() / f"{name_or_path}.yml"
    if template_p.is_file():
        with open(template_p, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return AudienceSpec.model_validate(data)

    raise FileNotFoundError(f"Audience template '{name_or_path}' not found in {default_templates_dir()}")


async def build_panel(
    spec: AudienceSpec,
    cfg: Config,
    *,
    mock: bool = True,
    sample_size: int | None = None,
    seed: int = 42,
    console: Console | None = None,
) -> AudiencePanel:
    """Generate a persistent panel of synthetic customer personas grounded in an AudienceSpec."""
    from .stages import personas as persona_stage

    n_personas = sample_size or spec.sample_size
    rng = random.Random(seed)

    brief = Brief(
        client=spec.title,
        product_name=spec.name,
        category=spec.category,
        one_liner=spec.target_market,
        description=spec.description or spec.target_market,
        target_market=spec.target_market,
        region=spec.region,
    )

    research_segs = [s.to_segment() for s in spec.segments]
    research = MarketResearch(
        summary=f"Audience specification for {spec.title}",
        segments=research_segs,
        sources=list(spec.sources),
    )
    models = cfg.role("personas").models
    sampled = sample_traits(research.segments, n_personas, rng, models)

    tracker = CostTracker(budget_usd=cfg.run.budget_usd)
    llm = LLM(cfg, tracker, Mocker(seed, brief) if mock else None)
    try:
        personas, errors = await persona_stage.run(llm, cfg, brief, research, sampled, lambda: None)
        if errors and len(personas) < n_personas * 0.8:
            raise RuntimeError(
                f"Failed to generate panel personas: {len(personas)}/{n_personas} succeeded. "
                f"Sample errors: {errors[:3]}"
            )
    finally:
        await llm.aclose()

    panel = AudiencePanel(
        spec=spec,
        sample_size=len(personas),
        mock=mock,
        seed=seed,
        created_at=datetime.now(timezone.utc).isoformat(),
        personas=personas,
        provenance={
            "models_used": sorted(list({p.model for p in personas})),
            "cost_usd": tracker.total,
            "errors_count": len(errors),
            "sources": spec.sources,
            "assumptions": spec.assumptions,
        },
    )
    return panel


def save_panel(panel: AudiencePanel, out_dir: Path | None = None) -> Path:
    """Save an AudiencePanel to JSON in the audiences directory."""
    dir_path = out_dir or default_audiences_dir()
    dir_path.mkdir(parents=True, exist_ok=True)
    target = dir_path / f"{panel.spec.name}.json"
    target.write_text(panel.model_dump_json(indent=2), encoding="utf-8")
    return target


def save_panel_from_run(
    brief: Brief,
    research: MarketResearch,
    personas: list[Persona],
    *,
    mock: bool,
    seed: int,
    run_id: str = "",
    cost_usd: float = 0.0,
    out_dir: Path | None = None,
) -> tuple[AudiencePanel, Path]:
    """Automatically archive a run's product-blind Base Personas into the Audience Vault for future briefs."""
    import re

    raw_slug = f"{brief.region}-{brief.category}".lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw_slug).strip("-")[:48] or "panel"
    if mock:
        slug = f"{slug}-mock"

    seg_specs = [
        SegmentSpec(
            name=s.name,
            description=s.description,
            share=s.share,
            age_min=s.age_min,
            age_max=s.age_max,
            income_levels=s.income_levels,
            key_needs=s.key_needs,
            likely_objections=s.likely_objections,
            buying_habits=s.buying_habits,
            price_anchors=s.price_anchors,
        )
        for s in research.segments
    ]
    spec = AudienceSpec(
        name=slug,
        title=f"{brief.region} — {brief.category.title()} Buyers",
        region=brief.region,
        category=brief.category,
        target_market=brief.target_market,
        description=research.summary,
        sample_size=max(5, len(personas)),
        sources=list(research.sources),
        assumptions=list(research.caveats),
        segments=seg_specs,
    )
    panel = AudiencePanel(
        spec=spec,
        sample_size=len(personas),
        mock=mock,
        seed=seed,
        created_at=datetime.now(timezone.utc).isoformat(),
        personas=personas,
        provenance={
            "source_run_id": run_id,
            "client": brief.client,
            "models_used": sorted(list({p.model for p in personas})),
            "cost_usd": cost_usd,
            "sources": list(research.sources),
            "scraped_urls": list(research.buyer_dossier.scraped_urls) if research.buyer_dossier else [],
            "assumptions": list(research.caveats),
        },
    )
    path = save_panel(panel, out_dir=out_dir)
    return panel, path


def load_panel(name_or_path: str | Path, audiences_dir: Path | None = None) -> AudiencePanel:
    """Load an AudiencePanel from a JSON file or named audience."""
    p = Path(name_or_path)
    if p.is_file():
        return AudiencePanel.model_validate_json(p.read_text(encoding="utf-8"))

    dir_path = audiences_dir or default_audiences_dir()
    target = dir_path / f"{name_or_path}.json"
    if target.is_file():
        return AudiencePanel.model_validate_json(target.read_text(encoding="utf-8"))

    raise FileNotFoundError(f"Saved audience panel '{name_or_path}' not found in {dir_path}")


def list_panels(audiences_dir: Path | None = None) -> list[dict]:
    """List all saved audience panels and available audience templates."""
    dir_path = audiences_dir or default_audiences_dir()
    results = []

    # Saved panels
    if dir_path.is_dir():
        for file in sorted(dir_path.glob("*.json")):
            try:
                panel = AudiencePanel.model_validate_json(file.read_text(encoding="utf-8"))
                results.append({
                    "name": panel.spec.name,
                    "title": panel.spec.title,
                    "region": panel.spec.region,
                    "category": panel.spec.category,
                    "sample_size": panel.sample_size,
                    "status": "saved",
                    "mock": panel.mock,
                    "created_at": panel.created_at,
                    "path": str(file),
                })
            except Exception:
                continue

    # Templates
    tmpl_dir = default_templates_dir()
    saved_names = {r["name"] for r in results}
    if tmpl_dir.is_dir():
        for file in sorted(tmpl_dir.glob("*.yaml")) + sorted(tmpl_dir.glob("*.yml")):
            try:
                with open(file, encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                name = data.get("name", file.stem)
                if name not in saved_names:
                    results.append({
                        "name": name,
                        "title": data.get("title", name),
                        "region": data.get("region", "Global"),
                        "category": data.get("category", ""),
                        "sample_size": data.get("sample_size", 50),
                        "status": "template",
                        "mock": True,
                        "created_at": "",
                        "path": str(file),
                    })
            except Exception:
                continue

    return results
