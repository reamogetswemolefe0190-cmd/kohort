"""Data models used across the Kohort concept-test pipeline.

Validators are intentionally lenient: LLMs sometimes return "4" instead of 4,
or price points out of order. We repair what we safely can instead of failing.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

IncomeBand = Literal["low", "lower-middle", "middle", "upper-middle", "high"]
AdopterType = Literal["innovator", "early adopter", "early majority", "late majority", "laggard"]
Level = Literal["low", "medium", "high"]

INCOME_BANDS: list[str] = ["low", "lower-middle", "middle", "upper-middle", "high"]


def _clamp_score(value, lo: int = 1, hi: int = 5) -> int | None:
    """Coerce a 1-5 rating from int/float/str and clamp it to the scale."""
    if value is None or value == "":
        return None
    try:
        num = round(float(str(value).strip().split("/")[0]))
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, num))


def _to_price(value) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    s = str(value).strip()
    s = re.sub(r"(?<=\d)[,\s](?=\d{3}(?:\D|$))", "", s)
    s = re.sub(r"(?<=\d),(?=\d{1,2}(?:\D|$))", ".", s)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        num = float(m.group(0))
    except ValueError:
        return None
    return num if num > 0 else None


# --------------------------------------------------------------------------- brief


class HookVariant(BaseModel):
    """A distinct messaging hook or value proposition tested against personas."""

    id: str
    name: str = ""
    hook_text: str
    angle: str = ""


class Brief(BaseModel):
    """What the client gives us."""

    client: str
    product_name: str
    category: str = Field(description="Product category, e.g. 'household cleaning products'")
    one_liner: str
    description: str
    key_features: list[str] = Field(default_factory=list)
    price: float | None = None
    currency: str = "EUR"
    price_unit: str = ""  # e.g. "per month"
    target_market: str
    region: str
    competitors: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    variants: list[HookVariant] = Field(default_factory=list)

    def price_label(self, amount: float | None = None) -> str:
        amount = self.price if amount is None else amount
        if amount is None:
            return "not specified"
        unit = f" {self.price_unit}" if self.price_unit else ""
        return f"{amount:,.2f} {self.currency}{unit}"

    def concept_board(self, include_price: bool, variant: HookVariant | None = None) -> str:
        """Neutral concept description shown to personas (no marketing hype)."""
        headline = variant.hook_text if (variant and variant.hook_text) else self.one_liner
        lines = [
            f"PRODUCT: {self.product_name}",
            f"IN ONE LINE: {headline}",
            "",
            self.description.strip(),
        ]
        if self.key_features:
            lines += ["", "KEY FEATURES:"] + [f"- {f}" for f in self.key_features]
        if include_price and self.price is not None:
            lines += ["", f"PRICE: {self.price_label()}"]
        return "\n".join(lines)


# --------------------------------------------------------------------------- research


class Competitor(BaseModel):
    name: str
    offering: str = ""
    price_point: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class BuyerDiscoveryQuery(BaseModel):
    focus: Literal["buying_habits", "frustrations", "price_anchors", "switching_triggers"] = "buying_habits"
    rationale: str = ""
    search_query: str


class BuyerDiscoveryPlan(BaseModel):
    """Sense-Maker output: translates a client brief into targeted scraper missions for buyer psychology."""

    buyer_persona_hypothesis: str
    category_context_needed: str
    queries: list[BuyerDiscoveryQuery] = Field(default_factory=list)


class BuyerDossier(BaseModel):
    """Scraped real-world buyer intelligence used to tailor persona personalities."""

    buying_habits: list[str] = Field(default_factory=list)
    real_frustrations: list[str] = Field(default_factory=list)
    price_anchors: list[str] = Field(default_factory=list)
    status_quo_alternatives: list[str] = Field(default_factory=list)
    cultural_signals: list[str] = Field(default_factory=list)
    scraped_urls: list[str] = Field(default_factory=list)


class Segment(BaseModel):
    name: str
    description: str = ""
    share: float = Field(ge=0)
    age_min: int = 18
    age_max: int = 75
    income_levels: list[IncomeBand] = Field(default_factory=lambda: ["lower-middle", "middle", "upper-middle"])
    key_needs: list[str] = Field(default_factory=list)
    likely_objections: list[str] = Field(default_factory=list)
    buying_habits: list[str] = Field(default_factory=list)
    price_anchors: list[str] = Field(default_factory=list)

    @field_validator("income_levels", mode="before")
    @classmethod
    def _filter_incomes(cls, v):
        if not isinstance(v, list):
            return v
        cleaned = [str(x).strip().lower() for x in v if str(x).strip().lower() in INCOME_BANDS]
        return cleaned or ["lower-middle", "middle", "upper-middle"]

    @model_validator(mode="after")
    def _fix_ages(self):
        self.age_min = max(18, min(self.age_min, 90))
        self.age_max = max(self.age_min, min(self.age_max, 90))
        return self


class MarketResearch(BaseModel):
    summary: str
    trends: list[str] = Field(default_factory=list)
    competitors: list[Competitor] = Field(default_factory=list)
    typical_price_range: str = ""
    segments: list[Segment]
    buyer_dossier: BuyerDossier | None = None
    discovery_plan: BuyerDiscoveryPlan | None = None
    sources: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    web_search_used: bool = False

    @model_validator(mode="after")
    def _normalise_shares(self):
        if not self.segments:
            raise ValueError("research must define at least one segment")
        total = sum(s.share for s in self.segments)
        if total <= 0:
            for s in self.segments:
                s.share = 1 / len(self.segments)
        else:
            for s in self.segments:
                s.share = s.share / total
        return self


# --------------------------------------------------------------------------- personas


class PersonaTraits(BaseModel):
    """Sampled in code (not by the LLM) to guarantee a realistic spread."""

    segment: str
    age: int
    gender: str
    income: IncomeBand
    household: str
    location: str
    adopter_type: AdopterType
    price_sensitivity: Level
    skepticism: Level


class PersonaProfile(BaseModel):
    """The part of a persona the LLM writes."""

    name: str
    occupation: str
    bio: str
    current_solution: str = ""
    values: list[str] = Field(default_factory=list)
    frustrations: list[str] = Field(default_factory=list)


class Persona(PersonaProfile):
    id: str
    model: str
    traits: PersonaTraits
    category_lens: dict = Field(default_factory=dict)
    trait_provenance: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------- interviews


class QA(BaseModel):
    question: str
    answer: str


class InterviewPart1(BaseModel):
    """Reactions to the concept *before* the price is revealed."""

    first_reaction: str
    appeal: int
    uniqueness: int
    believability: int
    likes: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    price_too_cheap: float | None = None
    price_bargain: float | None = None
    price_getting_expensive: float | None = None
    price_too_expensive: float | None = None
    client_questions: list[QA] = Field(default_factory=list)
    # Only used when the brief has no price (otherwise asked in part 2).
    purchase_intent: int | None = None
    would_buy_if: str = ""
    quote: str = ""

    @field_validator("appeal", "uniqueness", "believability", mode="before")
    @classmethod
    def v_scores(cls, v):
        return _clamp_score(v) or 3

    @field_validator("purchase_intent", mode="before")
    @classmethod
    def v_intent(cls, v):
        return _clamp_score(v)

    @field_validator(
        "price_too_cheap", "price_bargain", "price_getting_expensive", "price_too_expensive", mode="before"
    )
    @classmethod
    def v_prices(cls, v):
        return _to_price(v)


class InterviewPart2(BaseModel):
    """Reactions once the actual price is shown."""

    price_reaction: str = ""
    value_for_money: int
    purchase_intent: int
    would_buy_if: str = ""
    quote: str = ""

    @field_validator("value_for_money", "purchase_intent", mode="before")
    @classmethod
    def v_scores(cls, v):
        return _clamp_score(v) or 3


class InterviewResponse(BaseModel):
    persona_id: str
    model: str
    variant_id: str | None = None
    first_reaction: str
    appeal: int
    uniqueness: int
    believability: int
    purchase_intent: int
    value_for_money: int | None = None
    price_reaction: str = ""
    likes: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    would_buy_if: str = ""
    quote: str = ""
    client_questions: list[QA] = Field(default_factory=list)
    price_too_cheap: float | None = None
    price_bargain: float | None = None
    price_getting_expensive: float | None = None
    price_too_expensive: float | None = None
    price_points_repaired: bool = False

    @model_validator(mode="after")
    def _fix_price_order(self):
        pts = [self.price_too_cheap, self.price_bargain, self.price_getting_expensive, self.price_too_expensive]
        if any(p is None for p in pts):
            self.price_too_cheap = self.price_bargain = None
            self.price_getting_expensive = self.price_too_expensive = None
            return self
        ordered = sorted(pts)
        if ordered != pts:
            self.price_points_repaired = True
            (self.price_too_cheap, self.price_bargain,
             self.price_getting_expensive, self.price_too_expensive) = ordered
        return self


# --------------------------------------------------------------------------- synthesis


class Theme(BaseModel):
    id: str
    label: str
    description: str = ""


class ThemeCoding(BaseModel):
    themes: list[Theme]
    assignments: dict[str, str] = Field(default_factory=dict)  # item index -> theme id

    @field_validator("assignments", mode="before")
    @classmethod
    def _accept_list(cls, v):
        # Accept [{"item": 0, "theme": "T1"}, ...] as well as {"0": "T1"}.
        if isinstance(v, list):
            out = {}
            for row in v:
                if isinstance(row, dict):
                    idx = row.get("item", row.get("index"))
                    theme = row.get("theme", row.get("theme_id"))
                    if idx is not None and theme is not None:
                        out[str(idx)] = str(theme)
            return out
        if isinstance(v, dict):
            return {str(k): str(t) for k, t in v.items()}
        return v


class CriticIssue(BaseModel):
    severity: Literal["low", "medium", "high"] = "medium"
    issue: str
    recommendation: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def _sev(cls, v):
        v = str(v).lower().strip()
        return v if v in ("low", "medium", "high") else "medium"


class CriticReview(BaseModel):
    issues: list[CriticIssue] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    confidence_rationale: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        v = str(v).lower().strip()
        return v if v in ("low", "medium", "high") else "medium"


class ActionPlan(BaseModel):
    what_to_do_next: str = Field(description="A clear recommendation tied to their actual decision.")
    why_this_option: str = Field(description="Supporting evidence, alternatives, and the main uncertainty.")
    what_to_change: str = Field(description="Specific changes to the offer, message, campaign, or customer targeting.")
    how_to_implement: str = Field(description="Usable material: revised messaging, concept briefs, interview questions, or a test plan.")
    how_to_measure: str = Field(description="A measurable outcome and decision criteria suited to their budget.")
    what_happens_next: str = Field(description="Review the results, update the recommendation, and identify the next decision.")


class SegmentInsight(BaseModel):
    segment: str
    insight: str


class ReportNarrative(BaseModel):
    verdict: Literal["GO", "ITERATE", "NO-GO"]
    verdict_rationale: str
    executive_summary: str
    key_findings: list[str] = Field(default_factory=list)
    segment_insights: list[SegmentInsight] = Field(default_factory=list)
    action_plan: ActionPlan
    client_question_answers: list[QA] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    @field_validator("verdict", mode="before")
    @classmethod
    def _verdict(cls, v):
        v = str(v).upper().strip().replace("_", "-").replace(" ", "-")
        return "NO-GO" if v in ("NOGO", "NO-GO", "NO") else v


# --------------------------------------------------------------------------- shootout & statistical inference


class WilsonInterval(BaseModel):
    lower: float
    upper: float
    confidence: float = 0.95


class PairwiseComparison(BaseModel):
    other_variant_id: str
    diff_top2: float
    z_score: float
    p_value: float
    significant_95: bool
    significant_90: bool


class VariantMetrics(BaseModel):
    variant_id: str
    variant_name: str
    hook_text: str
    angle: str = ""
    n: int
    intent_mean: float
    intent_top2: float
    conversion_count: int
    wilson_ci: WilsonInterval
    trial_estimate: float
    appeal_mean: float
    pairwise: list[PairwiseComparison] = Field(default_factory=list)


class ShootoutAnalysis(BaseModel):
    enabled: bool = True
    variants: list[VariantMetrics] = Field(default_factory=list)
    winner_id: str | None = None
    winner_rationale: str = ""
    recommendation: str = ""


# --------------------------------------------------------------------------- deterministic data binding & provenance


class AdvisoryProvenance(BaseModel):
    source_metric_id: str  # e.g. "pricing.van_westendorp.optimal_price_point"
    metric_value: float | int | str | None = None
    advisory_interpretation: str
    is_editorial: bool = False


class AdvisoryRecommendation(BaseModel):
    strategic_verdict: str
    pricing_corridor_min: float | None = None
    pricing_corridor_max: float | None = None
    rationale: str = ""
    provenance: list[AdvisoryProvenance] = Field(default_factory=list)


class RawSimulationOutput(BaseModel):
    n: int
    metrics: dict
    trial_estimate: float
    intent_weights: dict[str, float]
    by: dict
    pricing: dict
    diagnostics: dict
    shootout: ShootoutAnalysis | None = None

