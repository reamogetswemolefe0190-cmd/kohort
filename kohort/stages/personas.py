"""Stage 2: build the synthetic customer panel (Two-Layer Persona Architecture).

Layer 1 (Base Persona): Fixed sampled traits + core identity (product-blind, reusable across runs).
Layer 2 (Category Lens): Scraped category habits, frustrations, price anchors, and status-quo alternatives
injected from Stage 1 research so each bot is tailored to the business request without product priming.
"""

from __future__ import annotations

from typing import Callable

from ..config import Config
from ..llm import LLM, LLMError, gather_cancel_on_error
from ..schemas import Brief, MarketResearch, Persona, PersonaProfile, PersonaTraits, Segment

SYSTEM = """You create realistic consumer or business-buyer personas for market research simulations.
Make each person specific, ordinary and believable - a real individual, not a marketing archetype.
CRITICAL TRAIT & NON-PRIMING RULES (never violate):
1. Gender & Pronouns: If gender is "non-binary", use ONLY they/them/their pronouns. If "woman", use she/her. If "man", use he/him.
2. Income Level & Price Sensitivity:
   - "low" or "lower-middle": modest budget, careful with cash flow and expenses.
   - "middle": steady middle-class income, balances practical value with budget limits.
   - "upper-middle" or "high": financially comfortable/established; NEVER describe them as cash-strapped, struggling, or watching every rand.
   - If `price_sensitivity` is "low", show they prioritize quality, reliability, or time saved over finding the cheapest option (never say they watch every rand or avoid expenses).
   - If `price_sensitivity` is "high", show they scrutinize costs and strongly resist recurring overhead.
3. Adopter Type, Segment & Skepticism:
   - Match their `segment`, `adopter_type`, and `skepticism` naturally without parroting raw schema labels like "late majority", "early adopter", "laggard", or "upper-middle" in the text.
   - If `skepticism` is "low", describe them as open and trusting toward service providers (never call them skeptical or guarded).
4. Product Blindness & Anti-Stereotype Rules:
   - The persona does NOT know what product or offer they will be shown later.
   - Do NOT prime `values` or `frustrations` around shopping for the test category or complaining about problems that a new vendor pitch solves.
   - NEVER use overused regional clichés such as "load-shedding" or "a website built years ago by a nephew/relative/friend".
   - Make `values` and `frustrations` specific to their assigned role, household, and operational focus.
Reply with ONLY a JSON object, no other text."""

USER = """Create persona #{persona_id} living in {region}.

FIXED TRAITS (every detail must be consistent with these; do not echo raw trait label strings verbatim)
- Segment: {segment} - {segment_desc}
- Age: {age}
- Gender: {gender} ({pronoun_hint})
- Income level: {income} ({income_hint})
- Household: {household} (mention this exact household situation naturally)
- Location type: {location} ({location_hint})
- Attitude to new products: {adopter_type} ({adopter_hint})
- Price sensitivity: {price_sensitivity} ({price_hint})
- General skepticism toward brands and advertising: {skepticism} ({skepticism_hint})

DIVERSITY & ANTI-UNIFORMITY GUIDANCE
- Full name: "{name_seed}"
- Role & sector angle: {role_seed} in {sector_hint}
- Everyday frustration angle to explore (avoid load-shedding clichés): {frustration_hint}
- Current posture in {category}: {posture_hint}

JSON FORMAT
{{
  "name": "{name_seed}",
  "occupation": "specific {role_seed} role in {sector_hint} matching age {age}, {location}, and {income} income",
  "bio": "4-6 sentences about their daily work, {household}, {location} setting, financial footing ({income_hint}), and mindset ({adopter_hint}; {price_hint}; {skepticism_hint})",
  "current_solution": "neutral, concrete description of how they currently handle {category} today ({posture_hint}) without mentioning a relative/nephew",
  "values": ["3 distinct personal, family, or core operational priorities matching their role and traits"],
  "frustrations": ["2-3 specific day-to-day operational or household frustrations around {frustration_hint} (no load-shedding or product-primed complaints)"]
}}"""

NAME_SEEDS_BY_GENDER = {
    "woman": [
        "Annelize Botha", "Priya Naidoo", "Zanele Khumalo", "Bongiwe Dlamini", "Lesedi Mahlangu",
        "Nadia Abrahams", "Marlene Steyn", "Puleng Modise", "Ayanda Nkosi", "Claire Fourie",
        "Lindiwe Chauke", "Elmarie Kruger", "Yasmin Cassim", "Nomsa Radebe", "Rethabile Mokoena",
        "Sanet Pretorius", "Kavitha Pillay", "Palesa Sithole", "Chantal du Plessis", "Busisiwe Zulu",
        "Meghan Louw", "Fatima Hendricks", "Tsholofelo Modiba", "Liezl Coetzee", "Sibongile Nkosi",
    ],
    "man": [
        "Sipho Ndlovu", "Kagiso Molefe", "Dawid Pretorius", "Tariq Patel", "Hendrik Coetzee",
        "Tshepo Sithole", "Sibusiso Zulu", "Rohan Pillay", "Cobus Venter", "Mpho Baloyi",
        "Kobus du Plessis", "Devan Govender", "Tumi Mabaso", "Mandla Tshabalala", "Jaco Oosthuizen",
        "Craig Louw", "Vusi Mabena", "Pieter Steyn", "Imran Cassim", "Kabelo Chauke",
        "Willem Fourie", "Keshav Naidoo", "Bongani Khumalo", "Gert Kruger", "Lunga Radebe",
    ],
    "non-binary": [
        "Alex Modise", "Lesedi Nkosi", "Sam Abrahams", "Mpho Sithole", "Robin Fourie",
        "Tumi Mahlangu", "Jamie Patel", "Kagiso Baloyi", "Jordan Louw", "Ayo Dlamini",
    ],
}

SECTOR_SEEDS = [
    "accounting, tax & bookkeeping practice",
    "private allied-health or dental clinic",
    "catering, bakery & event hospitality",
    "accredited safety & medical training provider",
    "boutique retail or specialty food store",
    "commercial electrical & solar maintenance",
    "regional courier, fleet & logistics operation",
    "HR, payroll & compliance consultancy",
    "architectural drafting or quantity surveying",
    "industrial PPE & engineering supply distributor",
    "veterinary clinic or pet care services",
    "independent insurance brokerage or financial advisory",
    "automotive service centre & panel beating",
    "property letting & sectional-title management",
    "physiotherapy & wellness studio",
    "custom furniture & joinery workshop",
    "agricultural & borehole equipment supplier",
    "private tutoring & early-childhood education",
    "security systems & access-control installer",
    "import/export wholesale distribution",
]

ROLE_SEEDS = [
    "Owner & Managing Director",
    "Founder & Practice Principal",
    "Co-founder & Operations Director",
    "Managing Partner",
    "Sole Proprietor & Owner",
    "Owner & General Manager",
    "Founder & Fleet Director",
    "Senior Managing Partner",
    "Principal & Director",
    "Co-owner & Commercial Director",
    "Practice Owner & Director",
    "Principal & Founder",
    "Owner-Operator",
    "Managing Director",
    "Founder & Studio Director",
    "Co-owner & Master Partner",
    "Managing Director & Co-owner",
    "Founder & Centre Director",
    "Owner & Technical Director",
    "Founder & Wholesale Director",
]

FRUSTRATION_SEEDS = [
    "SARS tax deadline bottlenecks and incomplete client paperwork",
    "medical-aid claim rejections and slow scheme reimbursements",
    "perishable ingredient price spikes and last-minute event headcount changes",
    "SETA accreditation audits and certificate turnaround delays",
    "import customs delays and shelf-life management for specialty stock",
    "shortage of certified trade technicians and safety-file signoffs",
    "diesel price volatility, fleet tyre wear, and depot loading bottlenecks",
    "CCMA regulatory updates and client payroll cut-off crunches",
    "municipal building-plan approval delays and scope creep on site",
    "60-day B2B procurement payment cycles and warehouse stockouts",
    "weekend emergency call-outs and veterinary medicine supplier backorders",
    "FSCA compliance audits and policy underwriting delays",
    "imported OEM spare-part backorders and insurer assessor delays",
    "body-corporate trustee disputes and contractor quote follow-ups",
    "patient no-shows and balancing school-run hours with afternoon clinics",
    "kiln-dried timber lead times and workshop dust-extraction maintenance",
    "seasonal rainfall swings affecting farm equipment demand and courier routes",
    "parent fee arrears and recruiting reliable afternoon subject tutors",
    "false-alarm callouts and supplier warranty replacements on hardware",
    "cross-border waybill paperwork and container freight rate fluctuations",
]


def _trait_hints(traits: PersonaTraits, seg: Segment | None, idx: int) -> dict[str, str]:
    pronoun_map = {
        "woman": "use she/her pronouns",
        "man": "use he/him pronouns",
        "non-binary": "MUST use they/them/their pronouns only",
    }
    income_map = {
        "low": "tight personal/business budget, small-scale operation",
        "lower-middle": "modest budget, watches monthly cash flow carefully",
        "middle": "steady, comfortable middle-class income",
        "upper-middle": "financially comfortable and established, NOT cash-strapped",
        "high": "affluent, well-resourced, high earner, NOT on a tight budget",
    }
    adopter_map = {
        "innovator": "enjoys experimenting with new digital tools and workflows early",
        "early adopter": "proactively adopts modern software and tools ahead of peers",
        "early majority": "pragmatic; adopts proven tools once practical value is clear",
        "late majority": "prefers familiar, established routines and only switches when necessary",
        "laggard": "sticks to traditional manual or relationship-based methods and avoids new tech",
    }
    price_map = {
        "low": "prioritizes quality, reliability, and time saved rather than the lowest price (do NOT describe as watching every rand)",
        "medium": "weighs fair value against practical return on investment",
        "high": "scrutinizes every rand spent and avoids non-essential recurring overhead",
    }
    skepticism_map = {
        "low": "generally open, trusting, and receptive to professional service providers",
        "medium": "balanced and practical when evaluating external vendors",
        "high": "guarded and questions sales promises until proven through references",
    }
    location_map = {
        "city": "urban commercial or city district",
        "suburb": "suburban business node or residential suburb",
        "small town / rural": "smaller outlying town or peri-rural area (do NOT set in a dense city CBD)",
    }
    posture_by_adopter = {
        "innovator": "actively uses modern cloud software and digital tools and enjoys testing newer capabilities",
        "early adopter": "maintains a modern digital setup and upgrades tools periodically as the business grows",
        "early majority": "uses practical, mainstream tools suited to their sector and is open to proven improvements",
        "late majority": "relies on familiar, long-standing channels and habits, only changing when clearly necessary",
        "laggard": "operates primarily through direct personal relationships, phone calls, and manual records",
    }
    posture = posture_by_adopter.get(traits.adopter_type, posture_by_adopter["early majority"])
    seg_desc = f"{seg.name}: {seg.description}" if seg else traits.segment
    posture += f" (must align with segment '{seg_desc}')"

    return {
        "pronoun_hint": pronoun_map.get(traits.gender, "use appropriate pronouns"),
        "income_hint": income_map.get(traits.income, ""),
        "adopter_hint": adopter_map.get(traits.adopter_type, ""),
        "price_hint": price_map.get(traits.price_sensitivity, ""),
        "skepticism_hint": skepticism_map.get(traits.skepticism, ""),
        "location_hint": location_map.get(traits.location, ""),
        "sector_hint": SECTOR_SEEDS[(idx - 1) % len(SECTOR_SEEDS)],
        "role_seed": ROLE_SEEDS[(idx - 1) % len(ROLE_SEEDS)],
        "frustration_hint": FRUSTRATION_SEEDS[(idx - 1) % len(FRUSTRATION_SEEDS)],
        "posture_hint": posture,
    }


def build_category_lens(research: MarketResearch, seg: Segment | None, category: str = "") -> dict:
    """Construct the Layer-2 Category Lens for a persona from scraped MarketResearch and segment data."""
    dossier = research.buyer_dossier
    habits = (seg.buying_habits if seg and seg.buying_habits else []) or (dossier.buying_habits if dossier else [])
    anchors = (seg.price_anchors if seg and seg.price_anchors else []) or (
        dossier.price_anchors if dossier else ([research.typical_price_range] if research.typical_price_range else [])
    )
    frustrations = (
        (seg.likely_objections if seg and seg.likely_objections else [])
        + (dossier.real_frustrations[:3] if dossier else [])
    )
    alternatives = (
        (dossier.status_quo_alternatives if dossier and dossier.status_quo_alternatives else [])
        or [c.name for c in research.competitors[:4]]
    )
    cultural = dossier.cultural_signals if dossier else []
    urls = (dossier.scraped_urls if dossier and dossier.scraped_urls else []) or list(research.sources[:5])
    return {
        "category": category,
        "segment": seg.name if seg else "",
        "buying_habits": habits[:4],
        "price_anchors": anchors[:3],
        "status_quo_alternatives": alternatives[:4],
        "real_frustrations": frustrations[:4],
        "cultural_signals": cultural[:3],
        "evidence_urls": urls[:5],
    }


def build_trait_provenance(research: MarketResearch) -> dict[str, str]:
    """Tag persona trait groups as 'sourced', 'modelled', or 'assumed' based on scraped evidence."""
    has_sources = bool(research.sources or (research.buyer_dossier and research.buyer_dossier.scraped_urls))
    return {
        "demographics": "modelled",
        "psychographics": "modelled",
        "category_habits": "sourced" if has_sources else "assumed",
        "price_anchors": "sourced" if (has_sources and research.typical_price_range) else "assumed",
        "frustrations": "sourced" if has_sources else "assumed",
    }


def apply_category_lens(personas: list[Persona], research: MarketResearch, category: str = "") -> list[Persona]:
    """Attach a freshly scraped Layer-2 Category Lens to a reused Base Persona panel."""
    segs = {s.name: s for s in research.segments}
    fallback_seg = research.segments[0] if research.segments else None
    provenance = build_trait_provenance(research)
    updated: list[Persona] = []
    for p in personas:
        seg = segs.get(p.traits.segment, fallback_seg)
        lens = build_category_lens(research, seg, category=category)
        merged_prov = {**p.trait_provenance, **provenance}
        updated.append(p.model_copy(update={"category_lens": lens, "trait_provenance": merged_prov}))
    return updated


def validate_persona_item(profile: PersonaProfile, traits: PersonaTraits, brief: Brief) -> list[str]:
    """Tier-1 deterministic check for single-persona defects before Jev sees the batch."""
    issues: list[str] = []
    text = f"{profile.bio} {profile.current_solution}".lower()
    for term in (brief.product_name, brief.client):
        clean = (term or "").strip().lower()
        if len(clean) >= 4 and clean in text:
            issues.append(f"Do not mention client or product '{term}' in bio or current_solution (must remain product-blind)")
    words = set(text.replace(".", " ").replace(",", " ").replace(";", " ").split())
    if traits.gender == "man" and ({"she", "her", "hers"} & words):
        issues.append("Gender is 'man'; use he/him pronouns only, never she/her")
    elif traits.gender == "woman" and ({"he", "him", "his"} & words):
        issues.append("Gender is 'woman'; use she/her pronouns only, never he/him")
    elif traits.gender == "non-binary" and ({"he", "him", "his", "she", "her", "hers"} & words):
        issues.append("Gender is 'non-binary'; use they/them/their pronouns only")
    return issues


async def _one(llm: LLM, cfg: Config, brief: Brief, research: MarketResearch, segs: dict[str, Segment],
               pid: str, traits: PersonaTraits, model: str) -> Persona:
    role = cfg.role("personas")
    seg = segs.get(traits.segment)
    lens = build_category_lens(research, seg, category=brief.category)
    provenance = build_trait_provenance(research)
    idx = int("".join(ch for ch in pid if ch.isdigit()) or "0")
    gender_names = NAME_SEEDS_BY_GENDER.get(traits.gender, NAME_SEEDS_BY_GENDER["non-binary"])
    name_seed = gender_names[(idx - 1) % len(gender_names)]
    hints = _trait_hints(traits, seg, idx)
    prompt = USER.format(
        persona_id=pid,
        region=brief.region,
        segment=traits.segment,
        segment_desc=seg.description if seg else "",
        category=brief.category,
        name_seed=name_seed,
        **hints,
        **traits.model_dump(exclude={"segment"}),
    )
    mock_ctx = {"persona_id": pid, "traits": traits.model_dump(), "category_lens": lens}
    profile, _ = await llm.structured(
        task="persona", schema=PersonaProfile, model=model, system=SYSTEM, user=prompt,
        temperature=role.temperature, max_tokens=role.max_tokens,
        mock_ctx=mock_ctx,
    )
    if not llm.is_mock and cfg.guardrails.self_heal_enabled:
        issues = validate_persona_item(profile, traits, brief)
        if issues:
            repair_prompt = prompt + "\n\nSELF-HEAL CORRECTION REQUIRED:\n- " + "\n- ".join(issues)
            profile, _ = await llm.structured(
                task="persona", schema=PersonaProfile, model=model, system=SYSTEM, user=repair_prompt,
                temperature=role.temperature, max_tokens=role.max_tokens,
                mock_ctx=mock_ctx,
            )
    return Persona(
        id=pid,
        model=model,
        traits=traits,
        category_lens=lens,
        trait_provenance=provenance,
        **profile.model_dump(),
    )


async def run(llm: LLM, cfg: Config, brief: Brief, research: MarketResearch,
              sampled: list[tuple[PersonaTraits, str]], on_done: Callable[[], None]) -> tuple[list[Persona], list[str]]:
    segs = {s.name: s for s in research.segments}

    async def task(i: int, traits: PersonaTraits, model: str):
        try:
            return await _one(llm, cfg, brief, research, segs, f"P{i + 1:03d}", traits, model)
        except LLMError as exc:
            return exc
        finally:
            on_done()

    results = await gather_cancel_on_error(*(task(i, t, m) for i, (t, m) in enumerate(sampled)))
    personas = [r for r in results if isinstance(r, Persona)]
    errors = [str(r) for r in results if not isinstance(r, Persona)]
    return personas, errors

