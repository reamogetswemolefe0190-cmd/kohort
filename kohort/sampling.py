"""Samples persona traits in code so the panel has a realistic, controlled spread.

LLMs left to invent personas drift toward young, affluent, enthusiastic people.
Here demographics and attitudes are drawn from explicit distributions; the LLM
only writes the human details on top.
"""

from __future__ import annotations

import hashlib
import random

from .schemas import HookVariant, PersonaTraits, Segment

# Rogers' diffusion of innovations curve.
ADOPTER_DIST = [
    ("innovator", 0.025),
    ("early adopter", 0.135),
    ("early majority", 0.34),
    ("late majority", 0.34),
    ("laggard", 0.16),
]

PRICE_SENSITIVITY_BY_INCOME = {  # (low, medium, high)
    "low": (0.00, 0.20, 0.80),
    "lower-middle": (0.00, 0.35, 0.65),
    "middle": (0.20, 0.55, 0.25),
    "upper-middle": (0.55, 0.45, 0.00),
    "high": (0.75, 0.25, 0.00),
}

SKEPTICISM_BY_ADOPTER = {  # (low, medium, high)
    "innovator": (0.60, 0.30, 0.10),
    "early adopter": (0.45, 0.40, 0.15),
    "early majority": (0.25, 0.50, 0.25),
    "late majority": (0.15, 0.45, 0.40),
    "laggard": (0.05, 0.35, 0.60),
}

LEVELS = ("low", "medium", "high")


def _household(age: int, rng: random.Random) -> str:
    if age < 30:
        options = [("single", 0.45), ("living with partner", 0.30), ("shared flat", 0.18), ("young family", 0.07)]
    elif age < 50:
        options = [("single", 0.18), ("couple, no kids", 0.22), ("family with young children", 0.45),
                   ("single parent", 0.15)]
    elif age < 65:
        options = [("couple, kids left home", 0.45), ("family with teenagers", 0.25), ("single", 0.25),
                   ("caring for elderly parent", 0.05)]
    else:
        options = [("retired couple", 0.50), ("living alone", 0.45), ("multigenerational household", 0.05)]
    return _pick(options, rng)


def _pick(options: list[tuple[str, float]], rng: random.Random) -> str:
    labels, weights = zip(*options)
    return rng.choices(labels, weights=weights, k=1)[0]


def allocate(n: int, shares: list[float]) -> list[int]:
    """Split n personas across segments by share (largest-remainder method)."""
    raw = [n * s for s in shares]
    counts = [int(x) for x in raw]
    remainder = n - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    # Every segment gets at least one voice if we have enough personas.
    if n >= len(shares):
        for i, c in enumerate(counts):
            if c == 0:
                donor = max(range(len(counts)), key=lambda j: counts[j])
                counts[donor] -= 1
                counts[i] = 1
    return counts


def _segment_aligned_traits(seg: Segment, income: str, rng: random.Random) -> tuple[str, str, str]:
    """Ensure sampled adopter_type, price_sensitivity, and skepticism do not contradict the segment title or income."""
    seg_name = seg.name.lower()
    if any(k in seg_name for k in ("growth", "innovat", "early", "digital", "custom", "modern", "tech", "complex", "established", "ambitious", "pain", "support", "active", "autom", "workflow", "improver", "trust", "process")):
        adopter = _pick([("innovator", 0.10), ("early adopter", 0.35), ("early majority", 0.55)], rng)
        if any(k in seg_name for k in ("skeptic", "sceptic", "cautious", "careful")):
            skepticism = _pick([("medium", 0.55), ("high", 0.45)], rng)
        else:
            skepticism = rng.choices(LEVELS, weights=SKEPTICISM_BY_ADOPTER[adopter])[0]
    elif any(k in seg_name for k in ("brochure", "diy", "template", "basic", "simple")):
        adopter = _pick([("early majority", 0.55), ("late majority", 0.45)], rng)
        skepticism = rng.choices(LEVELS, weights=SKEPTICISM_BY_ADOPTER[adopter])[0]
    elif any(k in seg_name for k in ("lukewarm", "indifferent", "do-nothing", "wait-and-see", "reluctant", "resistant", "laggard", "offline", "traditional", "skeptic", "sceptic", "satisfied", "status-quo", "fallback")):
        adopter = _pick([("late majority", 0.60), ("laggard", 0.40)], rng)
        skepticism = _pick([("medium", 0.40), ("high", 0.60)], rng)
    else:
        adopter = _pick(ADOPTER_DIST, rng)
        skepticism = rng.choices(LEVELS, weights=SKEPTICISM_BY_ADOPTER[adopter])[0]

    if any(k in seg_name for k in ("cost", "budget", "price", "cheap", "diy", "micro", "informal", "frugal", "brochure")):
        price_sens = "medium" if income in ("upper-middle", "high") else _pick([("medium", 0.25), ("high", 0.75)], rng)
    elif any(k in seg_name for k in ("premium", "higher-end", "high-end", "enterprise", "affluent")):
        price_sens = "medium" if income in ("low", "lower-middle") else _pick([("low", 0.60), ("medium", 0.40)], rng)
    else:
        price_sens = rng.choices(LEVELS, weights=PRICE_SENSITIVITY_BY_INCOME[income])[0]

    return adopter, price_sens, skepticism


def sample_traits(
    segments: list[Segment], n: int, rng: random.Random, models: list[str]
) -> list[tuple[PersonaTraits, str]]:
    """Return n (traits, assigned_model) pairs."""
    out: list[PersonaTraits] = []
    for seg, count in zip(segments, allocate(n, [s.share for s in segments])):
        for _ in range(count):
            age = rng.randint(seg.age_min, seg.age_max)
            income = rng.choice(seg.income_levels)
            adopter, price_sens, skepticism = _segment_aligned_traits(seg, income, rng)
            out.append(
                PersonaTraits(
                    segment=seg.name,
                    age=age,
                    gender=_pick([("woman", 0.49), ("man", 0.49), ("non-binary", 0.02)], rng),
                    income=income,
                    household=_household(age, rng),
                    location=_pick([("city", 0.45), ("suburb", 0.35), ("small town / rural", 0.20)], rng),
                    adopter_type=adopter,
                    price_sensitivity=price_sens,
                    skepticism=skepticism,
                )
            )
    # Shuffle, then assign models round-robin so every model covers every segment.
    rng.shuffle(out)
    return [(t, models[i % len(models)]) for i, t in enumerate(out)]


def assign_variant(persona_id: str, variants: list[HookVariant], seed: int = 42) -> HookVariant | None:
    """Deterministically assign a persona to exactly one variant in between-subjects isolation."""
    if not variants:
        return None
    h = hashlib.sha256(f"{seed}:{persona_id}".encode()).hexdigest()
    idx = int(h, 16) % len(variants)
    return variants[idx]


