"""Samples persona traits in code so the panel has a realistic, controlled spread.

LLMs left to invent personas drift toward young, affluent, enthusiastic people.
Here demographics and attitudes are drawn from explicit distributions; the LLM
only writes the human details on top.
"""

from __future__ import annotations

import random

from .schemas import PersonaTraits, Segment

# Rogers' diffusion of innovations curve.
ADOPTER_DIST = [
    ("innovator", 0.025),
    ("early adopter", 0.135),
    ("early majority", 0.34),
    ("late majority", 0.34),
    ("laggard", 0.16),
]

PRICE_SENSITIVITY_BY_INCOME = {  # (low, medium, high)
    "low": (0.10, 0.30, 0.60),
    "lower-middle": (0.15, 0.40, 0.45),
    "middle": (0.25, 0.45, 0.30),
    "upper-middle": (0.40, 0.45, 0.15),
    "high": (0.60, 0.32, 0.08),
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


def sample_traits(
    segments: list[Segment], n: int, rng: random.Random, models: list[str]
) -> list[tuple[PersonaTraits, str]]:
    """Return n (traits, assigned_model) pairs."""
    out: list[PersonaTraits] = []
    for seg, count in zip(segments, allocate(n, [s.share for s in segments])):
        for _ in range(count):
            age = rng.randint(seg.age_min, seg.age_max)
            income = rng.choice(seg.income_levels)
            adopter = _pick(ADOPTER_DIST, rng)
            out.append(
                PersonaTraits(
                    segment=seg.name,
                    age=age,
                    gender=_pick([("woman", 0.49), ("man", 0.49), ("non-binary", 0.02)], rng),
                    income=income,
                    household=_household(age, rng),
                    location=_pick([("city", 0.45), ("suburb", 0.35), ("small town / rural", 0.20)], rng),
                    adopter_type=adopter,
                    price_sensitivity=rng.choices(LEVELS, weights=PRICE_SENSITIVITY_BY_INCOME[income])[0],
                    skepticism=rng.choices(LEVELS, weights=SKEPTICISM_BY_ADOPTER[adopter])[0],
                )
            )
    # Shuffle, then assign models round-robin so every model covers every segment.
    rng.shuffle(out)
    return [(t, models[i % len(models)]) for i, t in enumerate(out)]
