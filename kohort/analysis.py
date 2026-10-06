"""Deterministic numbers: scores, segment cuts, pricing curves and quality checks.

All figures in the final report come from here (never from an LLM), so
numbers can't be hallucinated.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from .schemas import Brief, InterviewResponse, Persona

METRICS = ["purchase_intent", "appeal", "uniqueness", "believability", "value_for_money"]


def _share(values: list, predicate) -> float:
    return sum(1 for v in values if predicate(v)) / len(values) if values else 0.0


def _metric_block(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "top2": round(_share(values, lambda v: v >= 4), 3),
        "bottom2": round(_share(values, lambda v: v <= 2), 3),
        "dist": {str(k): round(_share(values, lambda v, k=k: v == k), 3) for k in range(1, 6)},
    }


def trial_estimate(intents: list[int], weights: dict[str, float]) -> float:
    """Calibrated trial: discount stated intent toward realistic behaviour."""
    return round(
        weights.get("definitely", 0.75) * _share(intents, lambda v: v == 5)
        + weights.get("probably", 0.25) * _share(intents, lambda v: v == 4),
        3,
    )


def _group_block(rows: list[InterviewResponse], weights: dict[str, float]) -> dict:
    intents = [r.purchase_intent for r in rows]
    return {
        "n": len(rows),
        "intent_mean": round(statistics.mean(intents), 2) if intents else None,
        "intent_top2": round(_share(intents, lambda v: v >= 4), 3),
        "appeal_mean": round(statistics.mean(r.appeal for r in rows), 2) if rows else None,
        "trial_estimate": trial_estimate(intents, weights),
    }


# --------------------------------------------------------------------------- pricing


def _crossing(grid: list[float], f: list[float], g: list[float]) -> float | None:
    """First price where curve f crosses curve g (linear interpolation)."""
    d = [a - b for a, b in zip(f, g)]
    for i in range(len(d) - 1):
        if d[i] == 0:
            return grid[i]
        if d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1])
            return grid[i] + t * (grid[i + 1] - grid[i])
    return None


def van_westendorp(rows: list[InterviewResponse]) -> dict | None:
    pts = [
        (r.price_too_cheap, r.price_bargain, r.price_getting_expensive, r.price_too_expensive)
        for r in rows
        if r.price_too_cheap is not None
    ]
    if len(pts) < 5:
        return None
    lo = min(p[0] for p in pts)
    hi = max(p[3] for p in pts)
    if hi <= lo:
        return None
    steps = 400
    grid = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
    n = len(pts)
    too_cheap = [sum(1 for p in pts if p[0] >= x) / n for x in grid]
    cheap = [sum(1 for p in pts if p[1] >= x) / n for x in grid]
    expensive = [sum(1 for p in pts if p[2] <= x) / n for x in grid]
    too_exp = [sum(1 for p in pts if p[3] <= x) / n for x in grid]
    not_cheap = [1 - c for c in cheap]
    not_exp = [1 - e for e in expensive]

    def r(x):
        return round(x, 2) if x is not None else None

    return {
        "n": n,
        "point_of_marginal_cheapness": r(_crossing(grid, too_cheap, not_cheap)),
        "optimal_price_point": r(_crossing(grid, too_cheap, too_exp)),
        "indifference_price_point": r(_crossing(grid, cheap, expensive)),
        "point_of_marginal_expensiveness": r(_crossing(grid, too_exp, not_exp)),
        "median_too_cheap": r(statistics.median(p[0] for p in pts)),
        "median_bargain": r(statistics.median(p[1] for p in pts)),
        "median_getting_expensive": r(statistics.median(p[2] for p in pts)),
        "median_too_expensive": r(statistics.median(p[3] for p in pts)),
    }


def price_acceptance(rows: list[InterviewResponse], price: float | None) -> dict | None:
    """How the stated price lands against each persona's own price thresholds."""
    pts = [r for r in rows if r.price_too_cheap is not None]
    if price is None or not pts:
        return None
    n = len(pts)
    return {
        "price": price,
        "too_cheap": round(sum(1 for r in pts if price <= r.price_too_cheap) / n, 3),
        "bargain_or_fair": round(sum(1 for r in pts if r.price_too_cheap < price <= r.price_getting_expensive) / n, 3),
        "expensive_but_considered": round(
            sum(1 for r in pts if r.price_getting_expensive < price < r.price_too_expensive) / n, 3),
        "too_expensive": round(sum(1 for r in pts if price >= r.price_too_expensive) / n, 3),
    }


# --------------------------------------------------------------------------- diagnostics


def diagnostics(rows: list[InterviewResponse], by_model: dict) -> dict:
    intents = [r.purchase_intent for r in rows]
    flags: list[str] = []
    std = statistics.pstdev(intents) if len(intents) > 1 else 0.0
    modal_share = Counter(intents).most_common(1)[0][1] / len(intents) if intents else 0.0
    quotes = [r.quote.strip().lower() for r in rows if r.quote.strip()]
    dup_ratio = 1 - len(set(quotes)) / len(quotes) if quotes else 0.0
    top2 = _share(intents, lambda v: v >= 4)
    means = [m["intent_mean"] for m in by_model.values() if m["n"] >= 5 and m["intent_mean"] is not None]
    spread = max(means) - min(means) if len(means) > 1 else 0.0
    repaired = _share(rows, lambda r: r.price_points_repaired)

    if std < 0.7:
        flags.append(f"Low variety in purchase intent (std dev {std:.2f}); personas may be too uniform.")
    if modal_share > 0.6:
        flags.append(f"{modal_share:.0%} of personas gave the same intent score; possible herding.")
    if top2 > 0.6:
        flags.append(f"Top-2-box intent of {top2:.0%} is unusually high; check for AI positivity bias.")
    if dup_ratio > 0.1:
        flags.append(f"{dup_ratio:.0%} of quotes are duplicates; persona voices are repetitive.")
    if spread > 1.0:
        flags.append(f"Mean intent differs by {spread:.1f} points between models; results are model-sensitive.")
    if repaired > 0.15:
        flags.append(f"{repaired:.0%} of personas gave price points out of order (auto-sorted).")

    return {
        "intent_std": round(std, 2),
        "modal_share": round(modal_share, 3),
        "duplicate_quote_ratio": round(dup_ratio, 3),
        "model_intent_spread": round(spread, 2),
        "price_points_repaired_share": round(repaired, 3),
        "flags": flags,
    }


# --------------------------------------------------------------------------- main


def summarize(brief: Brief, personas: list[Persona], rows: list[InterviewResponse],
              weights: dict[str, float]) -> dict:
    pmap = {p.id: p for p in personas}
    groups: dict[str, dict[str, list[InterviewResponse]]] = {
        "segment": defaultdict(list), "adopter_type": defaultdict(list), "model": defaultdict(list),
        "income": defaultdict(list), "age_band": defaultdict(list),
    }
    for r in rows:
        t = pmap[r.persona_id].traits
        groups["segment"][t.segment].append(r)
        groups["adopter_type"][t.adopter_type].append(r)
        groups["model"][r.model].append(r)
        groups["income"][t.income].append(r)
        band = "18-29" if t.age < 30 else "30-44" if t.age < 45 else "45-59" if t.age < 60 else "60+"
        groups["age_band"][band].append(r)

    by = {k: {name: _group_block(v, weights) for name, v in g.items()} for k, g in groups.items()}
    metrics = {m: _metric_block([getattr(r, m) for r in rows if getattr(r, m) is not None]) for m in METRICS}

    return {
        "n": len(rows),
        "metrics": metrics,
        "trial_estimate": trial_estimate([r.purchase_intent for r in rows], weights),
        "intent_weights": weights,
        "by": by,
        "pricing": {
            "stated_price": brief.price,
            "currency": brief.currency,
            "price_unit": brief.price_unit,
            "van_westendorp": van_westendorp(rows),
            "acceptance": price_acceptance(rows, brief.price),
        },
        "diagnostics": diagnostics(rows, by["model"]),
    }


def theme_stats(coding_items: list[tuple[str, str]], assignments: dict[str, str], themes: list[dict],
                personas: list[Persona], n_total: int) -> list[dict]:
    """Count how many personas mention each theme, overall and per segment.

    coding_items: list of (persona_id, text) in the order they were sent for coding.
    """
    pmap = {p.id: p for p in personas}
    per_theme: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[str]] = defaultdict(list)
    for idx, (pid, text) in enumerate(coding_items):
        tid = assignments.get(str(idx))
        if not tid:
            continue
        per_theme[tid].add(pid)
        if len(examples[tid]) < 3 and text not in examples[tid]:
            examples[tid].append(text)

    out = []
    for th in themes:
        pids = per_theme.get(th["id"], set())
        seg_counts = Counter(pmap[p].traits.segment for p in pids if p in pmap)
        out.append({
            **th,
            "personas": len(pids),
            "share": round(len(pids) / n_total, 3) if n_total else 0.0,
            "by_segment": dict(seg_counts.most_common()),
            "examples": examples.get(th["id"], []),
        })
    return sorted(out, key=lambda t: t["personas"], reverse=True)
