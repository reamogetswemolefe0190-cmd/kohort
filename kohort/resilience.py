"""Resilience Suite: 3-Tier Audit and Deterministic Data Binding.

Tier 1: Pure-Python deterministic assertions (monotonicity, sample conservation, sums).
Tier 2: Strict provenance logging (every advisory number traces to a raw simulation or research source).
Jev Boundary: Jev is strictly a passive auditor returning risk scores and routing flags.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


class AuditFailure(RuntimeError):
    """Raised when deterministic invariants or provenance checks fail."""


@dataclass
class Tier1AuditResult:
    passed: bool
    violations: list[str] = field(default_factory=list)


@dataclass
class Tier2AuditResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    untraced_figures: list[str] = field(default_factory=list)


def audit_tier1_invariants(analysis_data: dict, rows: list[Any] | None = None) -> Tier1AuditResult:
    """Enforces mathematical invariants on analysis.json output.

    Must pass without LLM intervention.
    """
    violations: list[str] = []
    n = analysis_data.get("n", 0)

    if n <= 0:
        violations.append("Total sample size n must be positive.")

    # 1. Row count reconciliation
    if rows is not None and len(rows) != n:
        violations.append(f"Row count mismatch: received {len(rows)} rows, analysis reports n={n}.")

    # 2. Sample conservation across segment cuts
    by = analysis_data.get("by", {})
    for group_name, group_data in by.items():
        if isinstance(group_data, dict):
            group_sum = sum(v.get("n", 0) for v in group_data.values() if isinstance(v, dict))
            if group_sum != n:
                violations.append(
                    f"Sample conservation failed in group '{group_name}': sum of subgroups ({group_sum}) != total n ({n})."
                )

    # 3. Distribution sums (Purchase intent distribution must sum to 1.0 +- 0.01)
    metrics = analysis_data.get("metrics", {})
    pi_dist = metrics.get("purchase_intent", {}).get("dist", {})
    if pi_dist:
        dist_sum = sum(pi_dist.values())
        if not (0.99 <= dist_sum <= 1.01):
            violations.append(f"Purchase intent distribution sum {dist_sum:.4f} is not within 1.0 +- 0.01.")

    # 4. Price acceptance distribution sums to 1.0 +- 0.01
    pricing = analysis_data.get("pricing", {})
    acc = pricing.get("acceptance")
    if acc:
        brackets = [
            acc.get("too_cheap", 0.0),
            acc.get("bargain_or_fair", 0.0),
            acc.get("expensive_but_considered", 0.0),
            acc.get("too_expensive", 0.0),
        ]
        acc_sum = sum(brackets)
        if not (0.99 <= acc_sum <= 1.01):
            violations.append(f"Price acceptance brackets sum {acc_sum:.4f} is not within 1.0 +- 0.01.")

    # 5. Van Westendorp monotonicity and price bounds
    vw = pricing.get("van_westendorp")
    if vw:
        med_cheap = vw.get("median_too_cheap")
        med_bargain = vw.get("median_bargain")
        med_exp = vw.get("median_getting_expensive")
        med_too_exp = vw.get("median_too_expensive")
        if all(x is not None for x in (med_cheap, med_bargain, med_exp, med_too_exp)):
            if not (med_cheap <= med_bargain <= med_exp <= med_too_exp):
                violations.append(
                    f"Van Westendorp median order violated: cheap={med_cheap}, bargain={med_bargain}, "
                    f"exp={med_exp}, too_exp={med_too_exp}."
                )

        opp = vw.get("optimal_price_point")
        pmc = vw.get("point_of_marginal_cheapness")
        pme = vw.get("point_of_marginal_expensiveness")
        if pmc is not None and pme is not None and pmc > pme:
            violations.append(f"Point of Marginal Cheapness ({pmc}) exceeds Point of Marginal Expensiveness ({pme}).")

    # 6. Hook Shootout mathematical bounds
    shootout = analysis_data.get("shootout")
    if shootout and shootout.get("variants"):
        shootout_n = sum(v.get("n", 0) for v in shootout["variants"])
        if shootout_n != n:
            violations.append(f"Shootout variants sum n={shootout_n} does not match total panel n={n}.")

        for v in shootout["variants"]:
            v_n = v.get("n", 0)
            conv = v.get("conversion_count", 0)
            top2 = v.get("intent_top2", 0.0)
            if v_n > 0:
                expected_top2 = round(conv / v_n, 3)
                if abs(top2 - expected_top2) > 0.005:
                    violations.append(
                        f"Variant '{v.get('variant_id')}' conversion rate {top2} does not match {conv}/{v_n} ({expected_top2})."
                    )

            ci = v.get("wilson_ci", {})
            lo, hi = ci.get("lower", 0.0), ci.get("upper", 1.0)
            if not (0.0 <= lo <= hi <= 1.0):
                violations.append(f"Variant '{v.get('variant_id')}' Wilson CI [{lo}, {hi}] is outside [0.0, 1.0].")

    return Tier1AuditResult(passed=len(violations) == 0, violations=violations)


def extract_numbers_from_text(text: str) -> set[float]:
    """Extracts scalar numbers, percentages, and currencies from deliverables text/HTML."""
    found: set[float] = set()
    # 1. Percentages: 45%, 45.2%
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
        val = float(m.group(1))
        found.add(round(val, 2))
        found.add(round(val / 100.0, 4))
        found.add(round(val / 100.0, 3))
        found.add(round(val / 100.0, 2))

    # 2. Currency: R3,021, $45.00, €1,200, 3,021 ZAR
    for m in re.finditer(r"(?:[$€£]|\bR)\s*(\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)", text):
        s = m.group(1).replace(",", "").replace(" ", "")
        try:
            val = float(s)
            found.add(round(val, 2))
            found.add(round(val, 0))
        except ValueError:
            pass

    # 3. Plain floating point numbers and integers
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", text):
        s = m.group(1)
        try:
            val = float(s)
            found.add(round(val, 2))
            found.add(round(val, 1))
            found.add(round(val, 0))
        except ValueError:
            pass

    return found


def harvest_grounded_numbers(
    analysis_data: dict, brief: Any = None, dossier: Any = None
) -> set[float]:
    """Recursively collects all grounded numbers from analysis.json, brief, and market research dossier."""
    numbers: set[float] = set()

    def _walk(obj):
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            val = float(obj)
            numbers.add(round(val, 4))
            numbers.add(round(val, 3))
            numbers.add(round(val, 2))
            numbers.add(round(val, 1))
            numbers.add(round(val, 0))
            numbers.add(float(math.floor(val)))
            numbers.add(float(math.ceil(val)))
            if 0.0 <= val <= 1.0:
                # Store also as percentage equivalent: e.g. 0.45 -> 45
                pct = val * 100.0
                numbers.add(round(pct, 2))
                numbers.add(round(pct, 1))
                numbers.add(round(pct, 0))
                numbers.add(float(math.floor(pct)))
                numbers.add(float(math.ceil(pct)))
        elif isinstance(obj, str):
            # Harvest numbers mentioned in research notes or dossier text
            for num in extract_numbers_from_text(obj):
                numbers.add(num)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                _walk(v)

    _walk(analysis_data)

    if brief:
        b_dict = brief.model_dump() if hasattr(brief, "model_dump") else brief if isinstance(brief, dict) else {}
        if b_dict.get("price") is not None:
            p = float(b_dict["price"])
            numbers.add(round(p, 2))
            numbers.add(round(p, 0))

    if dossier:
        _walk(dossier)

    # Add standard scale numbers (1-5 ratings, 0-100%, 0-20 sample indices)
    for i in range(0, 101):
        numbers.add(float(i))

    return numbers


def audit_tier2_provenance(
    deliverable_text: str,
    analysis_data: dict,
    brief: Any = None,
    dossier: Any = None,
    allowed_editorial: list[dict] | None = None,
) -> Tier2AuditResult:
    """Verifies that every figure in a deliverable traces to analysis.json, research, or an explicit editorial registry."""
    grounded = harvest_grounded_numbers(analysis_data, brief, dossier)
    editorial_set: set[float] = set()

    if allowed_editorial:
        for item in allowed_editorial:
            val = item.get("value")
            if val is not None:
                try:
                    fval = float(val)
                    editorial_set.add(round(fval, 2))
                    editorial_set.add(round(fval, 0))
                except (ValueError, TypeError):
                    pass

    total_allowed = grounded | editorial_set

    # Regex targeting numbers in deliverable text:
    untraced: list[str] = []
    suspicious_patterns = [
        r"(?:[$€£]|\bR)\s*(\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?)",  # Currency: R4,250
        r"(\d+(?:\.\d+)?)\s*%",  # Percentages: 90%
        r"(?:Mean|Score|Average|Intent|Trial|N\s*=|N\s*:)\s*(\d+(?:\.\d+)?)",  # Metrics
    ]

    for pat in suspicious_patterns:
        for m in re.finditer(pat, deliverable_text, re.IGNORECASE):
            raw_match = m.group(0)
            num_str = m.group(1).replace(",", "").replace(" ", "")
            try:
                val = float(num_str)
                # Check if val or val/100 or val*100 is in total_allowed
                in_allowed = (
                    round(val, 2) in total_allowed
                    or round(val, 0) in total_allowed
                    or round(val / 100.0, 3) in total_allowed
                    or round(val / 100.0, 2) in total_allowed
                    or round(val * 100.0, 1) in total_allowed
                )
                if not in_allowed:
                    untraced.append(f"{raw_match} (value: {val})")
            except ValueError:
                pass

    unique_untraced = sorted(set(untraced))
    passed = len(unique_untraced) == 0
    violations = [f"Untraced editorial number detected: {fig}" for fig in unique_untraced]

    return Tier2AuditResult(passed=passed, violations=violations, untraced_figures=unique_untraced)


class JevSupervisorBoundary:
    """Enforces Jev's strict boundary as a passive, read-only auditor."""

    @staticmethod
    def validate_jev_response(raw_jev_response: dict) -> dict:
        """Validates that Jev output strictly conforms to the read-only assessment schema."""
        if not isinstance(raw_jev_response, dict):
            raise AuditFailure("Jev response must be a JSON dictionary.")

        # Ensure Jev does not attempt to return mutated data structures
        forbidden_keys = {"analysis", "responses", "pricing", "metrics", "quant", "raw_model_output"}
        found_forbidden = forbidden_keys.intersection(raw_jev_response.keys())
        if found_forbidden:
            raise AuditFailure(f"Jev boundary violation: attempted to mutate pipeline data keys {found_forbidden}.")

        return raw_jev_response
