"""Deterministic Data Binding: Safe Jinja2 template rendering bound to analysis.json.

Ensures every number rendered in client HTML/Markdown deliverables is deterministically
drawn from analysis.json as the single source of truth, eliminating hardcoded editorial leaks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jinja2

from .resilience import AuditFailure, audit_tier2_provenance


def _filter_pct(value: Any, default: str = "-") -> str:
    if value is None or value == "":
        return default
    try:
        return f"{float(value):.0%}"
    except (ValueError, TypeError):
        return str(value)


def _filter_pct1(value: Any, default: str = "-") -> str:
    if value is None or value == "":
        return default
    try:
        return f"{float(value):.1%}"
    except (ValueError, TypeError):
        return str(value)


def _filter_money(value: Any, currency: str = "EUR", default: str = "-") -> str:
    if value is None or value == "":
        return default
    try:
        fval = float(value)
        if currency.upper() in ("ZAR", "R"):
            return f"R{fval:,.2f}"
        if currency == "USD" or currency == "$":
            return f"${fval:,.2f}"
        if currency == "EUR" or currency == "€":
            return f"€{fval:,.2f}"
        return f"{fval:,.2f} {currency}"
    except (ValueError, TypeError):
        return str(value)


def _filter_ci(ci_dict: Any, default: str = "-") -> str:
    if not isinstance(ci_dict, dict):
        return default
    lo = ci_dict.get("lower")
    hi = ci_dict.get("upper")
    if lo is None or hi is None:
        return default
    return f"[{float(lo):.1%}, {float(hi):.1%}]"


def _filter_bar(value: Any, width: int = 15) -> str:
    try:
        f = max(0.0, min(1.0, float(value)))
        filled = round(f * width)
        return "█" * filled + "░" * (width - filled)
    except (ValueError, TypeError):
        return ""


def get_jinja_environment() -> jinja2.Environment:
    """Creates a hardened Jinja2 environment with strict undefined checking and business filters."""
    env = jinja2.Environment(
        autoescape=jinja2.select_autoescape(["html", "xml"]),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = _filter_pct
    env.filters["pct1"] = _filter_pct1
    env.filters["money"] = _filter_money
    env.filters["ci"] = _filter_ci
    env.filters["bar"] = _filter_bar
    return env


def render_bound_template(
    template_str: str,
    analysis_data: dict,
    brief: Any = None,
    extra_context: dict | None = None,
    allowed_editorial: list[dict] | None = None,
    strict_provenance: bool = True,
) -> str:
    """Renders a Jinja2 template bound strictly to analysis.json and verifies provenance."""
    env = get_jinja_environment()
    tmpl = env.from_string(template_str)

    brief_dict = brief.model_dump() if hasattr(brief, "model_dump") else brief if isinstance(brief, dict) else {}
    context = {
        "analysis": analysis_data,
        "quant": analysis_data,
        "brief": brief_dict,
        "metrics": analysis_data.get("metrics", {}),
        "pricing": analysis_data.get("pricing", {}),
        "by": analysis_data.get("by", {}),
        "shootout": analysis_data.get("shootout"),
        **(extra_context or {}),
    }

    rendered = tmpl.render(context)

    if strict_provenance:
        audit = audit_tier2_provenance(
            deliverable_text=rendered,
            analysis_data=analysis_data,
            brief=brief,
            allowed_editorial=allowed_editorial,
        )
        if not audit.passed:
            raise AuditFailure(
                f"Deterministic binding failed Tier-2 provenance audit: {'; '.join(audit.violations)}"
            )

    return rendered
