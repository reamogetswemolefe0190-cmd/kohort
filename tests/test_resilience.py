"""Tests for the Resilience Suite (Tier-1 invariants, Tier-2 provenance, and Jinja2 data binding)."""

from __future__ import annotations

import unittest

from kohort.binding import render_bound_template
from kohort.resilience import (
    AuditFailure,
    JevSupervisorBoundary,
    audit_tier1_invariants,
    audit_tier2_provenance,
    harvest_grounded_numbers,
)
from kohort.schemas import Brief, HookVariant, InterviewResponse, Persona, PersonaTraits


class ResilienceTests(unittest.TestCase):
    def setUp(self):
        self.brief = Brief(
            client="Nova Site",
            product_name="Websites and Web Apps",
            category="Web Development",
            one_liner="Production-grade websites for established businesses.",
            description="High performance web design and custom apps.",
            price=3000.0,
            currency="ZAR",
            target_market="Business owners",
            region="South Africa",
            variants=[
                HookVariant(id="v1", name="Speed", hook_text="Ready in 7 days"),
                HookVariant(id="v2", name="Conversion", hook_text="Built to sell"),
            ],
        )

        self.valid_analysis = {
            "n": 20,
            "metrics": {
                "purchase_intent": {
                    "n": 20,
                    "mean": 3.45,
                    "top2": 0.55,
                    "bottom2": 0.20,
                    "dist": {"1": 0.10, "2": 0.10, "3": 0.25, "4": 0.35, "5": 0.20},
                },
                "appeal": {"n": 20, "mean": 3.80, "top2": 0.65, "bottom2": 0.10},
            },
            "trial_estimate": 0.238,
            "intent_weights": {"definitely": 0.75, "probably": 0.25},
            "by": {
                "segment": {
                    "Growth Seekers": {"n": 12, "intent_mean": 3.8, "intent_top2": 0.67, "trial_estimate": 0.3},
                    "Cautious Traditionalists": {"n": 8, "intent_mean": 2.9, "intent_top2": 0.38, "trial_estimate": 0.15},
                },
                "model": {
                    "gpt-4.1-mini": {"n": 20, "intent_mean": 3.45, "intent_top2": 0.55, "trial_estimate": 0.238},
                },
            },
            "pricing": {
                "stated_price": 3000.0,
                "currency": "ZAR",
                "acceptance": {
                    "price": 3000.0,
                    "too_cheap": 0.05,
                    "bargain_or_fair": 0.50,
                    "expensive_but_considered": 0.35,
                    "too_expensive": 0.10,
                },
                "van_westendorp": {
                    "n": 20,
                    "point_of_marginal_cheapness": 1500.0,
                    "optimal_price_point": 3021.0,
                    "indifference_price_point": 3200.0,
                    "point_of_marginal_expensiveness": 5200.0,
                    "median_too_cheap": 1200.0,
                    "median_bargain": 2200.0,
                    "median_getting_expensive": 3800.0,
                    "median_too_expensive": 5500.0,
                },
            },
            "diagnostics": {"flags": []},
            "shootout": {
                "enabled": True,
                "variants": [
                    {
                        "variant_id": "v1",
                        "variant_name": "Speed",
                        "hook_text": "Ready in 7 days",
                        "n": 10,
                        "intent_mean": 3.7,
                        "intent_top2": 0.6,
                        "conversion_count": 6,
                        "wilson_ci": {"lower": 0.3127, "upper": 0.8318, "confidence": 0.95},
                        "trial_estimate": 0.26,
                        "appeal_mean": 3.9,
                    },
                    {
                        "variant_id": "v2",
                        "variant_name": "Conversion",
                        "hook_text": "Built to sell",
                        "n": 10,
                        "intent_mean": 3.2,
                        "intent_top2": 0.5,
                        "conversion_count": 5,
                        "wilson_ci": {"lower": 0.2366, "upper": 0.7634, "confidence": 0.95},
                        "trial_estimate": 0.21,
                        "appeal_mean": 3.7,
                    },
                ],
                "winner_id": "v1",
                "winner_rationale": "Directional lead but CIs overlap",
                "recommendation": "Test in live ads",
            },
        }

    def test_tier1_audit_passes_on_valid_analysis(self):
        result = audit_tier1_invariants(self.valid_analysis)
        self.assertTrue(result.passed, f"Expected pass, got: {result.violations}")
        self.assertEqual(len(result.violations), 0)

    def test_tier1_audit_catches_sample_conservation_failure(self):
        corrupt = dict(self.valid_analysis)
        corrupt["by"] = {
            "segment": {
                "Growth Seekers": {"n": 10},  # Sum = 18 != 20
                "Cautious Traditionalists": {"n": 8},
            }
        }
        result = audit_tier1_invariants(corrupt)
        self.assertFalse(result.passed)
        self.assertTrue(any("Sample conservation failed" in v for v in result.violations))

    def test_tier1_audit_catches_distribution_sum_failure(self):
        corrupt = dict(self.valid_analysis)
        corrupt["metrics"] = {
            "purchase_intent": {
                "dist": {"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.5}  # Sum = 1.3
            }
        }
        result = audit_tier1_invariants(corrupt)
        self.assertFalse(result.passed)
        self.assertTrue(any("Purchase intent distribution sum" in v for v in result.violations))

    def test_tier1_audit_catches_van_westendorp_monotonicity_violation(self):
        corrupt = dict(self.valid_analysis)
        corrupt["pricing"] = {
            "van_westendorp": {
                "median_too_cheap": 2000.0,
                "median_bargain": 1500.0,  # Bargain < Cheap!
                "median_getting_expensive": 3000.0,
                "median_too_expensive": 4000.0,
            }
        }
        result = audit_tier1_invariants(corrupt)
        self.assertFalse(result.passed)
        self.assertTrue(any("Van Westendorp median order violated" in v for v in result.violations))

    def test_tier2_provenance_passes_on_grounded_text(self):
        text = (
            "We surveyed N=20 participants. The Optimal Price Point is R3,021 with an overall trial "
            "estimate of 23.8% and Top-2-Box intent of 55%. The stated price is R3,000."
        )
        result = audit_tier2_provenance(text, self.valid_analysis, self.brief)
        self.assertTrue(result.passed, f"Expected pass, got: {result.violations}")

    def test_tier2_provenance_fails_on_untraced_editorial_numbers(self):
        # Text includes R4,250 corridor and 90% rejection which have no raw provenance
        untraced_text = (
            "Recommended pricing corridor is R4,250 – R5,500. There was a 90% rejection of recurring models."
        )
        result = audit_tier2_provenance(untraced_text, self.valid_analysis, self.brief)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.untraced_figures), 0)
        self.assertTrue(any("R4,250" in fig or "90%" in fig for fig in result.untraced_figures))

    def test_tier2_provenance_with_registered_editorial_justification(self):
        # Register the strategic corridor with explicit justification
        allowed_editorial = [
            {"value": 4250.0, "source": "Strategic 40% margin markup over R3,021 OPP"},
            {"value": 5500.0, "source": "Van Westendorp median_too_expensive ceiling"},
            {"value": 90.0, "source": "Aggregated negative sentiment on subscription tier"},
        ]
        text = "Recommended pricing corridor is R4,250 – R5,500. There was a 90% rejection of recurring models."
        result = audit_tier2_provenance(text, self.valid_analysis, self.brief, allowed_editorial=allowed_editorial)
        self.assertTrue(result.passed, f"Expected pass with registry, got: {result.violations}")

    def test_jev_boundary_guard_blocks_mutation(self):
        # Jev response attempting to alter pricing data
        malicious_response = {
            "risk_score": 0.1,
            "pricing": {"optimal_price_point": 9999.0},  # Forbidden mutation
        }
        with self.assertRaises(AuditFailure):
            JevSupervisorBoundary.validate_jev_response(malicious_response)

    def test_deterministic_template_binding(self):
        template = """
        <h1>{{ brief.product_name }}</h1>
        <p>Stated Price: {{ pricing.stated_price | money(brief.currency) }}</p>
        <p>OPP: {{ pricing.van_westendorp.optimal_price_point | money(brief.currency) }}</p>
        <p>Trial Estimate: {{ quant.trial_estimate | pct }}</p>
        <p>Top-2-Box Intent: {{ metrics.purchase_intent.top2 | pct }}</p>
        {% if shootout %}
        <p>Shootout Winner: {{ shootout.winner_id }} (Wilson CI: {{ shootout.variants[0].wilson_ci | ci }})</p>
        {% endif %}
        """
        rendered = render_bound_template(
            template_str=template,
            analysis_data=self.valid_analysis,
            brief=self.brief,
            strict_provenance=True,
        )
        self.assertIn("R3,000.00", rendered)
        self.assertIn("R3,021.00", rendered)
        self.assertIn("24%", rendered)
        self.assertIn("55%", rendered)
        self.assertIn("[31.3%, 83.2%]", rendered)

    def test_deterministic_template_binding_catches_editorial_leak(self):
        leaky_template = """
        <p>OPP: {{ pricing.van_westendorp.optimal_price_point | money(brief.currency) }}</p>
        <p>Editorial un-sourced recommendation: R8,750</p>
        """
        with self.assertRaises(AuditFailure):
            render_bound_template(
                template_str=leaky_template,
                analysis_data=self.valid_analysis,
                brief=self.brief,
                strict_provenance=True,
            )


if __name__ == "__main__":
    unittest.main()
