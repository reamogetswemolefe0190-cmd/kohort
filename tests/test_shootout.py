"""Unit tests for the Hook Shootout engine (A/B testing, deterministic assignment, Wilson intervals)."""

from __future__ import annotations

import unittest
from collections import Counter

from kohort.analysis import analyze_shootout, summarize, two_proportion_z_test, wilson_score_interval
from kohort.sampling import assign_variant
from kohort.schemas import Brief, HookVariant, InterviewResponse, Persona, PersonaTraits, QA


class ShootoutTests(unittest.TestCase):
    def setUp(self):
        self.variants = [
            HookVariant(id="var_a", name="Pain: Freelancer Risk", hook_text="Stop losing clients to late delivery.", angle="loss_aversion"),
            HookVariant(id="var_b", name="Benefit: Rapid Turnaround", hook_text="Ship verified web apps in 48 hours.", angle="speed"),
            HookVariant(id="var_c", name="Social Proof: Top Tier", hook_text="Used by 50+ high-growth agencies.", angle="social_proof"),
        ]

    def test_assign_variant_is_deterministic_and_reproducible(self):
        """Same persona_id + seed must always yield the exact same variant."""
        p_id = "P042"
        v1 = assign_variant(p_id, self.variants, seed=123)
        v2 = assign_variant(p_id, self.variants, seed=123)
        self.assertIsNotNone(v1)
        self.assertEqual(v1.id, v2.id)

        # Different seed or persona should hash independently
        v3 = assign_variant(p_id, self.variants, seed=456)
        # Empty variants returns None
        self.assertIsNone(assign_variant(p_id, [], seed=123))

    def test_assign_variant_distribution_is_balanced(self):
        """Between-subjects assignment must spread personas evenly across variants."""
        counts = Counter()
        n_personas = 60
        for i in range(1, n_personas + 1):
            pid = f"P{i:03d}"
            v = assign_variant(pid, self.variants, seed=42)
            counts[v.id] += 1

        self.assertEqual(len(counts), 3)
        for var_id in ("var_a", "var_b", "var_c"):
            # With 60 personas across 3 variants, each should receive roughly 15-25 personas
            self.assertGreaterEqual(counts[var_id], 15)
            self.assertLessEqual(counts[var_id], 25)

    def test_wilson_score_interval_bounds(self):
        """Wilson score interval must maintain defensible bounds at small N and extreme rates."""
        # 0 successes out of 20: Wald gives [0, 0]; Wilson gives a defensible positive upper bound
        lo, hi = wilson_score_interval(0, 20, confidence=0.95)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.10)
        self.assertLess(hi, 0.20)

        # 20 successes out of 20: Wilson gives a defensible lower bound < 1.0
        lo, hi = wilson_score_interval(20, 20, confidence=0.95)
        self.assertGreater(lo, 0.80)
        self.assertEqual(hi, 1.0)

        # 10 successes out of 20 (50%): symmetrical around 0.5
        lo, hi = wilson_score_interval(10, 20, confidence=0.95)
        self.assertAlmostEqual((lo + hi) / 2.0, 0.5, places=2)
        self.assertLess(lo, 0.35)
        self.assertGreater(hi, 0.65)

        # Edge cases: 0 total
        self.assertEqual(wilson_score_interval(0, 0), (0.0, 0.0))

    def test_two_proportion_z_test(self):
        """Check z-score and two-tailed p-value calculation."""
        # Identical proportions
        z, p = two_proportion_z_test(10, 20, 10, 20)
        self.assertEqual(z, 0.0)
        self.assertEqual(p, 1.0)

        # Statistically significant difference (e.g. 18/20 vs 4/20)
        z, p = two_proportion_z_test(18, 20, 4, 20)
        self.assertGreater(z, 3.5)
        self.assertLess(p, 0.001)

        # Zero denominator edge case
        z, p = two_proportion_z_test(0, 0, 5, 10)
        self.assertEqual(z, 0.0)
        self.assertEqual(p, 1.0)

    def test_analyze_shootout_end_to_end(self):
        """Full shootout analysis with winner identification and overlap caveats."""
        rows = []
        # Variant A: 16 out of 20 buy (80% Top-2-Box)
        for i in range(1, 21):
            intent = 5 if i <= 12 else 4 if i <= 16 else 2
            rows.append(
                InterviewResponse(
                    persona_id=f"P{i:03d}",
                    model="test-model",
                    variant_id="var_a",
                    first_reaction="Great",
                    appeal=4,
                    uniqueness=4,
                    believability=4,
                    purchase_intent=intent,
                    price_too_cheap=100.0,
                    price_bargain=200.0,
                    price_getting_expensive=400.0,
                    price_too_expensive=600.0,
                )
            )

        # Variant B: 5 out of 20 buy (25% Top-2-Box)
        for i in range(21, 41):
            intent = 4 if i <= 25 else 2
            rows.append(
                InterviewResponse(
                    persona_id=f"P{i:03d}",
                    model="test-model",
                    variant_id="var_b",
                    first_reaction="Mediocre",
                    appeal=2,
                    uniqueness=3,
                    believability=3,
                    purchase_intent=intent,
                    price_too_cheap=100.0,
                    price_bargain=200.0,
                    price_getting_expensive=400.0,
                    price_too_expensive=600.0,
                )
            )

        shootout = analyze_shootout(rows, self.variants[:2])
        self.assertIsNotNone(shootout)
        self.assertTrue(shootout["enabled"])
        self.assertEqual(len(shootout["variants"]), 2)

        var_a_metric = next(m for m in shootout["variants"] if m["variant_id"] == "var_a")
        var_b_metric = next(m for m in shootout["variants"] if m["variant_id"] == "var_b")

        self.assertEqual(var_a_metric["n"], 20)
        self.assertEqual(var_a_metric["conversion_count"], 16)
        self.assertEqual(var_a_metric["intent_top2"], 0.8)
        self.assertGreater(var_a_metric["wilson_ci"]["lower"], 0.58)

        self.assertEqual(var_b_metric["n"], 20)
        self.assertEqual(var_b_metric["conversion_count"], 5)
        self.assertEqual(var_b_metric["intent_top2"], 0.25)
        self.assertLess(var_b_metric["wilson_ci"]["upper"], 0.47)

        # Pairwise comparison
        pair = var_a_metric["pairwise"][0]
        self.assertEqual(pair["other_variant_id"], "var_b")
        self.assertTrue(pair["significant_95"])
        self.assertLess(pair["p_value"], 0.01)

        # Winner
        self.assertEqual(shootout["winner_id"], "var_a")
        self.assertIn("statistically outperformed", shootout["winner_rationale"])
        self.assertIn("Deploy 'Stop losing clients to late delivery.'", shootout["recommendation"])

    def test_shootout_integrated_in_summarize(self):
        """summarize() must include shootout key when variants are present."""
        brief = Brief(
            client="Acme",
            product_name="Test Product",
            category="Software",
            one_liner="Fast software",
            description="Software description",
            target_market="Developers",
            region="Global",
            variants=self.variants,
        )
        persona = Persona(
            id="P001",
            name="Alice",
            occupation="Engineer",
            bio="Bio",
            model="test-model",
            traits=PersonaTraits(
                segment="Growth",
                age=30,
                gender="woman",
                income="middle",
                household="single",
                location="city",
                adopter_type="early adopter",
                price_sensitivity="medium",
                skepticism="low",
            ),
        )
        response = InterviewResponse(
            persona_id="P001",
            model="test-model",
            variant_id="var_a",
            first_reaction="Good",
            appeal=4,
            uniqueness=4,
            believability=4,
            purchase_intent=4,
            price_too_cheap=10.0,
            price_bargain=20.0,
            price_getting_expensive=40.0,
            price_too_expensive=60.0,
        )

        quant = summarize(brief, [persona], [response], weights={"definitely": 0.75, "probably": 0.25})
        self.assertIn("shootout", quant)
        self.assertEqual(quant["shootout"]["winner_id"], "var_a")


if __name__ == "__main__":
    unittest.main()
