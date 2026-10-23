"""Mock mode: simulated AI responses so the whole pipeline runs for free.

Responses are deterministic and react to persona traits (e.g. price-sensitive
laggards are less keen), so the report shows realistic-looking variation. They
are placeholders for testing the flow - never findings.
"""

from __future__ import annotations

import math
import random

from .schemas import Brief

FIRST = {
    "woman": ["Anna", "Sofia", "Emma", "Lea", "Maria", "Julia", "Sara", "Laura", "Nina", "Hanna", "Clara", "Eva",
              "Mia", "Lena", "Ines", "Fatima", "Chloe", "Ida", "Zoe", "Elif"],
    "man": ["Lukas", "Jonas", "David", "Ben", "Paul", "Elias", "Noah", "Felix", "Leon", "Max", "Tom", "Omar",
            "Mateo", "Erik", "Samuel", "Luca", "Hugo", "Adam", "Karim", "Nils"],
    "non-binary": ["Alex", "Robin", "Sam", "Kim", "Charlie", "Noa"],
}
LAST = ["Müller", "Schmidt", "Rossi", "Jansen", "Novak", "García", "Dubois", "Andersson", "Kowalski", "Weber",
        "Silva", "Meyer", "Fischer", "Bakker", "Moreau", "Nielsen", "Ivanova", "Yilmaz", "Hoffmann", "Costa"]
OCCUPATIONS = {
    "low": ["warehouse associate", "retail cashier", "care assistant", "delivery driver", "cleaner"],
    "lower-middle": ["administrative assistant", "bus driver", "hairdresser", "call-centre agent", "electrician"],
    "middle": ["primary school teacher", "nurse", "IT support technician", "logistics coordinator", "accountant"],
    "upper-middle": ["software engineer", "marketing manager", "pharmacist", "architect", "project manager"],
    "high": ["company director", "medical consultant", "lawyer", "engineering manager", "business owner"],
}
VALUES = ["family time", "saving money", "sustainability", "convenience", "health", "reliability",
          "trying new things", "tradition", "simplicity", "community", "quality over quantity"]
FRUSTRATIONS = ["never enough time", "rising prices", "products that break quickly", "too many subscriptions",
                "misleading marketing", "clutter at home", "complicated apps", "poor customer service"]

LIKES = {
    "Convenience": ["Saves me a trip to the shop", "Looks easy to fit into my routine",
                    "I like that it just turns up when I need it"],
    "Sustainability": ["Less waste is a good thing", "Feels better for the environment",
                       "Nice to cut down on plastic"],
    "Novelty": ["It's different from what's out there", "Fresh idea, haven't seen this before"],
    "Potential savings": ["Could save money in the long run", "Decent value if it works as promised"],
    "Specific features": [],  # filled from the brief
}
CONCERNS = {
    "Price": ["Too pricey for my budget", "Not sure it's worth the money", "Another monthly cost I don't need"],
    "Subscription fatigue": ["I already have too many subscriptions", "Worried it'll be hard to cancel"],
    "Doubts it works": ["Not convinced it works as well as what I use now",
                        "Sounds good on paper, but does it deliver?"],
    "Switching effort": ["What I use now works fine", "Can't be bothered to change my habits"],
    "Unknown brand": ["Never heard of this company", "I'd want reviews from real people first"],
}
TRIGGERS = {
    "Free trial": ["A free trial with no commitment", "If I could try it once without signing up"],
    "Lower price": ["If it were noticeably cheaper", "A big first-order discount"],
    "Social proof": ["If friends recommended it", "Lots of good reviews"],
    "Flexibility": ["Easy pause or cancel any time", "No minimum contract"],
    "Proof of quality": ["Seeing a side-by-side comparison", "A money-back guarantee"],
}

ADOPTER_EFFECT = {"innovator": 1.2, "early adopter": 0.7, "early majority": 0.1, "late majority": -0.5,
                  "laggard": -1.1}
LEVEL_EFFECT = {"low": 0.35, "medium": 0.0, "high": -0.6}
INCOME_PRICE_FACTOR = {"low": 0.7, "lower-middle": 0.85, "middle": 1.0, "upper-middle": 1.2, "high": 1.45}
PRICE_SENS_FACTOR = {"low": 1.2, "medium": 1.0, "high": 0.8}


def _clamp(x: float) -> int:
    return max(1, min(5, round(x)))


class Mocker:
    def __init__(self, seed: int, brief: Brief | None = None):
        self.seed = seed
        self.brief = brief
        self.phrase_theme: dict[str, str] = {}
        features = brief.key_features[:4] if brief and brief.key_features else []
        self.likes = {**LIKES, "Specific features": [f"The '{f}' part sounds useful" for f in features]
                      or ["The main feature sounds useful"]}

    def _rng(self, key: str) -> random.Random:
        return random.Random(f"{self.seed}:{key}")

    def respond(self, task: str, ctx: dict) -> dict:
        return getattr(self, f"_{task}")(ctx)

    def _alternative_map(self, ctx: dict) -> dict:
        brief = ctx.get('brief') or self.brief.model_dump()
        return {'buyer': brief['target_market'],
                'problem_and_desired_outcome': '[MOCK] ' + brief['one_liner'],
                'categories': [{'name': '[MOCK] Brief-supplied alternative: ' + name,
                                'buyer_problem_overlap': '[MOCK] Investigate overlap with ' + brief['category'],
                                'relevance_hypothesis': '[MOCK] Candidate supplied by the client, not researched.',
                                'search_query': name + ' ' + brief['region']}
                               for name in (brief.get('competitors') or ['Current solution'])[:4]],
                'unknowns': ['[MOCK] Category relevance and coverage have not been researched.']}

    def _competitive_scan(self, ctx: dict) -> dict:
        return {"competitors": [{
            "name": f"[MOCK] {name}", "offering": "[MOCK] Existing alternative; not researched.",
            "pricing": "Unknown; no live research in this demo.",
            "positioning": "[MOCK] Established alternative.", "observed_strengths": [],
            "possible_gaps": ["[MOCK] Investigate switching friction; no evidence collected."], "source_urls": [],
        } for name in (self.brief.competitors or ["Current solution"])],
                "coverage_limits": ["[MOCK] Competitor names came from the brief, not internet research."]}

    def _tool_scan(self, ctx: dict) -> dict:
        return {"tools": [{"name": "[MOCK] Landing page builder", "purpose": "[MOCK] Test a proposition.",
                           "pricing": "Unknown", "why_it_fits": "[MOCK] Collect interest before building.",
                           "first_step": "[MOCK] Draft two positioning variants.", "source_urls": []},
                          {"name": "[MOCK] Interview scheduling tool", "purpose": "[MOCK] Recruit customer interviews.",
                           "pricing": "Unknown", "why_it_fits": "[MOCK] Explore existing habits and barriers.",
                           "first_step": "[MOCK] Prepare neutral interview questions.", "source_urls": []}],
                "coverage_limits": ["[MOCK] No actual tools or prices were researched."]}

    def _positioning(self, ctx: dict) -> dict:
        return {"summary": "[MOCK] Demonstration of a competitive positioning plan, without researched findings.",
                "opportunities": [{"target_audience": self.brief.target_market if self.brief else "Target buyers",
                                    "positioning_statement": "[MOCK] A simpler way to meet an existing need.",
                                    "rationale": "[MOCK] Investigate whether current alternatives create avoidable friction.",
                                    "tradeoff": "[MOCK] A narrow proposition may exclude other customers.",
                                    "assumptions": ["[MOCK] The proposed friction is real and worth paying to remove."],
                                    "validation_test": "[MOCK] Interview customers, then compare two landing page messages.",
                                    "decision_criteria": "[MOCK] Define a meaningful improvement before testing; do not assume demand.",
                                    "source_urls": []}],
                "next_actions": ["[MOCK] Run live research.", "[MOCK] Validate the opportunity with real customers."],
                "caveats": ["[MOCK] This plan contains no internet evidence and should not guide purchasing decisions."]}

    def _teardown(self, ctx: dict) -> dict:
        name = ctx.get("competitor_name", "Competitor")
        url = ctx.get("target_url", f"https://{name.lower()}.com")
        return {
            "competitor_name": name,
            "target_url": url,
            "summary": f"[MOCK] Competitive deconstruction of {name}.",
            "positioning": {
                "target_audiences": ["[MOCK] Product managers and marketing agencies"],
                "value_proposition": f"[MOCK] Fast synthetic audience research for {name}.",
                "user_objections_and_complaints": [
                    "[MOCK] High credit pack pricing and lack of verifiable citations."
                ],
            },
            "pricing": {
                "model_type": "Usage-based credit tiers",
                "tiers": [
                    {
                        "name": "Pay As You Go",
                        "price": "$0.12 / response",
                        "features": ["[MOCK] Standard synthetic panel responses"],
                        "limits": ["200 profiles max"],
                    },
                    {
                        "name": "Pro",
                        "price": "$199 / seat / month",
                        "features": ["[MOCK] 5,000 pooled responses", "Shared workspaces"],
                        "limits": ["25 saved audiences"],
                    },
                ],
                "estimated_margins_and_markup": (
                    "[MOCK] Charges $0.12 per response vs estimated underlying LLM cost of $0.0005 (approx 240x markup)."
                ),
                "vulnerabilities": ["[MOCK] High per-response charges deter iterative hypothesis testing."],
            },
            "architecture": {
                "summary": "[MOCK] Client-facing web app with MCP server gateway to LLM agent generation.",
                "integrations_and_apis": ["[MOCK] Model Context Protocol (MCP) server", "REST export API"],
                "strengths": ["[MOCK] Seamless IDE/chat client integration via MCP."],
                "limitations": [
                    "[MOCK] Lacks mathematical hallucination gating or claim-level citation audits."
                ],
            },
            "roadmap": {
                "features_to_adopt": [
                    "Expose a lightweight MCP server endpoint for IDE and chat agent execution.",
                    "Save and reuse named audience profiles to avoid repeated generation.",
                ],
                "differentiators_to_exploit": [
                    "Transparent flat pricing instead of 200x marked-up response credits.",
                    "Jev System One mathematical guardrails preventing ungrounded hallucinations.",
                ],
                "client_pitch_angles": [
                    "Why pay $240+ for 2,000 synthetic responses when Kohort gives you audited decision roadmaps?",
                    "Audited claim-level citation evidence instead of black-box synthetic numbers.",
                ],
                "kohort_implementation_spec": [
                    "Add `--audience` caching and reuse flag to pipeline.",
                    "Package kohort as an MCP server with tools for concept testing and teardowns.",
                ],
            },
            "evidence_sources": [url],
        }

    def _pick_phrases(self, pool: dict[str, list[str]], k: int, rng: random.Random,
                      weights: dict[str, float] | None = None) -> list[str]:
        themes = [t for t in pool if pool[t]]
        w = [(weights or {}).get(t, 1.0) for t in themes]
        chosen: list[str] = []
        for _ in range(k * 3):
            if len(chosen) >= k:
                break
            theme = rng.choices(themes, weights=w)[0]
            phrase = rng.choice(pool[theme])
            if phrase not in chosen:
                chosen.append(phrase)
                self.phrase_theme[phrase] = theme
        return chosen

    # ------------------------------------------------------------------ stages

    def _buyer_discovery(self, ctx: dict) -> dict:
        b = self.brief
        return {
            "buyer_persona_hypothesis": f"[MOCK] Buyers of {b.category} in {b.region} weigh routine habits against price.",
            "category_context_needed": f"[MOCK] Buying occasions, price anchors, and frustrations in {b.category}.",
            "queries": [
                {
                    "focus": "buying_habits",
                    "rationale": "Understand where and how often buyers purchase in this category.",
                    "search_query": f"{b.category} buying habits {b.region}",
                },
                {
                    "focus": "frustrations",
                    "rationale": "Identify common complaints and packaging/product annoyances.",
                    "search_query": f"{b.category} customer complaints reviews {b.region}",
                },
                {
                    "focus": "price_anchors",
                    "rationale": "Establish typical retail price expectations.",
                    "search_query": f"{b.category} typical price {b.region}",
                },
                {
                    "focus": "switching_triggers",
                    "rationale": "Learn what triggers switching from status-quo brands.",
                    "search_query": f"why consumers switch {b.category} brands {b.region}",
                },
            ],
        }

    def _research(self, ctx: dict) -> dict:
        b = self.brief
        comps = b.competitors or ["Existing supermarket alternatives"]
        return {
            "summary": f"[MOCK] Simulated market overview for {b.product_name} in the {b.category} category "
                       f"({b.region}). Replace with a live run for real desk research.",
            "trends": ["[MOCK] Growing interest in convenience", "[MOCK] Cost-of-living pressure on households",
                       "[MOCK] Rising attention to sustainability claims"],
            "competitors": [{"name": c, "offering": "[MOCK] Comparable offering", "price_point": "varies",
                             "strengths": ["Established"], "weaknesses": ["Not differentiated"]} for c in comps],
            "typical_price_range": "[MOCK] Unknown in mock mode",
            "segments": [
                {"name": "Busy young families", "description": "Time-poor parents juggling work and kids",
                 "share": 0.30, "age_min": 28, "age_max": 45,
                 "income_levels": ["lower-middle", "middle", "upper-middle"],
                 "buying_habits": [f"[MOCK] Buys {b.category} during weekly family grocery runs"],
                 "price_anchors": ["[MOCK] Looks for multi-pack or family value pricing"]},
                {"name": "Eco-minded urban professionals", "description": "City dwellers who care about impact",
                 "share": 0.25, "age_min": 24, "age_max": 40, "income_levels": ["middle", "upper-middle", "high"],
                 "buying_habits": [f"[MOCK] Picks up {b.category} on commute or via quick-commerce apps"],
                 "price_anchors": ["[MOCK] Willing to pay a modest premium for quality"]},
                {"name": "Budget-focused households", "description": "Price comes first",
                 "share": 0.30, "age_min": 22, "age_max": 65, "income_levels": ["low", "lower-middle"],
                 "buying_habits": [f"[MOCK] Compares unit prices and sticks to discounted {b.category}"],
                 "price_anchors": ["[MOCK] Strict unit-price ceiling; suspicious of shrinkflation"]},
                {"name": "Established empty-nesters", "description": "Settled habits, quality-oriented",
                 "share": 0.15, "age_min": 52, "age_max": 75, "income_levels": ["middle", "upper-middle", "high"],
                 "buying_habits": [f"[MOCK] Loyal to familiar {b.category} brands for years"],
                 "price_anchors": ["[MOCK] Expects consistent size and quality for the price"]},
            ],
            "buyer_dossier": {
                "buying_habits": [
                    f"[MOCK] Buyers purchase {b.category} as part of weekly shopping or on-the-go convenience.",
                ],
                "real_frustrations": [
                    "[MOCK] Rising prices and suspected shrinkflation when packaging changes.",
                    "[MOCK] Inconsistent quality or inconvenient storage.",
                ],
                "price_anchors": [
                    f"[MOCK] Anchored to mainstream {b.category} pricing in {b.region}.",
                ],
                "status_quo_alternatives": list(comps),
                "cultural_signals": [f"[MOCK] Everyday consumer habits in {b.region}"],
                "scraped_urls": [],
            },
            "sources": [],
            "caveats": ["Mock mode: no real research was performed."],
        }

    def _persona(self, ctx: dict) -> dict:
        t = ctx["traits"]
        lens = ctx.get("category_lens") or {}
        rng = self._rng(f"persona:{ctx['persona_id']}")
        first = rng.choice(FIRST.get(t["gender"], FIRST["non-binary"]))
        occ = rng.choice(OCCUPATIONS[t["income"]])
        if t["age"] < 23 and rng.random() < 0.6:
            occ = "university student with a part-time job"
        elif t["age"] >= 67:
            occ = f"retired {occ}"
        values = rng.sample(VALUES, 3)
        habit_hint = (lens.get("buying_habits") or [f"Buys the usual {self.brief.category} from local shops or online."])[0]
        return {
            "name": f"{first} {rng.choice(LAST)}",
            "occupation": occ,
            "bio": f"{t['age']}-year-old {occ} living in a {t['location']} ({t['household']}). "
                   f"Cares about {values[0]} and {values[1]}. [MOCK persona]",
            "current_solution": habit_hint,
            "values": values,
            "frustrations": rng.sample(FRUSTRATIONS, 2),
        }

    def _latent(self, persona_id: str, t: dict, variant: dict | None = None) -> float:
        rng = self._rng(f"latent:{persona_id}")
        val = (3.0 + ADOPTER_EFFECT[t["adopter_type"]] + LEVEL_EFFECT[t["price_sensitivity"]] * 0.8
                + LEVEL_EFFECT[t["skepticism"]] * 0.7 + rng.gauss(0, 0.8))
        if variant and variant.get("id"):
            import hashlib
            var_hash = int(hashlib.sha256(f"{self.seed}:{variant['id']}".encode()).hexdigest()[:4], 16) % 100
            val += (var_hash - 50) / 100.0 * 0.6
        return val

    def _interview1(self, ctx: dict) -> dict:
        t, pid = ctx["traits"], ctx["persona_id"]
        variant = ctx.get("variant")
        rng = self._rng(f"int1:{pid}")
        latent = self._latent(pid, t, variant=variant)
        positive = latent >= 3.0
        concern_w = {"Price": 2.5 if t["price_sensitivity"] == "high" else 1.0,
                     "Doubts it works": 2.0 if t["skepticism"] == "high" else 1.0}
        likes = self._pick_phrases(self.likes, rng.randint(2, 3) if positive else rng.randint(0, 1), rng)
        concerns = self._pick_phrases(CONCERNS, rng.randint(1, 2) if positive else rng.randint(2, 3), rng,
                                      concern_w)

        base = (self.brief.price or 20.0) * INCOME_PRICE_FACTOR[t["income"]] * PRICE_SENS_FACTOR[
            t["price_sensitivity"]] * math.exp(rng.gauss(0, 0.2))

        def p(mult: float) -> float:
            return round(base * mult * math.exp(rng.gauss(0, 0.06)) * 2) / 2

        out = {
            "first_reaction": ("Interesting - I can see myself using this." if latent > 3.6 else
                               "Hmm, could be handy but I'm not sold yet." if latent > 2.6 else
                               "Not really for me, to be honest.") + " [MOCK]",
            "appeal": _clamp(latent + rng.gauss(0.2, 0.5)),
            "uniqueness": _clamp(3.2 + rng.gauss(0, 0.9)),
            "believability": _clamp(3.4 - (0.6 if t["skepticism"] == "high" else 0) + rng.gauss(0, 0.7)),
            "likes": likes,
            "concerns": concerns,
            "price_too_cheap": p(0.4), "price_bargain": p(0.7),
            "price_getting_expensive": p(1.05), "price_too_expensive": p(1.45),
            "client_questions": [
                {"question": q, "answer": ("Yes, that would matter to me. [MOCK]" if positive
                                           else "Not really a deciding factor for me. [MOCK]")}
                for q in self.brief.questions
            ],
        }
        if self.brief.price is None:
            trig = self._pick_phrases(TRIGGERS, 1, rng)[0]
            out.update(purchase_intent=_clamp(latent), would_buy_if=trig,
                       quote="I'd give it a go." if latent > 3.5 else "I'd probably pass. [MOCK]")
        return out

    def _interview2(self, ctx: dict) -> dict:
        t, pid, part1 = ctx["traits"], ctx["persona_id"], ctx["part1"]
        rng = self._rng(f"int2:{pid}")
        price = self.brief.price or 0
        if price <= part1["price_bargain"]:
            vfm = 5
        elif price <= part1["price_getting_expensive"]:
            vfm = 4
        elif price < part1["price_too_expensive"]:
            vfm = rng.choice([2, 3])
        else:
            vfm = 1
        intent = _clamp(part1["appeal"] - 0.6 + (vfm - 3) * 0.45 + LEVEL_EFFECT[t["price_sensitivity"]] * 0.5
                        + rng.gauss(0, 0.6))
        trig_w = {"Lower price": 3.0 if vfm <= 2 else 0.7, "Free trial": 1.5}
        trigger = self._pick_phrases(TRIGGERS, 1, rng, trig_w)[0]
        return {
            "price_reaction": ("Fair price." if vfm >= 4 else "A bit steep." if vfm >= 2 else "Way too much.")
                              + " [MOCK]",
            "value_for_money": vfm,
            "purchase_intent": intent,
            "would_buy_if": trigger,
            "quote": rng.choice({
                5: ["Honestly, sign me up.", "This solves a real problem for me."],
                4: ["I'd probably try it.", "Worth a shot if the first month goes well."],
                3: ["Maybe, if a friend tried it first.", "I'm on the fence."],
                2: ["Probably not, but never say never.", "Not convinced it's for me."],
                1: ["No, I'm happy with what I have.", "Definitely not at that price."],
            }[intent]) + f" ({pid}) [MOCK]",
        }

    def _themes(self, ctx: dict) -> dict:
        labels: dict[str, str] = {}
        assignments: dict[str, str] = {}
        for i, text in enumerate(ctx["items"]):
            label = self.phrase_theme.get(text, "Other")
            if label not in labels:
                labels[label] = f"T{len(labels) + 1}"
            assignments[str(i)] = labels[label]
        return {"themes": [{"id": tid, "label": lab, "description": f"[MOCK] {lab}"} for lab, tid in labels.items()],
                "assignments": assignments}

    def _critic(self, ctx: dict) -> dict:
        issues = [{"severity": "high", "issue": "Report generated in MOCK mode: every answer is a simulated "
                                                "placeholder, not real AI persona output.",
                   "recommendation": "Run with a live API key before using any finding."}]
        issues += [{"severity": "medium", "issue": f, "recommendation": "Review before sharing."}
                   for f in ctx.get("flags", [])]
        return {"issues": issues, "confidence": "low",
                "confidence_rationale": "Mock data cannot support business decisions."}

    def _report(self, ctx: dict) -> dict:
        q = ctx["quant"]
        trial = round(q["trial_estimate"], 2)
        verdict = "GO" if trial >= 0.25 else "ITERATE" if trial >= 0.12 else "NO-GO"
        top_like = ctx["top_likes"][0] if ctx["top_likes"] else "n/a"
        top_concern = ctx["top_concerns"][0] if ctx["top_concerns"] else "n/a"
        segs = q["by"]["segment"]
        best = max(segs, key=lambda s: segs[s]["trial_estimate"])
        worst = min(segs, key=lambda s: segs[s]["trial_estimate"])
        return {
            "verdict": verdict,
            "verdict_rationale": f"[MOCK] Calibrated trial estimate of {trial:.0%} drives this verdict.",
            "executive_summary": f"[MOCK] This is a placeholder summary for {self.brief.product_name}. The "
                                 f"strongest interest came from '{best}', the weakest from '{worst}'. The most "
                                 f"common positive theme was '{top_like}' and the main barrier was "
                                 f"'{top_concern}'.",
            "key_findings": [f"[MOCK] '{best}' is the most promising segment.",
                             f"[MOCK] '{top_concern}' is the most frequent concern.",
                             "[MOCK] Price perception varies strongly by income."],
            "segment_insights": [{"segment": s, "insight": f"[MOCK] Trial estimate {v['trial_estimate']:.0%}."}
                                 for s, v in segs.items()],
            "action_plan": {
                "what_to_do_next": "[MOCK] Validate with real customers before launch.",
                "why_this_option": "[MOCK] Simulated placeholders cannot support a business decision.",
                "what_to_change": f"[MOCK] Explore '{top_concern}' in offer design.",
                "how_to_implement": "[MOCK] Interview customers about their current solution and barriers.",
                "how_to_measure": "[MOCK] Set decision criteria before collecting real responses.",
                "what_happens_next": "[MOCK] Review actual responses and update the recommendation.",
            },
            "client_question_answers": [{"question": qq, "answer": "[MOCK] Placeholder answer."}
                                        for qq in self.brief.questions],
            "risks": ["[MOCK] Results are simulated placeholders."],
        }
