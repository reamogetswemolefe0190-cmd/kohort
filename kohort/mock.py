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
    def __init__(self, seed: int, brief: Brief):
        self.seed = seed
        self.brief = brief
        self.phrase_theme: dict[str, str] = {}
        self.likes = {**LIKES, "Specific features": [f"The '{f}' part sounds useful" for f in brief.key_features[:4]]
                      or ["The main feature sounds useful"]}

    def _rng(self, key: str) -> random.Random:
        return random.Random(f"{self.seed}:{key}")

    def respond(self, task: str, ctx: dict) -> dict:
        return getattr(self, f"_{task}")(ctx)

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
                 "income_levels": ["lower-middle", "middle", "upper-middle"]},
                {"name": "Eco-minded urban professionals", "description": "City dwellers who care about impact",
                 "share": 0.25, "age_min": 24, "age_max": 40, "income_levels": ["middle", "upper-middle", "high"]},
                {"name": "Budget-focused households", "description": "Price comes first",
                 "share": 0.30, "age_min": 22, "age_max": 65, "income_levels": ["low", "lower-middle"]},
                {"name": "Established empty-nesters", "description": "Settled habits, quality-oriented",
                 "share": 0.15, "age_min": 52, "age_max": 75, "income_levels": ["middle", "upper-middle", "high"]},
            ],
            "sources": [],
            "caveats": ["Mock mode: no real research was performed."],
        }

    def _persona(self, ctx: dict) -> dict:
        t = ctx["traits"]
        rng = self._rng(f"persona:{ctx['persona_id']}")
        first = rng.choice(FIRST.get(t["gender"], FIRST["non-binary"]))
        occ = rng.choice(OCCUPATIONS[t["income"]])
        if t["age"] < 23 and rng.random() < 0.6:
            occ = "university student with a part-time job"
        elif t["age"] >= 67:
            occ = f"retired {occ}"
        values = rng.sample(VALUES, 3)
        return {
            "name": f"{first} {rng.choice(LAST)}",
            "occupation": occ,
            "bio": f"{t['age']}-year-old {occ} living in a {t['location']} ({t['household']}). "
                   f"Cares about {values[0]} and {values[1]}. [MOCK persona]",
            "current_solution": f"Buys the usual {self.brief.category} from local shops or online.",
            "values": values,
            "frustrations": rng.sample(FRUSTRATIONS, 2),
        }

    def _latent(self, persona_id: str, t: dict) -> float:
        rng = self._rng(f"latent:{persona_id}")
        return (3.0 + ADOPTER_EFFECT[t["adopter_type"]] + LEVEL_EFFECT[t["price_sensitivity"]] * 0.8
                + LEVEL_EFFECT[t["skepticism"]] * 0.7 + rng.gauss(0, 0.8))

    def _interview1(self, ctx: dict) -> dict:
        t, pid = ctx["traits"], ctx["persona_id"]
        rng = self._rng(f"int1:{pid}")
        latent = self._latent(pid, t)
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
            "recommendations": [
                {"priority": 1, "action": f"[MOCK] Address '{top_concern}' in messaging and offer design.",
                 "rationale": "Most frequent barrier."},
                {"priority": 2, "action": f"[MOCK] Focus launch marketing on '{best}'.",
                 "rationale": "Highest calibrated trial."},
                {"priority": 3, "action": "[MOCK] Validate with 30-50 real customers before launch.",
                 "rationale": "Synthetic findings need real-world calibration."},
            ],
            "client_question_answers": [{"question": qq, "answer": "[MOCK] Placeholder answer."}
                                        for qq in self.brief.questions],
            "risks": ["[MOCK] Results are simulated placeholders."],
        }
