from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel, ValidationError
from rich.console import Console

from kohort.config import GuardrailCfg, load_config
from kohort.guardrails import GuardrailBlocked
from kohort.llm import BudgetExceeded, CostTracker, LLM, LLMError
from kohort.mock import Mocker
from kohort.pipeline import load_brief, run_concept_test
from kohort.schemas import CriticIssue, CriticReview

ROOT = Path(__file__).resolve().parent.parent
STAGES = ["brief", "research", "personas", "interviews", "analysis", "critic", "report"]
TASK_FOR_STAGE = {"research": "research", "personas": "persona", "interviews": "interview1",
                  "analysis": "themes", "critic": "critic", "report": "report"}


class FixtureLLM(LLM):
    """Runs real pipeline code with deterministic generation and decision fixtures."""
    def __init__(self, cfg, tracker, mocker=None):
        super().__init__(cfg, tracker, Mocker(42, load_brief(ROOT / "briefs/example_loopa.yaml")))
        self.tasks = []
        self.states = []
        self.decision_counts = Counter()
        self.target = None
        self.mode = "pass"
        self.high_critic_issue = False
        self.invalid_coding = False

    @property
    def is_mock(self):
        return False

    async def structured(self, **kwargs):
        self.tasks.append((kwargs["task"], self.stage_feedback, kwargs.get("mock_ctx")))
        if kwargs["task"] == "critic":
            issues = [CriticIssue(severity="high", issue="Unresolved upstream data flaw")] if self.high_critic_issue else []
            return CriticReview(confidence="low", issues=issues), "{}"
        obj, raw = await super().structured(**kwargs)
        if self.invalid_coding and kwargs["task"] == "themes" and obj.assignments:
            del obj.assignments[next(iter(obj.assignments))]
        return obj, raw

    async def decide(self, model, state, questions):
        stage = state.get("stage", "report")
        self.states.append(state)
        self.decision_counts[stage] += 1
        self.tracker.add(model, 100, 0, .0000042)
        fail = stage == self.target and (self.mode != "repair" or self.decision_counts[stage] == 1)
        if fail and self.mode == "error":
            raise LLMError("Fixture API unavailable")
        if fail and self.mode == "budget":
            raise BudgetExceeded("Fixture budget exhausted")
        probability = .95 if fail else .05
        if fail and self.mode == "uncertain":
            probability = .4
        return {"model": "typesafe/jev-1.13-test", "answers": {
            name: {"type": "noul", "noul": (.95 if fail and self.mode in ("repair", "repeat") else .05)
                   if name == "repairable" else probability} for name in questions
        }, "usage": {"input_tokens": 100, "output_tokens": 0, "cost": .0000042}}


class SupervisorTests(unittest.IsolatedAsyncioTestCase):
    async def execute(self, folder, configure=lambda llm: None, cfg=None):
        instances = []
        def factory(*args):
            llm = FixtureLLM(*args)
            configure(llm)
            instances.append(llm)
            return llm
        failure = None
        with patch("kohort.pipeline.LLM", side_effect=factory):
            try:
                await run_concept_test(ROOT / "briefs/example_loopa.yaml", cfg or load_config(),
                                       mock=False, n_personas=20, seed=42, out_root=Path(folder),
                                       console=Console(file=io.StringIO()))
            except (GuardrailBlocked, LLMError, BudgetExceeded) as exc:
                failure = exc
        run = next(Path(folder).iterdir())
        audit = json.loads((run / "supervision.json").read_text(encoding="utf-8"))
        return run, instances[0], audit, failure

    async def test_all_seven_checkpoints_approve_in_order(self):
        with tempfile.TemporaryDirectory() as folder:
            run, llm, audit, failure = await self.execute(folder)
            self.assertIsNone(failure)
            self.assertEqual(list(dict.fromkeys(c["stage"] for c in audit["checkpoints"])), STAGES)
            self.assertTrue(all(c["status"] == "passed" and c["action"] == "continue" for c in audit["checkpoints"]))
            self.assertTrue((run / "report.html").exists())
            self.assertGreater(json.loads((run / "costs.json").read_text())["total_usd"], 0)
            for checkpoint in audit["checkpoints"]:
                self.assertTrue((run / checkpoint["evidence_file"]).exists())
                self.assertTrue((run / checkpoint["output_file"]).exists())
            self.assertIn("| Jev checkpoint |", (run / "report.md").read_text(encoding="utf-8"))

    async def test_rejection_at_each_stage_prevents_every_downstream_stage(self):
        for stage in STAGES:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as folder:
                def configure(llm):
                    llm.target, llm.mode = stage, "hold"
                run, llm, audit, failure = await self.execute(folder, configure)
                self.assertIsInstance(failure, GuardrailBlocked)
                self.assertEqual(audit["stage"], stage)
                self.assertEqual(audit["action"], "hold")
                called = {t[0] for t in llm.tasks}
                for later in STAGES[STAGES.index(stage) + 1:]:
                    self.assertNotIn(TASK_FOR_STAGE[later], called)
                self.assertFalse((run / "report.html").exists())
                self.assertEqual((run / "draft-report.html").exists(), stage == "report")

    async def test_early_api_failure_and_budget_failure_never_advance(self):
        for mode in ["error", "budget", "uncertain"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                def configure(llm):
                    llm.target, llm.mode = "research", mode
                run, llm, audit, failure = await self.execute(folder, configure)
                self.assertIsInstance(failure, GuardrailBlocked)
                self.assertEqual(audit["action"], "hold")
                self.assertEqual(llm.decision_counts["research"], 1)
                self.assertNotIn("persona", [task[0] for task in llm.tasks])
                self.assertTrue((run / "costs.json").exists())

    async def test_repairable_defect_retries_only_current_stage(self):
        for stage in STAGES[1:]:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as folder:
                def configure(llm):
                    llm.target, llm.mode = stage, "repair"
                run, llm, audit, failure = await self.execute(folder, configure)
                self.assertIsNone(failure)
                events = [c for c in audit["checkpoints"] if c["stage"] == stage]
                self.assertEqual(events[0]["action"], "retry")
                self.assertEqual(events[-1]["attempt"], 2)
                self.assertEqual(events[-1]["action"], "continue")
                counts = Counter(t[0] for t in llm.tasks)
                for earlier in STAGES[1:STAGES.index(stage)]:
                    expected = {"persona": 20, "interview1": 20, "themes": 3}.get(TASK_FOR_STAGE[earlier], 1)
                    self.assertEqual(counts[TASK_FOR_STAGE[earlier]], expected)
                retry_calls = [t for t in llm.tasks if t[0] == TASK_FOR_STAGE[stage] and t[1]]
                self.assertTrue(retry_calls)
                self.assertTrue(all("Jev risk probability" in t[1] for t in retry_calls))
                self.assertEqual(llm.stage_feedback, "")
                if stage == "personas":
                    first = json.loads((run / "supervision/personas-attempt-1.json").read_text())
                    second = json.loads((run / "supervision/personas-attempt-2.json").read_text())
                    self.assertEqual([p["traits"] for p in first["personas"]], [p["traits"] for p in second["personas"]])

    async def test_retry_limit_holds_repeated_defect(self):
        with tempfile.TemporaryDirectory() as folder:
            def configure(llm):
                llm.target, llm.mode = "research", "repeat"
            run, llm, audit, failure = await self.execute(folder, configure)
            self.assertIsInstance(failure, GuardrailBlocked)
            self.assertEqual(llm.decision_counts["research"], 2)
            self.assertEqual(audit["action"], "hold")
            self.assertIn("retry limit", " ".join(audit["checkpoints"][-1]["reasons"]))
            self.assertNotIn("persona", [t[0] for t in llm.tasks])

    async def test_zero_retry_setting_and_brief_never_regenerate(self):
        for stage in ["brief", "research"]:
            with tempfile.TemporaryDirectory() as folder:
                cfg = load_config()
                cfg.guardrails.max_stage_retries = 0
                def configure(llm):
                    llm.target, llm.mode = stage, "repeat"
                _, llm, audit, failure = await self.execute(folder, configure, cfg)
                self.assertIsInstance(failure, GuardrailBlocked)
                self.assertEqual(llm.decision_counts[stage], 1)
                self.assertEqual(audit["action"], "hold")

    async def test_batches_cover_every_persona_response_and_coded_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = load_config()
            cfg.guardrails.batch_size = 7
            run, llm, audit, failure = await self.execute(folder, cfg=cfg)
            self.assertIsNone(failure)
            for stage, field in [("personas", "personas"), ("interviews", "responses")]:
                items = [item for state in llm.states if state.get("stage") == stage for item in state["evidence"][field]]
                ids = [item["id" if stage == "personas" else "persona_id"] for item in items]
                self.assertEqual(len(ids), 20)
                self.assertEqual(len(set(ids)), 20)
            coding = json.loads((run / "theme-coding.json").read_text())
            expected = sorted((kind, item["persona_id"], item["text"]) for kind, items in coding.items() for item in items)
            seen = sorted((item["kind"], item["persona_id"], item["text"])
                          for state in llm.states if state.get("stage") == "analysis"
                          for item in state["evidence"]["coded_answers"])
            self.assertEqual(seen, expected)

    async def test_high_critic_issue_stops_before_report_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            def configure(llm):
                llm.high_critic_issue = True
            run, llm, audit, failure = await self.execute(folder, configure)
            self.assertIsInstance(failure, GuardrailBlocked)
            self.assertEqual(audit["stage"], "critic")
            self.assertNotIn("report", [t[0] for t in llm.tasks])
            self.assertFalse((run / "narrative.json").exists())

    async def test_incomplete_theme_assignments_stop_before_critic(self):
        with tempfile.TemporaryDirectory() as folder:
            def configure(llm):
                llm.invalid_coding = True
            run, llm, audit, failure = await self.execute(folder, configure)
            self.assertIsInstance(failure, LLMError)
            self.assertEqual(audit["stage"], "analysis")
            self.assertEqual(audit["action"], "hold")
            self.assertNotIn("critic", [t[0] for t in llm.tasks])

    async def test_rework_feedback_is_in_actual_generation_prompt(self):
        class Ping(BaseModel):
            ok: bool
        llm = FixtureLLM(load_config(), CostTracker(1))
        llm.mocker = None
        llm.stage_feedback = "Research has unsupported facts; acknowledge uncertainty."
        with patch.object(llm, "_call", new_callable=AsyncMock, return_value='{"ok":true}') as call:
            await LLM.structured(llm, task="ping", schema=Ping, model="fixture", system="Respond with JSON.", user="Ping")
            system = call.call_args.args[1][0]["content"]
            self.assertIn(llm.stage_feedback, system)
            self.assertIn("Do not hide uncertainty", system)


class ConfigTests(unittest.TestCase):
    def test_retry_threshold_must_exceed_hold_threshold(self):
        with self.assertRaises(ValidationError):
            GuardrailCfg(max_risk_probability=.9, retry_min_probability=.8)


class SelfHealingGuardrailTests(unittest.IsolatedAsyncioTestCase):
    async def test_dead_zone_retry_and_advisory_soft_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            def configure(llm):
                llm.target = "personas"
                orig_decide = llm.decide
                async def custom_decide(model, state, questions):
                    stage = state.get("stage", "report")
                    if stage == "personas":
                        llm.states.append(state)
                        llm.decision_counts[stage] += 1
                        attempt = llm.decision_counts[stage]
                        # Attempt 1: 0.45 risk (in old 0.20-0.80 dead zone) with 0.40 repairable -> triggers retry
                        # Attempt 2: 0.30 risk (<= 0.35 advisory threshold) with 0.40 repairable -> advisory soft-pass
                        prob = 0.45 if attempt == 1 else 0.30
                        return {
                            "model": "typesafe/jev-1.13-test",
                            "answers": {
                                name: {"type": "noul", "noul": 0.40 if name == "repairable" else prob}
                                for name in questions
                            },
                            "usage": {"input_tokens": 100, "output_tokens": 0, "cost": 0.0000042},
                        }
                    return await orig_decide(model, state, questions)
                llm.decide = custom_decide

            run, llm, audit, failure = await SupervisorTests().execute(folder, configure)
            self.assertIsNone(failure)
            persona_events = [c for c in audit["checkpoints"] if c["stage"] == "personas"]
            self.assertEqual(persona_events[0]["action"], "retry")
            self.assertEqual(persona_events[-1]["attempt"], 2)
            self.assertEqual(persona_events[-1]["status"], "passed")
            self.assertEqual(persona_events[-1]["action"], "continue")
            self.assertTrue(persona_events[-1].get("advisory_pass"))

            guardrails_data = json.loads((run / "guardrails.json").read_text(encoding="utf-8"))
            self.assertIn("integrity", guardrails_data)
            self.assertGreater(guardrails_data["integrity"]["integrity_score"], 0.8)
            self.assertIn("personas", guardrails_data["integrity"]["self_healed_stages"])
            self.assertTrue(guardrails_data["integrity"]["advisory_warnings"])
            report_md = (run / "report.md").read_text(encoding="utf-8")
            self.assertIn("Evidence Integrity:", report_md)

    def test_tier1_deterministic_item_validators(self):
        from kohort.schemas import InterviewResponse, PersonaProfile, PersonaTraits
        from kohort.stages.interviews import validate_interview_item
        from kohort.stages.personas import validate_persona_item

        brief = load_brief(ROOT / "briefs/example_loopa.yaml")
        traits = PersonaTraits(
            segment="Test", age=35, gender="man", income="middle",
            household="living with partner", location="city",
            adopter_type="early majority", price_sensitivity="medium", skepticism="medium",
        )
        bad_profile = PersonaProfile(
            name="Sipho Ndlovu",
            occupation="Manager",
            bio=f"She works in the city and loves {brief.product_name}.",
            current_solution="Uses basic tools",
            values=["Quality"],
            frustrations=["Delays"],
        )
        persona_issues = validate_persona_item(bad_profile, traits, brief)
        self.assertEqual(len(persona_issues), 2)

        bad_interview = InterviewResponse(
            persona_id="P001", model="test", first_reaction="As an AI, I think it is okay.",
            appeal=1, uniqueness=3, believability=3, likes=[], concerns=[],
            price_too_cheap=0, price_bargain=80, price_getting_expensive=120, price_too_expensive=150,
            purchase_intent=5, would_buy_if="cheaper", quote="Nice",
        )
        interview_issues = validate_interview_item(bad_interview, brief)
        self.assertEqual(len(interview_issues), 3)


if __name__ == "__main__":
    unittest.main()

