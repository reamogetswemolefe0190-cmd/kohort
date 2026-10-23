from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from rich.console import Console

from kohort.config import load_config
from kohort.guardrails import CHECKS, GuardrailBlocked, decision_questions, review_report
from kohort.llm import BudgetExceeded, CostTracker, LLM, LLMError, gather_cancel_on_error
from kohort.mock import Mocker
from kohort.pipeline import load_brief, run_concept_test
from kohort.schemas import CriticIssue, CriticReview, MarketResearch, ReportNarrative
from kohort.stages import synthesis

ROOT = Path(__file__).resolve().parent.parent


def decision_response(probability=0.05, questions=None):
    return {"model": "typesafe/jev-1.13-20260917", "answers": {
        name: {"type": "noul", "noul": probability} for name in (questions or CHECKS)
    }, "usage": {"input_tokens": 100, "output_tokens": 0, "cost": 0.0000042}}


class ReviewTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = load_config()
        self.brief = load_brief(ROOT / "briefs/example_loopa.yaml")
        self.mocker = Mocker(42, self.brief)
        self.research = MarketResearch.model_validate(self.mocker.respond("research", {}))
        self.quant = {"diagnostics": {"flags": []}, "n": 20, "metrics": {}, "trial_estimate": .1,
                      "by": {"segment": {"Example": {"trial_estimate": .1}}}, "pricing": {}}
        self.narrative = ReportNarrative.model_validate(self.mocker.respond("report", {
            "quant": self.quant, "top_likes": [], "top_concerns": []
        }))
        self.critic = CriticReview(confidence="low", issues=[])
        self.llm = LLM(self.cfg, CostTracker(1))
        self.llm.decide = AsyncMock(return_value=decision_response())

    async def review(self):
        return await review_report(self.llm, self.cfg, self.brief, self.research,
                                   self.quant, {}, self.critic, self.narrative)

    async def test_low_risk_passes_and_sends_actual_narrative_and_evidence(self):
        result = await self.review()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["response"]["model"], "typesafe/jev-1.13-20260917")
        state = self.llm.decide.call_args.args[1]
        self.assertEqual(state["narrative"], self.narrative.model_dump())
        self.assertEqual(state["evidence"]["quantitative_results"], self.quant)
        self.assertEqual(state["evidence"]["research"], self.research.model_dump())

    async def test_each_risk_individually_holds_report(self):
        for name in CHECKS:
            response = decision_response()
            response["answers"][name]["noul"] = .9
            self.llm.decide.return_value = response
            result = await self.review()
            self.assertEqual(result["status"], "held", name)
            self.assertEqual(len(result["reasons"]), 1)

    async def test_uncertainty_holds_report(self):
        self.llm.decide.return_value = decision_response(.4)
        self.assertEqual((await self.review())["status"], "held")

    async def test_threshold_boundary(self):
        self.llm.decide.return_value = decision_response(.2)
        self.assertEqual((await self.review())["status"], "passed")
        self.llm.decide.return_value = decision_response(.20001)
        self.assertEqual((await self.review())["status"], "held")

    async def test_high_critic_issue_holds_even_when_jev_passes(self):
        self.critic.issues = [CriticIssue(severity="high", issue="Material data flaw")]
        self.assertEqual((await self.review())["status"], "held")

    async def test_provider_error_and_budget_exhaustion_do_not_approve(self):
        for error in [LLMError("Unavailable"), BudgetExceeded("No budget")]:
            self.llm.decide.side_effect = error
            self.assertEqual((await self.review())["status"], "error")

    async def test_mock_skips_jev_and_standard_critic_has_no_name_error(self):
        llm = LLM(self.cfg, CostTracker(1), self.mocker)
        llm.decide = AsyncMock(side_effect=AssertionError("Mock must not call Jev"))
        critic = await synthesis.critique(llm, self.cfg, self.brief, self.quant, {}, self.research)
        self.assertEqual(critic.confidence, "low")
        result = await review_report(llm, self.cfg, self.brief, self.research,
                                     self.quant, {}, critic, self.narrative)
        self.assertEqual(result["status"], "mock")
        llm.decide.assert_not_called()


class DecisionAPITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = load_config()
        self.tracker = CostTracker(1)
        self.llm = LLM(self.cfg, self.tracker)

    async def request(self, payload, status=200):
        captured = []
        def handle(request):
            captured.append(request)
            return httpx.Response(status, content=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
        client_cls = httpx.AsyncClient
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-placeholder"}), patch(
            "httpx.AsyncClient", side_effect=lambda **kw: client_cls(transport=httpx.MockTransport(handle), **kw)
        ):
            result = await self.llm.decide(self.cfg.guardrails.model, {"evidence": "test"}, decision_questions())
        return result, captured

    async def test_api_contract_and_real_usage_accounting(self):
        result, captured = await self.request(decision_response())
        request = captured[0]
        self.assertEqual(str(request.url), "https://openrouter.ai/api/alpha/decisions")
        body = json.loads(request.content)
        self.assertEqual(body["model"], "typesafe/jev-1.13")
        self.assertIsInstance(body["questions"], dict)
        self.assertEqual(body["questions"]["unsupported_claims"]["type"], "noul")
        self.assertIn("instructions", body["questions"]["unsupported_claims"])
        self.assertEqual(result["answers"]["contradictions"]["noul"], .05)
        self.assertEqual(self.tracker.calls, 1)
        self.assertAlmostEqual(self.tracker.total, .0000042)
        self.assertEqual(self.tracker.reserved_usd, 0)

    async def test_invalid_answers_are_rejected_and_billed_usage_kept(self):
        for value in [None, "0.1", True, -1, 2, float("nan"), float("inf")]:
            data = decision_response()
            data["answers"]["unsupported_claims"]["noul"] = value
            with self.assertRaises(LLMError):
                await self.request(data)
        for mutation in ["missing", "wrong_type", "wrong_key"]:
            data = decision_response()
            if mutation == "missing":
                del data["answers"]["contradictions"]
            elif mutation == "wrong_type":
                data["answers"]["contradictions"]["type"] = "choice"
            else:
                data["answers"]["contradictions"] = {"type": "noul", "probability": .05}
            with self.assertRaises(LLMError):
                await self.request(data)
        self.assertGreater(self.tracker.total, 0)

    async def test_http_error_and_missing_usage_hold_reservations(self):
        for data, status in [({"error": "Failure"}, 503), ({"answers": {}}, 200)]:
            with self.assertRaises(LLMError):
                await self.request(data, status)
        self.assertEqual(self.tracker.uncertain_calls, 4)  # three HTTP attempts plus malformed usage
        self.assertGreater(self.tracker.reserved_usd, 0)

    async def test_router_and_missing_key_are_rejected_before_network(self):
        with patch("httpx.AsyncClient", side_effect=AssertionError("No network")):
            with self.assertRaises(LLMError):
                await self.llm.decide("typesafe/jev-router", "test", decision_questions())
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}):
                with self.assertRaises(LLMError):
                    await self.llm.decide(self.cfg.guardrails.model, "test", decision_questions())


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_parallel_stage_cancels_and_drains_siblings(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        async def pending():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        async def failed():
            await started.wait()
            raise BudgetExceeded("No budget")
        with self.assertRaises(BudgetExceeded):
            await gather_cancel_on_error(pending(), failed())
        self.assertTrue(cancelled.is_set())

    async def test_concurrent_requests_cannot_bypass_reservations(self):
        cfg = load_config()
        cfg.run.concurrency = 2
        cfg.prices["budget-test"] = [0, 100000]  # One maximum output token costs $0.10.
        tracker = CostTracker(.15)
        llm = LLM(cfg, tracker)
        started = asyncio.Event()
        release = asyncio.Event()
        requests = []
        async def create(**kwargs):
            requests.append(kwargs)
            started.set()
            await release.wait()
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                                                         model_extra={"cost": .1}),
                                   choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(llm, "_client", return_value=client):
            first = asyncio.create_task(llm._call("budget-test", [], 0, 1, False))
            await started.wait()
            with self.assertRaises(BudgetExceeded):
                await llm._call("budget-test", [], 0, 1, False)
            release.set()
            await first
            with self.assertRaises(BudgetExceeded):
                await llm._call("budget-test", [], 0, 1, False)
        self.assertEqual(len(requests), 1)
        self.assertAlmostEqual(tracker.total, .1)
        self.assertEqual(tracker.reserved_usd, 0)

    async def test_unknown_pricing_never_becomes_free(self):
        llm = LLM(load_config(), CostTracker(1))
        with self.assertRaises(LLMError):
            llm._reservation("unknown-model", {}, 10)

    async def test_timeout_keeps_budget_for_potential_bill(self):
        cfg = load_config()
        llm = LLM(cfg, CostTracker(1))
        create = AsyncMock(side_effect=TimeoutError("Unknown outcome"))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(llm, "_client", return_value=client), self.assertRaises(TimeoutError):
            await llm._call(cfg.role("report").model, [], 0, 100, False)
        self.assertEqual(llm.tracker.uncertain_calls, 1)
        self.assertGreater(llm.tracker.reserved_usd, 0)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_early_failure_still_saves_costs_and_closes_clients(self):
        async def fail(llm, *args):
            llm.tracker.add("fixture", 100, 1, .01)
            raise BudgetExceeded("No budget")
        with tempfile.TemporaryDirectory() as folder, patch(
            "kohort.pipeline.research_stage.run", side_effect=fail
        ), patch.object(LLM, "aclose", new_callable=AsyncMock) as close:
            with self.assertRaises(BudgetExceeded):
                await run_concept_test(ROOT / "briefs/example_loopa.yaml", load_config(), mock=True,
                                       n_personas=20, seed=42, out_root=Path(folder),
                                       console=Console(file=io.StringIO()))
            run = next(Path(folder).iterdir())
            self.assertEqual(json.loads((run / "costs.json").read_text())["total_usd"], .01)
            self.assertFalse((run / "report.html").exists())
            close.assert_awaited_once()

    async def test_full_mock_pipeline_is_offline_and_renders_action_plan(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "httpx.AsyncClient", side_effect=AssertionError("Mock opened a network client")
        ), patch.object(LLM, "_client", side_effect=AssertionError("Mock opened a chat client")):
            run = await run_concept_test(ROOT / "briefs/example_loopa.yaml", load_config(), mock=True,
                                         n_personas=20, seed=42, out_root=Path(folder),
                                         console=Console(file=io.StringIO()))
            self.assertTrue((run / "report.html").exists())
            md = (run / "report.md").read_text(encoding="utf-8")
            self.assertIn("## 13. Action Plan", md)
            self.assertIn("MOCK RUN", md)
            self.assertIn("Jev was not called", md)
            self.assertEqual(json.loads((run / "guardrails.json").read_text())["status"], "mock")
            self.assertEqual(json.loads((run / "costs.json").read_text())["total_usd"], 0)

    async def test_live_output_policy_for_pass_hold_and_failure(self):
        for outcome in ["passed", "held", "error"]:
            class FixtureLiveLLM(LLM):
                def __init__(self, cfg, tracker, mocker=None):
                    super().__init__(cfg, tracker, Mocker(42, load_brief(ROOT / "briefs/example_loopa.yaml")))

                @property
                def is_mock(self):
                    return False

                async def structured(self, **kwargs):
                    if kwargs["task"] == "critic":
                        return CriticReview(confidence="low", issues=[]), "{}"
                    return await super().structured(**kwargs)

                async def decide(self, *args):
                    state, questions = args[1:]
                    if state.get("stage") != "report" and "narrative" not in state:
                        return decision_response(questions=questions)
                    if outcome == "error":
                        raise LLMError("Fixture service unavailable")
                    return decision_response(.9 if outcome == "held" else .05, questions)

            with tempfile.TemporaryDirectory() as folder, patch("kohort.pipeline.LLM", FixtureLiveLLM):
                call = run_concept_test(ROOT / "briefs/example_loopa.yaml", load_config(), mock=False,
                                        n_personas=20, seed=42, out_root=Path(folder),
                                        console=Console(file=io.StringIO()))
                if outcome == "passed":
                    await call
                else:
                    with self.assertRaises(GuardrailBlocked):
                        await call
                run = next(Path(folder).iterdir())
                result = json.loads((run / "guardrails.json").read_text())
                self.assertEqual(result["status"], outcome)
                self.assertTrue((run / "costs.json").exists())
                self.assertEqual((run / "report.html").exists(), outcome == "passed")
                self.assertEqual((run / "draft-report.html").exists(), outcome != "passed")
                if outcome != "passed":
                    self.assertIn("DRAFT", (run / "draft-report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
