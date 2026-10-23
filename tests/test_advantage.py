from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from rich.console import Console

from kohort.advantage import run_advantage
from kohort.config import load_config as _load_config
from kohort.guardrails import GuardrailBlocked
from kohort.llm import CostTracker, LLM, LLMError
from kohort.mock import Mocker
from kohort.pipeline import load_brief

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    """Preserve regression coverage of the optional legacy enforcement mode."""
    cfg = _load_config()
    cfg.guardrails.advantage_review_mode = 'enforce'
    return cfg

SOURCE = {"url": "https://example.com/product", "title": "Official product page (test fixture)",
          "content": "Fixture alternative offers cleaning refills. Current pricing is not stated."}


class FixtureAdvantageLLM(LLM):
    def __init__(self, cfg, tracker, mocker=None):
        cfg.scraping.enabled = False  # Live-shaped fixtures must remain offline.
        super().__init__(cfg, tracker, Mocker(42, load_brief(ROOT / "briefs/example_loopa.yaml")))
        self.mode = "pass"
        self.tasks = []
        self.requests = []
        self.repair_requested = False

    @property
    def is_mock(self):
        return False

    async def structured(self, **kwargs):
        self.tasks.append(kwargs["task"])
        self.requests.append(kwargs)
        if kwargs['schema'].__name__ == 'ReviewedItemRepair':
            payload = json.loads(kwargs['user'])
            obj = kwargs['schema'].model_validate(payload['draft'])
            self.repaired_name = obj.name
            self.repair_sources = payload['sources']
            return obj, json.dumps(obj.model_dump())
        obj, raw = await super().structured(**kwargs)
        if kwargs["task"] in ("competitive_scan", "tool_scan"):
            items = obj.competitors if kwargs["task"] == "competitive_scan" else obj.tools
            for item in items:
                item.source_urls = [SOURCE["url"] if self.mode != "invented_url" else "https://example.com/invented"]
            sources = [] if self.mode == "no_sources" else [dict(SOURCE)]
            if self.mode == "no_excerpts":
                sources[0]["content"] = ""
            self.search_log.append({"sources": sources, "retrieved_at": "2026-10-09T00:00:00+00:00"})
        elif kwargs["task"] == "positioning":
            for opportunity in obj.opportunities:
                opportunity.source_urls = [SOURCE["url"]]
        return obj, raw

    async def decide(self, model, state, questions):
        stage = state["stage"]
        probability = .9 if self.mode == "held" and stage == "positioning" else .05
        if self.mode == 'repair_item' and stage == 'competitive_research' and not self.repair_requested:
            probability = .28
            self.repair_requested = True
        if self.mode == 'partial' and stage == 'competitive_research' and state['evidence'].get('review_item', {}).get('kind') == 'competitors':
            entries = state['evidence']['research']['competitors']['competitors']
            if entries[0]['name'] == '[MOCK] Supermarket own-brand cleaners':
                probability = .28
        return {"answers": {name: {"type": "noul", "noul": probability if name != "repairable" else .05}
                            for name in questions}, "usage": {"cost": .0000042, "input_tokens": 100, "output_tokens": 0}}


class AdvantageTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_receives_varied_business_contexts(self):
        import yaml
        from kohort.advantage import DISCOVERY_SYSTEM, SYSTEM
        for category, buyer, alternative in [('meal delivery', 'Busy working parents', 'Supermarket ready meals'),
                                              ('commercial cleaning', 'Office facilities managers', 'In-house cleaners'),
                                              ('farm irrigation equipment', 'Smallholder farmers', 'Manual watering')]:
            with self.subTest(category=category), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                brief = load_brief(ROOT / 'briefs/example_loopa.yaml').model_dump()
                brief.update(category=category, target_market=buyer, competitors=[alternative],
                             product_name=category, one_liner='Proposed ' + category,
                             description='A business offering ' + category, region='South Africa')
                path = root / 'brief.yaml'
                path.write_text(yaml.safe_dump(brief), encoding='utf-8')
                run = await run_advantage(path, load_config(), mock=True, seed=42,
                                          out_root=root / 'runs', console=Console(file=io.StringIO()))
                plan = json.loads((run / 'alternative-map.json').read_text())
                self.assertEqual(plan['buyer'], buyer)
                self.assertIn(alternative, plan['categories'][0]['search_query'])
                self.assertNotIn('web development', json.dumps(plan).lower())
        for prompt in (DISCOVERY_SYSTEM, SYSTEM):
            for phrase in ('Nova Site', 'Wix', 'website hosting', 'partner plan', 'fixed-price agencies'):
                self.assertNotIn(phrase, prompt)

    async def test_disputed_item_does_not_block_review_of_other_items_or_leak_into_advice(self):
        instances = []
        def factory(*args):
            llm = FixtureAdvantageLLM(*args)
            llm.mode = 'partial'
            instances.append(llm)
            return llm
        with tempfile.TemporaryDirectory() as folder, patch('kohort.advantage.LLM', side_effect=factory):
            run = await run_advantage(ROOT / 'briefs/example_loopa.yaml', load_config(), mock=False,
                                     seed=42, out_root=Path(folder), console=Console(file=io.StringIO()))
            self.assertTrue((run / 'preliminary-advantage.html').exists())
            self.assertTrue((run / 'research-progress.html').exists())
            advisory = json.loads((run / 'advisory-evidence.json').read_text())
            self.assertNotIn('[MOCK] Supermarket own-brand cleaners', [i['name'] for i in advisory['competitors']['competitors']])
            self.assertTrue(advisory['tools']['tools'])
            plan_request = next(r for r in instances[0].requests if r['task'] == 'positioning')
            payload = json.loads(plan_request['user'])
            self.assertTrue(payload['preliminary'])
            self.assertTrue(payload['excluded_entries'])

    async def test_targeted_repair_uses_single_item_schema_and_preserves_other_scan(self):
        instances = []
        def factory(*args):
            llm = FixtureAdvantageLLM(*args)
            llm.mode = 'repair_item'
            instances.append(llm)
            return llm
        with tempfile.TemporaryDirectory() as folder, patch('kohort.advantage.LLM', side_effect=factory):
            await run_advantage(ROOT / 'briefs/example_loopa.yaml', load_config(), mock=False,
                                seed=42, out_root=Path(folder), console=Console(file=io.StringIO()))
            llm = instances[0]
            request = next(r for r in llm.requests if r['schema'].__name__ == 'ReviewedItemRepair')
            payload = json.loads(request['user'])
            self.assertEqual(llm.repaired_name, payload['draft']['name'])
            self.assertEqual(llm.repair_sources, [SOURCE])
            self.assertFalse(request['web_search'])
            with self.assertRaises(ValueError):
                request['schema'].model_validate({**payload['draft'], 'name': 'Unrelated replacement'})
            self.assertEqual(llm.tasks.count('tool_scan'), 2)

    async def test_offline_demo_has_no_network_and_contains_all_sections(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(LLM, "_client", side_effect=AssertionError("Network")), patch(
            "httpx.AsyncClient", side_effect=AssertionError("Network")
        ):
            run = await run_advantage(ROOT / "briefs/example_loopa.yaml", load_config(), mock=True,
                                      seed=42, out_root=Path(folder), console=Console(file=io.StringIO()))
            md = (run / "advantage.md").read_text(encoding="utf-8")
            self.assertIn("MOCK DEMO", md)
            for section in ["Competitors and alternatives", "Useful tools", "Positioning opportunities to test", "What to do next"]:
                self.assertIn(section, md)
            self.assertEqual(json.loads((run / "search-evidence.json").read_text()), [])
            self.assertEqual(json.loads((run / "costs.json").read_text())["total_usd"], 0)

    async def test_live_flow_releases_sourced_results_and_holds_flagged_plan(self):
        for mode in ["pass", "held"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                instances = []
                def factory(*args):
                    llm = FixtureAdvantageLLM(*args)
                    llm.mode = mode
                    instances.append(llm)
                    return llm
                with patch("kohort.advantage.LLM", side_effect=factory):
                    call = run_advantage(ROOT / "briefs/example_loopa.yaml", load_config(), mock=False,
                                          seed=42, out_root=Path(folder), console=Console(file=io.StringIO()))
                    if mode == "pass":
                        await call
                    else:
                        with self.assertRaises(GuardrailBlocked):
                            await call
                run = next(Path(folder).iterdir())
                self.assertEqual((run / "advantage.html").exists(), mode == "pass")
                self.assertEqual((run / "draft-advantage.html").exists(), mode == "held")
                self.assertTrue((run / "search-evidence.json").exists())
                self.assertEqual(instances[0].tasks, ["alternative_map", "competitive_scan", "competitive_scan", "tool_scan", "tool_scan", "positioning"])
                for index in (2, 4):
                    request = instances[0].requests[index]
                    self.assertFalse(request["web_search"])
                    self.assertEqual(json.loads(request["user"])["sources"], [SOURCE])
                report = (run / ("advantage.md" if mode == "pass" else "draft-advantage.md")).read_text(encoding="utf-8")
                self.assertIn(SOURCE["url"], report)
                self.assertIn("hypotheses", report)

    async def test_absent_excerpt_or_invented_source_stops_before_positioning(self):
        for mode in ["no_sources", "no_excerpts", "invented_url"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                instances = []
                def factory(*args):
                    llm = FixtureAdvantageLLM(*args)
                    llm.mode = mode
                    instances.append(llm)
                    return llm
                with patch("kohort.advantage.LLM", side_effect=factory), self.assertRaises(LLMError):
                    await run_advantage(ROOT / "briefs/example_loopa.yaml", load_config(), mock=False,
                                        seed=42, out_root=Path(folder), console=Console(file=io.StringIO()))
                self.assertNotIn("positioning", instances[0].tasks)
                run = next(Path(folder).iterdir())
                self.assertTrue((run / "costs.json").exists())
                self.assertFalse((run / "advantage.html").exists())
                self.assertEqual(json.loads((run / "supervision.json").read_text())["action"], "hold")


class WebSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_tool_limits_citations_and_cost_reservation(self):
        cfg = load_config()
        tracker = CostTracker(1)
        llm = LLM(cfg, tracker)
        message = SimpleNamespace(content='{"ok":true}', annotations=[
            {"type": "url_citation", "url_citation": SOURCE},
            {"type": "url_citation", "url_citation": {"url": "javascript:alert(1)"}},
        ])
        create = AsyncMock(return_value=SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=20, model_extra={"cost": .02}),
            choices=[SimpleNamespace(message=message)]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(llm, "_client", return_value=client):
            await llm._call(cfg.role("competitive_research").model, [], .2, 100, True)
        kwargs = create.call_args.kwargs
        self.assertFalse(kwargs["model"].endswith(":online"))
        body = kwargs["extra_body"]
        self.assertEqual(body['reasoning'], {'enabled': False})
        self.assertEqual(body["tools"][0]["type"], "openrouter:web_search")
        self.assertEqual(body["tools"][0]["parameters"]["max_uses"], 2)
        self.assertEqual(body["tools"][0]["parameters"]["max_total_results"], 10)
        self.assertEqual(body["max_tool_calls"], 2)
        self.assertEqual(llm.last_search_sources, [SOURCE])
        self.assertEqual(tracker.total, .02)
        model = cfg.role("competitive_research").model
        self.assertGreater(llm._reservation(model, {}, 100, web_search=True), llm._reservation(model, {}, 100))

    async def test_concurrent_calls_keep_their_citations_separate(self):
        llm = LLM(load_config(), CostTracker(1))
        async def create(**kwargs):
            url = kwargs["messages"][0]["content"]
            await asyncio.sleep(0)
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1, model_extra={"cost": .01}),
                                   choices=[SimpleNamespace(message=SimpleNamespace(content="{}", annotations=[
                                       {"type": "url_citation", "url_citation": {**SOURCE, "url": url}}
                                   ]))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        async def request(url):
            await llm._call(llm.cfg.role("competitive_research").model, [{"role": "user", "content": url}], .2, 100, True)
            return llm.last_search_sources[0]["url"]
        with patch.object(llm, "_client", return_value=client):
            urls = ["https://example.com/first", "https://example.com/second"]
            self.assertEqual(await asyncio.gather(*(request(url) for url in urls)), urls)


if __name__ == "__main__":
    unittest.main()
