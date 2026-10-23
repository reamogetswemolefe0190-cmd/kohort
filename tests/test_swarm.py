import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from kohort.advantage import run_advantage
from kohort.config import load_config
from kohort.llm import BudgetExceeded, LLM, LLMError
from kohort.swarm import (Comparison, PanelDesign, customer_text, coverage_complete, load_advisory,
                          prepare_boards, run_swarm, summarize, validate_audience_scope, validate_comparison)
from tests.test_advantage import FixtureAdvantageLLM, ROOT


def response(ids):
    return Comparison.model_validate({'options': [{'option_id': oid, 'relevance': 2+i, 'clarity': 3, 'credibility': 4,
        'objections': ['Clarify support'], 'suggested_changes': ['Show contract scope']} for i, oid in enumerate(ids)],
        'choice': 'keep_current', 'reason': 'Switching effort exceeds the benefit.', 'conditions_for_switching': ['Clear support terms'],
        'question_for_real_buyers': 'How much work would switching create?'})


class SwarmFixture(LLM):
    mode = 'pass'
    requests = []

    async def structured(self, **kwargs):
        self.requests.append(kwargs)
        if kwargs['web_search']:
            raise AssertionError('New searches are forbidden')
        payload = json.loads(kwargs['user'])
        if kwargs['task'] in ('swarm_design', 'swarm_design_repair'):
            if kwargs['task'] == 'swarm_design_repair' and self.mode == 'repair_error':
                raise LLMError('Repair temporarily unavailable')
            data = {'segments': [{'role': role, 'setting': payload['buyer_context']['target_market'], 'buying_responsibility': 'Evaluates purchases',
                                'inclusion_criteria': role + ' acting as sole evaluator', 'exclusion_criteria': ['Other role acts as sole evaluator'],
                                'audience_group': 'primary', 'scope_quote': payload['buyer_context']['target_market'], 'scope_reason': 'Explicit target audience'}
                                for role in ['Owner', 'Operations lead']], 'assumptions': ['Hypothetical roles, not measured shares.']}
        elif kwargs['task'] == 'swarm_panel_review':
            if self.mode == 'review_error':
                raise LLMError('Reviewer temporarily unavailable')
            first_review = sum(r['task'] == 'swarm_panel_review' for r in self.requests) == 1
            data = {'issues': ['Broad segment overlaps specialist segment'] if self.mode in ('overlap', 'repair_error') or (self.mode == 'repair' and first_review) else []}
        elif kwargs['task'] == 'swarm_persona':
            if self.mode == 'budget':
                raise BudgetExceeded('Fixture budget exhausted')
            data = {'current_workflow': 'Existing system', 'needs': ['Reliable support'], 'constraints': ['Limited time'], 'decision_process': 'Owner approval'}
        else:
            result = response([b['option_id'] for b in payload['offers']])
            if self.mode == 'invalid' or (self.mode == 'partial' and sum(r['task'] == 'swarm_comparison' for r in self.requests) == 1):
                result.options.pop()
            return result, '{}'
        return kwargs['schema'].model_validate(data), json.dumps(data)

    async def decide(self, model, state, questions):
        # Observer disagreement must not block structurally valid synthetic output.
        return {'answers': {key: {'noul': .99} for key in questions}}


class SwarmTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        async def passed(llm, cfg, stage, evidence, allow_retry):
            return {'stage': stage, 'status': 'passed', 'action': 'continue', 'reasons': []}
        class SourceFixture(FixtureAdvantageLLM):
            async def structured(self, **kwargs):
                obj, raw = await super().structured(**kwargs)
                if kwargs['task'] == 'positioning':
                    obj.opportunities.append(obj.opportunities[0].model_copy(update={'positioning_statement': 'Alternative test message'}))
                return obj, raw
        with patch('kohort.advantage.LLM', SourceFixture), patch('kohort.advantage.review_with_observer', passed):
            self.source = await run_advantage(ROOT / 'briefs/example_loopa.yaml', load_config(), mock=False, seed=42,
                out_root=self.root / 'advisory', console=Console(file=io.StringIO()))

    async def asyncTearDown(self):
        self.temp.cleanup()

    def test_blocked_sources_and_invalid_option_responses(self):
        for status in ('held', 'mock', 'running'):
            (self.source / 'release.json').write_text(json.dumps({'status': status, 'eligible_for_delivery': True}))
            with self.assertRaises(ValueError):
                load_advisory(self.source)
        for mode in ('missing', 'duplicate', 'choice'):
            result = response(['O1', 'O2'])
            if mode == 'missing':
                result.options.pop()
            elif mode == 'duplicate':
                result.options[1].option_id = 'O1'
            else:
                result.choice = 'O999'
            with self.assertRaises(LLMError):
                validate_comparison(result, ['O1', 'O2'])

    async def test_mock_is_offline_reproducible_and_order_varies_within_segments(self):
        folders = []
        with patch('httpx.AsyncClient', side_effect=AssertionError('Network')):
            for _ in range(2):
                folders.append(await run_swarm(self.source, load_config(), n_personas=12, mock=True, seed=17,
                    out_root=self.root / 'swarm', console=Console(file=io.StringIO())))
        self.assertEqual((folders[0] / 'responses.json').read_bytes(), (folders[1] / 'responses.json').read_bytes())
        assignments = json.loads((folders[0] / 'assignments.json').read_text())
        for segment in {a['segment_id'] for a in assignments}:
            self.assertGreater(len({tuple(a['option_order']) for a in assignments if a['segment_id'] == segment}), 1)
        release = json.loads((folders[0] / 'release.json').read_text())
        self.assertFalse(release['eligible_for_delivery'])
        self.assertEqual(json.loads((folders[0] / 'costs.json').read_text())['total_usd'], 0)
        report = (folders[0] / 'swarm-report.md').read_text(encoding='utf-8')
        self.assertIn('MOCK DEMO', report)
        self.assertIn('not interviews with real customers', report)

    async def test_reused_audience_keeps_profiles_but_gets_fresh_offers_and_answers(self):
        SwarmFixture.mode, SwarmFixture.requests = 'pass', []
        with patch('kohort.swarm.LLM', SwarmFixture):
            original = await run_swarm(self.source, load_config(), n_personas=6,
                out_root=self.root / 'original', console=Console(file=io.StringIO()))
        # Poison prior results: audience reuse must never read or prompt with them.
        (original / 'responses.json').write_text('PRIOR_ANSWERS_MUST_NOT_BE_READ')
        (original / 'panel-limitations.json').write_text(json.dumps({'issues': ['Saved concern']}))
        evidence_path = self.source / 'advisory-evidence.json'
        _, _, evidence, _ = load_advisory(self.source)
        evidence['plan']['opportunities'][0]['proposed_message'] = 'Revised offer for this comparison'
        extra = dict(evidence['plan']['opportunities'][0])
        extra['proposed_message'] = 'A third offer for comparison'
        evidence['plan']['opportunities'].append(extra)
        evidence_path.write_text(json.dumps(evidence))
        before = {p.name: p.read_bytes() for p in original.iterdir() if p.is_file()}
        SwarmFixture.requests = []
        with patch('kohort.swarm.LLM', SwarmFixture):
            new = await run_swarm(self.source, load_config(), n_personas=6, audience_from=original,
                out_root=self.root / 'reused', console=Console(file=io.StringIO()))
        self.assertEqual({r['task'] for r in SwarmFixture.requests}, {'swarm_comparison'})
        for req in SwarmFixture.requests:
            self.assertIn('Revised offer for this comparison', req['user'])
            self.assertIn('A third offer for comparison', req['user'])
            self.assertNotIn('PRIOR_ANSWERS_MUST_NOT_BE_READ', req['user'])
        old_people = json.loads((original / 'personas.json').read_text())
        new_people = json.loads((new / 'personas.json').read_text())
        for old, current in zip(old_people, new_people):
            for key in ('persona_id', 'profile', 'traits', 'model', 'segment_id', 'segment'):
                self.assertEqual(old[key], current[key])
            self.assertEqual(set(current['option_order']), {'O1', 'O2', 'O3'})
        self.assertEqual(len(json.loads((new / 'responses.json').read_text())), 6)
        self.assertEqual(before, {p.name: p.read_bytes() for p in original.iterdir() if p.is_file()})
        provenance = json.loads((new / 'audience-reuse.json').read_text())
        self.assertFalse(provenance['prior_answers_imported'])
        self.assertNotIn('responses.json', provenance['source_files_sha256'])
        self.assertEqual(json.loads((new / 'release.json').read_text())['scenario_review_issues'], ['Saved concern'])

    async def test_audience_reuse_rejects_incompatible_or_corrupt_inputs_before_calls(self):
        SwarmFixture.mode, SwarmFixture.requests = 'pass', []
        with patch('kohort.swarm.LLM', SwarmFixture):
            original = await run_swarm(self.source, load_config(), n_personas=6,
                out_root=self.root / 'original', console=Console(file=io.StringIO()))
        cases = [
            ('source-advisory.json', lambda d: d['brief'].update(client='Another client')),
            ('source-advisory.json', lambda d: d['brief'].update(target_market='Another audience')),
            ('run.json', lambda d: d.update(mock=True)),
            ('run.json', lambda d: d.update(swarm_protocol=3)),
            ('run.json', lambda d: d.update(requested_personas=12)),
            ('personas.json', lambda d: d.pop()),
            ('personas.json', lambda d: d[1].update(persona_id=d[0]['persona_id'])),
            ('personas.json', lambda d: d[0].update(segment_id='S999')),
            ('panel-limitations.json', lambda d: d.update(issues=None)),
        ]
        for filename, mutate in cases:
            with self.subTest(filename=filename):
                path = original / filename
                before = path.read_bytes()
                data = json.loads(before)
                mutate(data)
                path.write_text(json.dumps(data))
                try:
                    with patch('kohort.swarm.LLM', side_effect=AssertionError('No paid calls permitted')):
                        with self.assertRaises(ValueError):
                            await run_swarm(self.source, load_config(), n_personas=6, audience_from=original,
                                out_root=self.root / 'blocked', console=Console(file=io.StringIO()))
                finally:
                    path.write_bytes(before)
        self.assertFalse((self.root / 'blocked').exists())

    async def test_mock_audience_reuse_is_offline(self):
        with patch('httpx.AsyncClient', side_effect=AssertionError('Network')):
            original = await run_swarm(self.source, load_config(), n_personas=6, mock=True,
                out_root=self.root / 'mock-original', console=Console(file=io.StringIO()))
            reused = await run_swarm(self.source, load_config(), n_personas=6, mock=True, audience_from=original,
                out_root=self.root / 'mock-reused', console=Console(file=io.StringIO()))
        self.assertFalse(json.loads((reused / 'release.json').read_text())['eligible_for_delivery'])
        self.assertEqual(json.loads((reused / 'costs.json').read_text())['total_usd'], 0)

    async def test_live_fixture_neutral_prompts_counts_and_observer(self):
        SwarmFixture.mode, SwarmFixture.requests = 'pass', []
        with patch('kohort.swarm.LLM', SwarmFixture):
            folder = await run_swarm(self.source, load_config(), n_personas=6, out_root=self.root / 'swarm', console=Console(file=io.StringIO()))
        brief, plan, _, _ = load_advisory(self.source)
        for req in SwarmFixture.requests:
            payload = json.loads(req['user'])
            if req['task'] in ('swarm_design', 'swarm_persona', 'swarm_panel_review', 'swarm_design_repair'):
                self.assertNotIn(brief.product_name, req['user'])
                self.assertNotIn('offers', payload)
                self.assertNotIn('plan', payload)
            else:
                for board in payload['offers']:
                    self.assertEqual(set(board), {'option_id', 'message'})
                    self.assertEqual(board['message'], next(b for b in json.loads((folder / 'buyer-offers.json').read_text()) if b['option_id'] == board['option_id'])['message'])
                self.assertEqual(set(payload['persona']), {'profile', 'traits', 'segment'})
                self.assertNotIn('audience_group', payload['persona']['segment'])
                for alternative in payload['alternatives']:
                    self.assertNotIn('possible_gaps', alternative)
                    self.assertNotIn('why_buyers_consider_it', alternative)
        summary = json.loads((folder / 'analysis.json').read_text())
        self.assertNotIn('options', summary)
        self.assertEqual(sum(s['keep_current'] for s in summary['scenarios'].values()), 6)
        self.assertTrue(all(o['chosen'] == 0 and o['respondents'] == 3 for s in summary['scenarios'].values() for o in s['options']))
        self.assertNotIn('options', summary['audiences']['primary'])
        self.assertEqual(json.loads((folder / 'jev-observation.json').read_text())['status'], 'held')
        self.assertTrue(json.loads((folder / 'release.json').read_text())['eligible_for_delivery'])

    async def test_panel_overlap_repaired_before_any_personas(self):
        SwarmFixture.mode, SwarmFixture.requests = 'repair', []
        with patch('kohort.swarm.LLM', SwarmFixture):
            folder = await run_swarm(self.source, load_config(), n_personas=6, out_root=self.root / 'repair', console=Console(file=io.StringIO()))
        tasks = [r['task'] for r in SwarmFixture.requests]
        self.assertEqual(tasks[:4], ['swarm_design', 'swarm_panel_review', 'swarm_design_repair', 'swarm_panel_review'])
        self.assertTrue(json.loads((folder / 'panel-review-1.json').read_text())['issues'])
        self.assertEqual(json.loads((folder / 'panel-review-2.json').read_text())['issues'], [])
        self.assertTrue(json.loads((folder / 'release.json').read_text())['eligible_for_delivery'])

    async def test_unresolved_overlap_and_review_errors_continue_with_scenario_limits(self):
        for mode in ('overlap', 'review_error', 'repair_error'):
            SwarmFixture.mode, SwarmFixture.requests = mode, []
            with patch('kohort.swarm.LLM', SwarmFixture):
                folder = await run_swarm(self.source, load_config(), n_personas=6, out_root=self.root / mode, console=Console(file=io.StringIO()))
            state = json.loads((folder / 'release.json').read_text())
            self.assertTrue(state['eligible_for_delivery'])
            self.assertEqual(state['status'], 'released_with_limits')
            self.assertTrue(json.loads((folder / 'panel-limitations.json').read_text())['issues'])
            summary = json.loads((folder / 'analysis.json').read_text())
            self.assertEqual(summary['completed'], 6)
            self.assertEqual(len(summary['scenarios']), 2)
            self.assertNotIn('options', summary['audiences']['primary'])
            report = (folder / 'swarm-report.md').read_text(encoding='utf-8')
            self.assertIn('Unresolved scenario review:', report)
            self.assertIn('Scenario S1', report)
            self.assertIn('Scenario S2', report)

    async def test_failures_keep_partial_artifacts_unreleased(self):
        for mode in ('invalid', 'budget'):
            with self.subTest(mode=mode):
                SwarmFixture.mode, SwarmFixture.requests = mode, []
                with patch('kohort.swarm.LLM', SwarmFixture):
                    call = run_swarm(self.source, load_config(), n_personas=5, out_root=self.root / mode, console=Console(file=io.StringIO()))
                    if mode == 'budget':
                        with self.assertRaises(BudgetExceeded):
                            await call
                    else:
                        await call
                folder = next((self.root / mode).iterdir())
                state = json.loads((folder / 'release.json').read_text())
                self.assertEqual(state['status'], 'incomplete')
                self.assertFalse(state['eligible_for_delivery'])
                self.assertTrue((folder / 'swarm-report.html').exists())
                self.assertTrue((folder / 'costs.json').exists())

    async def test_coverage_threshold_releases_with_explicit_limits(self):
        SwarmFixture.mode, SwarmFixture.requests = 'partial', []
        with patch('kohort.swarm.LLM', SwarmFixture):
            folder = await run_swarm(self.source, load_config(), n_personas=5, out_root=self.root / 'partial', console=Console(file=io.StringIO()))
        self.assertEqual(json.loads((folder / 'analysis.json').read_text())['completed'], 4)
        state = json.loads((folder / 'release.json').read_text())
        self.assertTrue(state['eligible_for_delivery'])
        self.assertEqual(state['status'], 'released_with_limits')
        self.assertIn('4/5', (folder / 'swarm-report.md').read_text(encoding='utf-8'))

    def test_aggregation_uses_observed_denominator(self):
        rows = [{'segment_id': 'S1', 'response': response(['O1', 'O2']).model_dump()}]
        result = summarize(rows, ['O1', 'O2'])
        self.assertEqual(result['completed'], 1)
        self.assertEqual(result['scenarios']['S1']['options'][1]['relevance'], 3)
        self.assertEqual(result['scenarios']['S1']['keep_current'], 1)

    def test_presentation_removes_only_research_boilerplate(self):
        terms = 'Build is free in year one, then R1,000/month. Custom work is quoted separately; scope and VAT are unknown. No guaranteed results.'
        for phrase in ('This is a hypothesis to test, not a proven advantage.',
                       'Test this with real buyers before treating it as a claim.'):
            clean, removed = customer_text(terms + ' ' + phrase)
            self.assertEqual(clean, terms)
            self.assertEqual(removed, [phrase])
        mixed = 'Hypothesis: support is included, but this is unverified. Pricing is unknown.'
        self.assertEqual(customer_text(mixed), (mixed, []))
        _, plan, _, _ = load_advisory(self.source)
        plan.opportunities[0].proposed_message = terms + ' Test this with real buyers before treating it as a claim.'
        boards, audit = prepare_boards(plan)
        self.assertEqual(boards[0]['proposed_message'], terms)
        self.assertTrue(next(r for r in audit if r['field'] == 'proposed_message')['internal_notes'])

    def test_audiences_never_pool_preferences_or_completion(self):
        rows, assignments = [], []
        for i in range(10):
            group = 'primary' if i < 2 else 'secondary'
            assignment = {'segment_id': 'S1' if i < 2 else 'S2', 'audience_group': group}
            assignments.append(assignment)
            answer = response(['O1', 'O2'])
            if group == 'primary':
                answer.choice = 'O1'
            if i != 0:
                rows.append({**assignment, 'response': answer.model_dump()})
        summary = summarize(rows, ['O1', 'O2'], assignments)
        self.assertEqual(summary['completed'], 9)
        self.assertNotIn('options', summary)
        self.assertNotIn('keep_current', summary)
        self.assertEqual(summary['scenarios']['S1']['options'][0]['chosen'], 1)
        self.assertEqual(summary['scenarios']['S2']['keep_current'], 8)
        self.assertEqual(summary['audiences']['primary']['requested'], 2)
        # Overall completion is 90%, but only half the primary audience completed.
        self.assertFalse(coverage_complete(summary, rows, assignments))

    def test_overlapping_scenarios_in_same_audience_never_pool_choices(self):
        rows = [{'segment_id': sid, 'audience_group': 'primary', 'response': response(['O1', 'O2']).model_dump()} for sid in ('S1', 'S2')]
        rows[0]['response']['choice'] = 'O1'
        summary = summarize(rows, ['O1', 'O2'])
        self.assertEqual(summary['audiences']['primary'], {'completed': 2, 'requested': 2})
        self.assertEqual(summary['scenarios']['S1']['options'][0]['chosen'], 1)
        self.assertEqual(summary['scenarios']['S2']['keep_current'], 1)

    def test_audience_classification_requires_actual_context_quote(self):
        data = {'segments': [{'role': role, 'setting': 'Retail', 'buying_responsibility': 'Purchases',
                 'inclusion_criteria': role, 'exclusion_criteria': ['Other buying context'],
                 'audience_group': group, 'scope_quote': quote, 'scope_reason': 'Context explicitly identifies priority.'}
                 for role, group, quote in [('Owner', 'primary', 'Local shops'), ('Importer', 'secondary', 'Overseas enquiries are secondary')]],
                'assumptions': ['Synthetic panel']}
        design = PanelDesign.model_validate(data)
        context = {'target_market': 'Local shops. Overseas enquiries are secondary', 'region': 'Test region'}
        validate_audience_scope(design, context)
        design.segments[1].scope_quote = 'Invented priority'
        with self.assertRaises(LLMError):
            validate_audience_scope(design, context)
