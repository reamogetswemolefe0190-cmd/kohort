import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
from rich.console import Console

from kohort.advantage import CompetitorScan, run_advantage
from kohort.config import load_config
from kohort.evidence_review import ClaimCheck, EvidenceReview, ReferenceCheck, ReferenceReview, claim_packet, review_evidence, review_with_observer, validate_review
from kohort.llm import CostTracker, LLM, LLMError
from kohort.guardrails import GuardrailBlocked
from tests.test_advantage import FixtureAdvantageLLM

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = {'brief': {'client': 'Test'}, 'research': {'pricing': 'R100/month'},
            'sources': [{'url': 'https://example.com', 'content': 'The price is R100/month.'}]}


def good_review(source_id='https://example.com'):
    return EvidenceReview.model_validate({'checks': [{'claim_id': 'claim/pricing', 'verdict': 'supported',
        'explanation': 'Matches supplied price.', 'evidence': [{'source_id': source_id, 'excerpt_id': 0}]}]})


def reference_review(reference_id='E1'):
    data = good_review().model_dump()
    data['checks'][0]['evidence'] = [{'reference_id': reference_id}]
    return ReferenceReview.model_validate(data)


class EvidenceReviewTests(unittest.IsolatedAsyncioTestCase):
    def test_unknown_needs_no_quote_but_factual_verdict_does(self):
        unknown = ClaimCheck(claim_id='pricing', verdict='unknown', explanation='Price not established.')
        self.assertEqual(unknown.evidence, [])
        with self.assertRaises(ValidationError):
            ClaimCheck(claim_id='pricing', verdict='supported', explanation='Unsupported approval.')

    def test_reviewer_cannot_invent_or_omit_claims_or_quotes(self):
        claims, sources = claim_packet('competitive_research', EVIDENCE)
        validate_review(good_review(), claims, sources)
        for mode in ('missing', 'duplicate', 'invented', 'quote', 'noquote'):
            review = good_review()
            if mode == 'missing':
                review.checks = []
            elif mode == 'duplicate':
                review.checks *= 2
            elif mode == 'invented':
                review.checks[0].claim_id = 'not_a_claim'
            elif mode == 'quote':
                review.checks[0].evidence[0].excerpt_id = 999
            else:
                review.checks[0].evidence = []
            with self.subTest(mode=mode), self.assertRaises(LLMError):
                validate_review(review, claims, sources)

    async def test_jev_disagreement_and_outage_do_not_override_review(self):
        cfg = load_config()
        llm = LLM(cfg, CostTracker(1))
        for observation in ({'status': 'held', 'action': 'hold', 'reasons': ['Risk 99%']},
                            {'status': 'error', 'action': 'hold', 'reasons': ['API unavailable']}):
            with patch.object(llm, 'structured', AsyncMock(return_value=(reference_review(), '{}'))), patch(
                'kohort.evidence_review.review_stage', AsyncMock(return_value=observation)):
                result = await review_with_observer(llm, cfg, 'competitive_research', EVIDENCE, False)
                self.assertEqual(result['action'], 'continue')
                self.assertEqual(result['jev_observation'], observation)
                self.assertFalse(result['human_review_required'])
                citation = result['checks'][0]['evidence'][0]
                self.assertEqual(citation['source_url'], 'https://example.com')
                self.assertEqual(citation['quote'], 'The price is R100/month.')
        await llm.aclose()

    async def test_structured_review_failure_is_not_approval(self):
        cfg = load_config()
        llm = LLM(cfg, CostTracker(1))
        with patch.object(llm, 'structured', AsyncMock(side_effect=LLMError('outage'))):
            result = await review_evidence(llm, cfg, 'competitive_research', EVIDENCE)
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['action'], 'hold')
        await llm.aclose()

    async def test_invalid_reference_gets_one_bounded_correction(self):
        cfg = load_config()
        for corrected in (True, False):
            with self.subTest(corrected=corrected):
                llm = LLM(cfg, CostTracker(1))
                invalid = reference_review('E999')
                replies = [(invalid, '{}'), (reference_review() if corrected else invalid, '{}')]
                with patch.object(llm, 'structured', AsyncMock(side_effect=replies)) as call:
                    result = await review_evidence(llm, cfg, 'competitive_research', EVIDENCE)
                    self.assertEqual(call.await_count, 2)
                    payloads = [json.loads(c.kwargs['user']) for c in call.call_args_list]
                    self.assertEqual(payloads[0]['evidence_catalog'], payloads[1]['evidence_catalog'])
                    self.assertIn('unknown evidence ID', payloads[1]['validation_feedback'])
                    self.assertEqual(result['action'], 'continue' if corrected else 'hold')
                    self.assertEqual(result['review_attempts'][0]['status'], 'invalid')
                await llm.aclose()

    async def test_large_review_batches_cover_every_claim_and_preserve_defects(self):
        cfg = load_config()
        llm = LLM(cfg, CostTracker(1))
        evidence = {**EVIDENCE, 'research': {'findings': ['R100/month'] * 80}}
        seen = []
        async def respond(**kwargs):
            payload = json.loads(kwargs['user'])
            self.assertLessEqual(len(payload['claims']), 12)
            self.assertFalse(kwargs['web_search'])
            seen.extend(payload['claims'])
            checks = []
            for key in payload['claims']:
                issue = key == 'claim/findings/13'
                checks.append(ReferenceCheck(claim_id=key, verdict='unsupported' if issue else 'supported',
                    explanation='Fixture defect' if issue else 'Matches source',
                    evidence=[] if issue else [{'reference_id': 'E1'}],
                    correction='Remove the unsupported premise.' if issue else ''))
            return ReferenceReview(checks=checks), '{}'
        with patch.object(llm, 'structured', side_effect=respond) as call:
            result = await review_evidence(llm, cfg, 'competitive_research', evidence)
            self.assertEqual(call.await_count, 7)
            self.assertEqual(len(seen), 80)
            self.assertEqual(len(set(seen)), 80)
            self.assertEqual(len(result['checks']), 80)
            self.assertEqual(result['status'], 'held')
            self.assertIn('claim/findings/13', result['reasons'][0])
        await llm.aclose()

    async def test_missing_claim_feedback_requires_complete_replacement(self):
        cfg = load_config()
        llm = LLM(cfg, CostTracker(1))
        with patch.object(llm, 'structured', AsyncMock(side_effect=[
            (ReferenceReview(checks=[]), '{}'), (reference_review(), '{}')])) as call:
            result = await review_evidence(llm, cfg, 'competitive_research', EVIDENCE)
            self.assertEqual(result['status'], 'passed')
            payload = json.loads(call.call_args_list[1].kwargs['user'])
            self.assertIn('claim/pricing', payload['validation_feedback'])
            self.assertEqual(payload['required_check_count'], 1)
        await llm.aclose()

    def test_duplicate_competitors_rejected(self):
        item = {'name': 'Example', 'offering': 'Test', 'positioning': 'Test'}
        with self.assertRaises(ValidationError):
            CompetitorScan.model_validate({'competitors': [item, {**item, 'name': ' EXAMPLE '}]})

    async def test_mock_report_is_not_eligible_for_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            run = await run_advantage(ROOT / 'briefs/example_loopa.yaml', load_config(), mock=True,
                                     seed=42, out_root=Path(directory), console=Console(file=io.StringIO()))
            self.assertTrue((run / 'advantage.html').exists())
            self.assertTrue((run / 'evidence-review.html').exists())
            state = json.loads((run / 'release.json').read_text())
            self.assertFalse(state['eligible_for_delivery'])
            self.assertFalse(state['human_review_required'])
            self.assertEqual(state['status'], 'mock')

    async def test_automatic_release_pass_partial_hold_and_outage(self):
        for mode in ('pass', 'partial', 'held', 'outage', 'no_sources'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                instances = []
                def factory(*args):
                    llm = FixtureAdvantageLLM(*args)
                    if mode == 'no_sources':
                        llm.mode = mode
                    instances.append(llm)
                    return llm

                async def review(llm, cfg, stage, evidence, allow_retry):
                    status = 'passed'
                    if mode == 'held' and stage == 'positioning':
                        status = 'held'
                    if stage == 'competitive_research':
                        if mode == 'outage':
                            status = 'error'
                        if mode == 'partial' and evidence.get('review_item', {}).get('name') == '[MOCK] Supermarket own-brand cleaners':
                            status = 'held'
                    return {'stage': stage, 'status': status,
                            'action': 'continue' if status == 'passed' else 'hold',
                            'reviewer': 'Fixture structured review', 'reasons': [] if status == 'passed' else ['Fixture defect'],
                            'jev_observation': {'status': 'held', 'reasons': ['Observational only']}}

                with patch('kohort.advantage.LLM', side_effect=factory), patch(
                    'kohort.advantage.review_with_observer', side_effect=review):
                    call = run_advantage(ROOT / 'briefs/example_loopa.yaml', load_config(), mock=False,
                                         seed=42, out_root=Path(directory), console=Console(file=io.StringIO()))
                    if mode in ('pass', 'partial'):
                        await call
                    else:
                        with self.assertRaises((GuardrailBlocked, LLMError)):
                            await call
                run = next(Path(directory).iterdir())
                state = json.loads((run / 'release.json').read_text())
                expected = {'pass': 'released', 'partial': 'released_with_limits',
                            'held': 'held', 'outage': 'held', 'no_sources': 'failed'}
                self.assertEqual(state['status'], expected[mode])
                self.assertEqual(state['eligible_for_delivery'], mode in ('pass', 'partial'))
                self.assertFalse(state['human_review_required'])
                self.assertFalse((run / 'human-review.json').exists())
                if state['report']:
                    report = (run / state['report']).read_text(encoding='utf-8')
                    self.assertNotIn('HUMAN APPROVAL REQUIRED', report)
                else:
                    self.assertTrue((run / 'research-progress.html').exists())
                if mode == 'partial':
                    self.assertEqual(len(state['excluded_entries']), 1)
                    payload = json.loads(next(r for r in instances[0].requests if r['task'] == 'positioning')['user'])
                    self.assertNotIn('[MOCK] Supermarket own-brand cleaners',
                                     [c['name'] for c in payload['competitors']['competitors']])
