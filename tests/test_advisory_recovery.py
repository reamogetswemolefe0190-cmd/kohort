import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from kohort.advisory_recovery import recover_advisory
from kohort.advantage import run_advantage
from kohort.config import load_config
from kohort.evidence_review import claim_packet
from tests.test_advantage import FixtureAdvantageLLM, ROOT


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_recovery_releases_only_unchanged_passed_live_evidence(self):
        async def passed(llm, cfg, stage, evidence, allow_retry):
            return {'stage': stage, 'status': 'passed', 'action': 'continue', 'reasons': []}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch('kohort.advantage.LLM', FixtureAdvantageLLM), patch('kohort.advantage.review_with_observer', passed):
                source = await run_advantage(ROOT / 'briefs/example_loopa.yaml', load_config(), mock=False,
                    seed=42, out_root=root / 'source', console=Console(file=io.StringIO()))
            ledger_path = source / 'supervision.json'
            ledger_text = ledger_path.read_text()
            ledger = json.loads(ledger_text)
            evidence_path = source / ledger['checkpoints'][-1]['evidence_file']
            raw = evidence_path.read_bytes()
            evidence = json.loads(raw)
            claims, sources = claim_packet('positioning', evidence)
            review = {'stage': 'positioning', 'reviewer': 'Test fixture', 'status': 'passed', 'action': 'continue',
                'reasons': [], 'claims': claims, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'source_catalog': {f'S{i}': key for i, key in enumerate(sources, 1)},
                'checks': [{'claim_id': key, 'verdict': 'proposal', 'explanation': 'Fixture', 'evidence': [], 'correction': ''}
                           for key in claims]}
            evaluation = root / 'evaluation'
            evaluation.mkdir()
            (evaluation / 'evaluation-config.json').write_text(json.dumps({'stage': 'positioning', 'source_run': str(source)}))
            snapshot = evaluation / 'case-1-evidence.json'
            snapshot.write_bytes(raw)
            control = {'status': 'passed', 'action': 'continue', 'expected_control_action': 'continue'}
            result_path = evaluation / 'evaluation.json'
            good_result = json.dumps({'status': 'complete', 'cases': [review, control]})
            result_path.write_text(good_result)
            with patch('httpx.AsyncClient', side_effect=AssertionError('No API calls permitted')):
                recovered = recover_advisory(evaluation, root / 'output')
            state = json.loads((recovered / 'release.json').read_text())
            self.assertTrue(state['eligible_for_delivery'])
            self.assertEqual(state['status'], 'released')
            self.assertTrue((recovered / state['report']).exists())
            self.assertEqual(evidence_path.read_bytes(), raw)
            self.assertEqual(ledger_path.read_text(), ledger_text)

            for mode in ('held', 'missing_claim', 'defect', 'control', 'snapshot', 'source_changed', 'research_held', 'mock'):
                with self.subTest(mode=mode):
                    result = json.loads(good_result)
                    if mode == 'held':
                        result['cases'][0]['status'] = 'held'
                    elif mode == 'missing_claim':
                        result['cases'][0]['checks'].pop()
                    elif mode == 'defect':
                        result['cases'][0]['checks'][0].update(verdict='unsupported', correction='Remove fact')
                    elif mode == 'control':
                        result['cases'][1]['action'] = 'hold'
                    elif mode == 'snapshot':
                        snapshot.write_text('{}')
                    elif mode == 'source_changed':
                        evidence_path.write_text('{}')
                    elif mode == 'research_held':
                        changed = json.loads(ledger_text)
                        next(r for r in changed['checkpoints'] if r['stage'] == 'competitive_research')['status'] = 'held'
                        ledger_path.write_text(json.dumps(changed))
                    elif mode == 'mock':
                        run_path = source / 'run.json'
                        run_text = run_path.read_text()
                        run_config = json.loads(run_text)
                        run_config['mock'] = True
                        run_path.write_text(json.dumps(run_config))
                    result_path.write_text(json.dumps(result))
                    with self.assertRaises((ValueError, RuntimeError)):
                        recover_advisory(evaluation, root / 'rejected')
                    self.assertFalse((root / 'rejected').exists())
                    snapshot.write_bytes(raw)
                    evidence_path.write_bytes(raw)
                    ledger_path.write_text(ledger_text)
                    if mode == 'mock':
                        run_path.write_text(run_text)
