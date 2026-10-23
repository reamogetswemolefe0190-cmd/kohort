import json
import tempfile
import unittest
from pathlib import Path

from kohort.report_explorer import build_report, load_report, safe_markdown, safe_url


class ReportExplorerTests(unittest.TestCase):
    def fixture(self, root):
        run = root / 'run'
        run.mkdir()
        records = {
            'source-advisory.json': {'brief': {'client': 'Example', 'product_name': 'Example service'},
                'research': {'competitors': {'competitors': [], 'coverage_limits': ['Incomplete']},
                             'tools': {'tools': []}, 'sources': []}, 'plan': {'caveats': ['Synthetic only']}},
            'analysis.json': {'completed': 2, 'scenarios': {'S1': {'completed': 2, 'requested': 2,
                'audience_group': 'primary', 'options': [{'option_id': 'O1', 'chosen': 1}], 'keep_current': 1}}},
            'responses.json': [{'persona_id': 'P1', 'segment_id': 'S1', 'model': 'test', 'response': {'choice': 'O1'}},
                               {'persona_id': 'P2', 'segment_id': 'S1', 'model': 'test', 'response': {'choice': 'keep_current'}}],
            'buyer-offers.json': [{'option_id': 'O1', 'message': '</script><script>alert(1)</script>'}],
            'offer-boards.json': [{'option_id': 'O1'}],
            'panel-design.json': {'segments': [{'role': 'Buyer'}]},
            'release.json': {'status': 'released_with_limits', 'eligible_for_delivery': True},
            'run.json': {'swarm_protocol': 4, 'mock': False, 'requested_personas': 2, 'source_run': 'saved-advice'},
        }
        for name, value in records.items():
            (run / name).write_text(json.dumps(value), encoding='utf-8')
        brief = root / 'brief.md'
        brief.write_text('# Example\n\n## Our recommendation\nChoose deliberately.\n\n## What to do next\nTest the offer.', encoding='utf-8')
        return run, brief

    def test_counts_and_provenance_use_saved_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, brief = self.fixture(Path(tmp))
            report = load_report(run, brief)
            self.assertEqual(report['completed'], 2)
            self.assertEqual(report['scenarios'][0]['stats']['keep_current'], 1)
            self.assertEqual(len(report['files']['responses.json']), 64)
            self.assertEqual(report['sections'][1]['title'], 'What to do next')

    def test_inconsistent_saved_counts_stop_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, brief = self.fixture(Path(tmp))
            path = run / 'analysis.json'
            data = json.loads(path.read_text())
            data['scenarios']['S1']['options'][0]['chosen'] = 2
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                load_report(run, brief)

    def test_workflow_does_not_promote_mock_or_draft_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, brief = self.fixture(Path(tmp))
            report = load_report(run, brief)
            self.assertEqual(report['workflow'][1]['status'], 'No source records')
            self.assertEqual(report['workflow'][4]['status'], 'Eligible with limitations')
            self.assertEqual(report['workflow'][-1]['status'], 'Not included in this report')
            run_file = run / 'run.json'
            record = json.loads(run_file.read_text())
            record['mock'] = True
            run_file.write_text(json.dumps(record))
            report = load_report(run, brief)
            self.assertEqual(report['workflow'][3]['status'], 'Mock data')
            self.assertEqual(report['workflow'][4]['status'], 'Draft only')
            record['mock'] = False
            run_file.write_text(json.dumps(record))
            (run / 'release.json').write_text(json.dumps({'status': 'held', 'eligible_for_delivery': False}))
            self.assertEqual(load_report(run, brief)['workflow'][4]['status'], 'Draft only')

    def test_untrusted_content_cannot_break_out_of_json_or_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, brief = self.fixture(Path(tmp))
            out = build_report(run, brief, Path(tmp) / 'out' / 'index.html')
            html = out.read_text(encoding='utf-8')
            self.assertNotIn('</script><script>alert(1)</script>', html)
            self.assertIn('\\u003c/script\\u003e', html)
        safe = safe_markdown('<img src=x onerror="alert(1)">\n\n[bad](javascript:alert(1))\n\n**Safe**')
        self.assertNotIn('<img', safe)
        self.assertNotIn('href=', safe)
        self.assertIn('<strong>Safe</strong>', safe)
        self.assertEqual(safe_url('javascript:alert(1)'), '')
        self.assertEqual(safe_url('https://user:secret@example.com'), '')

    def test_original_run_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, brief = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'outside'):
                build_report(run, brief, run / 'analysis.json')


if __name__ == '__main__':
    unittest.main()
