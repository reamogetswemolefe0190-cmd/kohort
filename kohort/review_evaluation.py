"""Replay frozen evidence through the structured reviewer without rerunning searches."""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from pathlib import Path

from .evidence_review import review_evidence, write_review_report
from .llm import CostTracker, LLM
from .pipeline import _save


async def evaluate_saved_run(run_dir, cfg, out_root, console, *, budget=.25, limit=3, item=None,
                             stage='competitive_research'):
    run_dir = Path(run_dir).resolve()
    ledger = json.loads((run_dir / 'supervision.json').read_text(encoding='utf-8'))
    selected = []
    # First attempts preserve the original landscape before failed whole-scan repairs.
    for row in ledger['checkpoints']:
        if row.get('stage') != stage or row.get('attempt') != 1:
            continue
        if item and row.get('review_item', {}).get('name') != item:
            continue
        selected.append(row)
    # Include the first item, a later disputed example, and coverage when available.
    if len(selected) > limit:
        indexes = [0, (len(selected) - 1) // 2, len(selected) - 1]
        selected = [selected[i] for i in dict.fromkeys(indexes)][:limit]
    if not selected:
        raise ValueError(f'No saved {stage} checkpoints found.')
    destination = Path(out_root) / f'review-evaluation-{secrets.token_hex(4)}'
    destination.mkdir(parents=True)
    tracker = CostTracker(budget)
    cfg = cfg.model_copy(deep=True)
    cfg.run.max_retries = min(cfg.run.max_retries, 1)
    llm = LLM(cfg, tracker)
    records = []
    _save(destination, 'evaluation-config.json', {'source_run': str(run_dir), 'config': cfg.model_dump(),
         'budget_usd': budget, 'selected_item': item, 'stage': stage,
         'no_new_searches': True, 'historical_jev_reused': True})
    cases = []
    for row in selected:
        path = (run_dir / row['evidence_file']).resolve()
        if not path.is_relative_to(run_dir):
            raise ValueError('Evidence path escapes the saved run.')
        raw = path.read_bytes()
        cases.append((row.get('review_item', {}).get('name', row['evidence_file']),
                      json.loads(raw), row, hashlib.sha256(raw).hexdigest(), None))
    # Explicit artificial controls, not real market assertions or human-labelled saved cases.
    for label, price, expected in [('supported-control', 'R100 per month.', 'continue'),
                                    ('contradicted-control', 'R10 per month.', 'hold'),
                                    ('unknown-control', 'Conversion from free to paid is unproven; no buyer data is supplied.', 'continue'),
                                    ('proposal-control', 'Hypothesis: buyers may prefer one supplier. This is not established; test with real buyers.', 'continue'),
                                    ('false-premise-control', 'Hypothesis: offer support because Rival charges R10 per month.', 'hold')]:
        evidence = {'brief': {'client': 'TEST FIXTURE', 'category': 'Synthetic evaluation only'},
                    'research': {'pricing': price},
                    'sources': [{'url': 'https://example.com/control', 'content': 'Rival charges R100 per month.'}]}
        cases.append((label, evidence, None, None, expected))
    try:
        for index, (label, evidence, historical, fingerprint, expected) in enumerate(cases, 1):
            console.print(f'  Reviewing saved case {index}/{len(cases)}: {label}')
            _save(destination, f'case-{index}-evidence.json', evidence)
            start = time.monotonic()
            result = await review_evidence(llm, cfg, stage if expected is None else 'competitive_research', evidence)
            result.update(review_item={'kind': 'evaluation', 'name': label},
                          source_sha256=fingerprint, elapsed_seconds=round(time.monotonic() - start, 2),
                          expected_control_action=expected)
            if historical:
                observation = historical.get('jev_observation', historical)
                result['jev_observation'] = {'status': observation['status'],
                    'reasons': observation.get('reasons', []), 'probabilities': observation.get('probabilities', {}),
                    'historical': True, 'model': observation.get('response', {}).get('model', observation.get('model'))}
            records.append(result)
            _save(destination, f'case-{index}-review.json', result)
    finally:
        _save(destination, 'costs.json', tracker.summary())
        _save(destination, 'evaluation.json', {
            'status': 'complete' if len(records) == len(cases) else 'incomplete',
            'purpose': 'reviewer_evaluation', 'independent_labels_available': False,
            'limitation': 'Saved cases have no independent human labels. Disagreement is not a false-positive rate. '
                          'Artificial controls test obvious defects only, not broad research accuracy.',
            'controls_passed': sum(r['action'] == r['expected_control_action'] for r in records
                                   if r.get('expected_control_action')),
            'controls_total': sum(expected is not None for _, _, _, _, expected in cases), 'cases': records})
        try:
            report = write_review_report(destination, records)
            console.print(f'  Evaluation: {report}')
        finally:
            await llm.aclose()
    return destination
