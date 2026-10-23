"""Release an unchanged saved advisory after a successful positioning evaluation."""
from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

from pydantic import Field

from .advantage import AdvantagePlan, CompetitorScan, ToolScan, UsefulTool, render_advantage, validate_sources, _text
from .evidence_review import EvidenceReview, claim_packet, validate_review, write_review_report
from .pipeline import _save
from .report import render_html
from .schemas import Brief


class AdvisoryTools(ToolScan):
    # Filtering held tools may legitimately leave no tool recommendations.
    tools: list[UsefulTool] = Field(default_factory=list)


def _read(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Saved evidence path escapes its run directory.')
    return json.loads(path.read_text(encoding='utf-8'))


def recover_advisory(evaluation_dir, out_root):
    """No search, model calls or edits to the source run. Never release an unreviewed revision."""
    evaluation_dir = Path(evaluation_dir).resolve()
    metadata = _read(evaluation_dir, 'evaluation-config.json')
    evaluation = _read(evaluation_dir, 'evaluation.json')
    if metadata.get('stage') != 'positioning' or evaluation['status'] != 'complete':
        raise ValueError('A complete positioning evaluation is required.')
    controls = [r for r in evaluation['cases'] if r.get('expected_control_action') is not None]
    if not controls or any(r['action'] != r['expected_control_action'] or r['status'] == 'error' for r in controls):
        raise ValueError('Evaluation controls did not pass.')
    cases = [(i, r) for i, r in enumerate(evaluation['cases'], 1) if r.get('expected_control_action') is None]
    if len(cases) != 1:
        raise ValueError('Recovery requires exactly one reviewed positioning draft.')
    index, review = cases[0]
    if review['status'] != 'passed' or review['action'] != 'continue':
        raise ValueError('The positioning draft has not passed automated review.')

    source_run = Path(metadata['source_run']).resolve()
    run = _read(source_run, 'run.json')
    if run.get('mock') or run['config']['guardrails'].get('advantage_review_mode') != 'observe':
        raise ValueError('Recovery requires a live run with structured evidence review.')
    ledger = _read(source_run, 'supervision.json')
    positions = [r for r in ledger['checkpoints'] if r['stage'] == 'positioning']
    if not positions:
        raise ValueError('No saved positioning draft.')
    path = (source_run / positions[-1]['evidence_file']).resolve()
    if not path.is_relative_to(source_run):
        raise ValueError('Saved evidence path escapes its run directory.')
    if hashlib.sha256(path.read_bytes()).hexdigest() != review['source_sha256']:
        raise ValueError('The saved draft changed or the evaluation reviewed a different draft.')
    evidence = _read(evaluation_dir, f'case-{index}-evidence.json')
    if evidence != json.loads(path.read_text(encoding='utf-8')):
        raise ValueError('Evaluation snapshot differs from the saved draft.')
    review = dict(review)
    if positions[-1].get('jev_observation'):
        review['jev_observation'] = {**positions[-1]['jev_observation'], 'historical': True}
    claims, sources = claim_packet('positioning', evidence)
    catalog = {f'S{i}': key for i, key in enumerate(sources, 1)}
    if review['claims'] != claims or review['source_catalog'] != catalog:
        raise ValueError('Review does not describe this evidence snapshot.')
    parsed = EvidenceReview.model_validate({'checks': review['checks']})
    validate_review(parsed, claims, {key: sources[url] for key, url in catalog.items()})
    if any(c.verdict in ('unsupported', 'contradicted') for c in parsed.checks):
        raise ValueError('Unresolved factual defects remain in the review.')

    # A passing final review must not revive research excluded by the original run.
    latest = {}
    for row in ledger['checkpoints']:
        if row['stage'] == 'competitive_research' and row.get('review_item'):
            item = row['review_item']
            latest[(item['kind'], item['name'])] = row
    research = evidence['research']
    for kind in ('competitors', 'tools'):
        for item in research[kind][kind]:
            row = latest.get((kind, item['name']), {})
            if row.get('status') != 'passed' or row.get('action') != 'continue':
                raise ValueError(f'Research item was not passed: {item["name"]}')
            saved = _read(source_run, row['evidence_file'])
            if item not in saved['research'][kind][kind]:
                raise ValueError('Research item differs from its passed evidence.')
    brief = Brief.model_validate(evidence['brief'])
    plan = AdvantagePlan.model_validate(evidence['plan'])
    research = {**research, 'competitors': CompetitorScan.model_validate(research['competitors']),
                'tools': AdvisoryTools.model_validate(research['tools'])}
    for items in (research['competitors'].competitors, research['tools'].tools, plan.opportunities):
        validate_sources(items, research['sources'])
    preliminary = any(r['stage'] == 'competitive_research' and r['action'] != 'continue' for r in ledger['checkpoints'])
    disposition = source_run / 'research-disposition.json'
    excluded = _read(source_run, disposition.name).get('excluded', []) if disposition.exists() else []
    markdown = render_advantage(brief, research, plan, review, False)
    if preliminary:
        markdown = '> **PRELIMINARY ADVISORY:** Uses passed research only. Coverage is incomplete; excluded findings are listed below.\n\n' + markdown
        markdown += '\n\n## Excluded research\n\n' + '\n'.join(
            f'- {_text(row["name"])}: excluded from recommendation evidence.' for row in excluded)
    destination = Path(out_root) / f'advisory-recovered-{secrets.token_hex(4)}'
    destination.mkdir(parents=True)
    name = 'preliminary-advantage' if preliminary else 'advantage'
    (destination / f'{name}.md').write_text(markdown, encoding='utf-8')
    (destination / f'{name}.html').write_text(render_html(markdown, 'Competitive positioning advisory'), encoding='utf-8')
    _save(destination, 'advisory-evidence.json', evidence)
    _save(destination, 'guardrails.json', review)
    _save(destination, 'recovery.json', {'source_run': str(source_run), 'evaluation_run': str(evaluation_dir),
         'evidence_sha256': review['source_sha256'], 'no_new_searches': True, 'no_new_model_calls': True})
    write_review_report(destination, [review])
    _save(destination, 'release.json', {'status': 'released_with_limits' if preliminary else 'released',
         'eligible_for_delivery': True, 'automatic': True, 'human_review_required': False,
         'review_mode': 'observe', 'report': f'{name}.html', 'excluded_entries': excluded, 'reasons': []})
    return destination
