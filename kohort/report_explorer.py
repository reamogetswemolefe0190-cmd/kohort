"""Build an offline interactive report from a saved swarm and an editorial brief.

Usage: python -m kohort.report_explorer RUN --brief CLIENT.md --out REPORT.html
No model calls, source refetching, or changes to the original run.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import markdown


class SafeHTML(HTMLParser):
    tags = {'p', 'strong', 'em', 'blockquote', 'ul', 'ol', 'li', 'table', 'thead',
            'tbody', 'tr', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'hr', 'br', 'code', 'pre'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.tags:
            self.parts.append(f'<{tag}>')

    def handle_endtag(self, tag):
        if tag in self.tags and tag not in {'hr', 'br'}:
            self.parts.append(f'</{tag}>')

    def handle_data(self, data):
        self.parts.append(html.escape(data))


def safe_markdown(text):
    parser = SafeHTML()
    parser.feed(markdown.markdown(text, extensions=['tables']))
    return ''.join(parser.parts)


def safe_url(value):
    try:
        parsed = urlsplit(str(value))
        return str(value) if parsed.scheme in {'https', 'http'} and parsed.hostname and not parsed.username else ''
    except ValueError:
        return ''


def brief_sections(text):
    parts = re.split(r'^## (.+)$', text, flags=re.M)
    return [{'title': parts[i].strip(), 'html': safe_markdown(parts[i + 1])}
            for i in range(1, len(parts), 2)] or [{'title': 'Recommendation', 'html': safe_markdown(text)}]


def workflow_record(source, run, release, responses, offers, scenarios):
    """Describe saved artifacts, never infer live execution or customer validation."""
    research = source['research']
    competitors = research.get('competitors', {}).get('competitors', [])
    sources = research.get('sources', [])
    mock = bool(run.get('mock'))
    eligible = release.get('eligible_for_delivery') is True and not mock
    source_release = run.get('source_release', {})
    return [
        {'title': 'Business brief', 'status': 'Owner supplied', 'target': 'recommendation',
         'summary': 'The offer and intended buyers come from the supplied business brief. Capabilities are owner-reported.',
         'artifacts': ['source-advisory.json'], 'notes': []},
        {'title': 'Market research', 'status': 'Saved evidence' if sources else 'No source records', 'target': 'evidence',
         'summary': f'{len(competitors)} buyer alternatives and {len(sources)} source records in the saved advisory. Coverage is bounded.',
         'artifacts': ['source-advisory.json'],
         'notes': research.get('competitors', {}).get('coverage_limits', [])},
        {'title': 'Offers and buying scenarios', 'status': 'Saved comparison design', 'target': 'options',
         'summary': f'{len(offers)} messages compared across {len(scenarios)} scenarios. Scenarios are separate perspectives, not population weights.',
         'artifacts': ['buyer-offers.json', 'offer-boards.json', 'panel-design.json'], 'notes': []},
        {'title': 'Synthetic buyer responses', 'status': 'Mock data' if mock else 'Saved synthetic results', 'target': 'responses',
         'summary': f'{len(responses)} responses saved; {run.get("requested_personas", "unrecorded")} requested. Counts reconcile with the response records.',
         'artifacts': ['responses.json', 'analysis.json'], 'notes': []},
        {'title': 'Automated release review', 'status': 'Eligible with limitations' if eligible and release.get('status') == 'released_with_limits' else ('Eligible' if eligible else 'Draft only'),
         'target': 'method', 'summary': 'The saved release decision applies to the swarm output. It does not independently approve the editorial brief or validate demand.',
         'artifacts': ['release.json', 'run.json'],
         'notes': [f'Saved swarm status: {release.get("status", "unrecorded")}.',
                   f'Saved source advisory status: {source_release.get("status", "unrecorded")}.']
                  + list(release.get('scenario_review_issues', []))},
        {'title': 'Positioning recommendation', 'status': 'Editorial brief supplied', 'target': 'actions',
         'summary': 'A supplied editorial brief translates the findings into positioning, copy and next actions. This report build does not generate new advice.',
         'artifacts': ['editorial-brief.md'], 'notes': []},
        {'title': 'Customer validation', 'status': 'Not included in this report', 'target': 'actions',
         'summary': 'Use the recommended actions to collect real prospect feedback and quote outcomes. Synthetic selections do not establish customer demand.',
         'artifacts': [], 'notes': []},
    ]


def load_report(run_dir, brief_path):
    root, brief_path = Path(run_dir), Path(brief_path)
    files = {}

    def read(name):
        path = root / name
        raw = path.read_bytes()
        files[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode('utf-8'))

    source = read('source-advisory.json')
    analysis = read('analysis.json')
    responses = read('responses.json')
    offers = read('buyer-offers.json')
    boards = read('offer-boards.json')
    panel = read('panel-design.json')
    release = read('release.json')
    run = read('run.json')
    if run.get('swarm_protocol') != 4:
        raise ValueError('Report explorer requires swarm protocol 4 with separate scenario results.')
    if not analysis.get('scenarios'):
        raise ValueError('No saved scenario results are available.')
    option_ids = [o['option_id'] for o in offers]
    if len(option_ids) != len(set(option_ids)):
        raise ValueError('Duplicate offer IDs in saved data.')
    if len({r['persona_id'] for r in responses}) != len(responses):
        raise ValueError('Duplicate synthetic respondent IDs in saved data.')
    scenario_ids = set(analysis['scenarios'])
    if any(r['segment_id'] not in scenario_ids for r in responses):
        raise ValueError('Response refers to a missing scenario.')
    scenarios = []
    for sid, stats in analysis['scenarios'].items():
        rows = [r for r in responses if r['segment_id'] == sid]
        counts = Counter(r['response']['choice'] for r in rows)
        if set(counts) - set(option_ids) - {'keep_current'}:
            raise ValueError(f'Unknown offer selected in {sid}.')
        saved = {o['option_id']: o['chosen'] for o in stats['options']}
        if (set(saved) != set(option_ids) or len(rows) != stats['completed']
                or counts['keep_current'] != stats['keep_current']
                or any(counts[oid] != saved[oid] for oid in option_ids)):
            raise ValueError(f'Saved counts disagree with responses in {sid}.')
        # Protocol 4 assigns S1, S2, ... by panel order, not JSON key order.
        if not re.fullmatch(r'S[1-9][0-9]*', sid):
            raise ValueError(f'Invalid protocol-4 scenario ID: {sid}')
        segment = panel['segments'][int(sid[1:]) - 1]
        scenarios.append({'id': sid, 'segment': segment, 'stats': stats})
    if len(responses) != analysis['completed']:
        raise ValueError('Saved completion total disagrees with responses.')
    research = source['research']
    sources = [{**s, 'url': safe_url(s.get('url', ''))} for s in research.get('sources', [])]
    raw_brief = brief_path.read_bytes()
    files['editorial-brief.md'] = hashlib.sha256(raw_brief).hexdigest()
    return {
        'client': source['brief']['client'], 'product': source['brief']['product_name'],
        'runId': root.name, 'sourceRun': Path(run.get('source_run', '')).name,
        'builtAt': datetime.now(timezone.utc).isoformat(), 'release': release,
        'mock': bool(run.get('mock')), 'completed': len(responses),
        'requested': run.get('requested_personas'), 'protocol': run['swarm_protocol'],
        'sections': brief_sections(raw_brief.decode('utf-8')),
        'offers': offers, 'boards': boards, 'scenarios': scenarios, 'responses': responses,
        'competitors': research.get('competitors', {}).get('competitors', []),
        'coverage': research.get('competitors', {}).get('coverage_limits', []),
        'tools': research.get('tools', {}).get('tools', []), 'sources': sources,
        'caveats': source['plan'].get('caveats', []),
        'models': sorted({r.get('model', 'Unrecorded') for r in responses}),
        'workflow': workflow_record(source, run, release, responses, offers, scenarios),
        'files': files,
    }


def build_report(run_dir, brief_path, output):
    data = load_report(run_dir, brief_path)
    template = Path(__file__).with_name('report_explorer.html').read_text(encoding='utf-8')
    encoded = json.dumps(data, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    document = template.replace('__CLIENT_TITLE__', html.escape(data['client'])).replace('__REPORT_DATA__', encoded)
    output = Path(output)
    if output.resolve() == Path(brief_path).resolve() or output.resolve().parent == Path(run_dir).resolve():
        raise ValueError('Choose an output outside the original run and brief source file.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--brief', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    try:
        print(build_report(args.run, args.brief, args.out))
    except (ValueError, KeyError, OSError, IndexError) as exc:
        parser.exit(1, f'Report could not be built: {exc}\n')


if __name__ == '__main__':
    main()
