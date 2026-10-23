import asyncio
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scrapling import Selector

from kohort.config import ScrapingCfg, load_config
from kohort.scraping import enrich_sources, fetch_public, page_text, public_url, scrape_page


def page(body, status=200, headers=None):
    selector = Selector(body)
    return SimpleNamespace(body=body.encode(), status=status,
        headers=headers or {'content-type': 'text/html'}, css=selector.css, get_all_text=selector.get_all_text)


class ScrapingTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_urls_reject_local_dns_and_credentials(self):
        resolver = AsyncMock(return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 80))])
        with patch.object(asyncio.get_running_loop(), 'getaddrinfo', resolver):
            for url in (
                'file:///secret',
                'gopher://secret',
                'http://localhost/',
                'https://user:password@example.com/',
                'https://example.com:9999/',
                'http://127.0.0.1/',
                'http://169.254.169.254/latest/meta-data/',
                'http://100.100.100.200/',
                'http://[::ffff:127.0.0.1]/',
                'http://[::1]/',
                'http://10.0.0.1/',
                'http://192.168.1.1/',
                'https://private-alias.example/',
            ):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    await public_url(url)
        resolver.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))]
        with patch.object(asyncio.get_running_loop(), 'getaddrinfo', resolver):
            self.assertEqual(await public_url('https://example.com/pricing#plans'), 'https://example.com/pricing')

    def test_extraction_retains_terms_and_excludes_scripts_and_navigation(self):
        html = '<html><head><title>Pricing</title></head><body><nav>Navigation</nav><h1>Standard</h1><p>R100 per month. Hosting is included. VAT is excluded. Custom work is quoted separately and is not part of this package.</p><script>invented price R5</script></body></html>'
        title, text, truncated = page_text(Selector(html), 8000)
        self.assertEqual(title, 'Pricing')
        self.assertIn('VAT is excluded', text)
        self.assertNotIn('invented', text)
        self.assertNotIn('Navigation', text)
        self.assertFalse(truncated)
        self.assertTrue(page_text(Selector(html), 80)[2])
        with self.assertRaises(ValueError):
            page_text(Selector('<title>Just a moment...</title><p>' + 'wait ' * 30 + '</p>'), 8000)

    async def test_redirect_rechecks_url_and_robots_before_fetch(self):
        async def validate(url):
            if '127.0.0.1' in url:
                raise ValueError('private redirect')
            return url
        fetch = AsyncMock(return_value=page('', 302, {'location': 'http://127.0.0.1/admin'}))
        with patch('kohort.scraping.public_url', side_effect=validate), patch('scrapling.fetchers.AsyncFetcher.get', fetch):
            with self.assertRaisesRegex(ValueError, 'private redirect'):
                await fetch_public('https://example.com/', 1)
        self.assertEqual(fetch.await_count, 1)
        fetch.reset_mock()
        fetch.return_value = page('', 302, {'location': 'https://other.example/private'})
        with patch('kohort.scraping.public_url', side_effect=validate), patch('scrapling.fetchers.AsyncFetcher.get', fetch), patch('kohort.scraping.allowed_by_robots', AsyncMock(side_effect=[True, False])):
            with self.assertRaisesRegex(ValueError, 'disallows'):
                await fetch_public('https://example.com/', 1, check_robots=True)
        self.assertEqual(fetch.await_count, 1)

    async def test_robots_missing_denied_and_unavailable(self):
        from kohort.scraping import allowed_by_robots
        for response, expected in ((page('', 404), True),
                                   (page('User-agent: *\nDisallow: /private'), False),
                                   (page('', 503), None)):
            with patch('kohort.scraping.fetch_public', AsyncMock(return_value=(response, 'https://example.com/robots.txt'))):
                if expected is None:
                    with self.assertRaises(ValueError):
                        await allowed_by_robots('https://example.com/private', 1)
                else:
                    self.assertEqual(await allowed_by_robots('https://example.com/private', 1), expected)

    async def test_capture_metadata_and_reject_non_html(self):
        response = page('<title>Service</title><p>' + 'Actual service terms. ' * 10 + '</p>')
        with patch('kohort.scraping.public_url', AsyncMock(side_effect=lambda url: url)), patch('kohort.scraping.fetch_public', AsyncMock(return_value=(response, 'https://example.com/'))):
            record = await scrape_page('https://example.com/', ScrapingCfg())
            self.assertEqual(record['status'], 'fetched')
            self.assertEqual(len(record['sha256']), 64)
            self.assertIn('retrieved_at', record)
            response.headers = {'content-type': 'application/pdf'}
            with self.assertRaisesRegex(ValueError, 'Unsupported page type'):
                await scrape_page('https://example.com/', ScrapingCfg())

    async def test_enrichment_cache_limits_failure_and_original_evidence(self):
        sources = [{'url': f'https://example.com/{i}', 'content': f'Search {i}', 'title': 'Old'} for i in range(3)]
        success = {'status': 'fetched', 'content': 'Saved page evidence', 'title': 'New', 'retrieved_at': '2026-10-13', 'final_url': sources[0]['url']}
        cfg = ScrapingCfg(enabled=True, max_pages_per_scan=2)
        with tempfile.TemporaryDirectory() as directory, patch('kohort.scraping.scrape_page', AsyncMock(side_effect=[success, TimeoutError('slow page')])) as fetch:
            root = Path(directory)
            result = await enrich_sources(sources, cfg, root, 'test')
            self.assertEqual(fetch.await_count, 2)
            self.assertEqual(result[0]['content'], 'Search 0\n\nSaved page evidence')
            self.assertEqual(result[0]['search_content'], 'Search 0')
            self.assertEqual(result[1]['content'], 'Search 1')
            self.assertEqual(result[1]['scraping']['status'], 'unavailable')
            self.assertEqual(result[2]['scraping']['status'], 'skipped_limit')
            again = await enrich_sources(sources, cfg, root, 'second')
            self.assertEqual(again, result)
            self.assertEqual(fetch.await_count, 2)
            self.assertEqual(sources[0]['content'], 'Search 0')
            disabled = await enrich_sources(sources, ScrapingCfg(enabled=False), root, 'off')
            self.assertEqual(disabled, sources)

    async def test_scan_rewrite_receives_scraped_evidence_before_validation(self):
        from kohort.advantage import _scan, CompetitorScan
        from kohort.pipeline import load_brief
        from tests.test_advantage import FixtureAdvantageLLM, ROOT, SOURCE
        from kohort.llm import CostTracker
        cfg = load_config()
        llm = FixtureAdvantageLLM(cfg, CostTracker(1))
        llm.mode = 'no_excerpts'
        cfg.scraping.enabled = True
        record = {'status': 'fetched', 'title': 'Fetched source', 'content': SOURCE['content']}
        try:
            with tempfile.TemporaryDirectory() as directory, patch('kohort.scraping.scrape_page', AsyncMock(return_value=record)):
                _, sources = await _scan(llm, cfg, load_brief(ROOT / 'briefs/example_loopa.yaml'), 'competitive_scan', CompetitorScan, 'Research', Path(directory))
                rewrite = json.loads(llm.requests[-1]['user'])
                self.assertIn(SOURCE['content'], rewrite['sources'][0]['content'])
                self.assertEqual(sources[0]['scraping']['status'], 'fetched')
        finally:
            await llm.aclose()
