"""Bounded public-page retrieval; preserves search evidence when fetching fails."""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from .config import ScrapingCfg

USER_AGENT = 'KohortResearch/1.0'


def validate_ip_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """Strictly assert that an IP address is publicly routable on the global internet.

    Blocks all private networks (RFC 1918), loopbacks (127.0.0.0/8, ::1),
    link-local addresses (169.254.0.0/16, fe80::/10), cloud metadata endpoints
    (169.254.169.254, 100.100.100.200, fd00:ec2::254), multicast, broadcast,
    and IPv4-mapped IPv6 evasions.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
        or str(ip) in ("169.254.169.254", "100.100.100.200", "0.0.0.0", "255.255.255.255")
    ):
        raise ValueError(f"Source resolves to a non-public or internal address: {ip}")


async def public_url(url: str) -> str:
    """Reject local resources, credentials, internal subnets, and non-web ports, including DNS aliases."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    parts = urlsplit(url.strip())
    if parts.scheme not in ('http', 'https'):
        raise ValueError('Only public HTTP and HTTPS protocols are supported.')

    if not parts.hostname:
        raise ValueError('URL must include a valid hostname.')

    if parts.username or parts.password:
        raise ValueError('Embedded credentials in URLs are not supported.')

    if parts.port not in (None, 80, 443):
        raise ValueError('Only standard web ports (80, 443) are supported.')

    host = parts.hostname.strip().lower().rstrip('.')
    if host == 'localhost' or host.endswith(('.local', '.localhost', '.internal', '.lan', '.home')):
        raise ValueError('Local and private hostnames are not supported.')

    # Check for direct IP address literal in host
    clean_host = host.strip('[]')
    try:
        literal_ip = ipaddress.ip_address(clean_host)
        validate_ip_address(literal_ip)
    except ValueError as exc:
        if 'Source resolves to a non-public' in str(exc):
            raise

    # Resolve all DNS records
    target_port = parts.port or (443 if parts.scheme == 'https' else 80)
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            host, target_port, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host '{host}': {exc}")

    if not addresses:
        raise ValueError(f"Host '{host}' did not resolve to any addresses.")

    for row in addresses:
        resolved_ip_str = row[4][0]
        ip = ipaddress.ip_address(resolved_ip_str)
        validate_ip_address(ip)

    return urlunsplit((parts.scheme, parts.netloc, parts.path or '/', parts.query, ''))


async def fetch_public(url, timeout, *, check_robots=False):
    from scrapling.fetchers import AsyncFetcher
    current = url
    for _ in range(4):
        current = await public_url(current)
        if check_robots and not await allowed_by_robots(current, timeout):
            raise ValueError('robots.txt disallows this page.')
        response = await AsyncFetcher.get(current, timeout=timeout, retries=1,
            follow_redirects=False, stealthy_headers=False, impersonate=None,
            headers={'User-Agent': USER_AGENT, 'Accept': 'text/html,text/plain;q=0.9'},
            selector_config={'adaptive': False})
        cl = response.headers.get('content-length') or response.headers.get('Content-Length')
        if cl:
            try:
                if int(cl) > 2_000_000:
                    raise ValueError('Response exceeds 2 MB processing limit.')
            except ValueError:
                pass
        if response.status not in (301, 302, 303, 307, 308):
            return response, current
        location = response.headers.get('location') or response.headers.get('Location')
        if not location:
            raise ValueError('Redirect has no destination.')
        current = urljoin(current, location)
    raise ValueError('Too many redirects.')


async def allowed_by_robots(url, timeout):
    parts = urlsplit(url)
    robots_url = urlunsplit((parts.scheme, parts.netloc, '/robots.txt', '', ''))
    response, _ = await fetch_public(robots_url, timeout)
    if response.status in (404, 410):
        return True
    if response.status != 200:
        raise ValueError(f'Robots policy unavailable (HTTP {response.status}); page skipped.')
    raw = response.body.decode('utf-8', errors='replace')
    if len(raw) > 500_000 or '<html' in raw[:500].lower():
        raise ValueError('Robots policy could not be read; page skipped.')
    policy = RobotFileParser()
    policy.parse(raw.splitlines())
    return policy.can_fetch(USER_AGENT, url)


def page_text(page, limit):
    title = str(page.css('title::text').get() or '').strip()
    text = str(page.get_all_text(separator='\n', strip=True,
        ignore_tags=('script', 'style', 'noscript', 'svg', 'nav', 'footer', 'head')))
    lines = [' '.join(line.split()) for line in text.splitlines() if line.strip()]
    clean = '\n'.join(lines)
    # Do not treat an HTTP-200 challenge page as product evidence.
    probe = (title + ' ' + clean[:1500]).lower()
    if any(marker in probe for marker in ('just a moment...', 'verify you are human', 'checking your browser', 'enable javascript and cookies to continue')):
        raise ValueError('Challenge page returned; kept search evidence.')
    if len(clean) < 80:
        raise ValueError('Page has too little readable text; it may require JavaScript.')
    return title, clean[:limit], len(clean) > limit


async def scrape_page(url, cfg):
    async with asyncio.timeout(cfg.timeout_seconds * 2 + 5):
        url = await public_url(url)
        page, final_url = await fetch_public(url, cfg.timeout_seconds, check_robots=True)
        if page.status != 200:
            raise ValueError(f'HTTP {page.status}; kept search evidence.')
        content_type = (page.headers.get('content-type') or page.headers.get('Content-Type') or '').lower()
        if not any(kind in content_type for kind in ('text/html', 'application/xhtml+xml', 'text/plain')):
            raise ValueError('Unsupported page type; HTML or plain text required.')
        if len(page.body) > 2_000_000:
            raise ValueError('Page exceeds the 2 MB processing limit.')
        title, content, truncated = page_text(page, cfg.max_chars_per_page)
        return {'status': 'fetched', 'url': url, 'final_url': final_url, 'title': title,
                'retrieved_at': datetime.now(timezone.utc).isoformat(), 'content': content,
                'truncated': truncated, 'sha256': hashlib.sha256(page.body).hexdigest(),
                'extractor': 'scrapling-http', 'content_type': content_type}


async def enrich_sources(sources, cfg: ScrapingCfg, run_dir: Path, task: str):
    """Only fetch URLs discovered by search; never fetch model-invented item URLs."""
    if not cfg.enabled:
        return sources
    folder = Path(run_dir) / 'scraping'
    folder.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(cfg.concurrency)
    # Stable deduplication; cache is deliberately limited to this run.
    unique = list({source['url']: source for source in sources}.values())

    async def enrich(index, source):
        url = source['url']
        record = {'url': url, 'status': 'skipped_limit'}
        cache_path = folder / (hashlib.sha256(url.encode()).hexdigest() + '.json')
        async with semaphore:
            if cache_path.exists():
                record = json.loads(cache_path.read_text(encoding='utf-8'))
            elif index < cfg.max_pages_per_scan:
                try:
                    record = await scrape_page(url, cfg)
                except Exception as exc:
                    record = {'url': url, 'status': 'unavailable', 'reason': str(exc) or type(exc).__name__,
                              'retrieved_at': datetime.now(timezone.utc).isoformat()}
                cache_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        result = dict(source)
        original = source.get('content', '')
        result['search_content'] = original
        result['scraping'] = {key: value for key, value in record.items() if key != 'content'}
        if record['status'] == 'fetched':
            result['content'] = '\n\n'.join(part for part in (original, record['content']) if part)
            result['title'] = record['title'] or source.get('title', url)
        return result

    output = await asyncio.gather(*(enrich(i, source) for i, source in enumerate(unique)))
    (folder / f'{task}-sources.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    return output
