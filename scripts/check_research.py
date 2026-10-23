"""One isolated, budgeted check of reasoning control and web evidence. No full run."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from kohort.config import load_config
from kohort.llm import CostTracker, LLM


async def main():
    load_dotenv(ROOT / '.env')
    cfg = load_config()
    cfg.run.web_search_max_uses = 1
    cfg.run.web_search_max_results = 2
    tracker = CostTracker(.08)
    llm = LLM(cfg, tracker)
    outcome = {}
    try:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get('https://openrouter.ai/api/v1/models')
            response.raise_for_status()
            metadata = next(m for m in response.json()['data'] if m['id'] == cfg.role('competitive_research').model)
            outcome['model_reasoning_metadata'] = metadata.get('reasoning')
        raw = await llm._call(cfg.role('competitive_research').model,
            [{'role': 'system', 'content': 'Use web search once. Return a concise JSON object only. Do not reason at length.'},
             {'role': 'user', 'content': 'Find the official Wix plans page. Return its URL and one sentence stating whether the retrieved excerpt establishes a current South African ZAR retail price. Do not infer prices from fragmented text.'}],
            0, 6000, True)
        parsed = json.loads(raw)
        evidence = llm.last_search_sources
        if not evidence or not any(s.get('content') for s in evidence):
            raise ValueError('Missing retrieved excerpts')
        outcome.update(status='passed', result=parsed, source_count=len(evidence))
    except Exception as exc:
        outcome.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        outcome['calls'] = llm.search_log
        outcome['costs'] = tracker.summary()
        destination = ROOT / 'runs' / 'research-connection-check.json'
        destination.write_text(json.dumps(outcome, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in outcome.items() if k != 'calls'}, indent=2))
        await llm.aclose()
    if outcome['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
