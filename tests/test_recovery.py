import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rich.console import Console
from kohort.config import load_config
from kohort.llm import LLM, LLMError, CostTracker
from kohort.pipeline import _save
from kohort.supervisor import PipelineSupervisor
from kohort.guardrails import GuardrailBlocked


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_replies_stop_after_two_attempts(self):
        from kohort.schemas import Brief
        llm = LLM(load_config(), CostTracker(1))
        with patch.object(llm, '_call', AsyncMock(side_effect=ValueError('empty reply: finish_reason=length'))) as call, patch('kohort.llm.asyncio.sleep', AsyncMock()):
            with self.assertRaises(LLMError):
                await llm.structured(task='competitive_scan', schema=Brief,
                                     model='deepseek/deepseek-v4.1-flash', system='Return JSON',
                                     user='test', temperature=0, max_tokens=6000)
            self.assertEqual(call.await_count, 2)
        await llm.aclose()

    async def test_transient_retry_and_auth_failure(self):
        llm = LLM(load_config(), CostTracker(1))
        with patch.object(llm, '_decide_once', AsyncMock(side_effect=[LLMError('Jev Decisions API failed: HTTP 530.'), {'ok': True}])) as call, patch('kohort.llm.asyncio.sleep', AsyncMock()):
            self.assertEqual(await llm.decide('model', {}, {}), {'ok': True})
            self.assertEqual(call.await_count, 2)
        with patch.object(llm, '_decide_once', AsyncMock(side_effect=LLMError('Jev Decisions API failed: HTTP 401.'))) as call:
            with self.assertRaises(LLMError):
                await llm.decide('model', {}, {})
            self.assertEqual(call.await_count, 1)
        await llm.aclose()

    async def test_repair_uses_previous_output_and_still_holds(self):
        llm = LLM(load_config(), CostTracker(1))
        produce = AsyncMock(return_value={'sources': ['saved'], 'text': 'original'})
        repair = AsyncMock(return_value={'sources': ['saved'], 'text': 'revised'})
        def held(*args, **kwargs):
            return {'status': 'held', 'action': 'hold', 'reasons': ['Unsupported claim'], 'probabilities': {'risk': .28}}
        with tempfile.TemporaryDirectory() as directory, patch('kohort.supervisor.review_stage', AsyncMock(side_effect=held)):
            supervisor = PipelineSupervisor(llm, load_config(), Path(directory), Console(file=io.StringIO()), _save)
            with self.assertRaises(GuardrailBlocked):
                await supervisor.run('competitive_research', produce, lambda out: [out], repair=repair)
            self.assertEqual(produce.await_count, 1)
            self.assertEqual(repair.await_count, 1)
            self.assertEqual(repair.call_args.args[0]['sources'], ['saved'])
            self.assertEqual(supervisor.checkpoints[-1]['action'], 'hold')
        await llm.aclose()
