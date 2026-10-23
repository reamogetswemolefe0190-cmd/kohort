import json
import tempfile
import unittest
from pathlib import Path

from kohort.config import load_config
from kohort.teardown import TeardownReport, _render_teardown_markdown, run_teardown


class TeardownTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = load_config()

    async def test_mock_teardown_runs_offline_and_creates_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            run_dir = await run_teardown("getminds.ai", self.cfg, mock=True, out_root=out_root)
            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "teardown.json").exists())
            self.assertTrue((run_dir / "teardown.md").exists())
            self.assertTrue((run_dir / "teardown.html").exists())
            self.assertTrue((run_dir / "costs.json").exists())

            data = json.loads((run_dir / "teardown.json").read_text(encoding="utf-8"))
            report = TeardownReport.model_validate(data)
            self.assertEqual(report.competitor_name, "getminds.ai")
            self.assertIn("Pay As You Go", [t.name for t in report.pricing.tiers])
            self.assertTrue(len(report.roadmap.features_to_adopt) > 0)
            self.assertTrue(len(report.roadmap.differentiators_to_exploit) > 0)

            # Test markdown rendering contains key sections
            md = (run_dir / "teardown.md").read_text(encoding="utf-8")
            self.assertIn("Pricing Model & Unit Economics", md)
            self.assertIn("Technical Architecture & System Deconstruction", md)
            self.assertIn("Actionable Roadmap & Counter-Attack Strategy", md)

            # Check that cost in mock mode is zero
            costs = json.loads((run_dir / "costs.json").read_text(encoding="utf-8"))
            self.assertEqual(costs["total_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
