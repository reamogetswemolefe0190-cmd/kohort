"""Unit tests for the Persistent Audience Panel Builder and reuse layer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kohort.audiences import (
    AudiencePanel,
    AudienceSpec,
    SegmentSpec,
    build_panel,
    list_panels,
    load_audience_template,
    load_panel,
    save_panel,
)
from kohort.config import load_config
from kohort.pipeline import run_concept_test


class AudienceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = load_config()
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_default_template(self):
        spec = load_audience_template("sa-tech-founders")
        self.assertEqual(spec.name, "sa-tech-founders")
        self.assertEqual(spec.region, "South Africa")
        self.assertGreater(len(spec.segments), 1)

    async def test_build_and_save_panel_mock(self):
        spec = load_audience_template("sa-tech-founders")
        panel = await build_panel(spec, self.cfg, mock=True, sample_size=10, seed=123)

        self.assertEqual(panel.sample_size, 10)
        self.assertEqual(len(panel.personas), 10)
        self.assertTrue(panel.mock)

        # Save to temp directory
        saved_path = save_panel(panel, out_dir=self.out_dir)
        self.assertTrue(saved_path.is_file())

        # Load back
        loaded = load_panel(saved_path)
        self.assertEqual(loaded.spec.name, "sa-tech-founders")
        self.assertEqual(len(loaded.personas), 10)
        self.assertEqual(loaded.personas[0].id, panel.personas[0].id)

    async def test_run_concept_test_with_prebuilt_audience(self):
        spec = load_audience_template("sa-tech-founders")
        panel = await build_panel(spec, self.cfg, mock=True, sample_size=8, seed=42)
        saved_path = save_panel(panel, out_dir=self.out_dir)

        # Run concept test with brief pointing to saved audience
        brief_path = Path("briefs/example_loopa.yaml")
        run_dir = await run_concept_test(
            brief_path,
            self.cfg,
            mock=True,
            n_personas=8,
            seed=42,
            out_root=self.out_dir,
            console=None,
            audience=saved_path,
        )

        self.assertTrue((run_dir / "report.html").is_file())
        self.assertTrue((run_dir / "analysis.json").is_file())
        self.assertTrue((run_dir / "buyer-dossier.json").is_file())
        self.assertTrue((run_dir / "personas.json").is_file())
        self.assertTrue((run_dir / "responses.json").is_file())

        # Verify that reused personas received the new brief's Category Lens (household cleaning products)
        import json
        personas_data = json.loads((run_dir / "personas.json").read_text(encoding="utf-8"))
        self.assertEqual(len(personas_data), 8)
        self.assertEqual(personas_data[0]["category_lens"]["category"], "household cleaning products")
        self.assertTrue(len(personas_data[0]["category_lens"]["buying_habits"]) > 0)

    async def test_auto_vaulting_and_mock_mismatch_guard(self):
        brief_path = Path("briefs/example_loopa.yaml")
        run_dir = await run_concept_test(
            brief_path,
            self.cfg,
            mock=True,
            n_personas=6,
            seed=99,
            out_root=self.out_dir,
            console=None,
        )
        # Fresh run should auto-save a reusable base panel under run_dir / "audiences"
        vault_files = list((run_dir / "audiences").glob("*.json"))
        self.assertEqual(len(vault_files), 1)
        vaulted_panel = load_panel(vault_files[0])
        self.assertEqual(vaulted_panel.sample_size, 6)
        self.assertTrue(vaulted_panel.mock)
        self.assertIn("demographics", vaulted_panel.personas[0].trait_provenance)

        # Attempting to use a mock panel in a live run must fail immediately
        with self.assertRaises(ValueError):
            await run_concept_test(
                brief_path,
                self.cfg,
                mock=False,
                n_personas=6,
                seed=99,
                out_root=self.out_dir,
                console=None,
                audience=vault_files[0],
            )


if __name__ == "__main__":
    unittest.main()

