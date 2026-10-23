"""Build-failing reconciliation test for client deliverables.

Ensures that every number, percentage, and currency figure in generated client deliverables
reconciles directly to analysis.json or an explicit, audited editorial provenance registry.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from kohort.resilience import audit_tier1_invariants, audit_tier2_provenance

DELIVERABLES_DIR = (
    Path(__file__).resolve().parent.parent
    / "deliverables"
    / "novasite"
    / "2026-10-18-concept-test-feedback"
)


class DeliverableReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.analysis_path = DELIVERABLES_DIR / "analysis.json"
        self.assertTrue(self.analysis_path.exists(), "analysis.json must exist in deliverable directory")
        self.analysis_data = json.loads(self.analysis_path.read_text(encoding="utf-8"))

        self.dossier_path = DELIVERABLES_DIR / "buyer-dossier.json"
        self.dossier_data = None
        if self.dossier_path.exists():
            self.dossier_data = json.loads(self.dossier_path.read_text(encoding="utf-8"))

        self.editorial_path = DELIVERABLES_DIR / "editorial-provenance.json"
        self.editorial_registry = []
        if self.editorial_path.exists():
            data = json.loads(self.editorial_path.read_text(encoding="utf-8"))
            self.editorial_registry = data.get("editorial_provenance_registry", [])

    def test_novasite_analysis_json_satisfies_tier1_invariants(self):
        """Nova Site analysis.json must pass all deterministic mathematical invariants."""
        result = audit_tier1_invariants(self.analysis_data)
        self.assertTrue(
            result.passed,
            f"Nova Site analysis.json failed Tier-1 invariants: {'; '.join(result.violations)}",
        )

    def test_client_feedback_deliverable_reconciles_to_analysis_json(self):
        """client-feedback.html and .md must contain zero untraced figures."""
        for filename in ("client-feedback.html", "client-feedback.md", "report.html", "report.md"):
            doc_path = DELIVERABLES_DIR / filename
            if not doc_path.exists():
                continue
            content = doc_path.read_text(encoding="utf-8")
            audit = audit_tier2_provenance(
                deliverable_text=content,
                analysis_data=self.analysis_data,
                dossier=self.dossier_data,
                allowed_editorial=self.editorial_registry,
            )
            self.assertTrue(
                audit.passed,
                f"Deliverable '{filename}' contains untraced figures without provenance: {audit.untraced_figures}",
            )

    def test_mckinsey_and_one_pager_deliverables_reconcile_to_provenance(self):
        """Executive deliverables must strictly reconcile against raw analysis or registered editorial rationale."""
        for filename in (
            "mckinsey-executive-brief.html",
            "one-page-summary.html",
            "one-page-summary.md",
        ):
            doc_path = DELIVERABLES_DIR / filename
            if not doc_path.exists():
                continue
            content = doc_path.read_text(encoding="utf-8")
            audit = audit_tier2_provenance(
                deliverable_text=content,
                analysis_data=self.analysis_data,
                dossier=self.dossier_data,
                allowed_editorial=self.editorial_registry,
            )
            self.assertTrue(
                audit.passed,
                f"Deliverable '{filename}' contains untraced figures without provenance: {audit.untraced_figures}",
            )


if __name__ == "__main__":
    unittest.main()
