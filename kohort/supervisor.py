"""Checkpoint every stage before downstream work, with bounded, audited rework."""

from __future__ import annotations

import json
from .guardrails import GuardrailBlocked, review_stage


class PipelineSupervisor:
    def __init__(self, llm, cfg, run_dir, console, save, evidence_reviewer=None):
        self.llm, self.cfg, self.run_dir, self.console, self.save = llm, cfg, run_dir, console, save
        self.checkpoints = []
        self.evidence_reviewer = evidence_reviewer
        (run_dir / "supervision").mkdir()

    def _record(self, result):
        self.checkpoints.append(result)
        self.save(self.run_dir, "supervision.json", {
            "status": result["status"], "action": result["action"], "stage": result["stage"],
            "max_stage_retries": self.cfg.guardrails.max_stage_retries,
            "checkpoints": self.checkpoints,
        })

    async def run(self, stage, produce, evidence_batches, *, retry_allowed=True,
                  reviewer=None, stop_on_hold=True, repair=None, review_all=False):
        """Regenerate only this stage; every replacement is reviewed again."""
        feedback = ""
        if self.evidence_reviewer:
            retry_allowed = False  # Evaluate stable evidence; no blind rewriting during calibration.
        limit = self.cfg.guardrails.max_stage_retries if retry_allowed else 0
        for attempt in range(limit + 1):
            output_file = f"supervision/{stage}-attempt-{attempt + 1}.json"
            self.llm.stage_feedback = feedback
            try:
                output = await repair(output, feedback) if attempt and repair else await produce()
            except Exception as exc:
                result = {"stage": stage, "attempt": attempt + 1, "status": "error", "action": "hold",
                          "reasons": [f"Stage execution failed: {type(exc).__name__}: {exc}"]}
                self._record(result)
                raise
            finally:
                self.llm.stage_feedback = ""
            self.save(self.run_dir, output_file, output)
            batches = evidence_batches(output)
            if not batches:
                raise ValueError(f"No review evidence for {stage}")
            results = []
            for batch, evidence in enumerate(batches, 1):
                evidence_file = f"supervision/{stage}-attempt-{attempt + 1}-batch-{batch}-evidence.json"
                self.save(self.run_dir, evidence_file, evidence)
                evidence = json.loads((self.run_dir / evidence_file).read_text(encoding="utf-8"))
                if self.evidence_reviewer:
                    result = await self.evidence_reviewer(self.llm, self.cfg, stage, evidence, retry_allowed)
                elif reviewer:
                    result = await reviewer(output, retry_allowed)
                    result.setdefault("action", "continue" if result["status"] in ("passed", "mock") else "hold")
                else:
                    result = await review_stage(self.llm, self.cfg, stage, evidence, allow_retry=retry_allowed)
                result.update(stage=stage, attempt=attempt + 1, batch=batch,
                              output_file=output_file, evidence_file=evidence_file)
                if evidence.get("review_item"):
                    result["review_item"] = evidence["review_item"]
                    if result["action"] != "continue":
                        result["reasons"].insert(0, f"Review item: {evidence['review_item']['kind']} / {evidence['review_item']['name']}.")
                        self.console.print(f"  Review held: {evidence['review_item']['name']}")
                if (repair and retry_allowed and attempt < limit and result["status"] == "held"
                        and not result.get("hard_reasons")):
                    result["action"] = "retry"
                    result["repair_mode"] = "revise_from_saved_evidence"
                    result["reasons"].append("One bounded revision from saved evidence; replacement must pass every check.")
                if result["action"] == "retry" and attempt == limit:
                    max_prob = max((result.get("probabilities") or {}).values(), default=1.0)
                    if (attempt > 0 and self.cfg.guardrails.self_heal_enabled
                            and not result.get("hard_reasons")
                            and max_prob <= self.cfg.guardrails.advisory_max_probability):
                        prior_reasons = list(result.get("reasons") or [])
                        result["status"] = "passed"
                        result["action"] = "continue"
                        result["advisory_pass"] = True
                        result.setdefault("advisory_notes", []).extend(prior_reasons)
                        result["reasons"].append(
                            f"Advisory soft-pass after self-heal retry (max risk {max_prob:.1%} <= "
                            f"{self.cfg.guardrails.advisory_max_probability:.0%} advisory limit)."
                        )
                    else:
                        result["action"] = "hold"
                        result["reasons"].append("Stage retry limit reached; further work needs review.")
                self._record(result)
                results.append(result)
                if result["action"] != "continue" and (not review_all or result["status"] == "error"):
                    break
            failed = next((r for r in results if r["action"] != "continue"), None)
            if failed and failed["action"] == "retry":
                feedback = f"Stage: {stage}. " + " ".join(failed["reasons"])
                self.console.print(f"  [yellow]Jev requested rework of {stage}; retry {attempt + 1}/{limit}.[/]")
                continue
            final = dict(failed or results[-1])
            final["batches_reviewed"] = len(results)
            final["batch_results"] = results
            self.save(self.run_dir, "supervision.json", {
                "status": final["status"], "action": final["action"], "stage": stage,
                "stage_result": final, "max_stage_retries": self.cfg.guardrails.max_stage_retries,
                "checkpoints": self.checkpoints})
            label = 'Evidence review' if self.evidence_reviewer else 'Jev'
            self.console.print(f"  {label} {stage}: {final['action']} ({final['status']})")
            if failed and stop_on_hold:
                raise GuardrailBlocked(f"Pipeline held at {stage}: {'; '.join(final['reasons'])} "
                                       f"Evidence and decisions saved in {self.run_dir / 'supervision.json'}")
            return output, final
        raise AssertionError("Supervisor retry loop ended without a decision")
