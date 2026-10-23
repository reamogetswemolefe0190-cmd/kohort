# Minds backend reference investigation

Investigated 14 October 2026. Scope: public product documentation and API contracts, compared with Kohort's local implementation. No private source, authenticated studies, client uploads or paid Minds calls were accessed. This is a functional investigation, not a recovery of Minds' backend code.

## Findings and Kohort decisions

| Publicly documented capability | Evidence | Kohort before this change | Decision |
| --- | --- | --- | --- |
| Reuse a saved audience across studies | [Audience guide](https://getminds.ai/guide/audiences) | Every swarm designs scenarios and generates profiles again | Implement reuse of saved profiles for a fresh comparison |
| Keep individual responses connected to a saved study; move into follow-up interviews | [Studies guide](https://getminds.ai/guide/studies) | Response records exist, but no follow-up runner | Future extension with explicit conversation scope |
| Design audience composition from evidence while withholding historical outcomes from new research | [Audience guide](https://getminds.ai/guide/audiences) | Swarm traits are random modelling assumptions; scenarios receive roughly equal allocation | Future source-backed composition layer; do not describe current profiles as representative |
| Compare synthetic answers with external surveys | [Audience guide](https://getminds.ai/guide/audiences) | No independent customer or survey benchmark in the current client study | Future blind benchmark evaluator; requires suitable real data |
| Expose durable runs, status, knowledge and exports through APIs | [API overview](https://getminds.ai/docs/api/overview) | CLI with saved JSON and HTML; partial outputs preserved | Future job runner with checkpoints and bounded retries before adding a public API |

These are vendor-documented capabilities, not independently reproduced performance results. The pages do not establish Minds' internal model prompts, source selection logic, database schema, infrastructure or accuracy on Nova Site's market. No performance percentage is adopted as a Kohort claim.

## Implemented: reusable profiles for new comparisons

`swarm --audience-from SAVED_SWARM` loads the prior design, assignments and complete profiles. A new run compares the current advisory's offers against those profiles. It skips scenario design, scenario repair/review and persona-generation calls, retaining existing review limitations. It still runs comparison calls and the existing Jev observation. It does not resume an interrupted study or carry over prior answers.

Implementation: `kohort/audiences.py`, `kohort/swarm.py`, `kohort/cli.py`.

The loader requires protocol 4, matching client and exact region/category/target-market fields, the original complete panel size, compatible configured persona models, consistent scenario assignments and distinct IDs. Mock and live inputs cannot mix. Rejection happens before creating a run or model client. Current advisory eligibility and citation checks still apply.

Only declared profile and assignment fields enter the new comparison. Previous answers are not loaded. The current offer IDs determine a fresh presentation order, rotated within each scenario. Saved profiles and input files remain unchanged. The new run records file hashes and lineage in `audience-reuse.json`, with reuse metadata in `run.json` and a report limitation.

Hashes identify input bytes; they are not external authentication or independent validation. Reuse inherits the original profile assumptions, review limitations and model bias. It reduces repeated generation, not the need for grounded audience data. Stochastic outputs can still differ between runs. A changed buyer context requires a new panel.

## Verification

Offline tests cover fresh offers and responses using identical profiles, omission of old answers, source preservation, inherited concerns, incompatible clients/contexts, wrong panel sizes, duplicate/missing profiles, mismatched scenarios, mock boundaries and zero-cost mock reuse. Existing swarm and report checks are run alongside them.

Result: `python -m unittest tests.test_swarm tests.test_report_explorer -q` passed all 20 tests. CLI help exposes `--audience-from`. No new paid simulation was executed.

The saved Nova Site panel passed a read-only compatibility check: 200 profiles, four scenarios and six source artifacts hashed. This did not execute a new live comparison or update the client's findings.

## Recommended next backend sequence

1. Ground panel composition in cited audience evidence and label every assumption.
2. Introduce a durable comparison job with per-person completion, configuration fingerprints and retry of missing responses only.
3. Add a held-out benchmark dataset and deterministic scoring before making accuracy claims.
4. Add follow-up questions and richer study methods once the input/output contracts are stable.

These are proposals, not implemented features. Neither Minds' paid API nor a Minds dependency is required by the delivered reuse feature.
