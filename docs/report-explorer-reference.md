# Report explorer reference study

Investigated 13 October 2026. This is a reusable, offline presentation layer over saved Kohort runs and an existing editorial brief. It does not synthesize a new recommendation or run models.

## MiroFish observations

Reference: https://github.com/666ghj/MiroFish/blob/main/frontend/src/components/Step4Report.vue

The report component separates generated sections from a workflow/evidence timeline. Completed sections collapse; tool activity expands; the completed report leads into a separate interaction stage. The public demo exposes a staged graph / split / workbench workflow and labels its agent interaction as a static demo without an LLM connection.

## Kohort adaptation

- Preserve existing report colors (#16181d ink, #5f6672 secondary text, #4b3df5 accent) and system sans stack.
- Read mode: recommendation first, persistent section navigation, exact buyer messages next to scenario counts, source and response disclosure on demand.
- No simulated live progress or inactive chat. All actions work on saved local data.
- Recommendation is existing editorial judgment. Research excerpts and synthetic responses retain distinct labels.
- Counts are checked against saved response records before rendering. No aggregate market preference percentage.
- Mobile navigation wraps and comparison columns stack. Native disclosure, keyboard focus, print styling and JSON export.

Source was inspected for interaction patterns only. The implementation is authored for Kohort; no MiroFish source code, styles or assets were copied. REA was not required for this source-available investigation and was not installed.

## Scope and verification

Prototype command: `python -m kohort.report_explorer RUN --brief BRIEF.md --out REPORT.html`.
The client brief remains manually authored. There is no automatic marketing plan generation, report chat or live customer validation in this prototype.

## Implementation notes

This is a scoped extension of `kohort/report.py`, not a replacement visual system. The explorer inherits its ink, secondary text, violet accent, white surfaces, pale canvas and Inter/system sans approach. Inter is used only when available locally; the HTML does not load a web font. The implementation lives in `kohort/report_explorer.py` and `kohort/report_explorer.html`.

Read mode keeps the editorial recommendation first, with section navigation to option comparisons, individual synthetic responses, saved market evidence, proposed copy/actions and limitations. Scenario counts remain separate. Option drilldown sets response filters; search, pagination and native expandable disclosures expose supporting detail. Preserve these distinctions when extending the report.

The generated HTML embeds its data, CSS and JavaScript and runs offline in a browser with JavaScript enabled. It needs no CDN, frontend framework or new model calls. Building it uses Python and the existing `markdown` dependency. Opening an original source link requires network access. Evidence export includes the saved response data and file hashes; print includes all report sections but only the currently filtered response page, with its selection and pagination context stated in the printout.

The MiroFish influence is limited to navigation and evidence-disclosure patterns. The Kohort implementation is independently authored; no MiroFish source, styles or assets were copied.

## Verification record — 13 October 2026

- Four Python report-explorer tests passed during implementation.
- Desktop browser checks covered section navigation, scenario comparisons and option-to-response drilldown, response search (including an empty result and an ownership query), evidence expansion and the evidence download button.
- A 390px viewport DOM check found no horizontal overflow. Mobile screenshot capture was unreliable, so a complete mobile visual review remains outstanding.
- Print CSS and source behavior were inspected; an actual print-dialog/printed-output check remains outstanding.
- Independent review confirmed fixes for skip-link focus and missing response-selection context in print, and confirmed that template and generated scripts match. Browser logs were empty during the reported checks. Mobile visual and printed-output checks remain outstanding as noted above.

## Completed reference investigation — 14 October 2026

The user confirmed that REA means finishing the MiroFish investigation and applying improvements, rather than installing a separate repository. Scope: report navigation, evidence disclosure and explanation of the workflow. This is not a performance audit of the entire MiroFish backend or a replacement of Kohort's research engine.

Re-inspected the public [Step4Report.vue source](https://github.com/666ghj/MiroFish/blob/main/frontend/src/components/Step4Report.vue). Its report sections sit beside a workflow overview and expandable activity records; completed reports link onward to interaction. Kohort's saved artifacts support a retrospective stage map, not a live tool-call timeline. No external code or assets were copied.

| Before | After | Why |
| --- | --- | --- |
| Separate report sections without an explanation of how they relate | “How we reached this” connects seven stages, with links to findings and expandable supporting records | Readers can follow the path from brief to recommendation |
| Run provenance mostly presented in the method section or download | Each relevant stage exposes artifact names and SHA-256 hashes, plus saved research limits | Make the supporting records inspectable without inventing new evidence |
| Release wording could imply that the editorial recommendation passed the swarm review | Explicit separation of saved swarm eligibility, editorial advice and missing customer validation | Preserve the scope of the automated decision |
| Section navigation animated on every change, including keyboard use | Immediate section changes with heading focus preserved | Keep repeated navigation predictable |
| Long positioning brief as the client entry point | Separate concise client feedback note with message, offer clarification and three actions | Give the client a practical handoff without internal engineering notes |

Implementation is reusable: stage descriptions derive from loaded run data, with no Nova Site-specific conditions. Mock and held runs retain draft status; missing source records are labelled. The client feedback note is editorial, derived from the existing brief, and is not generated by the stage map. It is prepared for the user to share; it has not been sent to the client.

Five report tests pass, including draft/mock release handling, count reconciliation, injection escaping and source preservation. Browser checks verified the journey navigation, heading focus and expandable research record. Existing mobile visual and actual print-output limitations remain. No new model calls or competitor searches were run for the client.

### Boundaries and future work

This completes the scoped report-reference pass. Live execution events, resume/retry controls, report chat, automated editorial synthesis and real customer validation require separate backend work and are not represented as working features here. The research coverage gap for AI-assisted development and fixed-price agencies remains a research task; a new presentation does not resolve it.
