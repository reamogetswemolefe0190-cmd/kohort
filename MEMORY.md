# Kohort project memory

## Audience Builder direction — saved 14 October 2026

The user asked to preserve this direction for future work: collect internet and research data to build reusable groups of synthetic personas for different markets. Examples discussed include South African X users, Instagram users, gamers, and employed/unemployed young people. These are candidate audiences, not claims that relevant data or validated panels already exist.

Kohort should serve an agency across clients and industries. Nova Site is a client example, not the basis for hardcoded audience rules.

### Proposed architecture

Audience definition → evidence collection → source-backed composition → profile generation → reusable simulations → comparison with real responses.

- Define audience membership and the research question before collecting data.
- Combine surveys/statistics for composition with relevant public discussions and reviews for experiences and concerns. A general-population statistic does not establish a platform-specific audience distribution.
- Extend the existing evidence collection layer. Preserve source, date, population, excerpt and limitations. Use appropriate platform connectors where ordinary page scraping is insufficient.
- Allocate profile characteristics in code from supported distributions. Preserve observed relationships where available. Label unsupported allocations as assumptions; do not guess a person's age or employment from posts.
- Use an existing language model grounded in the evidence initially. Fine-tuning a new model is a separate later decision, not a prerequisite.
- Label profile characteristics as sourced, modelled or unknown. Keep generation separate from the offer being tested and withhold prior outcomes from new comparisons.
- Save the resulting audience for repeated studies with fresh responses. Simulated buyers are not real customer records or proof of demand.
- Compare synthetic responses with real responses to the same questions, withholding the real answers until scoring. Record discrepancies and population limits.

### Suggested pilot, not yet selected or implemented

One narrow adult audience, approximately 50 synthetic profiles. The example proposed was South African mobile gamers aged 18–34. The user has asked to remember the direction, not explicitly chosen that audience or authorized a new data collection or paid run. A small pilot diagnoses problems; it does not validate a whole market. Teenagers were mentioned as a possible future segment, not the initial pilot.

### Current implementation and next gap

Existing capabilities include source-page retrieval, saved research, synthetic buyer comparisons, evidence review and offline reports. The recent `swarm --audience-from` change reuses complete protocol-4 saved profiles for the same client and exact buyer context, preserving limitations and omitting prior answers. It does not make existing profiles source-backed or representative.

The next proposed component is the **evidence-to-audience layer**: source-backed composition, traceable profile characteristics and a validation process. It has not been built yet. Full model training, platform ingestion connectors, population calibration and live customer validation are not completed features.

Related investigation: `docs/minds-backend-reference.md`. Minds' public audience/study separation informed reuse; its private backend was not accessed or copied.

### Source starting points

- Stats SA labour-force survey: https://isibaloweb.statssa.gov.za/pages/surveys/pss/qlfs/2026/qlfs2026.php
- X search documentation: https://github.com/xdevplatform/docs/blob/main/x-api/posts/search/quickstart/recent-search.mdx
- Minds audience guide: https://getminds.ai/guide/audiences

Recheck access, suitability and current documentation before implementation. These links do not establish that a dataset covers the chosen pilot population.
