# Paper results

Results follow the paper's argument rather than the historical order in which
scripts were developed:

1. `01_sample`: corpus flow, measurement audit, and declared exceptions.
2. `02_reporting`: temporal and venue reporting completeness.
3. `03_gpu_scale`: reported GPU count, generation, capacity, models, and memory.
4. `04_contexts`: country, institution, and NLP-topic patterns.
5. `05_scholarly_impact`: concentration, citation, award, robustness models,
   and the Main Conference + Findings appendix extension.

Each analysis module contains `scripts/` and an immutable `reference/` directory.
`reference/source_data/` or `reference/tables/` contains frozen numerical
artifacts; `reference/figures/` contains the canonical reproducible render;
`reference/publication_figures/` contains the paper's editorial-layout render
when it differs. New runs go only to `results/reproduced/`.

The authoritative crosswalk is
[`../code/results_manifest.csv`](../code/results_manifest.csv).

The Findings extension is self-contained under
[`05_scholarly_impact/analyses/track_extension`](05_scholarly_impact/analyses/track_extension/README.md).
