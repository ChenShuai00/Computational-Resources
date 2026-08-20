# Reproduction contract

`run_all.py` executes the public workflows in a fixed order and writes only to
`results/reproduced/`. `verify.py` then checks:

- every analysis-ready input's SHA-256 digest, shape, and frozen sample counts;
- regenerated CSV schemas, row order, text fields, and numerical values
  (`rtol=1e-8`, `atol=1e-10`);
- canonical regenerated figure dimensions and color modes; and
- frozen publication-figure dimensions and documented panel/label contracts.

PNG files are not required to be byte-identical because renderer metadata and
compression can vary. Numerical source data are the authoritative comparison
layer. Values printed in the paper are additionally checked at the paper's
displayed precision for headline claims.

`results_manifest.csv` maps paper labels to evidence. `reported_only` means the
paper value is preserved and disclosed but the necessary frozen input is absent;
it never means verified.

`export_analysis_ready.py` is a maintainer utility for rebuilding the public CSV
layer from local legacy workbooks. It is not part of the reader-facing offline
reproduction command.
