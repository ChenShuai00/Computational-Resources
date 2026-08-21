# Main Conference + Findings appendix extension

This module tests whether the paper's main scholarly-impact patterns extend to
Findings papers and to the pooled Main Conference + Findings sample.

It reproduces three publication-ready tables:

1. sample sizes, reporting rates, citation sample sizes, and reported-compute
   medians by track;
2. five core citation regressions with coefficient, robust standard error,
   incremental R-squared where defined, and a fully interacted
   Findings-minus-Main test;
3. high-capability/high-impact concentration rates by track.

Run from the repository root:

```bash
uv run python results/05_scholarly_impact/analyses/track_extension/scripts/analyze_track_extension.py
```

Frozen results are in `reference/tables/` and the formatted appendix text is in
`reference/reports/track_extension.md`. New runs go to `reproduced/` or the
directory supplied with `--output-dir`.

## Interpretation boundary

Reported GPU capability is text-reported evidence, not verified resource
consumption. The models estimate conditional observational associations. The
track-difference tests use a stacked, fully interacted specification; they do
not identify causal effects. Pooled percentile and high-capability thresholds
are recalculated in the pooled sample.
