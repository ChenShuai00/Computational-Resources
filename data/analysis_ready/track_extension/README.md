# Main Conference + Findings inputs

This directory contains the minimal paper-level inputs needed for the public
track-extension appendix. It does not contain raw PDFs, extraction caches, or
the local source workbooks.

- `track_extension_membership.csv`: all 23,838 papers and reporting-membership
  flags.
- `track_extension_papers.csv`: the 12,724 model-reported papers and only the
  fields used by the extension analysis.
- `manifest.csv`: released file shapes and SHA-256 checksums.
- `provenance.json`: hashes of the frozen local source artifacts used by the
  maintainer export utility.

To rebuild these public inputs from the maintainers' frozen local bundles, use
`code/export_track_extension.py`. Ordinary users do not need those private
bundles: the released CSVs are sufficient for `code/run_all.py`.
