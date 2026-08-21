# Analysis-ready data

Key tables include:

- `paper_sample_membership.csv`: one row per corpus paper with model-reported
  and strict model-plus-count membership flags.
- `compute_papers.csv` and `paper_compute_rows.csv`: paper- and row-level
  standardized reported GPU evidence.
- `topics.csv`, `awards.csv`, and `openalex_metadata.csv`: topic, formal paper
  award, and bibliographic outcome data used by the paper models.
- `paper_organizations.csv`, `paper_organization_variables.csv`, and
  `organization_year_panel.csv`: minimized organizational context tables.
- `paper_confounder_controls.csv`: pre-publication controls used in robustness
  models.
- `consumption_audit_labels.csv`: minimized frozen labels for the 240-paper
  consumption-visibility audit.
- `track_extension/`: minimized Main Conference + Findings inputs for the
  appendix sample comparison, citation regressions, and high-impact
  concentration table. Its nested manifest and provenance file make this
  extension independently auditable.
