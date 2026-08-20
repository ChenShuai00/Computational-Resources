# Rebuttal: topic-year field-normalized citation impact

## Draft reviewer response

We thank the reviewer for noting that citation practices differ across NLP
subfields. We added a topic-year field-normalized citation percentile, comparing
each paper only with papers in the same NLP topic and publication year. The
association between reported GPU compute and citation impact is then
re-estimated using the same control structure as Section 4.4, with an additional
sensitivity excluding sparse topic-year cells.

## Main takeaway

Using the strict raw GPU sample from 2020-2023 and the same spec-7 controls as
Section 4.4, a 10x increase in reported GPU capacity is associated with a
3.52 percentile-point change in
topic-year field-normalized citation percentile (N=2,194,
p=0.002). When rows from sparse topic-year cells
with fewer than 10 cited papers are excluded, the estimate is
3.16 percentile points
(N=2,112, p=0.007).

## Sample audit

- GPU-only master papers: 6,900
- Topic-year citation reference papers: 6,900
- Topic-year cells: 173
- Topic-year cells with n < 10: 31
- Reference papers in n < 10 cells: 163
- 2020-2023 strict raw GPU sample papers: 2,195
- Primary field-normalized estimation sample: 2,194
- Sensitivity estimation sample after dropping n < 10 cells: 2,112

## Main results

| Sample | Spec | Controls | N | Topic-year cells | Min cell N | Rows in n<10 cells | Coef. | SE | p | 95% CI | 10x effect, percentile points | R2 | Adj. R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All topic-year cells | 7 | + Topic FE + Team + Org-count | 2,194 | 114 | 1 | 82 | 0.035 | 0.011 | 0.002 | [0.013, 0.058] | 3.52 | 0.090 | 0.072 |
| Sensitivity: topic-year cell n >= 10 | 7 | + Topic FE + Team + Org-count | 2,112 | 87 | 10 | 0 | 0.032 | 0.012 | 0.007 | [0.009, 0.054] | 3.16 | 0.093 | 0.077 |

## Complete regression results

| Sample | Spec | Controls | Family | Cov. | N | Topic-year cells | Min cell N | Rows in n<10 cells | Coef. | SE | p | 95% CI | 10x effect, percentile points | R2 | Adj. R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All topic-year cells | 1 | Year x Venue FE | ols | HC3 | 2,194 | 114 | 1 | 82 | 0.043 | 0.011 | 7.00e-05 | [0.022, 0.064] | 4.32 | 0.064 | 0.059 |
| All topic-year cells | 2 | + Topic FE | ols | HC3 | 2,194 | 114 | 1 | 82 | 0.046 | 0.011 | 4.62e-05 | [0.024, 0.068] | 4.62 | 0.070 | 0.053 |
| All topic-year cells | 3 | + Team size | ols | HC3 | 2,194 | 114 | 1 | 82 | 0.032 | 0.011 | 0.003 | [0.011, 0.054] | 3.22 | 0.081 | 0.076 |
| All topic-year cells | 4 | + Org-count group | ols | HC3 | 2,194 | 114 | 1 | 82 | 0.041 | 0.011 | 1.70e-04 | [0.020, 0.063] | 4.11 | 0.070 | 0.065 |
| All topic-year cells | 5 | + Topic FE + Team | ols | HC3 | 2,194 | 114 | 1 | 82 | 0.035 | 0.011 | 0.002 | [0.013, 0.057] | 3.50 | 0.088 | 0.071 |
| All topic-year cells | 6 | + Topic FE + Org-count | ols | HC3 | 2,194 | 114 | 1 | 82 | 0.044 | 0.011 | 1.04e-04 | [0.022, 0.067] | 4.42 | 0.076 | 0.059 |
| All topic-year cells | 7 | + Topic FE + Team + Org-count | ols | HC3 | 2,194 | 114 | 1 | 82 | 0.035 | 0.011 | 0.002 | [0.013, 0.058] | 3.52 | 0.090 | 0.072 |
| Sensitivity: topic-year cell n >= 10 | 1 | Year x Venue FE | ols | HC3 | 2,112 | 87 | 10 | 0 | 0.040 | 0.011 | 2.43e-04 | [0.019, 0.062] | 4.03 | 0.068 | 0.064 |
| Sensitivity: topic-year cell n >= 10 | 2 | + Topic FE | ols | HC3 | 2,112 | 87 | 10 | 0 | 0.043 | 0.011 | 1.68e-04 | [0.021, 0.066] | 4.32 | 0.072 | 0.057 |
| Sensitivity: topic-year cell n >= 10 | 3 | + Team size | ols | HC3 | 2,112 | 87 | 10 | 0 | 0.029 | 0.011 | 0.010 | [0.007, 0.050] | 2.85 | 0.087 | 0.082 |
| Sensitivity: topic-year cell n >= 10 | 4 | + Org-count group | ols | HC3 | 2,112 | 87 | 10 | 0 | 0.038 | 0.011 | 5.40e-04 | [0.017, 0.060] | 3.84 | 0.075 | 0.069 |
| Sensitivity: topic-year cell n >= 10 | 5 | + Topic FE + Team | ols | HC3 | 2,112 | 87 | 10 | 0 | 0.031 | 0.012 | 0.007 | [0.009, 0.054] | 3.14 | 0.091 | 0.076 |
| Sensitivity: topic-year cell n >= 10 | 6 | + Topic FE + Org-count | ols | HC3 | 2,112 | 87 | 10 | 0 | 0.041 | 0.012 | 3.51e-04 | [0.019, 0.064] | 4.13 | 0.079 | 0.063 |
| Sensitivity: topic-year cell n >= 10 | 7 | + Topic FE + Team + Org-count | ols | HC3 | 2,112 | 87 | 10 | 0 | 0.032 | 0.012 | 0.007 | [0.009, 0.054] | 3.16 | 0.093 | 0.077 |

## Model specification

For each paper, citations are ranked within the paper's `primary_topic` and
publication year. The percentile uses midpoint ranks:

`topic_year_citation_percentile = (average_rank - 0.5) / topic_year_cell_n`

This gives singleton cells a percentile of 0.5 and gives higher-cited papers
higher percentiles within their topic-year field. Venue is not included in the
normalization cell because topic-year-venue cells are too sparse for a stable
field-normalized outcome.

The regression sample and control structure follow the RQ3 citation-impact
workflow. The outcome is `topic_year_citation_percentile`; the compute regressor
is strict raw `log10_max_compute`, derived from
`paper_max_row_compute_capability`. Spec 1 includes year-by-venue fixed effects;
specs 2-7 add primary-topic fixed effects, team-size group, and
organization-count group controls as shown in the `Controls` column.

## Notes

- Each coefficient is the estimated coefficient on `log10_max_compute`.
- Because the outcome is a 0-1 percentile, `10x effect, percentile points`
  reports `100 * coef`.
- The primary rebuttal keeps all topic-year cells; the sensitivity model drops
  rows where `topic_year_citation_cell_n < 10`.
- The full machine-readable regression table and topic-year cell audit are
  exported as CSV files in the `data` directory.

## Source artifacts

- Model results: `data/rebuttal_field_normalized_citation.csv`
- Topic-year cell audit: `data/rebuttal_field_normalized_citation_cell_audit.csv`
