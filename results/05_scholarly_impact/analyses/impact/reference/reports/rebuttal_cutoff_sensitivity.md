# Rebuttal: high-compute and high-impact cutoff sensitivity

## Draft reviewer response

We thank the reviewer for pointing out that the 20% compute and 10% citation
cutoffs may appear arbitrary. We added a cutoff-sensitivity analysis varying
the yearly high-compute threshold over top 10%, 20%, and 30%, and the
year-venue high-impact citation threshold over top 5%, 10%, and 20%. Across all
nine combinations, high-compute papers remain more likely to be high-impact:
the relative-risk range is 1.49-2.14. The LPM robustness checks
likewise show positive compute associations across the tested citation
thresholds.

## Main takeaway

The original 20% compute / 10% citation cell is reproduced exactly: top-20%
yearly high-compute papers contain 92
of 322 year-venue top-10% cited papers. Their
high-impact rate is 14.5%, compared
with 9.1% for other
GPU-quantifiable papers (relative risk 1.59x).

Across the descriptive 3x3 grid, the relative risk remains above 1.0 in every
cell. The continuous-compute LPM is positive in 3 of
3 citation-threshold models, and the binary high-compute LPM
is positive in 9 of 9 threshold-combination
models. All binary high-compute LPM cells are positive; the following cells are less precise at p >= 0.05: compute top 10% / citation top 10% (p=0.124).

## Descriptive cutoff grid

| Compute cutoff | Citation cutoff | High-compute impact rate | Other impact rate | Relative risk | High-impact captured | High-compute/high-impact N | High-impact N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Top 10% | Top 5% | 9.7% | 4.7% | 2.10x | 19.0% | 31 | 163 |
| Top 10% | Top 10% | 15.4% | 9.6% | 1.60x | 15.2% | 49 | 322 |
| Top 10% | Top 20% | 28.6% | 19.2% | 1.49x | 14.3% | 91 | 637 |
| Top 20% | Top 5% | 9.0% | 4.2% | 2.14x | 35.0% | 57 | 163 |
| Top 20% | Top 10% | 14.5% | 9.1% | 1.59x | 28.6% | 92 | 322 |
| Top 20% | Top 20% | 27.9% | 18.2% | 1.53x | 27.8% | 177 | 637 |
| Top 30% | Top 5% | 7.8% | 4.0% | 1.93x | 45.4% | 74 | 163 |
| Top 30% | Top 10% | 13.6% | 8.7% | 1.55x | 40.1% | 129 | 322 |
| Top 30% | Top 20% | 26.2% | 17.6% | 1.49x | 39.1% | 249 | 637 |

## Figure

![Relative-risk cutoff sensitivity](../fig/rebuttal_cutoff_sensitivity_relative_risk.png)

## Continuous-compute LPM sensitivity

These models retain the Section 4.4 continuous compute regressor
`log10_max_compute` and vary only the high-citation outcome threshold.

| Citation cutoff | N | Mean outcome | Coef., pp | SE, pp | p | 95% CI, pp | Delta R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Top 5% | 2,194 | 5.7% | 4.04 | 1.13 | 3.74e-04 | [1.81, 6.26] | 0.0085 |
| Top 10% | 2,194 | 10.8% | 3.84 | 1.38 | 0.005 | [1.14, 6.53] | 0.0043 |
| Top 20% | 2,194 | 21.6% | 7.20 | 1.73 | 3.25e-05 | [3.80, 10.59] | 0.0086 |

## Binary high-compute LPM sensitivity

These supplementary models use the yearly high-compute group as the compute
regressor and vary both thresholds.

| Compute cutoff | Citation cutoff | N | Mean outcome | Coef., pp | SE, pp | p | 95% CI, pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Top 10% | Top 5% | 2,194 | 5.7% | 4.18 | 2.07 | 0.044 | [0.12, 8.24] |
| Top 10% | Top 10% | 2,194 | 10.8% | 3.88 | 2.53 | 0.124 | [-1.07, 8.83] |
| Top 10% | Top 20% | 2,194 | 21.6% | 8.77 | 3.24 | 0.007 | [2.42, 15.11] |
| Top 20% | Top 5% | 2,194 | 5.7% | 4.92 | 1.51 | 0.001 | [1.96, 7.87] |
| Top 20% | Top 10% | 2,194 | 10.8% | 4.45 | 1.87 | 0.018 | [0.77, 8.12] |
| Top 20% | Top 20% | 2,194 | 21.6% | 7.94 | 2.37 | 8.08e-04 | [3.29, 12.58] |
| Top 30% | Top 5% | 2,194 | 5.7% | 3.93 | 1.25 | 0.002 | [1.47, 6.38] |
| Top 30% | Top 10% | 2,194 | 10.8% | 3.82 | 1.61 | 0.017 | [0.67, 6.98] |
| Top 30% | Top 20% | 2,194 | 21.6% | 6.70 | 2.05 | 0.001 | [2.68, 10.71] |

## Model specification

High-compute status is computed separately within each publication year. High
citation status is computed within each publication-year-by-venue cell, using
the same citation-rank convention as Section 4.4. All LPM specifications use
the Section 4.4 spec-7 controls: year-by-venue fixed effects, primary-topic
fixed effects, team-size group, and organization-count group.

## Source artifacts

- Descriptive grid: `data/rebuttal_cutoff_sensitivity_descriptive.csv`
- Continuous-compute LPM: `data/rebuttal_cutoff_sensitivity_lpm_continuous.csv`
- Binary high-compute LPM: `data/rebuttal_cutoff_sensitivity_lpm_binary.csv`
- Figure: `fig/rebuttal_cutoff_sensitivity_relative_risk.png`
