# RQ3 GPU-only citation modeling

This report re-runs the citation-impact modeling workflow on the GPU-only input bundle. All regression models use the strict raw-compute sample.

## Sample audit

- GPU-only master papers: 6900
- 2020-2023 LB1/GFIMP GPU sample papers: 3156
- 2020-2023 strict raw GPU sample papers: 2195
- 2020-2025 LB1/GFIMP GPU sample papers: 6900
- 2020-2025 strict raw GPU sample papers: 5360

## Main results

| Outcome/model | Spec | Controls | N | Coef. | SE | p | 95% CI | R2 | Adj. R2 | 10x effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Strict raw GPU OLS: log1p citations | 7 | + Topic FE + Team + Org-count | 2,194 | 0.172 | 0.049 | 4.69e-04 | [0.076, 0.268] | 0.212 | 0.197 | 18.8% |
| Strict raw GPU PPML: citation count | 7 | + Topic FE + Team + Org-count | 2,194 | 0.477 | 0.085 | 2.06e-08 | [0.311, 0.644] |  |  | 61.2% |
| Strict raw GPU OLS: normalized citation percentile | 7 | + Topic FE + Team + Org-count | 2,194 | 0.013 | 0.009 | 0.151 | [-0.005, 0.030] | 0.092 | 0.074 |  |
| Strict raw GPU LPM: high cited all-yv top10 | 7 | + Topic FE + Team + Org-count | 2,194 | 0.038 | 0.014 | 0.005 | [0.011, 0.065] | 0.045 | 0.026 | 3.84 pp |
| Strict raw GPU LPM: award | 7 | + Topic FE + Team + Org-count | 5,357 | 0.009 | 0.004 | 0.056 | [-2.15e-04, 0.017] | 0.031 | 0.022 | 0.86 pp |

## Complete regression results

| Outcome/model | Spec | Controls | Family | Cov. | N | Coef. | SE | p | 95% CI | R2 | Adj. R2 | 10x effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Strict raw GPU OLS: log1p citations | 1 | Year x Venue FE | ols | HC3 | 2,194 | 0.210 | 0.047 | 9.18e-06 | [0.117, 0.303] | 0.154 | 0.150 | 23.4% |
| Strict raw GPU OLS: log1p citations | 2 | + Topic FE | ols | HC3 | 2,194 | 0.213 | 0.048 | 9.84e-06 | [0.119, 0.308] | 0.196 | 0.182 | 23.7% |
| Strict raw GPU OLS: log1p citations | 3 | + Team size | ols | HC3 | 2,194 | 0.165 | 0.048 | 5.73e-04 | [0.071, 0.259] | 0.170 | 0.166 | 18.0% |
| Strict raw GPU OLS: log1p citations | 4 | + Org-count group | ols | HC3 | 2,194 | 0.201 | 0.048 | 2.49e-05 | [0.108, 0.295] | 0.159 | 0.154 | 22.3% |
| Strict raw GPU OLS: log1p citations | 5 | + Topic FE + Team | ols | HC3 | 2,194 | 0.171 | 0.049 | 4.69e-04 | [0.075, 0.267] | 0.210 | 0.196 | 18.7% |
| Strict raw GPU OLS: log1p citations | 6 | + Topic FE + Org-count | ols | HC3 | 2,194 | 0.205 | 0.048 | 2.25e-05 | [0.110, 0.300] | 0.201 | 0.187 | 22.8% |
| Strict raw GPU OLS: log1p citations | 7 | + Topic FE + Team + Org-count | ols | HC3 | 2,194 | 0.172 | 0.049 | 4.69e-04 | [0.076, 0.268] | 0.212 | 0.197 | 18.8% |
| Strict raw GPU PPML: citation count | 1 | Year x Venue FE | poisson | HC0 | 2,194 | 0.507 | 0.094 | 6.95e-08 | [0.323, 0.691] |  |  | 66.0% |
| Strict raw GPU PPML: citation count | 2 | + Topic FE | poisson | HC0 | 2,194 | 0.510 | 0.080 | 1.96e-10 | [0.353, 0.668] |  |  | 66.6% |
| Strict raw GPU PPML: citation count | 3 | + Team size | poisson | HC0 | 2,194 | 0.465 | 0.096 | 1.31e-06 | [0.276, 0.653] |  |  | 59.1% |
| Strict raw GPU PPML: citation count | 4 | + Org-count group | poisson | HC0 | 2,194 | 0.505 | 0.094 | 7.48e-08 | [0.321, 0.689] |  |  | 65.8% |
| Strict raw GPU PPML: citation count | 5 | + Topic FE + Team | poisson | HC0 | 2,194 | 0.475 | 0.085 | 2.34e-08 | [0.308, 0.641] |  |  | 60.8% |
| Strict raw GPU PPML: citation count | 6 | + Topic FE + Org-count | poisson | HC0 | 2,194 | 0.509 | 0.080 | 1.99e-10 | [0.352, 0.666] |  |  | 66.4% |
| Strict raw GPU PPML: citation count | 7 | + Topic FE + Team + Org-count | poisson | HC0 | 2,194 | 0.477 | 0.085 | 2.06e-08 | [0.311, 0.644] |  |  | 61.2% |
| Strict raw GPU OLS: normalized citation percentile | 1 | Year x Venue FE | ols | HC3 | 2,194 | 0.017 | 0.008 | 0.038 | [9.64e-04, 0.033] | 0.040 | 0.036 |  |
| Strict raw GPU OLS: normalized citation percentile | 2 | + Topic FE | ols | HC3 | 2,194 | 0.020 | 0.009 | 0.020 | [0.003, 0.037] | 0.075 | 0.059 |  |
| Strict raw GPU OLS: normalized citation percentile | 3 | + Team size | ols | HC3 | 2,194 | 0.010 | 0.008 | 0.257 | [-0.007, 0.026] | 0.057 | 0.052 |  |
| Strict raw GPU OLS: normalized citation percentile | 4 | + Org-count group | ols | HC3 | 2,194 | 0.015 | 0.008 | 0.065 | [-9.75e-04, 0.032] | 0.047 | 0.042 |  |
| Strict raw GPU OLS: normalized citation percentile | 5 | + Topic FE + Team | ols | HC3 | 2,194 | 0.013 | 0.009 | 0.146 | [-0.004, 0.030] | 0.090 | 0.073 |  |
| Strict raw GPU OLS: normalized citation percentile | 6 | + Topic FE + Org-count | ols | HC3 | 2,194 | 0.018 | 0.009 | 0.034 | [0.001, 0.035] | 0.082 | 0.065 |  |
| Strict raw GPU OLS: normalized citation percentile | 7 | + Topic FE + Team + Org-count | ols | HC3 | 2,194 | 0.013 | 0.009 | 0.151 | [-0.005, 0.030] | 0.092 | 0.074 |  |
| Strict raw GPU LPM: high cited all-yv top10 | 1 | Year x Venue FE | lpm | HC3 | 2,194 | 0.045 | 0.013 | 6.28e-04 | [0.019, 0.071] | 0.008 | 0.003 | 4.50 pp |
| Strict raw GPU LPM: high cited all-yv top10 | 2 | + Topic FE | lpm | HC3 | 2,194 | 0.044 | 0.014 | 0.001 | [0.017, 0.071] | 0.039 | 0.022 | 4.40 pp |
| Strict raw GPU LPM: high cited all-yv top10 | 3 | + Team size | lpm | HC3 | 2,194 | 0.038 | 0.013 | 0.004 | [0.012, 0.064] | 0.014 | 0.009 | 3.81 pp |
| Strict raw GPU LPM: high cited all-yv top10 | 4 | + Org-count group | lpm | HC3 | 2,194 | 0.044 | 0.013 | 8.34e-04 | [0.018, 0.070] | 0.010 | 0.004 | 4.44 pp |
| Strict raw GPU LPM: high cited all-yv top10 | 5 | + Topic FE + Team | lpm | HC3 | 2,194 | 0.038 | 0.014 | 0.006 | [0.011, 0.065] | 0.044 | 0.027 | 3.77 pp |
| Strict raw GPU LPM: high cited all-yv top10 | 6 | + Topic FE + Org-count | lpm | HC3 | 2,194 | 0.044 | 0.014 | 0.001 | [0.017, 0.071] | 0.041 | 0.023 | 4.37 pp |
| Strict raw GPU LPM: high cited all-yv top10 | 7 | + Topic FE + Team + Org-count | lpm | HC3 | 2,194 | 0.038 | 0.014 | 0.005 | [0.011, 0.065] | 0.045 | 0.026 | 3.84 pp |
| Strict raw GPU LPM: award | 1 | Year x Venue FE | lpm | HC3 | 5,357 | 0.010 | 0.004 | 0.022 | [0.001, 0.018] | 0.017 | 0.014 | 0.97 pp |
| Strict raw GPU LPM: award | 2 | + Topic FE | lpm | HC3 | 5,357 | 0.010 | 0.004 | 0.026 | [0.001, 0.018] | 0.030 | 0.022 | 0.98 pp |
| Strict raw GPU LPM: award | 3 | + Team size | lpm | HC3 | 5,357 | 0.009 | 0.004 | 0.042 | [3.17e-04, 0.017] | 0.017 | 0.014 | 0.88 pp |
| Strict raw GPU LPM: award | 4 | + Org-count group | lpm | HC3 | 5,357 | 0.010 | 0.004 | 0.025 | [0.001, 0.018] | 0.017 | 0.014 | 0.97 pp |
| Strict raw GPU LPM: award | 5 | + Topic FE + Team | lpm | HC3 | 5,357 | 0.009 | 0.004 | 0.054 | [-1.63e-04, 0.017] | 0.031 | 0.022 | 0.86 pp |
| Strict raw GPU LPM: award | 6 | + Topic FE + Org-count | lpm | HC3 | 5,357 | 0.010 | 0.004 | 0.031 | [8.76e-04, 0.018] | 0.030 | 0.022 | 0.96 pp |
| Strict raw GPU LPM: award | 7 | + Topic FE + Team + Org-count | lpm | HC3 | 5,357 | 0.009 | 0.004 | 0.056 | [-2.15e-04, 0.017] | 0.031 | 0.022 | 0.86 pp |

## Robustness analyses

### Sample selection check

| Sample | Rows | Valid compute | Mean cites | High-cited rate | Award rate |
| --- | --- | --- | --- | --- | --- |
| gpu_lb1_2020_2023 | 3,156 | 3,156 | 24.060 | 10.2% | 1.5% |
| strict_raw_2020_2023 | 2,195 | 2,195 | 24.297 | 10.8% | 1.7% |

### Alternative compute and team controls

| Compute var | Team control | N | Coef. | SE | p | 10x effect |
| --- | --- | --- | --- | --- | --- | --- |
| log10_max_compute | group | 2,194 | 0.172 | 0.049 | 4.69e-04 | 18.8% |
| log10_max_compute | continuous | 2,194 | 0.139 | 0.049 | 0.005 | 15.0% |
| log10_compute | group | 2,194 | 0.168 | 0.049 | 5.48e-04 | 18.3% |
| log10_compute | continuous | 2,194 | 0.137 | 0.048 | 0.005 | 14.6% |

### Outlier sensitivity

| Sample | N | Coef. | SE | p | 10x effect |
| --- | --- | --- | --- | --- | --- |
| baseline | 2,194 | 0.172 | 0.049 | 4.69e-04 | 18.8% |
| drop citation top 1% | 2,172 | 0.115 | 0.047 | 0.015 | 12.2% |
| drop compute top 1% | 2,173 | 0.173 | 0.052 | 7.83e-04 | 18.9% |
| drop both top 1% | 2,152 | 0.114 | 0.049 | 0.021 | 12.1% |

### Clustered standard errors

| Cluster variable | Clusters | N | Coef. | SE | p |
| --- | --- | --- | --- | --- | --- |
| year_venue | 10 | 2,194 | 0.172 | 0.064 | 0.007 |
| primary_topic | 29 | 2,194 | 0.172 | 0.062 | 0.006 |
| year_str | 4 | 2,194 | 0.172 | 0.075 | 0.022 |

### Leave-one-out sensitivity

| Leave-out type | Left out | N | Coef. | SE | p |
| --- | --- | --- | --- | --- | --- |
| year | 2020 | 1,970 | 0.166 | 0.050 | 8.24e-04 |
| year | 2021 | 1,722 | 0.175 | 0.053 | 0.001 |
| year | 2022 | 1,547 | 0.109 | 0.061 | 0.073 |
| year | 2023 | 1,343 | 0.252 | 0.069 | 2.69e-04 |
| venue | acl | 1,363 | 0.164 | 0.063 | 0.009 |
| venue | emnlp | 976 | 0.204 | 0.073 | 0.005 |
| venue | naacl | 2,049 | 0.170 | 0.051 | 8.88e-04 |

### Institution-history controls

| Model | N | Coef. | SE | p | R2 |
| --- | --- | --- | --- | --- | --- |
| strict baseline on institution-control sample | 2,194 | 0.172 | 0.049 | 4.69e-04 | 0.212 |
| strict plus prior org history/collab controls | 2,194 | 0.172 | 0.051 | 7.64e-04 | 0.224 |

### Award sparsity diagnostics

| Metric | Value |
| --- | --- |
| nobs | 5,357 |
| award_events | 111 |
| event_rate | 0.021 |
| parameters_spec7 | 49 |
| events_per_parameter_spec7 | 2.265 |
| year_venue_cells | 16 |
| zero_award_cells | 6 |

### Effect size and incremental R2

| Model | N | Beta | SE | p | 10x effect | 100x effect | 1000x effect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ols_7 | 2,194 | 0.172 | 0.049 | 4.69e-04 | 18.8% | 41.0% | 67.5% |

| Spec | N | Full R2 | R2 without compute | Delta R2 |
| --- | --- | --- | --- | --- |
| 7 | 2,194 | 0.212 | 0.207 | 0.005 |

## Model specification

RQ3 uses year-by-venue fixed effects, primary-topic fixed effects, team-size group,
and organization-count group controls. It intentionally excludes `contribution_type`
and all contribution-label proxy controls. The compute regressor is strict raw
`log10_max_compute`, derived from `paper_max_row_compute_capability`.

## Notes

- Each coefficient is the estimated coefficient on `log10_max_compute`.
- Spec 1 includes year-by-venue fixed effects only; specs 2-7 add topic,
  team-size, and organization-count controls as shown in the `Controls` column.
- `N` is the estimation sample size after outcome and covariate filtering.
- `SE` is the standard error using the listed covariance estimator; `95% CI`
  is the confidence interval for the compute coefficient.
- `10x effect` reports `exp(coef) - 1` for log-link/log-outcome models and
  percentage-point effects for linear probability models.
- The full machine-readable regression table is also exported as
  `all_model_effect_tables.csv` in the `data` directory.
- Robustness-analysis tables are exported as CSV files in the same `data`
  directory using the section names shown above.

