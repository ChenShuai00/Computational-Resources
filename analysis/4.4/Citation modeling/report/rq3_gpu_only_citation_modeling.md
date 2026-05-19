# RQ3 GPU-only citation modeling

This report re-runs the citation-impact modeling workflow on the GPU-only input bundle.

## Sample audit

- GPU-only master papers: 6900
- 2020-2023 LB1/GFIMP GPU sample papers: 3156
- 2020-2023 strict raw GPU sample papers: 2195
- 2020-2025 LB1/GFIMP GPU sample papers: 6900

## Main results

| Outcome/model | Spec | Controls | N | Coef. | SE | p | 95% CI | R2 | Adj. R2 | 10x effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU-only OLS: log1p citations | 7 | + Topic FE + Team + Org-count | 3,155 | 0.178 | 0.034 | 2.11e-07 | [0.111, 0.246] | 0.202 | 0.191 | 19.5% |
| Strict raw GPU OLS: log1p citations | 7 | + Topic FE + Team + Org-count | 2,194 | 0.172 | 0.049 | 4.69e-04 | [0.076, 0.268] | 0.212 | 0.197 | 18.8% |
| GPU-only PPML: citation count | 7 | + Topic FE + Team + Org-count | 3,155 | 0.355 | 0.063 | 1.64e-08 | [0.232, 0.478] |  |  | 42.6% |
| GPU-only OLS: normalized citation percentile | 7 | + Topic FE + Team + Org-count | 3,154 | 0.021 | 0.006 | 6.21e-04 | [0.009, 0.034] | 0.093 | 0.081 |  |
| GPU-only LPM: high cited all-yv top10 | 7 | + Topic FE + Team + Org-count | 3,155 | 0.035 | 0.009 | 7.53e-05 | [0.017, 0.052] | 0.043 | 0.030 | 3.46 pp |
| GPU-only LPM: award | 7 | + Topic FE + Team + Org-count | 6,895 | 0.005 | 0.003 | 0.171 | [-0.002, 0.011] | 0.030 | 0.023 | 0.46 pp |

## Complete regression results

| Outcome/model | Spec | Controls | Family | Cov. | N | Coef. | SE | p | 95% CI | R2 | Adj. R2 | 10x effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU-only OLS: log1p citations | 1 | Year x Venue FE | ols | HC3 | 3,155 | 0.218 | 0.034 | 1.01e-10 | [0.152, 0.284] | 0.147 | 0.144 | 24.4% |
| GPU-only OLS: log1p citations | 2 | + Topic FE | ols | HC3 | 3,155 | 0.213 | 0.034 | 3.75e-10 | [0.147, 0.280] | 0.186 | 0.176 | 23.8% |
| GPU-only OLS: log1p citations | 3 | + Team size | ols | HC3 | 3,155 | 0.181 | 0.034 | 1.03e-07 | [0.114, 0.247] | 0.165 | 0.161 | 19.8% |
| GPU-only OLS: log1p citations | 4 | + Org-count group | ols | HC3 | 3,155 | 0.211 | 0.034 | 5.07e-10 | [0.144, 0.278] | 0.151 | 0.147 | 23.5% |
| GPU-only OLS: log1p citations | 5 | + Topic FE + Team | ols | HC3 | 3,155 | 0.179 | 0.034 | 1.78e-07 | [0.112, 0.246] | 0.201 | 0.191 | 19.6% |
| GPU-only OLS: log1p citations | 6 | + Topic FE + Org-count | ols | HC3 | 3,155 | 0.207 | 0.034 | 1.39e-09 | [0.140, 0.274] | 0.189 | 0.179 | 23.0% |
| GPU-only OLS: log1p citations | 7 | + Topic FE + Team + Org-count | ols | HC3 | 3,155 | 0.178 | 0.034 | 2.11e-07 | [0.111, 0.246] | 0.202 | 0.191 | 19.5% |
| Strict raw GPU OLS: log1p citations | 1 | Year x Venue FE | ols | HC3 | 2,194 | 0.210 | 0.047 | 9.18e-06 | [0.117, 0.303] | 0.154 | 0.150 | 23.4% |
| Strict raw GPU OLS: log1p citations | 2 | + Topic FE | ols | HC3 | 2,194 | 0.213 | 0.048 | 9.84e-06 | [0.119, 0.308] | 0.196 | 0.182 | 23.7% |
| Strict raw GPU OLS: log1p citations | 3 | + Team size | ols | HC3 | 2,194 | 0.165 | 0.048 | 5.73e-04 | [0.071, 0.259] | 0.170 | 0.166 | 18.0% |
| Strict raw GPU OLS: log1p citations | 4 | + Org-count group | ols | HC3 | 2,194 | 0.201 | 0.048 | 2.49e-05 | [0.108, 0.295] | 0.159 | 0.154 | 22.3% |
| Strict raw GPU OLS: log1p citations | 5 | + Topic FE + Team | ols | HC3 | 2,194 | 0.171 | 0.049 | 4.69e-04 | [0.075, 0.267] | 0.210 | 0.196 | 18.7% |
| Strict raw GPU OLS: log1p citations | 6 | + Topic FE + Org-count | ols | HC3 | 2,194 | 0.205 | 0.048 | 2.25e-05 | [0.110, 0.300] | 0.201 | 0.187 | 22.8% |
| Strict raw GPU OLS: log1p citations | 7 | + Topic FE + Team + Org-count | ols | HC3 | 2,194 | 0.172 | 0.049 | 4.69e-04 | [0.076, 0.268] | 0.212 | 0.197 | 18.8% |
| GPU-only PPML: citation count | 1 | Year x Venue FE | poisson | HC0 | 3,155 | 0.397 | 0.068 | 4.91e-09 | [0.264, 0.530] |  |  | 48.8% |
| GPU-only PPML: citation count | 2 | + Topic FE | poisson | HC0 | 3,155 | 0.386 | 0.062 | 4.81e-10 | [0.265, 0.508] |  |  | 47.1% |
| GPU-only PPML: citation count | 3 | + Team size | poisson | HC0 | 3,155 | 0.362 | 0.068 | 1.01e-07 | [0.229, 0.496] |  |  | 43.7% |
| GPU-only PPML: citation count | 4 | + Org-count group | poisson | HC0 | 3,155 | 0.394 | 0.068 | 7.40e-09 | [0.261, 0.528] |  |  | 48.3% |
| GPU-only PPML: citation count | 5 | + Topic FE + Team | poisson | HC0 | 3,155 | 0.355 | 0.063 | 1.59e-08 | [0.232, 0.478] |  |  | 42.7% |
| GPU-only PPML: citation count | 6 | + Topic FE + Org-count | poisson | HC0 | 3,155 | 0.382 | 0.062 | 7.71e-10 | [0.261, 0.504] |  |  | 46.6% |
| GPU-only PPML: citation count | 7 | + Topic FE + Team + Org-count | poisson | HC0 | 3,155 | 0.355 | 0.063 | 1.64e-08 | [0.232, 0.478] |  |  | 42.6% |
| GPU-only OLS: normalized citation percentile | 1 | Year x Venue FE | ols | HC3 | 3,154 | 0.026 | 0.006 | 1.80e-05 | [0.014, 0.038] | 0.046 | 0.043 |  |
| GPU-only OLS: normalized citation percentile | 2 | + Topic FE | ols | HC3 | 3,154 | 0.027 | 0.006 | 9.57e-06 | [0.015, 0.040] | 0.078 | 0.066 |  |
| GPU-only OLS: normalized citation percentile | 3 | + Team size | ols | HC3 | 3,154 | 0.020 | 0.006 | 0.001 | [0.008, 0.031] | 0.063 | 0.060 |  |
| GPU-only OLS: normalized citation percentile | 4 | + Org-count group | ols | HC3 | 3,154 | 0.024 | 0.006 | 5.47e-05 | [0.013, 0.036] | 0.051 | 0.047 |  |
| GPU-only OLS: normalized citation percentile | 5 | + Topic FE + Team | ols | HC3 | 3,154 | 0.022 | 0.006 | 5.37e-04 | [0.009, 0.034] | 0.092 | 0.080 |  |
| GPU-only OLS: normalized citation percentile | 6 | + Topic FE + Org-count | ols | HC3 | 3,154 | 0.026 | 0.006 | 2.53e-05 | [0.014, 0.038] | 0.082 | 0.070 |  |
| GPU-only OLS: normalized citation percentile | 7 | + Topic FE + Team + Org-count | ols | HC3 | 3,154 | 0.021 | 0.006 | 6.21e-04 | [0.009, 0.034] | 0.093 | 0.081 |  |
| GPU-only LPM: high cited all-yv top10 | 1 | Year x Venue FE | lpm | HC3 | 3,155 | 0.042 | 0.008 | 5.31e-07 | [0.026, 0.059] | 0.009 | 0.006 | 4.24 pp |
| GPU-only LPM: high cited all-yv top10 | 2 | + Topic FE | lpm | HC3 | 3,155 | 0.040 | 0.009 | 5.01e-06 | [0.023, 0.057] | 0.037 | 0.025 | 3.96 pp |
| GPU-only LPM: high cited all-yv top10 | 3 | + Team size | lpm | HC3 | 3,155 | 0.037 | 0.008 | 1.32e-05 | [0.020, 0.053] | 0.015 | 0.012 | 3.68 pp |
| GPU-only LPM: high cited all-yv top10 | 4 | + Org-count group | lpm | HC3 | 3,155 | 0.042 | 0.009 | 1.12e-06 | [0.025, 0.058] | 0.011 | 0.007 | 4.15 pp |
| GPU-only LPM: high cited all-yv top10 | 5 | + Topic FE + Team | lpm | HC3 | 3,155 | 0.034 | 0.009 | 7.62e-05 | [0.017, 0.051] | 0.042 | 0.030 | 3.44 pp |
| GPU-only LPM: high cited all-yv top10 | 6 | + Topic FE + Org-count | lpm | HC3 | 3,155 | 0.039 | 0.009 | 8.55e-06 | [0.022, 0.056] | 0.038 | 0.026 | 3.89 pp |
| GPU-only LPM: high cited all-yv top10 | 7 | + Topic FE + Team + Org-count | lpm | HC3 | 3,155 | 0.035 | 0.009 | 7.53e-05 | [0.017, 0.052] | 0.043 | 0.030 | 3.46 pp |
| GPU-only LPM: award | 1 | Year x Venue FE | lpm | HC3 | 6,895 | 0.005 | 0.003 | 0.145 | [-0.002, 0.011] | 0.017 | 0.014 | 0.47 pp |
| GPU-only LPM: award | 2 | + Topic FE | lpm | HC3 | 6,895 | 0.005 | 0.003 | 0.121 | [-0.001, 0.012] | 0.029 | 0.023 | 0.51 pp |
| GPU-only LPM: award | 3 | + Team size | lpm | HC3 | 6,895 | 0.004 | 0.003 | 0.172 | [-0.002, 0.011] | 0.017 | 0.014 | 0.44 pp |
| GPU-only LPM: award | 4 | + Org-count group | lpm | HC3 | 6,895 | 0.005 | 0.003 | 0.160 | [-0.002, 0.011] | 0.017 | 0.014 | 0.46 pp |
| GPU-only LPM: award | 5 | + Topic FE + Team | lpm | HC3 | 6,895 | 0.005 | 0.003 | 0.165 | [-0.002, 0.011] | 0.029 | 0.023 | 0.46 pp |
| GPU-only LPM: award | 6 | + Topic FE + Org-count | lpm | HC3 | 6,895 | 0.005 | 0.003 | 0.139 | [-0.002, 0.012] | 0.029 | 0.023 | 0.49 pp |
| GPU-only LPM: award | 7 | + Topic FE + Team + Org-count | lpm | HC3 | 6,895 | 0.005 | 0.003 | 0.171 | [-0.002, 0.011] | 0.030 | 0.023 | 0.46 pp |

## Robustness analyses

### Sample selection check

| Sample | Rows | Valid compute | Mean cites | High-cited rate | Award rate |
| --- | --- | --- | --- | --- | --- |
| gpu_lb1_2020_2023 | 3,156 | 3,156 | 24.060 | 10.2% | 1.5% |
| strict_raw_2020_2023 | 2,195 | 2,195 | 24.297 | 10.8% | 1.7% |

### Alternative compute and team controls

| Compute var | Team control | N | Coef. | SE | p | 10x effect |
| --- | --- | --- | --- | --- | --- | --- |
| log10_max_compute | group | 3,155 | 0.178 | 0.034 | 2.11e-07 | 19.5% |
| log10_max_compute | continuous | 3,155 | 0.157 | 0.034 | 4.16e-06 | 17.0% |
| log10_compute | group | 3,155 | 0.175 | 0.034 | 2.63e-07 | 19.2% |
| log10_compute | continuous | 3,155 | 0.155 | 0.034 | 4.62e-06 | 16.7% |

### Outlier sensitivity

| Sample | N | Coef. | SE | p | 10x effect |
| --- | --- | --- | --- | --- | --- |
| baseline | 3,155 | 0.178 | 0.034 | 2.11e-07 | 19.5% |
| drop citation top 1% | 3,123 | 0.135 | 0.033 | 4.27e-05 | 14.5% |
| drop compute top 1% | 3,127 | 0.179 | 0.036 | 5.78e-07 | 19.5% |
| drop both top 1% | 3,091 | 0.141 | 0.035 | 5.15e-05 | 15.1% |

### Clustered standard errors

| Cluster variable | Clusters | N | Coef. | SE | p |
| --- | --- | --- | --- | --- | --- |
| year_venue | 10 | 3,155 | 0.178 | 0.038 | 2.79e-06 |
| primary_topic | 29 | 3,155 | 0.178 | 0.038 | 3.40e-06 |
| year_str | 4 | 3,155 | 0.178 | 0.045 | 7.76e-05 |

### Leave-one-out sensitivity

| Leave-out type | Left out | N | Coef. | SE | p |
| --- | --- | --- | --- | --- | --- |
| year | 2020 | 2,694 | 0.201 | 0.036 | 2.78e-08 |
| year | 2021 | 2,414 | 0.155 | 0.039 | 5.75e-05 |
| year | 2022 | 2,280 | 0.149 | 0.042 | 3.52e-04 |
| year | 2023 | 2,077 | 0.209 | 0.043 | 1.34e-06 |
| venue | acl | 1,966 | 0.182 | 0.044 | 3.02e-05 |
| venue | emnlp | 1,394 | 0.196 | 0.051 | 1.17e-04 |
| venue | naacl | 2,950 | 0.170 | 0.036 | 2.30e-06 |

### Institution-history controls

| Model | N | Coef. | SE | p | R2 |
| --- | --- | --- | --- | --- | --- |
| baseline on institution-control sample | 3,155 | 0.178 | 0.034 | 2.11e-07 | 0.202 |
| plus prior org history/collab controls | 3,155 | 0.175 | 0.036 | 8.07e-07 | 0.209 |

### Award sparsity diagnostics

| Metric | Value |
| --- | --- |
| nobs | 6,895 |
| award_events | 140 |
| event_rate | 0.020 |
| parameters_spec7 | 49 |
| events_per_parameter_spec7 | 2.857 |
| year_venue_cells | 16 |
| zero_award_cells | 5 |

### Effect size and incremental R2

| Model | N | Beta | SE | p | 10x effect | 100x effect | 1000x effect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ols_7 | 3,155 | 0.178 | 0.034 | 2.11e-07 | 19.5% | 42.9% | 70.8% |

| Spec | N | Full R2 | R2 without compute | Delta R2 |
| --- | --- | --- | --- | --- |
| 7 | 3,155 | 0.202 | 0.194 | 0.008 |

## Model specification

RQ3 uses year-by-venue fixed effects, primary-topic fixed effects, team-size group,
and organization-count group controls. It intentionally excludes `contribution_type`
and all contribution-label proxy controls.

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

Amplifier interaction models are exported separately under `4.4/Amplifier interaction modeling`.
