# RQ2 Institution Fixed-Effects Models

## Method

This analysis uses a paper-level panel, not full-count organization-paper rows. The regression outcome is `log10(paper_max_row_compute_capability / 1e12)`, interpreted as strict raw max-row GPU compute in TFLOP/s.

The main table estimates five one-focal-variable-at-a-time OLS models with HC3 robust standard errors. Each model controls year fixed effects, topic fixed effects, and venue fixed effects. The separated specification is intentional because company participation, industry-academia collaboration, cross-sector collaboration, international collaboration, and organization count overlap conceptually and empirically.

Regression sample: `is_strict == 1`; paper-level rows: 6,895; strict-valid papers: 5,357; main-model observations: 5,357-5,357; fixed effects: 6 years, 29 topics, 3 venues.

## Main Models

| model   | term_label             |   nobs |   coef |   std_err | p_value   | percent_change   |   r_squared |
|:--------|:-----------------------|-------:|-------:|----------:|:----------|:-----------------|------------:|
| M1      | Company                |   5357 |  0.267 |     0.016 | <0.001    | 84.8%            |       0.191 |
| M2      | Industry-academia      |   5357 |  0.195 |     0.017 | <0.001    | 56.8%            |       0.17  |
| M3      | Cross-sector           |   5357 |  0.16  |     0.015 | <0.001    | 44.6%            |       0.164 |
| M4      | International          |   5357 |  0.03  |     0.016 | 0.068     | 7.0%             |       0.148 |
| M5      | log(1+n organizations) |   5357 |  0.149 |     0.022 | <0.001    | 41.0%            |       0.155 |

Coefficient interpretation is on a log10 outcome scale. The `percent_change` column reports `(10^coef - 1) * 100`. The largest absolute main-model coefficient is Company: coef=0.267, corresponding to 84.8% difference in max-row compute under the one-focal FE specification.

## Appendix: Full Conditional Model

The full model includes all five institutional terms simultaneously and should be read as a conditional association check, not the primary estimand.

| model   | term_label             |   nobs |   coef |   std_err | p_value   | percent_change   |   r_squared |
|:--------|:-----------------------|-------:|-------:|----------:|:----------|:-----------------|------------:|
| A1      | Company                |   5357 |  0.474 |     0.042 | <0.001    | 197.8%           |       0.201 |
| A1      | Industry-academia      |   5357 | -0.299 |     0.047 | <0.001    | -49.8%           |       0.201 |
| A1      | Cross-sector           |   5357 |  0.057 |     0.023 | 0.015     | 13.9%            |       0.201 |
| A1      | International          |   5357 | -0.023 |     0.019 | 0.222     | -5.2%            |       0.201 |
| A1      | log(1+n organizations) |   5357 |  0.085 |     0.031 | 0.006     | 21.6%            |       0.201 |

## Strict Single-Focal Compatibility Table

This table preserves the previous strict-output file shape. It uses the same strict raw outcome and `is_strict == 1` sample as the main regression table.

| model   | term_label             |   nobs |   coef |   std_err | p_value   | percent_change   |   r_squared |
|:--------|:-----------------------|-------:|-------:|----------:|:----------|:-----------------|------------:|
| S1      | Company                |   5357 |  0.267 |     0.016 | <0.001    | 84.8%            |       0.191 |
| S2      | Industry-academia      |   5357 |  0.195 |     0.017 | <0.001    | 56.8%            |       0.17  |
| S3      | Cross-sector           |   5357 |  0.16  |     0.015 | <0.001    | 44.6%            |       0.164 |
| S4      | International          |   5357 |  0.03  |     0.016 | 0.068     | 7.0%             |       0.148 |
| S5      | log(1+n organizations) |   5357 |  0.149 |     0.022 | <0.001    | 41.0%            |       0.155 |

## Concentration of Reported GPU Capacity

Country concentration uses full-count paper-country reported GPU capacity, following the country analysis convention that folds HK and TW into CN before country de-duplication. Region, organization-type, and topic concentration metrics use the yearly top-20% high-compute tail.

| level             | metric                                   | detail                                                                                                                                                                                     |   value | numerator   | denominator   |
|:------------------|:-----------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------:|:------------|:--------------|
| Country           | Top-1 share of reported GPU capacity     | CN                                                                                                                                                                                         |   0.281 | 6,017,654   | 21,390,397    |
| Country           | Top-5 share of reported GPU capacity     | CN, US, ES, KR, GB                                                                                                                                                                         |   0.828 | 17,704,973  | 21,390,397    |
| Country           | HHI of reported GPU capacity             |                                                                                                                                                                                            |   0.195 |             |               |
| Country           | Gini of reported GPU capacity            |                                                                                                                                                                                            |   0.917 |             |               |
| Region            | Share of high-compute tail               | East Asia and Pacific                                                                                                                                                                      |   0.478 | 1,046       | 2,187         |
| Region            | Share of high-compute tail               | North America                                                                                                                                                                              |   0.366 | 800         | 2,187         |
| Region            | Share of high-compute tail               | Europe and Central Asia                                                                                                                                                                    |   0.115 | 252         | 2,187         |
| Region            | Share of high-compute tail               | Middle East and North Africa                                                                                                                                                               |   0.024 | 53          | 2,187         |
| Region            | Share of high-compute tail               | South Asia                                                                                                                                                                                 |   0.013 | 28          | 2,187         |
| Region            | Share of high-compute tail               | Sub-Saharan Africa                                                                                                                                                                         |   0.003 | 6           | 2,187         |
| Region            | Share of high-compute tail               | Latin America and Caribbean                                                                                                                                                                |   0.001 | 2           | 2,187         |
| Organization type | Company high-compute share               |                                                                                                                                                                                            |   0.58  | 997         | 1,720         |
| Organization type | Academia-only high-compute share         |                                                                                                                                                                                            |   0.41  | 705         | 1,720         |
| Organization type | Industry-academia high-compute share     |                                                                                                                                                                                            |   0.476 | 819         | 1,720         |
| Topic             | Top-5 topic share of high-compute papers | Multimodality and Language Grounding to Vision, Robotics and Beyond; Language Modeling; Efficient Methods for NLP; Multilingualism and Cross-Lingual NLP; Dialogue and Interactive Systems |   0.391 | 672         | 1,720         |

## Access-Regime Summary

Access regimes are mutually exclusive paper-level categories: academic-only, industry-only, industry-academia collaboration, other cross-sector collaboration, and other/mixed. Reported GPU capacity is lower-bound imputed maximum GPU-row TFLOP/s; the high-compute tail is the yearly top-20% group.

| access_regime                    |   papers | paper_share   |   median_reported_gpu_capacity_tflops |   q25_reported_gpu_capacity_tflops | q75_reported_gpu_capacity_tflops   |   top20_compute_tail_papers | top20_compute_tail_share   |
|:---------------------------------|---------:|:--------------|--------------------------------------:|-----------------------------------:|:-----------------------------------|----------------------------:|:---------------------------|
| Academic-only                    |     3005 | 43.6%         |                                   312 |                              113.8 | 714.6                              |                         472 | 15.7%                      |
| Industry-only                    |      341 | 4.9%          |                                   756 |                              113.8 | 2,496.0                            |                         162 | 47.5%                      |
| Industry-academia collaboration  |     2336 | 33.9%         |                                   448 |                              119.1 | 2,494.4                            |                         819 | 35.1%                      |
| Other cross-sector collaboration |     1140 | 16.5%         |                                   312 |                              119.1 | 1,248.0                            |                         251 | 22.0%                      |
| Other / mixed                    |       73 | 1.1%          |                                   312 |                              112   | 1,248.0                            |                          16 | 21.9%                      |

## Outputs

- `4.3/Institution/data/institution_fe_panel_lb1.csv`
- `4.3/Institution/data/institution_fe_main_models.csv`
- `4.3/Institution/data/institution_fe_full_model.csv`
- `4.3/Institution/data/institution_fe_strict_robustness.csv`
- `4.3/Institution/data/institution_concentration_metrics.csv`
- `4.3/Institution/data/institution_access_regime_summary.csv`
- `4.3/Institution/data/institution_fe_audit.json`
- `4.3/Institution/fig/institution_access_regime_compute.png`

## Review Risks

These models are descriptive fixed-effects associations, not causal estimates. All regression tables are now estimated on the stricter raw-compute sample, so coverage is narrower than the lower-bound imputed descriptive summaries.
