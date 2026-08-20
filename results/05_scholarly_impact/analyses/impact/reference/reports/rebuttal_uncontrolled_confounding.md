# Revision: measured and unmeasured confounding

## Interpretation

The analyses in this report estimate conditional associations between reported
GPU compute and scholarly-recognition outcomes. They do not identify a causal
effect. Author and institution controls use only information from the three
calendar years preceding each focal paper. Public-artifact availability is a
secondary robustness control because it may be contemporaneous with, or follow,
the compute choice. Social-media attention and direct industrial promotion are
not treated as pre-publication confounders because reliable time-stamped measures
are unavailable and these factors may be mediators.

## Coverage audit

| Sample | N | History complete | Author IDs complete | Institution IDs complete | Artifact rate | Eligible for main text |
| --- | --- | --- | --- | --- | --- | --- |
| citation_2020_2023 | 2,195 | 94.6% | 96.5% | 94.7% | 30.8% | 1 |
| award_2020_2025 | 5,360 | 49.6% | 87.2% | 49.7% | 28.7% | 0 |

The expanded model is eligible for main-text interpretation only when at least
90% of the relevant sample has complete pre-publication history controls.

## Nested citation controls

| Model | Controls | N | Coef. | SE | p | CI low | CI high | R2 | Delta R2 | Attenuation vs M0 | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | Baseline Spec 7 | 2,077 | 0.162 | 0.051 | 0.001 | 0.062 | 0.261 | 0.215 | 0.004 | 0.0% | ok |
| M1 | + pre-publication author history | 2,077 | 0.126 | 0.050 | 0.012 | 0.028 | 0.224 | 0.242 | 0.003 | 22.0% | ok |
| M2 | + pre-publication institution visibility and collaboration | 2,077 | 0.167 | 0.053 | 0.002 | 0.064 | 0.270 | 0.227 | 0.004 | -3.4% | ok |
| M3 | + all pre-publication confounder proxies | 2,077 | 0.144 | 0.052 | 0.006 | 0.042 | 0.246 | 0.249 | 0.003 | 11.1% | ok |
| M4 | + public artifact (secondary robustness) | 2,077 | 0.140 | 0.052 | 0.007 | 0.039 | 0.242 | 0.251 | 0.003 | 13.2% | ok |

On the common N=2,077 sample, adding all pre-publication controls changes the coefficient from 0.162 to 0.144 (11.1% attenuation; p=0.006). Adding public-artifact availability gives 0.140 (13.2% attenuation; p=0.007). This is stability to the measured controls, not evidence of causal identification.

## Alternative citation outcomes under M3

| Outcome | Family | Model | N | Coef. | SE | p | CI low | CI high | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cited_by_count | poisson | M3 | 2,077 | 0.407 | 0.094 | 1.54e-05 | 0.223 | 0.592 | ok |
| citation_normalized_percentile | ols | M3 | 2,077 | 0.011 | 0.009 | 0.264 | -0.008 | 0.029 | ok |
| is_highly_cited_all_yv | lpm | M3 | 2,077 | 0.027 | 0.014 | 0.056 | -6.72e-04 | 0.055 | ok |

## Listed-author fixed effects

| Model | N | Authors | Coef. | SE | p | CI low | CI high | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first-listed author FE | 388 | 168 | -0.218 | 0.216 | 0.312 | -0.640 | 0.204 | ok |
| last-listed author FE | 1,182 | 322 | 0.089 | 0.097 | 0.358 | -0.101 | 0.278 | ok |

The first-listed-author estimate is -0.218 (p=0.312) and the last-listed-author estimate is 0.089 (p=0.358); both confidence intervals include zero, so these subset comparisons do not provide supportive fixed-effect evidence.

These subset models use only listed authors with at least two papers and
within-author variation in reported compute. Standard errors are clustered by
the corresponding listed-author identifier; last-listed authors are not assumed
to be senior authors.

## Unobserved-confounding sensitivity

| Model | N | Partial R2 | RV, zero | RV, p>=.05 | Benchmark x | Assumed treatment R2 | Assumed outcome R2 | Bias-adjusted coef. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 2,194 | 0.007 | 0.078 | 0.038 | 1 | 0.089 | 0.048 | 0.025 |
| M0 | 2,194 | 0.007 | 0.078 | 0.038 | 2 | 0.178 | 0.097 | -0.136 |
| M3 | 2,077 | 0.004 | 0.064 | 0.022 | 1 | 0.074 | 0.045 | 0.014 |
| M3 | 2,077 | 0.004 | 0.064 | 0.022 | 2 | 0.148 | 0.089 | -0.126 |

The M3 association is sensitive to omitted confounding: RV(p>=.05)=0.022; a confounder benchmarked at the strongest observed control group reduces the coefficient from 0.144 to 0.014, and the 2x benchmark gives -0.126. The evidence should therefore be described as fragile.

Robustness values and benchmark adjustments follow the Cinelli-Hazlett omitted-
variable-bias framework. Sensitivity calculations use classical OLS standard
errors, while the reported control-ladder estimates use HC3 standard errors.

## Exploratory award models

| Model | N | Awards | OR per 10x | OR CI low | OR CI high | p | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 2,660 | 48 | 1.270 | 0.786 | 1.992 | 0.319 | ok |
| M3 | 2,660 | 48 | 1.357 | 0.831 | 2.155 | 0.215 | ok |
| M4 | 2,660 | 48 | 1.332 | 0.816 | 2.112 | 0.243 | ok |

The expanded Firth award estimate uses N=2,660 with 48 awards and gives OR=1.357 (95% CI [0.831, 2.155], p=0.215). Because award-history coverage is below the 90% threshold, this remains an appendix-only exploratory result.

Award models use Firth bias reduction and profile-likelihood intervals. Because
awards are rare, these results remain exploratory regardless of statistical
significance.

## Suggested response to the reviewer

We agree that measured controls cannot eliminate all confounding in the
compute-recognition relationship. We therefore added pre-publication author and
institution visibility controls, organization-history and industry-collaboration
controls, listed-author fixed-effect comparisons, and a formal omitted-variable-
bias sensitivity analysis. We additionally treat public-artifact availability as
a secondary robustness control and use bias-reduced logistic regression for the
sparse award outcome. We now describe all estimates as conditional associations,
report non-supportive alternative outcomes, and explicitly retain residual
confounding as a limitation. The formal sensitivity analysis indicates that omitted confounding of a plausible observed-control magnitude could materially attenuate or reverse the estimate, so we characterize the evidence as fragile.
