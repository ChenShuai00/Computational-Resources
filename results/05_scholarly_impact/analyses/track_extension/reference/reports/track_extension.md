# Main Conference and Findings track-extension robustness

## Scope

This appendix analysis extends the ACL/EMNLP/NAACL main-conference corpus to
Findings papers. It preserves the paper's text-reported GPU measurement boundary:
reported capability is theoretical peak configuration capacity, not GPU-hours,
realized utilization, cost, energy, or causal treatment intensity.

## Sample comparison

| Characteristic | Main Conference | Findings | Main + Findings |
|---|---:|---:|---:|
| Total papers | 13,921 | 9,917 | 23,838 |
| Papers reporting standardized GPU model | 6,900 | 5,824 | 12,724 |
| Reporting rate | 49.6% | 58.7% | 53.4% |
| Papers reporting GPU model + count | 5,360 | 4,186 | 9,546 |
| Strict reporting rate | 38.5% | 42.2% | 40.0% |
| Citation-analysis sample | 2,194 | 1,620 | 3,814 |
| Median reported GPU count | 4 | 2 | 3 |
| Median reported GPU capability | 455.2 | 359.7 | 448.0 |

Median GPU count and capability are calculated among 2020-2025 strict papers
that report both a standardized GPU model and an explicit count.

## Core citation regressions

| Outcome | Main Conference β (SE) [ΔR²] | Findings β (SE) [ΔR²] | Pooled β (SE) [ΔR²] | Findings − Main difference (p) |
|---|---:|---:|---:|---:|
| NLP topic-year citation percentile | 0.035 (0.011) [0.0042] | 0.046 (0.014) [0.0071] | 0.039 (0.009) [0.0051] | +0.011 (0.551) |
| OpenAlex field-normalized percentile | 0.013 (0.009) [0.0009] | 0.040 (0.011) [0.0073] | 0.024 (0.007) [0.0029] | +0.027 (0.051) |
| log(1+citations) | 0.172 (0.049) [0.0052] | 0.207 (0.054) [0.0083] | 0.185 (0.036) [0.0061] | +0.035 (0.630) |
| Citation count, PPML | 0.477 (0.085) [—] | 0.390 (0.090) [—] | 0.456 (0.066) [—] | -0.088 (0.481) |
| Top-10% cited | 0.038 (0.014) [0.0043] | 0.059 (0.018) [0.0093] | 0.047 (0.011) [0.0062] | +0.020 (0.372) |
| N | 2,194 | 1,620 | 3,814 | 3,814 |

Cells report β (robust SE) [incremental R²]. Incremental R² is the full Spec-7
model R² minus the controls-only model R² on the identical sample. OLS and LPM
use HC3 robust standard errors; PPML uses HC0 robust standard errors and has no
ordinary R². The final column is the Findings-minus-main slope difference with
its two-sided Wald-test p-value. Holm-adjusted difference-test p-values are
available in the machine-readable table.

The pooled primary estimate is 3.90 percentile
points per tenfold increase in reported GPU capability
(N=3,814, p=6.95e-06,
ΔR²=0.0051). None of the five
Main-versus-Findings slope differences is significant at the 0.05 level.

## Does more reported capability ensure high impact?

| Track | High-impact rate among high-capability papers | High-impact rate among other papers | Risk ratio | High-capability share among high-impact papers | High-capability papers not high-impact |
|---|---:|---:|---:|---:|---:|
| Main Conference | 14.5% | 9.1% | 1.59 | 28.6% | 85.5% |
| Findings | 15.8% | 9.0% | 1.75 | 30.6% | 84.2% |
| Main + Findings | 14.7% | 9.2% | 1.60 | 28.6% | 85.3% |

In Findings, the high-impact rate is higher among annual top-20% capability
papers (15.8% versus
9.0%; risk ratio
1.75), but 84.2%
of high-capability papers are not venue-year-track top-10% cited. Reported
capability is therefore positively associated with citation impact but neither
sufficient nor necessary for high impact.

## Interpretation boundaries

- Results are conditional associations among papers with reportable and
  standardizable GPU evidence; they are not causal effects.
- Citation outcomes are incomplete proxies for scholarly value.
- The extension broadens publication tracks within ACL-family venues but does
  not establish generalizability to journals, workshops, arXiv, industrial
  reports, or other machine-learning venues.
- Findings lacks a directly comparable formal-award outcome, so award models are
  intentionally excluded.
