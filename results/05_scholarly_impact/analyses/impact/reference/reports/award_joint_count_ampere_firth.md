# Joint GPU-count plus Ampere Firth award model

## Result

| Term | N | Awards | Beta | LRT SE | Profile-likelihood 95% CI | Penalized-LRT p | Holm p | OR | OR 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU count | 5357 | 111 | 0.5380 | 0.1883 | [0.1730, 0.8907] | 0.004 | 0.009 | 1.713 | [1.189, 2.437] |
| Ampere-or-newer/equivalent | 5357 | 111 | -0.1115 | 0.2734 | [-0.6322, 0.4362] | 0.683 | 0.683 | 0.894 | [0.531, 1.547] |

The model uses the same 5,357-paper 2020-2025 award sample as
the joint LPM and contains 111 award-positive papers. A
tenfold increase in reported max-row GPU count has OR=1.713
(profile-likelihood 95% CI [1.189, 2.437],
penalized-LRT p=0.004265). Conditional on count and the
shared controls, Ampere-or-newer/equivalent hardware has OR=0.894
(profile-likelihood 95% CI [0.531, 1.547],
penalized-LRT p=0.6833). The 2-df joint penalized-LRT
statistic is 8.3178 (p=0.01563).

## Locked specification

`Award ~ log10(max-row GPU count) + Ampere-or-newer/equivalent + year-by-venue FE + topic FE + team-size groups + organization-count groups`

- The two hardware terms enter additively; there is no interaction.
- The GPU count and generation indicator refer to the same maximum-capability
  reported GPU row used by the unified main-table analysis.
- Single-term p-values are penalized likelihood-ratio tests, SEs are LRT
  back-corrected, and confidence intervals are profile-likelihood intervals.
- Holm-adjusted p-values across the two prespecified focal terms are included in
  the machine export as secondary multiplicity diagnostics.
- The omnibus test jointly constrains both focal coefficients to zero within the
  same full Firth design matrix.
- This is a sparse-outcome robustness analysis and does not identify a causal
  effect of hardware on awards.

## Source artifacts

- Full term-level results: `data/award_joint_count_ampere_firth_models.csv`
- Sample, estimator, input-hash, and convergence audit: `data/award_joint_count_ampere_firth_audit.json`
