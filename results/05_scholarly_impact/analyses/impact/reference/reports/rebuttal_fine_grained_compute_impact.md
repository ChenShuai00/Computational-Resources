# Rebuttal: fine-grained compute-impact analysis

We added a rebuttal-only analysis that decomposes the existing reported GPU
capacity measure into available hardware dimensions. The analysis remains
associational: it tests whether reported GPU quantity and hardware strength are
associated with citation and award outcomes after the same controls used in
Section 4.4.

## Main takeaway

The clearest fine-grained association is GPU quantity. A 10x increase in the
number of GPUs in the max-capacity row is positively associated with log
citations, expected citation counts, high-citation status, and awards. In
contrast, single-GPU capability and the Ampere-or-newer generation indicator are
not consistently significant once the same controls and GPU quantity are
included.

We do not analyze GPU-hours or training time in this rebuttal analysis.
No systematic GPU-hour, training-time, or wall-clock duration columns were found in the released paper-level or row-level GPU inputs.

## Validation and samples

- Max-row validation matched 6,900 of 6,900 GPU papers.
- Positive strict max-row compute rows: 5,360.
- Citation sample: 2,195 strict raw GPU papers from 2020-2023.
- Award sample: 5,360 strict raw GPU papers from 2020-2025.
- Citation-sample max-row generations: Ampere: 878, Volta: 837, Turing: 469, Ada Lovelace: 6, CDNA 2: 2, Hopper: 2, Pascal: 1.
- Award-sample max-row generations: Ampere: 3298, Volta: 1033, Turing: 551, Hopper: 263, Ada Lovelace: 205, CDNA 2: 6, Pascal: 1, Vega: 1, Blackwell: 1, CDNA 3: 1.

## Fine-grained model results

| Outcome | Model | Term | N | Coef. | SE | p | 95% CI | Effect | Effect scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLS: log1p citations | Number of GPUs in max-capacity row | Number of GPUs | 2,194 | 0.222 | 0.057 | 1.03e-04 | [0.110, 0.334] | 24.8% | percent per 10x |
| OLS: log1p citations | Single-GPU capability in max row | Single-GPU capability | 2,194 | 0.050 | 0.125 | 0.686 | [-0.194, 0.295] | 5.2% | percent per 10x |
| OLS: log1p citations | GPU count + single-GPU capability | Number of GPUs | 2,194 | 0.226 | 0.058 | 9.49e-05 | [0.112, 0.339] | 25.3% | percent per 10x |
| OLS: log1p citations | GPU count + single-GPU capability | Single-GPU capability | 2,194 | -0.048 | 0.126 | 0.704 | [-0.295, 0.199] | -4.7% | percent per 10x |
| OLS: log1p citations | GPU count + Ampere-or-newer indicator | Number of GPUs | 2,194 | 0.221 | 0.057 | 1.12e-04 | [0.109, 0.333] | 24.7% | percent per 10x |
| OLS: log1p citations | GPU count + Ampere-or-newer indicator | Ampere-or-newer generation | 2,194 | 0.140 | 0.055 | 0.011 | [0.031, 0.248] | 15.0% | percent vs older generation |
| OLS: log1p citations | Total paper GPU count robustness | Total paper GPU count | 2,194 | 0.206 | 0.056 | 2.40e-04 | [0.096, 0.316] | 22.8% | percent per 10x |
| PPML: citation count | Number of GPUs in max-capacity row | Number of GPUs | 2,194 | 0.532 | 0.084 | 2.65e-10 | [0.367, 0.697] | 70.3% | percent per 10x |
| PPML: citation count | Single-GPU capability in max row | Single-GPU capability | 2,194 | 0.486 | 0.335 | 0.148 | [-0.172, 1.143] | 62.5% | percent per 10x |
| PPML: citation count | GPU count + single-GPU capability | Number of GPUs | 2,194 | 0.520 | 0.082 | 2.27e-10 | [0.360, 0.681] | 68.3% | percent per 10x |
| PPML: citation count | GPU count + single-GPU capability | Single-GPU capability | 2,194 | 0.204 | 0.330 | 0.537 | [-0.443, 0.850] | 22.6% | percent per 10x |
| PPML: citation count | GPU count + Ampere-or-newer indicator | Number of GPUs | 2,194 | 0.526 | 0.084 | 4.64e-10 | [0.361, 0.692] | 69.3% | percent per 10x |
| PPML: citation count | GPU count + Ampere-or-newer indicator | Ampere-or-newer generation | 2,194 | 0.270 | 0.132 | 0.041 | [0.011, 0.529] | 31.0% | percent vs older generation |
| PPML: citation count | Total paper GPU count robustness | Total paper GPU count | 2,194 | 0.514 | 0.084 | 8.21e-10 | [0.350, 0.679] | 67.3% | percent per 10x |
| LPM: high-cited top 10% | Number of GPUs in max-capacity row | Number of GPUs | 2,194 | 0.044 | 0.016 | 0.006 | [0.013, 0.076] | 4.43 pp | percentage points per 10x |
| LPM: high-cited top 10% | Single-GPU capability in max row | Single-GPU capability | 2,194 | 0.039 | 0.034 | 0.241 | [-0.026, 0.105] | 3.94 pp | percentage points per 10x |
| LPM: high-cited top 10% | GPU count + single-GPU capability | Number of GPUs | 2,194 | 0.043 | 0.016 | 0.009 | [0.010, 0.075] | 4.26 pp | percentage points per 10x |
| LPM: high-cited top 10% | GPU count + single-GPU capability | Single-GPU capability | 2,194 | 0.021 | 0.034 | 0.542 | [-0.046, 0.088] | 2.08 pp | percentage points per 10x |
| LPM: high-cited top 10% | GPU count + Ampere-or-newer indicator | Number of GPUs | 2,194 | 0.044 | 0.016 | 0.006 | [0.012, 0.076] | 4.41 pp | percentage points per 10x |
| LPM: high-cited top 10% | GPU count + Ampere-or-newer indicator | Ampere-or-newer generation | 2,194 | 0.022 | 0.015 | 0.148 | [-0.008, 0.051] | 2.17 pp | percentage points vs older generation |
| LPM: high-cited top 10% | Total paper GPU count robustness | Total paper GPU count | 2,194 | 0.047 | 0.016 | 0.003 | [0.016, 0.078] | 4.68 pp | percentage points per 10x |
| LPM: award | Number of GPUs in max-capacity row | Number of GPUs | 5,357 | 0.013 | 0.006 | 0.017 | [0.002, 0.024] | 1.33 pp | percentage points per 10x |
| LPM: award | Single-GPU capability in max row | Single-GPU capability | 5,357 | -0.004 | 0.009 | 0.693 | [-0.021, 0.014] | -0.35 pp | percentage points per 10x |
| LPM: award | GPU count + single-GPU capability | Number of GPUs | 5,357 | 0.014 | 0.006 | 0.012 | [0.003, 0.025] | 1.41 pp | percentage points per 10x |
| LPM: award | GPU count + single-GPU capability | Single-GPU capability | 5,357 | -0.009 | 0.009 | 0.329 | [-0.026, 0.009] | -0.88 pp | percentage points per 10x |
| LPM: award | GPU count + Ampere-or-newer indicator | Number of GPUs | 5,357 | 0.013 | 0.006 | 0.017 | [0.002, 0.024] | 1.33 pp | percentage points per 10x |
| LPM: award | GPU count + Ampere-or-newer indicator | Ampere-or-newer generation | 5,357 | -0.002 | 0.005 | 0.645 | [-0.013, 0.008] | -0.25 pp | percentage points vs older generation |
| LPM: award | Total paper GPU count robustness | Total paper GPU count | 5,357 | 0.012 | 0.005 | 0.024 | [0.002, 0.023] | 1.22 pp | percentage points per 10x |

## Increment over controls-only models

This table compares `y ~ controls` with
`y ~ log10(number of GPUs) + controls` on the same estimation sample. For OLS
and LPM outcomes, the table reports the R2 increase. For the PPML citation-count
model, R2 is not defined, so the table reports AIC improvement instead.

| Outcome | N | GPU-count coef. | SE | p | Controls-only R2 | + GPU-count R2 | Delta R2 | AIC improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLS: log1p citations | 2,194 | 0.222 | 0.057 | 1.03e-04 | 0.207 | 0.213 | 0.006 |  |
| PPML: citation count | 2,194 | 0.532 | 0.084 | 2.65e-10 |  |  |  | 4058.1 |
| LPM: high-cited top 10% | 2,194 | 0.044 | 0.016 | 0.006 | 0.041 | 0.045 | 0.004 |  |
| LPM: award | 5,357 | 0.013 | 0.006 | 0.017 | 0.030 | 0.032 | 0.002 |  |

## Source artifacts

- Full results: `data/rebuttal_fine_grained_compute_impact.csv`
- Controls-only comparison: `data/rebuttal_fine_grained_compute_incremental_fit.csv`
- Audit: `data/rebuttal_fine_grained_compute_impact_audit.json`
