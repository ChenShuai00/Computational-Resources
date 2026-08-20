# Overall GPU capability models: unified export

## Unified table

| Outcome | N | Capacity β | SE capacity | p capacity | Capacity effect | Count β | SE count | p count | Count effect | Ampere β | SE ampere | p ampere | Ampere effect | ΔR² capacity | ΔR² joint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NLP topic-year percentile | 2194 | 0.0352 | 0.0115 | 0.002 | 3.52 pp | 0.0457 | 0.0133 | 5.61e-04 | 4.57 pp | 0.0416 | 0.0135 | 0.002 | 4.16 pp | 0.004207 | 0.009258 |
| OpenAlex field-normalized citation percentile | 2194 | 0.0126 | 0.0088 | 0.151 | 1.26 pp | 0.0196 | 0.0101 | 0.052 | 1.96 pp | 0.0202 | 0.0105 | 0.055 | 2.02 pp | 9.27e-04 | 0.003283 |
| Log citations | 2194 | 0.1719 | 0.0491 | 4.69e-04 | 18.8% | 0.2206 | 0.0571 | 1.12e-04 | 24.7% | 0.1396 | 0.0552 | 0.011 | 15.0% | 0.005157 | 0.008647 |
| PPML citations | 2194 | 0.4775 | 0.0852 | 2.06e-08 | 61.2% | 0.5264 | 0.0845 | 4.64e-10 | 69.3% | 0.2699 | 0.1321 | 0.041 | 31.0% | -- | -- |
| Year-by-venue top-decile cited | 2194 | 0.0384 | 0.0138 | 0.005 | 3.84 pp | 0.0441 | 0.0162 | 0.006 | 4.41 pp | 0.0217 | 0.0150 | 0.148 | 2.17 pp | 0.004251 | 0.005128 |
| Award | 5357 | 0.0086 | 0.0045 | 0.056 | 0.86 pp | 0.0133 | 0.0056 | 0.017 | 1.33 pp | -0.0025 | 0.0053 | 0.645 | -0.25 pp | 0.001087 | 0.001839 |

## Locked specification

- Primary outcome: NLP topic-year citation percentile, computed in the full
  6,900-paper analysis master corpus before selecting the strict estimation sample.
- Capacity model: `y ~ log10(max-row GPU capability) + controls`.
- Joint model: `y ~ log10(max-row GPU count) + Ampere-or-newer/equivalent + controls`.
- Controls: year-by-venue fixed effects, primary-topic fixed effects, team-size
  groups, and organization-count groups.
- OLS and LPM use HC3 standard errors; PPML uses HC0 sandwich standard errors.
- Citation outcomes use 2020-2023. Award uses 2020-2025 and has
  111 positives.
- The five citation outcomes use identical paper IDs: `True`.
- Raw p-values are displayed. `p_holm_secondary` in the full CSV applies Holm
  correction across the five secondary outcomes separately for capacity, count,
  and Ampere-or-newer/equivalent terms.

## Exact primary-outcome incremental fit

- Full-model R2: `0.090175967071377872`
- Controls-only R2: `0.085968715401789852`
- Delta R2: `0.0042072516695880191`

## Source artifacts

- Full machine-readable results: `data/overall_gpu_capability_models.csv`
- Compact requested table: `data/overall_gpu_capability_table.csv`
- Audit and frozen-input hashes: `data/overall_gpu_capability_model_audit.json`
