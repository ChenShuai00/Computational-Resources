# Common-sample confounder controls for citation outcomes

## Main table

| Outcome | Specification | N | β | SE | 95% CI | p | R² controls-only | R² full | ΔR² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NLP topic-year percentile | Common-sample baseline | 2077 | 0.0313 | 0.0118 | [0.0081, 0.0545] | 0.008 | 0.089544 | 0.092777 | 0.003233 |
| NLP topic-year percentile | + pre-publication controls | 2077 | 0.0274 | 0.0121 | [0.0037, 0.0512] | 0.024 | 0.133011 | 0.135354 | 0.002342 |
| NLP topic-year percentile | + public artifact | 2077 | 0.0265 | 0.0121 | [0.0028, 0.0502] | 0.028 | 0.136231 | 0.138415 | 0.002184 |
| OpenAlex field-normalized percentile | Common-sample baseline | 2077 | 0.0116 | 0.0091 | [-0.0062, 0.0295] | 0.202 | 0.090854 | 0.091635 | 7.80e-04 |
| OpenAlex field-normalized percentile | + pre-publication controls | 2077 | 0.0105 | 0.0094 | [-0.0080, 0.0290] | 0.264 | 0.117182 | 0.117783 | 6.01e-04 |
| OpenAlex field-normalized percentile | + public artifact | 2077 | 0.0102 | 0.0095 | [-0.0084, 0.0287] | 0.282 | 0.117935 | 0.118497 | 5.62e-04 |
| Log citations | Common-sample baseline | 2077 | 0.1617 | 0.0507 | [0.0623, 0.2611] | 0.001 | 0.210911 | 0.215346 | 0.004435 |
| Log citations | + pre-publication controls | 2077 | 0.1438 | 0.0521 | [0.0417, 0.2458] | 0.006 | 0.245451 | 0.248755 | 0.003304 |
| Log citations | + public artifact | 2077 | 0.1404 | 0.0520 | [0.0385, 0.2422] | 0.007 | 0.247679 | 0.250826 | 0.003147 |

## Specification and interpretation

- All outcomes and specifications use the same 2,077
  paper complete-case sample. The 2,194-paper baseline
  sample is not used for the baseline row in this table.
- The common-sample baseline includes reported max-row GPU capability,
  year-by-venue fixed effects, primary-topic fixed effects, team-size groups, and
  organization-count groups.
- Pre-publication controls are three-year author and institution
  visibility/history proxies: team-member maximum prior citations, team-member
  mean prior publications, maximum prior institutional citations, prior
  organization publications and partner-organization histories, and corporate,
  industry-academia, and international-collaboration indicators.
- Public-artifact availability is added only after all pre-publication proxies
  and is secondary robustness because it may be contemporaneous with or follow
  the compute choice.
- For every row, controls-only R2 removes only GPU capability and retains every
  other control in that row. Delta R2 is full R2 minus that matched controls-only R2.
- Estimates are conditional associations and do not identify causal effects.

## Complete five-step ladders

| Outcome | Specification | N | β | SE | 95% CI | p | R² controls-only | R² full | ΔR² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NLP topic-year percentile | Common-sample baseline | 2077 | 0.0313 | 0.0118 | [0.0081, 0.0545] | 0.008 | 0.089544 | 0.092777 | 0.003233 |
| NLP topic-year percentile | + author-history proxies | 2077 | 0.0224 | 0.0117 | [-5.62e-04, 0.0453] | 0.056 | 0.123870 | 0.125508 | 0.001638 |
| NLP topic-year percentile | + institution/collaboration proxies | 2077 | 0.0332 | 0.0122 | [0.0092, 0.0572] | 0.007 | 0.106156 | 0.109602 | 0.003446 |
| NLP topic-year percentile | + pre-publication controls | 2077 | 0.0274 | 0.0121 | [0.0037, 0.0512] | 0.024 | 0.133011 | 0.135354 | 0.002342 |
| NLP topic-year percentile | + public artifact | 2077 | 0.0265 | 0.0121 | [0.0028, 0.0502] | 0.028 | 0.136231 | 0.138415 | 0.002184 |
| OpenAlex field-normalized percentile | Common-sample baseline | 2077 | 0.0116 | 0.0091 | [-0.0062, 0.0295] | 0.202 | 0.090854 | 0.091635 | 7.80e-04 |
| OpenAlex field-normalized percentile | + author-history proxies | 2077 | 0.0067 | 0.0090 | [-0.0110, 0.0244] | 0.461 | 0.109379 | 0.109632 | 2.54e-04 |
| OpenAlex field-normalized percentile | + institution/collaboration proxies | 2077 | 0.0140 | 0.0095 | [-0.0046, 0.0327] | 0.140 | 0.100279 | 0.101348 | 0.001070 |
| OpenAlex field-normalized percentile | + pre-publication controls | 2077 | 0.0105 | 0.0094 | [-0.0080, 0.0290] | 0.264 | 0.117182 | 0.117783 | 6.01e-04 |
| OpenAlex field-normalized percentile | + public artifact | 2077 | 0.0102 | 0.0095 | [-0.0084, 0.0287] | 0.282 | 0.117935 | 0.118497 | 5.62e-04 |
| Log citations | Common-sample baseline | 2077 | 0.1617 | 0.0507 | [0.0623, 0.2611] | 0.001 | 0.210911 | 0.215346 | 0.004435 |
| Log citations | + author-history proxies | 2077 | 0.1261 | 0.0500 | [0.0282, 0.2240] | 0.012 | 0.239712 | 0.242385 | 0.002673 |
| Log citations | + institution/collaboration proxies | 2077 | 0.1672 | 0.0527 | [0.0639, 0.2704] | 0.002 | 0.222430 | 0.226916 | 0.004486 |
| Log citations | + pre-publication controls | 2077 | 0.1438 | 0.0521 | [0.0417, 0.2458] | 0.006 | 0.245451 | 0.248755 | 0.003304 |
| Log citations | + public artifact | 2077 | 0.1404 | 0.0520 | [0.0385, 0.2422] | 0.007 | 0.247679 | 0.250826 | 0.003147 |

## Source artifacts

- Full machine-readable results: `data/common_sample_confounder_models.csv`
- Requested display table: `data/common_sample_confounder_table.csv`
- Sample and input audit: `data/common_sample_confounder_model_audit.json`
