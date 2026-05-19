# Country-Level Compute Association

Core framing: this analysis describes compute associated with papers involving organizations from a country. It does not estimate national compute ownership or country-level compute capacity.

## Method

Input data: GPU-only paper-level compute and the GPU-only paper-organization long table. Quantifiable papers require positive `paper_max_row_compute_capability_gfimp_lb1`, `is_lb1_gfimp == 1`, and a valid publication year. The compute measure is paper-level maximum GPU-row normalized compute capacity in TFLOP/s.

Country assignment uses full counting. Hong Kong (HK) and Taiwan (TW) organization country codes are folded into China (CN) before de-duplication. For each paper, duplicate organization rows from the same country are collapsed to one paper-country observation. If a paper involves organizations from multiple countries, it contributes one full observation to each associated country. Therefore, the sum of country counts can exceed the number of papers.

The top-20% compute flag is computed within each publication year using the inclusive P80 cutoff. Ties at the cutoff can make the observed top-20% share differ from exactly 20%.

## Audit

- Quantifiable papers: 6,900
- Full-count paper-country observations: 9,628
- Countries observed: 87
- Main-text country threshold: full_count_n >= 50
- Countries meeting main threshold: 18

## Main-Threshold Countries With Highest Top-20% Compute Share

- CN: 30.4% (887/2915 full-count country-paper observations)
- CA: 27.3% (69/253 full-count country-paper observations)
- US: 26.6% (767/2884 full-count country-paper observations)
- JP: 26.6% (47/177 full-count country-paper observations)
- AE: 25.3% (24/95 full-count country-paper observations)
- KR: 21.8% (90/412 full-count country-paper observations)
- SG: 21.7% (75/346 full-count country-paper observations)
- AU: 20.8% (36/173 full-count country-paper observations)

## Largest Main-Threshold Country-Associated Paper Volumes

- CN: 2915 full-count quantifiable papers
- US: 2884 full-count quantifiable papers
- GB: 534 full-count quantifiable papers
- KR: 412 full-count quantifiable papers
- DE: 380 full-count quantifiable papers
- SG: 346 full-count quantifiable papers
- CA: 253 full-count quantifiable papers
- IN: 187 full-count quantifiable papers

## Figure Captions

**Country compute volume and top-20% compute concentration.** Each point represents a country associated with at least 50 quantifiable papers. Hong Kong and Taiwan are included under China. Multi-country papers are counted once for each associated country, after collapsing duplicate countries within the same paper. The x-axis shows full-count paper volume, the y-axis shows the share of papers in the within-year top 20% compute group, and point size is proportional to the full-count number of top-20% papers. The dashed line marks the 0.20 nominal top-20% baseline.

**Country-level concentration of top-20% compute papers.** The lollipop plot displays the 12 countries with the highest top-20% compute share among countries meeting the main sample threshold. The vertical reference line at 0.20 corresponds to the nominal expected share under no country-level concentration. Countries above this line are overrepresented among within-year top-20% compute papers among their country-associated quantifiable papers.
