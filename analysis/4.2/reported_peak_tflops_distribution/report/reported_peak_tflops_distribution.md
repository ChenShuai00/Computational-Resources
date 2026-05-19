# RQ1: Reported Peak TFLOP/s Distribution

## Figure Contract
Core conclusion: GPU-only ACL/EMNLP/NAACL papers report a highly right-skewed peak-compute distribution, with the typical paper moving from hundreds to low-thousands of TFLOP/s after 2023 while a small number of very large GPU allocations form the extreme tail.
Figure archetype: quantitative grid.
Target output: PNG plus source CSV tables.
Backend: Python/matplotlib only.
Final size: standalone single-panel figures for distribution and annual distribution.
Panel map: each single-panel figure is exported separately without panel-letter labels or plot titles.
Evidence hierarchy: the distribution figure is the hero evidence for skew; the annual distribution figure tests whether the distribution shifts over time.
Statistics needed: descriptive counts, quantiles, and coverage of papers with both reported GPU quantity and benchmark peak performance; no inferential test is used.
Source data needed: GPU normalized rows with `gpu_num` and `benchmark_max_performance`, aggregated to unique papers.
Image-integrity notes: PNG exports are generated directly from source tables; no raster image manipulation is used.
Reviewer risk: the metric combines paper-reported GPU counts with benchmark database peak performance, so it estimates reported peak capacity rather than measured training compute or realized utilization.

## Method
Input data: `data/acl_emnlp_naacl_2020_2025_gpu_normalized_gpu_only.xlsx`, sheet `merged_gpu_normalized`.
The raw `benchmark_max_performance` field is stored in FLOP/s. For each GPU row, reported peak TFLOP/s is calculated as `gpu_num * benchmark_max_performance / 1e12`. Rows must have a positive reported GPU quantity and a positive benchmark peak performance. If a paper reports multiple GPU rows, paper-level reported peak TFLOP/s is the maximum valid GPU-row value within that paper.
Coverage: 5,505/6,900 GPU-only papers (79.8%) have enough information for this metric.
The extreme-value-excluded figure uses a reproducible p99 trimming rule: papers above 39,936 TFLOP/s are excluded, leaving 5,454 papers.
The loose `is_lb1_gfimp` sensitivity version uses `data/paper_compute_level_gpu_only.xlsx`: row-level TFLOP/s is `compute_capability_gfimp_lb1 / 1e12`, equivalent to `gpu_num_filled * effective_flops_gfimp / 1e12`, and paper-level values again take the maximum row within each paper.

## Main Result
Across 5,505 papers, the median reported peak capacity is 624 TFLOP/s, the IQR is 125-2,496 TFLOP/s, and the p95 is 7,168 TFLOP/s. The distribution is strongly long-tailed: the p99 reaches 39,936 TFLOP/s and the maximum reaches 8,626,176 TFLOP/s.
The annual median peaks in 2025 at 1,321 TFLOP/s, while the annual p95 peaks in 2024 at 9,984 TFLOP/s.

## Annual Distribution
- 2020: median 91, IQR 12-448, p95 4,992 TFLOP/s (n=330).
- 2021: median 112, IQR 33-448, p95 4,992 TFLOP/s (n=504).
- 2022: median 284, IQR 112-896, p95 4,992 TFLOP/s (n=654).
- 2023: median 448, IQR 112-1,234, p95 4,992 TFLOP/s (n=859).
- 2024: median 896, IQR 310-4,992, p95 9,984 TFLOP/s (n=1156).
- 2025: median 1,321, IQR 624-4,992, p95 9,984 TFLOP/s (n=2002).

## Venue Distribution
- ACL: median 624, p95 9,984 TFLOP/s (n=2180).
- EMNLP: median 624, p95 6,048 TFLOP/s (n=2809).
- NAACL: median 624, p95 6,048 TFLOP/s (n=516).

## Largest Reported Allocations
- 2024.acl-long.799: 8,626,176 TFLOP/s, 13824 reported GPU units.
- 2025.acl-long.123: 774,144 TFLOP/s, 1024 reported GPU units.
- 2022.naacl-main.380: 638,976 TFLOP/s, 1024 reported GPU units.
- 2021.emnlp-main.274: 638,976 TFLOP/s, 1024 reported GPU units.
- 2025.acl-long.100: 606,208 TFLOP/s, 2368 reported GPU units.
- 2025.acl-long.1591: 506,624 TFLOP/s, 768 reported GPU units.
- 2024.acl-long.841: 392,192 TFLOP/s, 1240 reported GPU units.
- 2023.acl-long.856: 319,488 TFLOP/s, 640 reported GPU units.

## Outputs
- `4.2/reported_peak_tflops_distribution/data/paper_reported_peak_tflops.csv`: paper-level source table used for plotting.
- `4.2/reported_peak_tflops_distribution/data/reported_peak_distribution_summary.csv`: overall quantile summary.
- `4.2/reported_peak_tflops_distribution/data/reported_peak_by_year.csv`: annual quantile summary.
- `4.2/reported_peak_tflops_distribution/data/reported_peak_by_venue.csv`: venue quantile summary.
- `4.2/reported_peak_tflops_distribution/data/reported_peak_coverage_by_year_venue.csv`: metric coverage by year and venue.
- `4.2/reported_peak_tflops_distribution/data/paper_reported_peak_tflops_p99_trimmed.csv`: paper-level source table after excluding values above the full-sample p99.
- `4.2/reported_peak_tflops_distribution/data/reported_peak_by_year_p99_trimmed.csv`: annual quantile summary after p99 trimming.
- `4.2/reported_peak_tflops_distribution/fig/reported_peak_tflops_distribution_p99_trimmed_a.png`: p99-trimmed standalone distribution histogram.
- `4.2/reported_peak_tflops_distribution/fig/reported_peak_tflops_distribution_p99_trimmed_c.png`: p99-trimmed standalone annual distribution.
- `4.2/reported_peak_tflops_distribution/data/paper_reported_peak_tflops_lb1_gfimp_p99_trimmed.csv`: loose `is_lb1_gfimp` paper-level table after p99 trimming.
- `4.2/reported_peak_tflops_distribution/data/reported_peak_by_year_lb1_gfimp_p99_trimmed.csv`: loose `is_lb1_gfimp` annual quantile summary after p99 trimming.
