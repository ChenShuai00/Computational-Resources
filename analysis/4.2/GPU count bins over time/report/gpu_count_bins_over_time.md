# RQ1: GPU Count Bins over Time

## Figure Contract
Core conclusion: GPU-reporting ACL/EMNLP/NAACL papers moved away from mostly one- or two-GPU reports toward more frequent 3-8 GPU usage after 2023, while very large GPU counts remained a minority.
Figure archetype: quantitative grid.
Target output: PNG plus source CSV tables.
Backend: Python/matplotlib only.
Final size: two separate 183 mm wide double-column style figures.
Figure map: the composition figure shows annual GPU-count bin composition; the scale-class figure compresses the bins into 1-2, 3-8, and 9+ GPU classes.
Evidence hierarchy: the composition figure is the main evidence for count-bin redistribution; the scale-class figure makes the small-to-mid-scale shift readable without hiding large-count papers.
Statistics needed: descriptive counts, percentages, medians, and upper quantiles of unique GPU-reporting papers; no inferential test is used.
Source data needed: paper-level GPU count fields.
Image-integrity notes: vector line/text exports are generated directly from source tables; no raster image adjustment.
Reviewer risk: papers without an explicit GPU quantity are shown as `Unspecified`; filled-count summary statistics treat these as one GPU, so medians are conservative lower-bound style estimates.

## Method
Input data: `data/compute_paper_level_gpu_only.xlsx`.
Each row is one GPU-reporting paper. The analysis counts unique papers by publication year and bins `paper_gpu_num_total` as `1`, `2`, `3-4`, `5-8`, `9-16`, `17-32`, `33-64`, and `65+`. Papers with missing raw GPU counts are kept as `Unspecified` in the bin figure.
Annual denominators: 2020 n=461, 2021 n=741, 2022 n=876, 2023 n=1078, 2024 n=1440, 2025 n=2304.
Filled GPU count statistics use `paper_gpu_num_filled_total`, where missing quantities are filled by the upstream data as one GPU.

## Main Result
The share of papers reporting 1-2 GPUs decreased from 49.5% in 2020 to 35.9% in 2025, while 3-8 GPU reports increased from 26.9% to 40.6%.
The 9+ GPU class changed from 6.1% in 2020 to 11.7% in 2025. The filled-count median rose from 1 GPU in 2020 to 3 GPUs in 2025.
Unspecified quantity reports decreased from 17.6% in 2020 to 11.8% in 2025.

## Annual Reported-Bin Leaders
- 2020: 1, 194 papers (42.1%).
- 2021: 1, 310 papers (41.8%).
- 2022: 1, 328 papers (37.4%).
- 2023: 1, 420 papers (39.0%).
- 2024: 1, 369 papers (25.6%).
- 2025: 1, 577 papers (25.0%).

## Outputs
- `4.2/GPU count bins over time/data/gpu_count_bins_by_year.csv`: annual bin counts and shares, including unspecified quantities.
- `4.2/GPU count bins over time/data/gpu_count_bin_share_matrix.csv`: share matrix used for the composition figure.
- `4.2/GPU count bins over time/data/gpu_count_bin_count_matrix.csv`: count matrix used for both split figures.
- `4.2/GPU count bins over time/data/gpu_count_distribution_by_year.csv`: annual median, mean, upper quantiles, max, and unspecified-rate diagnostics.
- `4.2/GPU count bins over time/fig/gpu_count_bin_composition_over_time.png`: annual GPU-count bin composition figure export.
- `4.2/GPU count bins over time/fig/gpu_count_scale_classes_over_time.png`: collapsed scale-class trend figure export.
