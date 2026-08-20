# RQ1: GPU Memory over Time

## Figure Contract
Core conclusion: GPU-reporting ACL/EMNLP/NAACL papers moved from mostly 16-24 GB-class GPUs toward 40-80 GB-class GPUs, and paper-level lower-bound VRAM rose even faster because GPU counts also increased.
Figure archetype: quantitative grid.
Backend: Python/matplotlib only.
Target output: PNG, source CSV tables, and this written summary.
Figure map: panel a shows median/IQR/P90 maximum per-GPU memory per paper; panel b shows annual memory-bin composition; panel c shows lower-bound total VRAM per paper; panel d compares venue-level medians.
Evidence hierarchy: the maximum per-GPU memory trend is the primary evidence for hardware memory class changes; total VRAM is secondary evidence for aggregate compute-memory scale; memory-bin composition makes the categorical shift auditable.
Statistics used: descriptive counts, percentages, medians, IQRs, and P90 values. No inferential test is used.
Reviewer risk: paper-level total VRAM uses `gpu_num_filled`, which treats missing GPU quantities as one GPU, so it is a conservative lower-bound style estimate rather than an exact cluster inventory.

## Method
Input row table: `data/analysis_ready/paper_compute_rows.csv`.
Input hardware catalog: `data/analysis_ready/hardware_catalog.csv`.
Each row is a standardized GPU mention in a GPU-reporting paper. Memory is merged from the hardware catalog by `benchmark_gpu_name`. When catalog memory is missing but the standardized or extracted GPU name explicitly contains a memory size such as `16GB` or `80 GB`, that explicit size is used and flagged as `name_explicit_gb`.
Memory is expressed as bytes / 1e9 GB. Paper-level maximum memory is the largest known GPU memory in that paper. Paper-level total VRAM is the sum of `memory_gb * gpu_num_filled` across known-memory GPU rows.
Annual denominators: 2020 n=461, 2021 n=741, 2022 n=876, 2023 n=1078, 2024 n=1440, 2025 n=2304.
Known-memory coverage: 6887/6900 papers and 8324/8357 GPU rows.
Memory source counts: catalog=7923, name-explicit GB=401, missing=33.

## Main Result
The median maximum per-GPU memory increased from 16 GB in 2020 to 48 GB in 2025, a 3.0x increase.
The P90 maximum per-GPU memory increased from 32 GB to 80 GB.
Papers using at least 40 GB-class GPUs rose from 8.7% to 83.3%.
Papers using at least 80 GB-class GPUs rose from 0.0% to 38.3%.
The combined `64-80 GB` plus `>80 GB` memory-bin share changed from 0.0% in 2020 to 38.4% in 2025.
The median lower-bound total VRAM per paper increased from 22 GB to 160 GB, a 7.3x increase.

## Venue Pattern
- ACL: median max GPU memory 16 GB in 2020 to 48 GB in 2025.
- EMNLP: median max GPU memory 16 GB in 2020 to 48 GB in 2025.
- NAACL: median max GPU memory 16 GB in 2021 to 48 GB in 2025.

## Interpretation
The memory transition is not just an extreme-tail phenomenon. The annual median moved from the V100/P100-era 16 GB class into 40-48 GB by 2023-2025, while the upper tail increasingly reflects A100/H100/H20/H200/MI-series style devices. The total-VRAM trend rises more sharply than per-GPU memory because later papers combine larger-memory GPUs with larger reported or filled GPU counts.

## Outputs
- `4.2/memory/data/memory_row_level_enriched.csv`: GPU-row table with catalog/inferred memory fields and source flags.
- `4.2/memory/data/memory_paper_level.csv`: paper-level maximum, weighted, and total VRAM summaries.
- `4.2/memory/data/memory_summary_by_year.csv`: annual descriptive statistics.
- `4.2/memory/data/memory_bins_by_year.csv`: annual paper memory-bin counts and shares.
- `4.2/memory/data/memory_bin_share_matrix.csv`: matrix used for the stacked-bin panel.
- `4.2/memory/data/memory_summary_by_venue_year.csv`: venue-year medians and upper quantiles.
- `4.2/memory/data/top_memory_gpu_models_by_year.csv`: ranked standardized GPU models by annual mention count.
- `4.2/memory/fig/memory_trends_over_time.png`: main four-panel figure.
- `4.2/memory/fig/memory_distribution_by_year.png`: supplementary distribution figure.
