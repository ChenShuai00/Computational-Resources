# RQ1: Top GPU Models by Year

## Method
Input data: `data/paper_compute_level_gpu_only.xlsx`.
The analysis counts unique papers per normalized `benchmark_gpu_name` and publication year. This is a prevalence measure, not an inventory measure; the companion table also reports summed `gpu_num_filled` as estimated GPU units.

## Main Result
The annual leader shifts from V100-class GPUs in 2020-2022 to A100-class GPUs from 2023 onward, with H100 entering the top tier in 2025.

## Annual Leaders
- 2020: NVIDIA Tesla V100 PCIe 16 GB, 120 papers (26.0% of GPU-reporting papers).
- 2021: NVIDIA Tesla V100 PCIe 16 GB, 226 papers (30.5% of GPU-reporting papers).
- 2022: NVIDIA Tesla V100 PCIe 16 GB, 239 papers (27.3% of GPU-reporting papers).
- 2023: NVIDIA A100, 285 papers (26.4% of GPU-reporting papers).
- 2024: NVIDIA A100, 503 papers (34.9% of GPU-reporting papers).
- 2025: NVIDIA A100, 717 papers (31.1% of GPU-reporting papers).

## Overall Most Frequent Models
- NVIDIA A100: 1732 papers.
- NVIDIA Tesla V100 PCIe 16 GB: 937 papers.
- NVIDIA A100 PCIe 80GB: 866 papers.
- NVIDIA RTX A6000: 661 papers.
- NVIDIA GeForce RTX 3090: 533 papers.

## Outputs
- `4.2/top_gpu_models_by_year/data/top_gpu_models_by_year.csv`: annual top 10 models with counts, shares and estimated units.
- `4.2/top_gpu_models_by_year/data/top_gpu_model_year_matrix.csv`: annual count matrix for the most frequent models.
- `4.2/top_gpu_models_by_year/data/top_gpu_model_trajectories.csv`: source data for the trajectory panel.
- `4.2/top_gpu_models_by_year/data/top_gpu_model_annual_leaders.csv`: source data for the annual-leader panel.
- `4.2/top_gpu_models_by_year/fig/top_gpu_models_by_year.png`: publication-oriented figure export.