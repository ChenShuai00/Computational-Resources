# RQ1: GPU Generation and Family over Time

## Figure Contract
Core conclusion: GPU-reporting ACL/EMNLP/NAACL papers shifted from Tesla/Volta-era hardware toward Ampere datacenter GPUs after 2022, with Hopper and Ada Lovelace appearing mainly in 2024-2025.
Figure archetype: quantitative grid.
Target output: PNG plus source CSV tables.
Backend: Python/matplotlib only.
Final size: 183 mm wide single-panel figures.
Figure map: one figure shows annual main-GPU generation trajectories; the companion figure shows annual main-GPU family trajectories.
Evidence hierarchy: the generation trajectory is the hero evidence for the generational transition; the family trajectory validates that the transition is specifically driven by Datacenter A-series growth and Tesla decline.
Statistics needed: descriptive counts and percentages of unique GPU-reporting papers; no inferential test is used.
Source data needed: paper-level main GPU generation/family fields.
Image-integrity notes: vector line/text exports are generated directly from source tables; no raster image adjustment.
Reviewer risk: main-GPU assignment compresses multi-GPU papers to one dominant GPU family/generation, so the result is a prevalence measure rather than complete hardware inventory.

## Method
Input data: `data/analysis_ready/compute_papers.csv`.
Each row is one GPU-reporting paper. The analysis counts unique papers by publication year and the paper-level `paper_main_gpu_generation` and `paper_main_gpu_family` fields.
Annual denominators: 2020 n=461, 2021 n=741, 2022 n=876, 2023 n=1078, 2024 n=1440, 2025 n=2304.
Rare generations and families are grouped into `Other` in the figure; full grouped counts and shares are preserved in the output CSV files.

## Main Result
Ampere rose from 35.3% of GPU-reporting papers in 2022 to 73.4% in 2025. By 2025, Hopper reached 12.0% and Ada Lovelace reached 7.8%, while older Volta/Turing/Pascal generations contracted.
At the family level, Datacenter A-series increased from 5.4% in 2020 to 57.7% in 2025, whereas Tesla decreased from 46.6% to 4.3%.

## Annual Generation Leaders
- 2020: Volta, 153 papers (33.2%).
- 2021: Volta, 284 papers (38.3%).
- 2022: Volta, 317 papers (36.2%).
- 2023: Ampere, 652 papers (60.5%).
- 2024: Ampere, 1121 papers (77.8%).
- 2025: Ampere, 1692 papers (73.4%).

## Annual Family Leaders
- 2020: Tesla, 215 papers (46.6%).
- 2021: Tesla, 332 papers (44.8%).
- 2022: Tesla, 340 papers (38.8%).
- 2023: Datacenter A-series, 399 papers (37.0%).
- 2024: Datacenter A-series, 848 papers (58.9%).
- 2025: Datacenter A-series, 1330 papers (57.7%).

## Outputs
- `4.2/gpu_generation_family_over_time/data/gpu_generation_by_year.csv`: grouped annual generation counts and shares.
- `4.2/gpu_generation_family_over_time/data/gpu_family_by_year.csv`: grouped annual family counts and shares.
- `4.2/gpu_generation_family_over_time/data/gpu_generation_year_share_matrix.csv`: generation share matrix used for the generation trajectory.
- `4.2/gpu_generation_family_over_time/data/gpu_family_year_share_matrix.csv`: family share matrix used for the family trajectory.
- `4.2/gpu_generation_family_over_time/fig/gpu_generation_over_time.png`: generation figure export.
- `4.2/gpu_generation_family_over_time/fig/gpu_family_over_time.png`: family figure export.