# Rebuttal: GPU concentration calculation

We agree that the concentration calculation should be made clearer. We compute this
statistic separately within each publication year. For each year, we restrict to
papers with positive reported GPU capacity, sort papers in descending order of
paper-level reported GPU capacity, select the top
`k_y = ceil(0.20 N_y)` papers, and divide their summed capacity by the summed
capacity of all GPU-quantifiable papers in that year.

The calculation is:

`S_y = sum_{i=1}^{k_y} C_{(i)y} / sum_{i=1}^{N_y} C_{(i)y}`

where `C_(i)y` is the paper-level reported GPU capacity after sorting year `y`
from highest to lowest capacity. This yields 2020: 87.9%, 2021: 89.9%, 2022: 85.6%, 2023: 83.9%. Thus, the statement means
that the highest-capacity approximately 20% of GPU-quantifiable papers within
each year account for 83.9%-89.9% of that year's total
reported GPU capacity.

## Values

| Year | GPU-quantifiable papers | Top papers | Paper share | Capacity share | Top-group threshold (TFLOP/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2020 | 461 | 93 | 20.2% | 87.9% | 448 |
| 2021 | 741 | 149 | 20.1% | 89.9% | 448 |
| 2022 | 876 | 176 | 20.1% | 85.6% | 896 |
| 2023 | 1,078 | 216 | 20.0% | 83.9% | 896 |

## Rebuttal figure

![Cumulative reported GPU capacity](../fig/rq3_rebuttal_gpu_capacity_pareto.png)

The vertical dashed line marks the 20% paper-share point. The labeled dots are
the yearly top-20% cutoff values used in the rebuttal text.

## Source artifacts

- Figure: `fig/rq3_rebuttal_gpu_capacity_pareto.png`
- Pareto source data: `data/rq3_rebuttal_gpu_capacity_pareto.csv`
- Top-20 share source data: `data/rq3_top20_compute_concentration.csv`
