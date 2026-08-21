# More Computational Resources Do Not Ensure Higher Scholarly Impact: Evidence from Leading NLP Conference Papers

![](./assets/framework.png)

Official result-reproduction repository for:

> **More Computational Resources Do Not Ensure Higher Scholarly Impact: Evidence from Leading NLP Conference Papers**  
> Shuai Chen, Tong Bao, Jitong Peng, and Chengzhi Zhang

------

## Results at a glance

| Result | Paper value |
|---|---:|
| Full paper corpus | 13,921 |
| Papers reporting a GPU model | 6,900 (49.6%) |
| Strict model-and-count sample | 5,360 (38.5%) |
| Primary adjusted citation-percentile association per 10x capacity | +3.52 pp |
| Incremental model fit for the primary capacity specification | ΔR² = 0.0042 |
| Citation / award model samples | 2,194 / 5,357 |
| Main + Findings corpus | 23,838 |
| Main + Findings citation sample | 3,814 |
| Pooled adjusted citation-percentile association per 10x capacity | +3.90 pp (ΔR² = 0.0051) |

## Repository structure

```text
data/analysis_ready/   Analysis-ready inputs used by the paper-result scripts
data/gpu_info/         Paper-level GPU extraction evidence and released workbooks
code/                  One-command runner, verifier, and result contracts
results/               Paper-first sections with scripts and frozen references
  01_sample/
  02_reporting/
  03_gpu_scale/
  04_contexts/
  05_scholarly_impact/
```

The appendix robustness extension reproduces the Main Conference, Findings,
and pooled sample comparison, five core citation specifications, and the
high-capability/high-impact concentration matrix. See
[`results/05_scholarly_impact/analyses/track_extension/`](results/05_scholarly_impact/analyses/track_extension/README.md).

## Reproduce all results

Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync
uv run python code/run_all.py
uv run python code/verify.py
```

For a fast smoke run:

```bash
uv run python code/run_all.py --quick
uv run python code/verify.py --quick
```

## License

Code is released under the Apache License 2.0; see [`LICENSE`](LICENSE).
Third-party source records retain their original terms and are not relicensed by
this repository.

## Citation

Please cite the paper and this release.
