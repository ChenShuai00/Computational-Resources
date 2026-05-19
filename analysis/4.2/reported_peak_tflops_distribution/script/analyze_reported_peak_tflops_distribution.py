from __future__ import annotations

import os
from pathlib import Path
import time

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_analysis_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        candidate = parent / "data" / "acl_emnlp_naacl_2020_2025_gpu_normalized_gpu_only.xlsx"
        if candidate.exists():
            return parent
    raise FileNotFoundError("Could not find analysis root containing data.")


ROOT = find_analysis_root(Path(__file__).resolve())
INPUT = ROOT / "data" / "acl_emnlp_naacl_2020_2025_gpu_normalized_gpu_only.xlsx"
INPUT_LB1_GFIMP = ROOT / "data" / "paper_compute_level_gpu_only.xlsx"
BUNDLE = Path(__file__).resolve().parents[1]
OUT_DATA = BUNDLE / "data"
OUT_FIG = BUNDLE / "fig"
OUT_REPORT = BUNDLE / "report"


PALETTE = {
    "ink": "#222222",
    "muted": "#6F6F6F",
    "grid": "#E6E6E6",
    "blue": "#305F9F",
    "teal": "#3B8F8C",
    "amber": "#D99035",
    "purple": "#8D6A9F",
    "red": "#C85250",
    "fill": "#D8E3F2",
}

HIST_AXIS_LABEL = "Reported peak GPU configuration capacity (TFLOP/s, log10 scale)"
YEAR_AXIS_LABEL = "Reported peak GPU configuration capacity (TFLOP/s)"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": PALETTE["ink"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
    }
)


def ensure_dirs() -> None:
    for path in (OUT_DATA, OUT_FIG, OUT_REPORT):
        path.mkdir(parents=True, exist_ok=True)


def parse_year(paper_id: object) -> int:
    value = str(paper_id).split(".", maxsplit=1)[0]
    if not value.isdigit():
        raise ValueError(f"Could not parse year from paper_id={paper_id!r}")
    return int(value)


def parse_venue(paper_id: object) -> str:
    parts = str(paper_id).split(".")
    if len(parts) < 3:
        return "unknown"
    venue = ".".join(parts[1:-1]).removeprefix("findings-")
    return venue.split("-")[0].upper()


def load_reported_peak() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(INPUT, sheet_name="merged_gpu_normalized")
    required = {"paper_id", "gpu_name", "gpu_num", "benchmark_gpu_name", "benchmark_max_performance"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["year"] = df["paper_id"].map(parse_year)
    df["venue"] = df["paper_id"].map(parse_venue)
    df["gpu_num"] = pd.to_numeric(df["gpu_num"], errors="coerce")
    df["benchmark_max_performance"] = pd.to_numeric(df["benchmark_max_performance"], errors="coerce")
    df["has_reported_quantity"] = df["gpu_num"].gt(0)
    df["has_peak_benchmark"] = df["benchmark_max_performance"].gt(0)

    valid = df[df["has_reported_quantity"] & df["has_peak_benchmark"]].copy()
    valid["row_reported_peak_tflops"] = valid["gpu_num"] * valid["benchmark_max_performance"] / 1e12

    paper = (
        valid.groupby("paper_id", as_index=False)
        .agg(
            year=("year", "first"),
            venue=("venue", "first"),
            n_reported_gpu_rows=("gpu_name", "size"),
            n_unique_benchmark_gpus=("benchmark_gpu_name", "nunique"),
            reported_gpu_units=("gpu_num", "sum"),
            reported_peak_tflops=("row_reported_peak_tflops", "max"),
        )
        .sort_values(["year", "paper_id"])
    )

    all_paper_ids = df[["paper_id", "year", "venue"]].drop_duplicates()
    coverage = all_paper_ids.merge(
        paper[["paper_id", "reported_peak_tflops"]], on="paper_id", how="left"
    )
    coverage["has_reported_peak_tflops"] = coverage["reported_peak_tflops"].notna()
    coverage = (
        coverage.groupby(["year", "venue"], as_index=False)
        .agg(
            gpu_papers=("paper_id", "nunique"),
            papers_with_reported_peak_tflops=("has_reported_peak_tflops", "sum"),
        )
        .sort_values(["year", "venue"])
    )
    coverage["coverage_pct"] = (
        coverage["papers_with_reported_peak_tflops"] / coverage["gpu_papers"] * 100
    )

    return paper, coverage


def load_lb1_gfimp_peak() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(INPUT_LB1_GFIMP)
    required = {
        "paper_id",
        "gpu_name",
        "benchmark_gpu_name",
        "is_lb1_gfimp",
        "gpu_num_filled",
        "compute_capability_gfimp_lb1",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in lb1_gfimp input: {missing}")

    df = df.copy()
    df["year"] = df["paper_id"].map(parse_year)
    df["venue"] = df["paper_id"].map(parse_venue)
    df["is_lb1_gfimp"] = pd.to_numeric(df["is_lb1_gfimp"], errors="coerce").fillna(0).astype(int)
    df["gpu_num_filled"] = pd.to_numeric(df["gpu_num_filled"], errors="coerce")
    df["compute_capability_gfimp_lb1"] = pd.to_numeric(
        df["compute_capability_gfimp_lb1"], errors="coerce"
    )
    df["has_lb1_gfimp_peak"] = df["is_lb1_gfimp"].eq(1) & df[
        "compute_capability_gfimp_lb1"
    ].gt(0)

    valid = df[df["has_lb1_gfimp_peak"]].copy()
    valid["row_reported_peak_tflops"] = valid["compute_capability_gfimp_lb1"] / 1e12

    paper = (
        valid.groupby("paper_id", as_index=False)
        .agg(
            year=("year", "first"),
            venue=("venue", "first"),
            n_reported_gpu_rows=("gpu_name", "size"),
            n_unique_benchmark_gpus=("benchmark_gpu_name", "nunique"),
            reported_gpu_units=("gpu_num_filled", "sum"),
            reported_peak_tflops=("row_reported_peak_tflops", "max"),
        )
        .sort_values(["year", "paper_id"])
    )

    all_paper_ids = df[["paper_id", "year", "venue"]].drop_duplicates()
    coverage = all_paper_ids.merge(
        paper[["paper_id", "reported_peak_tflops"]], on="paper_id", how="left"
    )
    coverage["has_reported_peak_tflops"] = coverage["reported_peak_tflops"].notna()
    coverage = (
        coverage.groupby(["year", "venue"], as_index=False)
        .agg(
            gpu_papers=("paper_id", "nunique"),
            papers_with_reported_peak_tflops=("has_reported_peak_tflops", "sum"),
        )
        .sort_values(["year", "venue"])
    )
    coverage["coverage_pct"] = (
        coverage["papers_with_reported_peak_tflops"] / coverage["gpu_papers"] * 100
    )

    return paper, coverage


def summarize_distribution(paper: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = paper["reported_peak_tflops"]
    overall = pd.DataFrame(
        [
            {
                "scope": "overall",
                "papers": int(values.notna().sum()),
                "min_tflops": values.min(),
                "q25_tflops": values.quantile(0.25),
                "median_tflops": values.median(),
                "q75_tflops": values.quantile(0.75),
                "p90_tflops": values.quantile(0.90),
                "p95_tflops": values.quantile(0.95),
                "p99_tflops": values.quantile(0.99),
                "max_tflops": values.max(),
                "mean_tflops": values.mean(),
            }
        ]
    )

    def grouped_summary(group_cols: list[str]) -> pd.DataFrame:
        rows = []
        for keys, sub in paper.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            values = sub["reported_peak_tflops"]
            row = dict(zip(group_cols, keys, strict=True))
            row.update(
                {
                    "papers": int(values.notna().sum()),
                    "median_tflops": values.median(),
                    "q25_tflops": values.quantile(0.25),
                    "q75_tflops": values.quantile(0.75),
                    "p90_tflops": values.quantile(0.90),
                    "p95_tflops": values.quantile(0.95),
                    "max_tflops": values.max(),
                    "mean_tflops": values.mean(),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows).sort_values(group_cols)

    by_year = grouped_summary(["year"])
    by_venue = grouped_summary(["venue"])
    return overall, by_year, by_venue


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color="black",
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", labelsize=6.4, width=0.6, length=3)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)


def format_tflops(value: float) -> str:
    return f"{value:,.0f}"


def log_tick_values(max_value: float) -> np.ndarray:
    base = np.array([10, 100, 1_000, 10_000, 100_000, 1_000_000])
    if max_value <= 100_000:
        return base[:5]
    return base


def save_pub_figure(fig: plt.Figure, output_stem: Path) -> None:
    def save_atomic(ext: str, **kwargs: object) -> None:
        target = output_stem.with_suffix(f".{ext}")
        tmp = target.with_name(f"{target.stem}.tmp.{os.getpid()}.{ext}")
        fig.savefig(tmp, bbox_inches="tight", **kwargs)
        for attempt in range(5):
            try:
                if target.exists():
                    target.unlink()
                tmp.replace(target)
                return
            except OSError:
                if attempt == 4:
                    print(f"Warning: could not overwrite {target}; leaving existing file in place.")
                    if tmp.exists():
                        tmp.unlink(missing_ok=True)
                    return
                time.sleep(0.25)

    save_atomic("png", dpi=300)
    plt.close(fig)


def plot_hist_panel(ax_hist: plt.Axes, values: np.ndarray, log_values: np.ndarray) -> None:
    bins = np.linspace(np.floor(log_values.min()), np.ceil(log_values.max()), 36)
    ax_hist.hist(
        log_values,
        bins=bins,
        color=PALETTE["fill"],
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )
    median = np.median(values)
    p95 = np.quantile(values, 0.95)
    ax_hist.axvline(np.log10(median), color=PALETTE["blue"], linewidth=1.35)
    ax_hist.axvline(np.log10(p95), color=PALETTE["red"], linewidth=1.10, linestyle=(0, (3, 2)))
    ax_hist.text(
        np.log10(median) - 0.08,
        ax_hist.get_ylim()[1] * 0.93,
        f"median {format_tflops(median)}",
        color=PALETTE["blue"],
        fontsize=6.4,
        va="top",
        ha="right",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.92},
    )
    ax_hist.text(
        np.log10(p95) + 0.035,
        ax_hist.get_ylim()[1] * 0.78,
        f"p95 {format_tflops(p95)}",
        color=PALETTE["red"],
        fontsize=6.4,
        va="top",
    )
    ax_hist.set_xlabel(HIST_AXIS_LABEL)
    ax_hist.set_ylabel("Papers")
    tick_values = log_tick_values(float(np.max(values)))
    ax_hist.set_xticks(np.log10(tick_values))
    ax_hist.set_xticklabels([format_tflops(v) for v in tick_values])
    style_axis(ax_hist)


def plot_year_panel(
    ax_box: plt.Axes, years: list[int], year_values: list[np.ndarray], by_year: pd.DataFrame
) -> None:
    box = ax_box.boxplot(
        year_values,
        positions=np.arange(len(years)),
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.3},
        boxprops={"facecolor": PALETTE["blue"], "edgecolor": PALETTE["blue"], "linewidth": 0.8},
        whiskerprops={"color": PALETTE["blue"], "linewidth": 0.8},
        capprops={"color": PALETTE["blue"], "linewidth": 0.8},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.78)

    p95_line = by_year["p95_tflops"].to_numpy(dtype=float)
    ax_box.plot(
        np.arange(len(years)),
        p95_line,
        color=PALETTE["amber"],
        marker="o",
        markersize=3.8,
        linewidth=1.6,
        label="Annual p95",
    )
    for _, row in by_year.iterrows():
        xpos = years.index(int(row["year"]))
        label_above = xpos % 2 == 0 or xpos == len(years) - 1
        label_y = row["p95_tflops"] * (1.32 if label_above else 0.78)
        ax_box.text(
            xpos + (0.12 if xpos == len(years) - 1 else 0),
            label_y,
            format_tflops(row["p95_tflops"]),
            ha="center",
            va="bottom" if label_above else "top",
            fontsize=5.8,
            color=PALETTE["amber"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.92},
        )

    ax_box.set_yscale("log")
    ax_box.set_ylim(3, 12_000_000)
    ax_box.set_xticks(np.arange(len(years)))
    ax_box.set_xticklabels(years)
    ax_box.set_ylabel(YEAR_AXIS_LABEL)
    ax_box.set_xlabel("Publication year")
    y_ticks = log_tick_values(max(float(np.max(values)) for values in year_values))
    y_top = 12_000_000 if y_ticks[-1] >= 1_000_000 else 180_000
    ax_box.set_ylim(3, y_top)
    ax_box.set_yticks(y_ticks)
    ax_box.set_yticklabels([format_tflops(v) for v in y_ticks])
    ax_box.legend(loc="upper left", bbox_to_anchor=(0.01, 0.97), fontsize=6.4)
    style_axis(ax_box)


def plot_distribution(paper: pd.DataFrame, by_year: pd.DataFrame, output_stem: Path) -> None:
    values = paper["reported_peak_tflops"].dropna().to_numpy(dtype=float)
    log_values = np.log10(values)
    years = sorted(paper["year"].unique())
    year_values = [
        paper.loc[paper["year"].eq(year), "reported_peak_tflops"].dropna().to_numpy(dtype=float)
        for year in years
    ]

    fig_a = plt.figure(figsize=(4.25, 3.05), facecolor="white", constrained_layout=False)
    gs_a = fig_a.add_gridspec(
        1,
        1,
        left=0.16,
        right=0.97,
        top=0.94,
        bottom=0.22,
    )
    plot_hist_panel(fig_a.add_subplot(gs_a[0, 0]), values, log_values)
    save_pub_figure(fig_a, output_stem.with_name(f"{output_stem.name}_a"))

    fig_c = plt.figure(figsize=(7.2, 3.05), facecolor="white", constrained_layout=False)
    gs_c = fig_c.add_gridspec(
        1,
        1,
        left=0.10,
        right=0.98,
        top=0.94,
        bottom=0.20,
    )
    plot_year_panel(fig_c.add_subplot(gs_c[0, 0]), years, year_values, by_year)
    save_pub_figure(fig_c, output_stem.with_name(f"{output_stem.name}_c"))


def write_report(
    paper: pd.DataFrame,
    coverage: pd.DataFrame,
    overall: pd.DataFrame,
    by_year: pd.DataFrame,
    by_venue: pd.DataFrame,
    trimmed_cutoff: float,
    trimmed_papers: int,
) -> None:
    out = OUT_REPORT / "reported_peak_tflops_distribution.md"
    stats = overall.iloc[0]
    annual_peak = by_year.loc[by_year["median_tflops"].idxmax()]
    p95_peak = by_year.loc[by_year["p95_tflops"].idxmax()]
    total_gpu_papers = int(coverage["gpu_papers"].sum())
    covered_papers = int(coverage["papers_with_reported_peak_tflops"].sum())
    coverage_pct = covered_papers / total_gpu_papers * 100

    top_outliers = paper.sort_values("reported_peak_tflops", ascending=False).head(8)
    outlier_lines = [
        f"- {row.paper_id}: {format_tflops(row.reported_peak_tflops)} TFLOP/s, "
        f"{row.reported_gpu_units:.0f} reported GPU units."
        for row in top_outliers.itertuples(index=False)
    ]
    year_lines = [
        f"- {int(row.year)}: median {format_tflops(row.median_tflops)}, "
        f"IQR {format_tflops(row.q25_tflops)}-{format_tflops(row.q75_tflops)}, "
        f"p95 {format_tflops(row.p95_tflops)} TFLOP/s (n={int(row.papers)})."
        for row in by_year.itertuples(index=False)
    ]
    venue_lines = [
        f"- {row.venue}: median {format_tflops(row.median_tflops)}, "
        f"p95 {format_tflops(row.p95_tflops)} TFLOP/s (n={int(row.papers)})."
        for row in by_venue.itertuples(index=False)
    ]

    content = f"""# RQ1: Reported Peak TFLOP/s Distribution

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
Coverage: {covered_papers:,}/{total_gpu_papers:,} GPU-only papers ({coverage_pct:.1f}%) have enough information for this metric.
The extreme-value-excluded figure uses a reproducible p99 trimming rule: papers above {format_tflops(trimmed_cutoff)} TFLOP/s are excluded, leaving {trimmed_papers:,} papers.
The loose `is_lb1_gfimp` sensitivity version uses `data/paper_compute_level_gpu_only.xlsx`: row-level TFLOP/s is `compute_capability_gfimp_lb1 / 1e12`, equivalent to `gpu_num_filled * effective_flops_gfimp / 1e12`, and paper-level values again take the maximum row within each paper.

## Main Result
Across {int(stats.papers):,} papers, the median reported peak capacity is {format_tflops(stats.median_tflops)} TFLOP/s, the IQR is {format_tflops(stats.q25_tflops)}-{format_tflops(stats.q75_tflops)} TFLOP/s, and the p95 is {format_tflops(stats.p95_tflops)} TFLOP/s. The distribution is strongly long-tailed: the p99 reaches {format_tflops(stats.p99_tflops)} TFLOP/s and the maximum reaches {format_tflops(stats.max_tflops)} TFLOP/s.
The annual median peaks in {int(annual_peak.year)} at {format_tflops(annual_peak.median_tflops)} TFLOP/s, while the annual p95 peaks in {int(p95_peak.year)} at {format_tflops(p95_peak.p95_tflops)} TFLOP/s.

## Annual Distribution
{chr(10).join(year_lines)}

## Venue Distribution
{chr(10).join(venue_lines)}

## Largest Reported Allocations
{chr(10).join(outlier_lines)}

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
"""
    out.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    paper, coverage = load_reported_peak()
    overall, by_year, by_venue = summarize_distribution(paper)
    trimmed_cutoff = float(paper["reported_peak_tflops"].quantile(0.99))
    trimmed_paper = paper[paper["reported_peak_tflops"].le(trimmed_cutoff)].copy()
    trimmed_overall, trimmed_by_year, trimmed_by_venue = summarize_distribution(trimmed_paper)
    lb1_paper, lb1_coverage = load_lb1_gfimp_peak()
    lb1_overall, lb1_by_year, lb1_by_venue = summarize_distribution(lb1_paper)
    lb1_trimmed_cutoff = float(lb1_paper["reported_peak_tflops"].quantile(0.99))
    lb1_trimmed_paper = lb1_paper[
        lb1_paper["reported_peak_tflops"].le(lb1_trimmed_cutoff)
    ].copy()
    lb1_trimmed_overall, lb1_trimmed_by_year, lb1_trimmed_by_venue = summarize_distribution(
        lb1_trimmed_paper
    )

    paper.to_csv(OUT_DATA / "paper_reported_peak_tflops.csv", index=False)
    coverage.to_csv(OUT_DATA / "reported_peak_coverage_by_year_venue.csv", index=False)
    overall.to_csv(OUT_DATA / "reported_peak_distribution_summary.csv", index=False)
    by_year.to_csv(OUT_DATA / "reported_peak_by_year.csv", index=False)
    by_venue.to_csv(OUT_DATA / "reported_peak_by_venue.csv", index=False)
    trimmed_paper.to_csv(OUT_DATA / "paper_reported_peak_tflops_p99_trimmed.csv", index=False)
    trimmed_overall.to_csv(OUT_DATA / "reported_peak_distribution_summary_p99_trimmed.csv", index=False)
    trimmed_by_year.to_csv(OUT_DATA / "reported_peak_by_year_p99_trimmed.csv", index=False)
    trimmed_by_venue.to_csv(OUT_DATA / "reported_peak_by_venue_p99_trimmed.csv", index=False)
    lb1_paper.to_csv(OUT_DATA / "paper_reported_peak_tflops_lb1_gfimp.csv", index=False)
    lb1_coverage.to_csv(OUT_DATA / "reported_peak_coverage_by_year_venue_lb1_gfimp.csv", index=False)
    lb1_overall.to_csv(OUT_DATA / "reported_peak_distribution_summary_lb1_gfimp.csv", index=False)
    lb1_by_year.to_csv(OUT_DATA / "reported_peak_by_year_lb1_gfimp.csv", index=False)
    lb1_by_venue.to_csv(OUT_DATA / "reported_peak_by_venue_lb1_gfimp.csv", index=False)
    lb1_trimmed_paper.to_csv(
        OUT_DATA / "paper_reported_peak_tflops_lb1_gfimp_p99_trimmed.csv", index=False
    )
    lb1_trimmed_overall.to_csv(
        OUT_DATA / "reported_peak_distribution_summary_lb1_gfimp_p99_trimmed.csv",
        index=False,
    )
    lb1_trimmed_by_year.to_csv(
        OUT_DATA / "reported_peak_by_year_lb1_gfimp_p99_trimmed.csv", index=False
    )
    lb1_trimmed_by_venue.to_csv(
        OUT_DATA / "reported_peak_by_venue_lb1_gfimp_p99_trimmed.csv", index=False
    )

    plot_distribution(
        trimmed_paper,
        trimmed_by_year,
        OUT_FIG / "reported_peak_tflops_distribution_p99_trimmed",
    )
    write_report(
        paper,
        coverage,
        overall,
        by_year,
        by_venue,
        trimmed_cutoff,
        len(trimmed_paper),
    )
    print(f"Wrote outputs to {BUNDLE}")
    print(
        "Papers with reported peak TFLOP/s:",
        f"{len(paper):,}; median={overall.iloc[0]['median_tflops']:.1f};",
        f"p95={overall.iloc[0]['p95_tflops']:.1f}; max={overall.iloc[0]['max_tflops']:.1f}",
    )
    print(
        "P99-trimmed figure:",
        f"cutoff={trimmed_cutoff:.1f};",
        f"kept={len(trimmed_paper):,}; excluded={len(paper) - len(trimmed_paper):,}",
    )
    print(
        "LB1 GFIMP p99-trimmed sensitivity:",
        f"cutoff={lb1_trimmed_cutoff:.1f};",
        f"kept={len(lb1_trimmed_paper):,}; excluded={len(lb1_paper) - len(lb1_trimmed_paper):,};",
        f"median={lb1_trimmed_overall.iloc[0]['median_tflops']:.1f};",
        f"p95={lb1_trimmed_overall.iloc[0]['p95_tflops']:.1f}",
    )


if __name__ == "__main__":
    main()





