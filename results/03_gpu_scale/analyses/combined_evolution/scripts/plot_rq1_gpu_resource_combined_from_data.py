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
        if (parent / "data" / "analysis_ready" / "compute_papers.csv").exists():
            return parent
    raise FileNotFoundError("Could not find analysis root containing data.")


ROOT = find_analysis_root(Path(__file__).resolve())
PAPER_LEVEL_INPUT = ROOT / "data" / "analysis_ready" / "compute_papers.csv"
NORMALIZED_INPUT = ROOT / "data" / "analysis_ready" / "gpu_rows.csv"
BUNDLE = Path(os.environ.get("REPRO_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "reproduced"))
OUT_DATA = BUNDLE / "source_data"
OUT_FIG = BUNDLE / "figures"


COUNT_BIN_ORDER = [
    "Unspecified",
    "1",
    "2",
    "3-4",
    "5-8",
    "9-16",
    "17-32",
    "33-64",
    "65+",
]

GENERATION_ORDER = [
    "Pascal",
    "Volta",
    "Turing",
    "Ampere",
    "Ada Lovelace",
    "Hopper",
    "Other",
]

GENERATION_COLORS = {
    "Pascal": "#6F6F6F",
    "Volta": "#4E79A7",
    "Turing": "#59A14F",
    "Ampere": "#E15759",
    "Ada Lovelace": "#B07AA1",
    "Hopper": "#F28E2B",
    "Other": "#BAB0AC",
}

PALETTE = {
    "ink": "#222222",
    "grid": "#E6E6E6",
    "small": "#6F6F6F",
    "mid": "#4E79A7",
    "large": "#E15759",
    "blue": "#305F9F",
    "amber": "#D99035",
}

CM_TO_IN = 1 / 2.54
EMNLP_TEXTWIDTH_CM = 16.0
EMNLP_FIGURE_HEIGHT_CM = 5.8


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 5.2,
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
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)


def parse_year(paper_id: object) -> int:
    value = str(paper_id).split(".", maxsplit=1)[0]
    if not value.isdigit():
        raise ValueError(f"Could not parse year from paper_id={paper_id!r}")
    return int(value)


def assign_count_bin(value: float, raw_missing: bool) -> str:
    if raw_missing:
        return "Unspecified"
    if value <= 1:
        return "1"
    if value <= 2:
        return "2"
    if value <= 4:
        return "3-4"
    if value <= 8:
        return "5-8"
    if value <= 16:
        return "9-16"
    if value <= 32:
        return "17-32"
    if value <= 64:
        return "33-64"
    return "65+"


def load_paper_level() -> pd.DataFrame:
    df = pd.read_csv(PAPER_LEVEL_INPUT)
    required = {
        "paper_id",
        "paper_gpu_num_total",
        "paper_gpu_num_filled_total",
        "paper_main_gpu_generation",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in paper-level input: {missing}")

    df = df.copy()
    df["year"] = df["paper_id"].map(parse_year)
    df["gpu_count_raw"] = pd.to_numeric(df["paper_gpu_num_total"], errors="coerce")
    df["gpu_count_filled"] = pd.to_numeric(
        df["paper_gpu_num_filled_total"], errors="coerce"
    )
    df["count_unspecified"] = df["gpu_count_raw"].isna()
    df["gpu_count_bin"] = [
        assign_count_bin(value, missing)
        for value, missing in zip(df["gpu_count_filled"], df["count_unspecified"])
    ]
    df["gpu_count_bin"] = pd.Categorical(
        df["gpu_count_bin"], categories=COUNT_BIN_ORDER, ordered=True
    )
    df["generation"] = df["paper_main_gpu_generation"].fillna("Unknown").astype(str)
    df["generation_group"] = np.where(
        df["generation"].isin(GENERATION_ORDER[:-1]), df["generation"], "Other"
    )
    return df


def summarize_scale_classes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_totals = (
        df.groupby("year")["paper_id"].nunique().rename("gpu_papers_in_year").reset_index()
    )
    summary = (
        df.groupby(["year", "gpu_count_bin"], observed=False)["paper_id"]
        .nunique()
        .rename("papers")
        .reset_index()
        .merge(annual_totals, on="year", how="left")
    )
    summary["share_pct"] = summary["papers"] / summary["gpu_papers_in_year"] * 100
    summary["gpu_count_bin"] = summary["gpu_count_bin"].astype(str)

    count_matrix = (
        summary.pivot(index="year", columns="gpu_count_bin", values="papers")
        .reindex(columns=COUNT_BIN_ORDER)
        .fillna(0)
        .astype(int)
    )
    distribution = (
        df.groupby("year")
        .agg(
            gpu_papers=("paper_id", "nunique"),
            median_filled_gpu_count=("gpu_count_filled", "median"),
        )
        .reset_index()
    )
    return count_matrix, distribution


def summarize_generations(df: pd.DataFrame) -> pd.DataFrame:
    annual = df.groupby("year")["paper_id"].nunique().rename("gpu_papers").reset_index()
    generation = (
        df.groupby(["year", "generation_group"], as_index=False)
        .agg(papers=("paper_id", "nunique"))
        .merge(annual, on="year", how="left")
    )
    generation["paper_share_pct"] = 100 * generation["papers"] / generation["gpu_papers"]
    matrix = (
        generation.pivot(index="generation_group", columns="year", values="paper_share_pct")
        .reindex(GENERATION_ORDER)
        .fillna(0)
    )
    generation.to_csv(OUT_DATA / "gpu_generation_by_year.csv", index=False)
    matrix.to_csv(OUT_DATA / "gpu_generation_year_share_matrix.csv")
    return matrix


def load_reported_peak() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(NORMALIZED_INPUT)
    required = {
        "paper_id",
        "gpu_name",
        "gpu_num",
        "benchmark_gpu_name",
        "benchmark_max_performance",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in normalized input: {missing}")

    df = df.copy()
    df["year"] = df["paper_id"].map(parse_year)
    df["gpu_num"] = pd.to_numeric(df["gpu_num"], errors="coerce")
    df["benchmark_max_performance"] = pd.to_numeric(
        df["benchmark_max_performance"], errors="coerce"
    )
    valid = df[df["gpu_num"].gt(0) & df["benchmark_max_performance"].gt(0)].copy()
    valid["row_reported_peak_tflops"] = (
        valid["gpu_num"] * valid["benchmark_max_performance"] / 1e12
    )

    paper = (
        valid.groupby("paper_id", as_index=False)
        .agg(
            year=("year", "first"),
            n_reported_gpu_rows=("gpu_name", "size"),
            n_unique_benchmark_gpus=("benchmark_gpu_name", "nunique"),
            reported_gpu_units=("gpu_num", "sum"),
            reported_peak_tflops=("row_reported_peak_tflops", "max"),
        )
        .sort_values(["year", "paper_id"])
    )

    cutoff = float(paper["reported_peak_tflops"].quantile(0.99))
    trimmed = paper[paper["reported_peak_tflops"].le(cutoff)].copy()
    by_year = summarize_tflops_by_year(trimmed)

    paper.to_csv(OUT_DATA / "paper_reported_peak_tflops.csv", index=False)
    trimmed.to_csv(OUT_DATA / "paper_reported_peak_tflops_p99_trimmed.csv", index=False)
    by_year.to_csv(OUT_DATA / "reported_peak_by_year_p99_trimmed.csv", index=False)
    return trimmed, by_year


def summarize_tflops_by_year(paper: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, sub in paper.groupby("year", dropna=False):
        values = sub["reported_peak_tflops"]
        rows.append(
            {
                "year": int(year),
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
    return pd.DataFrame(rows).sort_values("year")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.12,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color="black",
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", labelsize=4.8, width=0.6, length=2.2, pad=1.5)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.set_axisbelow(True)


def format_tflops(value: float) -> str:
    return f"{value:,.0f}"


def log_tick_values(max_value: float) -> np.ndarray:
    base = np.array([10, 100, 1_000, 10_000, 100_000, 1_000_000])
    if max_value <= 100_000:
        return base[:5]
    return base


def plot_scale_panel(
    ax: plt.Axes, count_matrix: pd.DataFrame, distribution: pd.DataFrame
) -> None:
    years = count_matrix.index.to_numpy(dtype=int)
    x = np.arange(len(years))
    reported = count_matrix.drop(columns=["Unspecified"]).copy()
    small = reported[["1", "2"]].sum(axis=1)
    mid = reported[["3-4", "5-8"]].sum(axis=1)
    large = reported[["9-16", "17-32", "33-64", "65+"]].sum(axis=1)
    denom = count_matrix.sum(axis=1).replace(0, np.nan)

    ax.plot(
        x,
        (small / denom * 100).to_numpy(),
        color=PALETTE["small"],
        marker="o",
        markersize=3.2,
        linewidth=1.25,
        label="1-2 GPUs",
    )
    ax.plot(
        x,
        (mid / denom * 100).to_numpy(),
        color=PALETTE["mid"],
        marker="o",
        markersize=3.2,
        linewidth=1.25,
        label="3-8 GPUs",
    )
    ax.plot(
        x,
        (large / denom * 100).to_numpy(),
        color=PALETTE["large"],
        marker="o",
        markersize=3.2,
        linewidth=1.25,
        label="9+ GPUs",
    )

    ax.set_ylabel("Papers (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 62)
    style_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.54, 1.01),
        ncol=3,
        fontsize=4.8,
        columnspacing=0.62,
        handlelength=1.15,
        handletextpad=0.35,
    )

    medians = distribution.set_index("year").loc[years, "median_filled_gpu_count"]
    for i, median in enumerate(medians):
        ax.text(
            i,
            -0.19,
            f"med. {median:g}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=4.7,
            color="#555555",
        )
    add_panel_label(ax, "A")


def plot_generation_panel(ax: plt.Axes, generation_matrix: pd.DataFrame) -> None:
    years = generation_matrix.columns.to_numpy(dtype=int)
    x = np.arange(len(years))

    for name in ["Pascal", "Volta", "Turing", "Ampere", "Ada Lovelace", "Hopper"]:
        values = generation_matrix.loc[name].to_numpy(dtype=float)
        is_signal = name in {"Ampere", "Ada Lovelace", "Hopper"}
        ax.plot(
            x,
            values,
            label=name,
            color=GENERATION_COLORS[name],
            linewidth=2.0 if is_signal else 1.35,
            marker="o",
            markersize=4.0 if is_signal else 3.2,
            markeredgewidth=0,
            alpha=1.0 if is_signal else 0.82,
        )

    ax.set_xlim(-0.15, len(years) - 0.02)
    ax.set_ylim(0, 84)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Papers (%)")
    ax.set_xlabel("Publication year")
    style_axis(ax)
    ax.spines["left"].set_bounds(0, 80)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 1.21),
        ncol=3,
        fontsize=4.6,
        columnspacing=0.55,
        handlelength=1.0,
        handletextpad=0.30,
    )
    add_panel_label(ax, "B")


def plot_tflops_panel(
    ax: plt.Axes, paper: pd.DataFrame, by_year: pd.DataFrame
) -> None:
    years = sorted(paper["year"].unique())
    year_values = [
        paper.loc[paper["year"].eq(year), "reported_peak_tflops"].dropna().to_numpy(dtype=float)
        for year in years
    ]
    box = ax.boxplot(
        year_values,
        positions=np.arange(len(years)),
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.3},
        boxprops={
            "facecolor": PALETTE["blue"],
            "edgecolor": PALETTE["blue"],
            "linewidth": 0.8,
        },
        whiskerprops={"color": PALETTE["blue"], "linewidth": 0.8},
        capprops={"color": PALETTE["blue"], "linewidth": 0.8},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.78)

    p95_line = by_year["p95_tflops"].to_numpy(dtype=float)
    ax.plot(
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
        ax.text(
            xpos + (0.08 if xpos == len(years) - 1 else 0),
            row["p95_tflops"] * (1.30 if label_above else 0.80),
            format_tflops(row["p95_tflops"]),
            ha="center",
            va="bottom" if label_above else "top",
            fontsize=4.3,
            color=PALETTE["amber"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.92},
        )

    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years)
    ax.set_ylabel("TFLOP/s")
    ax.set_xlabel("Publication year")
    y_ticks = log_tick_values(max(float(np.max(values)) for values in year_values))
    ax.set_ylim(3, 180_000 if y_ticks[-1] <= 100_000 else 12_000_000)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([format_tflops(v) for v in y_ticks])
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), fontsize=4.8)
    style_axis(ax)
    add_panel_label(ax, "C")


def save_pub_figure(fig: plt.Figure, output_stem: Path) -> None:
    def save_atomic(ext: str, **kwargs: object) -> None:
        target = output_stem.with_suffix(f".{ext}")
        tmp = target.with_name(f"{target.stem}.tmp.{os.getpid()}.{ext}")
        fig.savefig(tmp, **kwargs)
        for attempt in range(5):
            try:
                if target.exists():
                    target.unlink()
                tmp.replace(target)
                return
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.25)

    save_atomic("png", dpi=300)


def main() -> None:
    ensure_dirs()
    paper_level = load_paper_level()
    count_matrix, scale_distribution = summarize_scale_classes(paper_level)
    generation_matrix = summarize_generations(paper_level)
    tflops_paper, tflops_by_year = load_reported_peak()

    count_matrix.to_csv(OUT_DATA / "gpu_count_bin_count_matrix.csv")
    scale_distribution.to_csv(OUT_DATA / "gpu_count_distribution_by_year.csv", index=False)

    fig = plt.figure(
        figsize=(EMNLP_TEXTWIDTH_CM * CM_TO_IN, EMNLP_FIGURE_HEIGHT_CM * CM_TO_IN),
        facecolor="white",
        constrained_layout=False,
    )
    gs = fig.add_gridspec(
        1,
        3,
        left=0.060,
        right=0.995,
        top=0.72,
        bottom=0.29,
        wspace=0.44,
    )
    plot_scale_panel(fig.add_subplot(gs[0, 0]), count_matrix, scale_distribution)
    plot_generation_panel(fig.add_subplot(gs[0, 1]), generation_matrix)
    plot_tflops_panel(fig.add_subplot(gs[0, 2]), tflops_paper, tflops_by_year)

    output_stem = OUT_FIG / "rq1_gpu_resource_combined_from_data_horizontal_emnlp2026_no_titles_b_legend"
    save_pub_figure(fig, output_stem)
    plt.close(fig)

    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Source data: {PAPER_LEVEL_INPUT}")
    print(f"Source data: {NORMALIZED_INPUT}")


if __name__ == "__main__":
    main()


