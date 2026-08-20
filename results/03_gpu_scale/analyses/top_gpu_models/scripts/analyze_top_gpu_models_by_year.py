import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_analysis_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "data" / "analysis_ready" / "paper_compute_rows.csv").exists():
            return parent
    raise FileNotFoundError("Could not find analysis root containing data.")


ROOT = find_analysis_root(Path(__file__).resolve())
INPUT = ROOT / "data" / "analysis_ready" / "paper_compute_rows.csv"
BUNDLE = Path(os.environ.get("REPRO_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "reproduced"))
OUT_DATA = BUNDLE / "source_data"
OUT_FIG = BUNDLE / "figures"
OUT_REPORT = BUNDLE / "reports"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#2B2B2B",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
    }
)


def ensure_dirs() -> None:
    for path in (OUT_DATA, OUT_FIG, OUT_REPORT):
        path.mkdir(parents=True, exist_ok=True)


def load_gpu_rows() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    required = {
        "paper_id",
        "benchmark_gpu_name",
        "gpu_num_filled",
        "benchmark_generation",
        "benchmark_family",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["year"] = df["paper_id"].astype(str).str.extract(r"^(\d{4})")[0].astype(int)
    df = df[df["benchmark_gpu_name"].notna()].copy()
    df["gpu_num_filled"] = pd.to_numeric(df["gpu_num_filled"], errors="coerce").fillna(0)
    return df


def summarize_top_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annual_totals = (
        df.groupby("year")["paper_id"].nunique().rename("gpu_papers_in_year").reset_index()
    )
    summary = (
        df.groupby(["year", "benchmark_gpu_name"], as_index=False)
        .agg(
            papers=("paper_id", "nunique"),
            estimated_gpu_units=("gpu_num_filled", "sum"),
            generation=("benchmark_generation", lambda x: x.dropna().mode().iat[0] if not x.dropna().empty else ""),
            family=("benchmark_family", lambda x: x.dropna().mode().iat[0] if not x.dropna().empty else ""),
        )
        .merge(annual_totals, on="year", how="left")
    )
    summary["paper_share_pct"] = 100 * summary["papers"] / summary["gpu_papers_in_year"]
    summary = summary.sort_values(
        ["year", "papers", "estimated_gpu_units", "benchmark_gpu_name"],
        ascending=[True, False, False, True],
    )
    summary["rank"] = summary.groupby("year").cumcount() + 1
    summary = summary[
        [
            "year",
            "rank",
            "benchmark_gpu_name",
            "papers",
            "paper_share_pct",
            "estimated_gpu_units",
            "gpu_papers_in_year",
            "generation",
            "family",
        ]
    ]

    top_by_year = summary[summary["rank"].le(10)].copy()

    top_models = (
        summary.groupby("benchmark_gpu_name")["papers"]
        .sum()
        .sort_values(ascending=False)
        .head(18)
        .index
    )
    matrix = (
        summary[summary["benchmark_gpu_name"].isin(top_models)]
        .pivot(index="benchmark_gpu_name", columns="year", values="papers")
        .fillna(0)
        .astype(int)
    )
    matrix["total"] = matrix.sum(axis=1)
    matrix = matrix.sort_values("total", ascending=True).drop(columns="total")
    return summary, top_by_year, matrix


def plot_top_models(summary: pd.DataFrame, top_by_year: pd.DataFrame, matrix: pd.DataFrame) -> None:
    years = list(matrix.columns)
    annual_leaders = top_by_year[top_by_year["rank"].eq(1)].sort_values("year").copy()
    line_models = (
        matrix.sum(axis=1)
        .sort_values(ascending=False)
        .head(8)
        .index
    )
    line_data = matrix.loc[line_models].T

    fig = plt.figure(figsize=(7.2, 3.8), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.38, 1.0],
        left=0.08,
        right=0.985,
        top=0.90,
        bottom=0.14,
        wspace=0.30,
    )

    ax_trend = fig.add_subplot(gs[0, 0])
    line_palette = [
        "#235789",
        "#59A14F",
        "#F28E2B",
        "#8E6C8A",
        "#4E79A7",
        "#7F7F7F",
        "#76B7B2",
        "#B07AA1",
    ]
    x_values = np.array(years)
    for idx, model in enumerate(line_models):
        counts = line_data[model].to_numpy(dtype=float)
        ax_trend.plot(
            x_values,
            counts,
            color=line_palette[idx],
            linewidth=1.5,
            marker="o",
            markersize=3.0,
            markeredgewidth=0,
        )

    last_year = years[-1]
    last_counts = line_data.loc[last_year].sort_values(ascending=False)
    label_positions = {}
    min_gap = 28
    previous = None
    for model, count in last_counts.items():
        y = float(count)
        if previous is not None and previous - y < min_gap:
            y = previous - min_gap
        label_positions[model] = max(y, 8)
        previous = label_positions[model]

    for idx, model in enumerate(line_models):
        end_count = float(line_data.loc[last_year, model])
        label = model.replace("NVIDIA ", "")
        ax_trend.text(
            last_year + 0.06,
            label_positions[model],
            f"{label} ({int(end_count)})",
            va="center",
            fontsize=4.9,
            color=line_palette[idx],
        )

    ax_trend.set_xlim(min(years) - 0.15, max(years) + 1.1)
    ax_trend.set_ylim(0, max(760, line_data.to_numpy().max() * 1.08))
    ax_trend.set_xticks(years)
    ax_trend.set_xlabel("Publication year")
    ax_trend.set_ylabel("Papers")
    ax_trend.set_title("a  Paper-count trajectories for leading GPU models", loc="left", pad=6, fontsize=7.5)
    ax_trend.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    ax_trend.tick_params(axis="both", labelsize=6.2)

    ax_leader = fig.add_subplot(gs[0, 1])
    y_positions = np.arange(len(annual_leaders))[::-1]
    leader_colors = annual_leaders["benchmark_gpu_name"].map(
        {
            "NVIDIA Tesla V100 PCIe 16 GB": "#59A14F",
            "NVIDIA A100": "#235789",
        }
    ).fillna("#7F7F7F")
    ax_leader.barh(
        y_positions,
        annual_leaders["paper_share_pct"],
        height=0.58,
        color=leader_colors,
        edgecolor="none",
    )
    for y, row in zip(y_positions, annual_leaders.itertuples(index=False)):
        label = row.benchmark_gpu_name.replace("NVIDIA ", "")
        ax_leader.text(
            row.paper_share_pct + 0.45,
            y,
            f"{label} ({int(row.papers)})",
            va="center",
            fontsize=5.6,
            color="#242424",
        )
        ax_leader.text(
            row.paper_share_pct - 0.6,
            y,
            f"{row.paper_share_pct:.0f}%",
            ha="right",
            va="center",
            fontsize=5.5,
            color="white",
        )

    ax_leader.set_yticks(y_positions)
    ax_leader.set_yticklabels(annual_leaders["year"])
    ax_leader.set_xlim(0, max(40, annual_leaders["paper_share_pct"].max() * 1.35))
    ax_leader.set_xlabel("Share of GPU-reporting papers (%)")
    ax_leader.set_title("b  Annual leading GPU model", loc="left", pad=6, fontsize=7.5)
    ax_leader.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    ax_leader.spines["left"].set_visible(False)
    ax_leader.tick_params(axis="y", length=0, labelsize=6.2)
    ax_leader.tick_params(axis="x", labelsize=6.2)

    base = OUT_FIG / "top_gpu_models_by_year"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(df: pd.DataFrame, top_by_year: pd.DataFrame) -> None:
    leaders = top_by_year[top_by_year["rank"].eq(1)].copy()
    leader_lines = [
        f"- {int(row.year)}: {row.benchmark_gpu_name}, {int(row.papers)} papers ({row.paper_share_pct:.1f}% of GPU-reporting papers)."
        for row in leaders.itertuples(index=False)
    ]
    overall_top = (
        df.groupby("benchmark_gpu_name")["paper_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(5)
    )
    overall_lines = [
        f"- {model}: {int(count)} papers." for model, count in overall_top.items()
    ]

    report = "\n".join(
        [
            "# RQ1: Top GPU Models by Year",
            "",
            "## Method",
            "Input data: `data/analysis_ready/paper_compute_rows.csv`.",
            "The analysis counts unique papers per normalized `benchmark_gpu_name` and publication year. "
            "This is a prevalence measure, not an inventory measure; the companion table also reports summed `gpu_num_filled` as estimated GPU units.",
            "",
            "## Main Result",
            "The annual leader shifts from V100-class GPUs in 2020-2022 to A100-class GPUs from 2023 onward, with H100 entering the top tier in 2025.",
            "",
            "## Annual Leaders",
            *leader_lines,
            "",
            "## Overall Most Frequent Models",
            *overall_lines,
            "",
            "## Outputs",
            "- `4.2/top_gpu_models_by_year/data/top_gpu_models_by_year.csv`: annual top 10 models with counts, shares and estimated units.",
            "- `4.2/top_gpu_models_by_year/data/top_gpu_model_year_matrix.csv`: annual count matrix for the most frequent models.",
            "- `4.2/top_gpu_models_by_year/data/top_gpu_model_trajectories.csv`: source data for the trajectory panel.",
            "- `4.2/top_gpu_models_by_year/data/top_gpu_model_annual_leaders.csv`: source data for the annual-leader panel.",
            "- `4.2/top_gpu_models_by_year/fig/top_gpu_models_by_year.png`: publication-oriented figure export.",
        ]
    )
    (OUT_REPORT / "top_gpu_models_by_year.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_gpu_rows()
    summary, top_by_year, matrix = summarize_top_models(df)
    top_by_year.to_csv(OUT_DATA / "top_gpu_models_by_year.csv", index=False)
    matrix.to_csv(OUT_DATA / "top_gpu_model_year_matrix.csv")
    trajectory_models = matrix.sum(axis=1).sort_values(ascending=False).head(8).index
    matrix.loc[trajectory_models].T.to_csv(OUT_DATA / "top_gpu_model_trajectories.csv")
    top_by_year[top_by_year["rank"].eq(1)].to_csv(OUT_DATA / "top_gpu_model_annual_leaders.csv", index=False)
    plot_top_models(summary, top_by_year, matrix)
    write_report(df, top_by_year)


if __name__ == "__main__":
    main()


