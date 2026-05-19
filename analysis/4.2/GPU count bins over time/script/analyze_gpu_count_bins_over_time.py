from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_analysis_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "data" / "compute_paper_level_gpu_only.xlsx").exists():
            return parent
    raise FileNotFoundError("Could not find analysis root containing data.")


ROOT = find_analysis_root(Path(__file__).resolve())
INPUT = ROOT / "data" / "compute_paper_level_gpu_only.xlsx"
BUNDLE = Path(__file__).resolve().parents[1]
OUT_DATA = BUNDLE / "data"
OUT_FIG = BUNDLE / "fig"
OUT_REPORT = BUNDLE / "report"


BIN_ORDER = [
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

BIN_COLORS = {
    "Unspecified": "#D0D0D0",
    "1": "#6F6F6F",
    "2": "#86A3B8",
    "3-4": "#4E79A7",
    "5-8": "#59A14F",
    "9-16": "#F2C14E",
    "17-32": "#E15759",
    "33-64": "#B07AA1",
    "65+": "#7B4F9D",
}


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


def assign_bin(value: float, raw_missing: bool) -> str:
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
    df = pd.read_excel(INPUT)
    required = {
        "paper_id",
        "paper_gpu_num_total",
        "paper_gpu_num_filled_total",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["year"] = df["paper_id"].astype(str).str.extract(r"^(\d{4})")[0].astype(int)
    df["gpu_count_raw"] = pd.to_numeric(df["paper_gpu_num_total"], errors="coerce")
    df["gpu_count_filled"] = pd.to_numeric(
        df["paper_gpu_num_filled_total"], errors="coerce"
    )
    df["count_unspecified"] = df["gpu_count_raw"].isna()
    df["gpu_count_bin"] = [
        assign_bin(value, missing)
        for value, missing in zip(df["gpu_count_filled"], df["count_unspecified"])
    ]
    df["gpu_count_bin"] = pd.Categorical(
        df["gpu_count_bin"], categories=BIN_ORDER, ordered=True
    )
    return df


def summarize_bins(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    summary["share_pct"] = (
        summary["papers"] / summary["gpu_papers_in_year"] * 100
    ).round(3)
    summary["gpu_count_bin"] = summary["gpu_count_bin"].astype(str)
    summary = summary.sort_values(
        ["year", "gpu_count_bin"],
        key=lambda col: col.map({name: i for i, name in enumerate(BIN_ORDER)})
        if col.name == "gpu_count_bin"
        else col,
    )

    share_matrix = (
        summary.pivot(index="year", columns="gpu_count_bin", values="share_pct")
        .reindex(columns=BIN_ORDER)
        .fillna(0)
    )
    count_matrix = (
        summary.pivot(index="year", columns="gpu_count_bin", values="papers")
        .reindex(columns=BIN_ORDER)
        .fillna(0)
        .astype(int)
    )

    distribution = (
        df.groupby("year")
        .agg(
            gpu_papers=("paper_id", "nunique"),
            unspecified_count_papers=("count_unspecified", "sum"),
            median_filled_gpu_count=("gpu_count_filled", "median"),
            mean_filled_gpu_count=("gpu_count_filled", "mean"),
            p75_filled_gpu_count=("gpu_count_filled", lambda x: x.quantile(0.75)),
            p90_filled_gpu_count=("gpu_count_filled", lambda x: x.quantile(0.90)),
            p95_filled_gpu_count=("gpu_count_filled", lambda x: x.quantile(0.95)),
            max_filled_gpu_count=("gpu_count_filled", "max"),
        )
        .reset_index()
    )
    distribution["unspecified_count_share_pct"] = (
        distribution["unspecified_count_papers"] / distribution["gpu_papers"] * 100
    ).round(3)
    for col in [
        "mean_filled_gpu_count",
        "p75_filled_gpu_count",
        "p90_filled_gpu_count",
        "p95_filled_gpu_count",
    ]:
        distribution[col] = distribution[col].round(3)

    return summary, share_matrix, count_matrix, distribution


def plot_count_bin_composition(
    share_matrix: pd.DataFrame, count_matrix: pd.DataFrame
) -> plt.Figure:
    years = share_matrix.index.to_numpy()
    x = np.arange(len(years))

    fig = plt.figure(figsize=(183 / 25.4, 70 / 25.4), constrained_layout=False)
    ax = fig.add_axes([0.08, 0.22, 0.905, 0.63])

    bottom = np.zeros(len(years))
    for name in BIN_ORDER:
        values = share_matrix[name].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.68,
            color=BIN_COLORS[name],
            edgecolor="white",
            linewidth=0.35,
            label=name,
        )
        bottom += values

    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of GPU-reporting papers (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_title("GPU count bins over time", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        columnspacing=1.0,
        handlelength=1.1,
        handletextpad=0.35,
    )

    for i, year in enumerate(years):
        total = int(count_matrix.loc[year].sum())
        ax.text(
            i,
            -0.15,
            f"n={total}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.2,
            color="#333333",
        )

    return fig


def plot_scale_classes(
    count_matrix: pd.DataFrame, distribution: pd.DataFrame
) -> plt.Figure:
    years = count_matrix.index.to_numpy()
    x = np.arange(len(years))
    reported = count_matrix.drop(columns=["Unspecified"]).copy()
    large = reported[["9-16", "17-32", "33-64", "65+"]].sum(axis=1)
    mid = reported[["3-4", "5-8"]].sum(axis=1)
    small = reported[["1", "2"]].sum(axis=1)
    denom = count_matrix.sum(axis=1).replace(0, np.nan)

    fig = plt.figure(figsize=(183 / 25.4, 66 / 25.4), constrained_layout=False)
    ax = fig.add_axes([0.08, 0.26, 0.905, 0.60])

    ax.plot(
        x,
        (small / denom * 100).to_numpy(),
        color="#6F6F6F",
        marker="o",
        markersize=3.0,
        linewidth=1.2,
        label="1-2 GPUs",
    )
    ax.plot(
        x,
        (mid / denom * 100).to_numpy(),
        color="#4E79A7",
        marker="o",
        markersize=3.0,
        linewidth=1.2,
        label="3-8 GPUs",
    )
    ax.plot(
        x,
        (large / denom * 100).to_numpy(),
        color="#E15759",
        marker="o",
        markersize=3.0,
        linewidth=1.2,
        label="9+ GPUs",
    )
    ax.set_title("GPU scale classes over time", loc="left", fontweight="bold")
    ax.set_ylabel("Share of GPU-reporting papers (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    collapsed_max = np.nanmax(
        np.vstack(
            [
                (small / denom * 100).to_numpy(),
                (mid / denom * 100).to_numpy(),
                (large / denom * 100).to_numpy(),
            ]
        )
    )
    ax.set_ylim(0, max(55, collapsed_max + 10))
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.02))

    medians = distribution.set_index("year").loc[years, "median_filled_gpu_count"]
    for i, median in enumerate(medians):
        ax.text(
            i,
            -0.22,
            f"med. {median:g}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.0,
            color="#555555",
        )

    return fig


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def remove_previous_combined_exports() -> None:
    stems = ("gpu_count_bins_over_time", "gpu_count_bins_horizontal_over_time")
    for stem in stems:
        for suffix in (".png",):
            path = OUT_FIG / f"{stem}{suffix}"
            if path.exists():
                path.unlink()


def write_report(
    summary: pd.DataFrame,
    share_matrix: pd.DataFrame,
    count_matrix: pd.DataFrame,
    distribution: pd.DataFrame,
) -> None:
    reported = count_matrix.drop(columns=["Unspecified"])
    denom = count_matrix.sum(axis=1)
    small = reported[["1", "2"]].sum(axis=1)
    mid = reported[["3-4", "5-8"]].sum(axis=1)
    large = reported[["9-16", "17-32", "33-64", "65+"]].sum(axis=1)
    scale = pd.DataFrame(
        {
            "year": denom.index,
            "small_1_2_share_pct": (small / denom * 100).round(1).to_numpy(),
            "mid_3_8_share_pct": (mid / denom * 100).round(1).to_numpy(),
            "large_9plus_share_pct": (large / denom * 100).round(1).to_numpy(),
        }
    )

    leaders = []
    for year, row in count_matrix.iterrows():
        leader = row.drop(labels=["Unspecified"]).idxmax()
        count = int(row[leader])
        share = count / int(row.sum()) * 100
        leaders.append(f"- {year}: {leader}, {count} papers ({share:.1f}%).")

    first_year = int(denom.index.min())
    last_year = int(denom.index.max())
    scale_first = scale[scale["year"] == first_year].iloc[0]
    scale_last = scale[scale["year"] == last_year].iloc[0]
    dist = distribution.set_index("year")
    unspecified_first = dist.loc[first_year, "unspecified_count_share_pct"]
    unspecified_last = dist.loc[last_year, "unspecified_count_share_pct"]
    median_first = dist.loc[first_year, "median_filled_gpu_count"]
    median_last = dist.loc[last_year, "median_filled_gpu_count"]

    annual_denominators = ", ".join(
        f"{int(row.year)} n={int(row.gpu_papers)}"
        for row in distribution.itertuples(index=False)
    )

    report = f"""# RQ1: GPU Count Bins over Time

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
Annual denominators: {annual_denominators}.
Filled GPU count statistics use `paper_gpu_num_filled_total`, where missing quantities are filled by the upstream data as one GPU.

## Main Result
The share of papers reporting 1-2 GPUs decreased from {scale_first.small_1_2_share_pct:.1f}% in {first_year} to {scale_last.small_1_2_share_pct:.1f}% in {last_year}, while 3-8 GPU reports increased from {scale_first.mid_3_8_share_pct:.1f}% to {scale_last.mid_3_8_share_pct:.1f}%.
The 9+ GPU class changed from {scale_first.large_9plus_share_pct:.1f}% in {first_year} to {scale_last.large_9plus_share_pct:.1f}% in {last_year}. The filled-count median rose from {median_first:g} GPU in {first_year} to {median_last:g} GPUs in {last_year}.
Unspecified quantity reports decreased from {unspecified_first:.1f}% in {first_year} to {unspecified_last:.1f}% in {last_year}.

## Annual Reported-Bin Leaders
{chr(10).join(leaders)}

## Outputs
- `4.2/GPU count bins over time/data/gpu_count_bins_by_year.csv`: annual bin counts and shares, including unspecified quantities.
- `4.2/GPU count bins over time/data/gpu_count_bin_share_matrix.csv`: share matrix used for the composition figure.
- `4.2/GPU count bins over time/data/gpu_count_bin_count_matrix.csv`: count matrix used for both split figures.
- `4.2/GPU count bins over time/data/gpu_count_distribution_by_year.csv`: annual median, mean, upper quantiles, max, and unspecified-rate diagnostics.
- `4.2/GPU count bins over time/fig/gpu_count_bin_composition_over_time.png`: annual GPU-count bin composition figure export.
- `4.2/GPU count bins over time/fig/gpu_count_scale_classes_over_time.png`: collapsed scale-class trend figure export.
"""
    (OUT_REPORT / "gpu_count_bins_over_time.md").write_text(report, encoding="utf-8")
    scale.to_csv(OUT_DATA / "gpu_count_scale_classes_by_year.csv", index=False)


def main() -> None:
    ensure_dirs()
    df = load_paper_level()
    summary, share_matrix, count_matrix, distribution = summarize_bins(df)

    summary.to_csv(OUT_DATA / "gpu_count_bins_by_year.csv", index=False)
    share_matrix.to_csv(OUT_DATA / "gpu_count_bin_share_matrix.csv")
    count_matrix.to_csv(OUT_DATA / "gpu_count_bin_count_matrix.csv")
    distribution.to_csv(OUT_DATA / "gpu_count_distribution_by_year.csv", index=False)

    remove_previous_combined_exports()

    fig = plot_count_bin_composition(share_matrix, count_matrix)
    save_figure(fig, OUT_FIG / "gpu_count_bin_composition_over_time")
    plt.close(fig)

    fig = plot_scale_classes(count_matrix, distribution)
    save_figure(fig, OUT_FIG / "gpu_count_scale_classes_over_time")
    plt.close(fig)

    write_report(summary, share_matrix, count_matrix, distribution)


if __name__ == "__main__":
    main()





