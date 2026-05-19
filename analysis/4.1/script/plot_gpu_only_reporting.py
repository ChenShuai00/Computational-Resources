from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]


PALETTE = {
    "blue_dark": "#0F4D92",
    "blue_mid": "#3775BA",
    "blue_light": "#B4C0E4",
    "teal": "#42949E",
    "neutral_dark": "#4D4D4D",
    "neutral_mid": "#767676",
    "neutral_light": "#D8D8D8",
    "neutral_faint": "#F2F2F2",
    "paper": "#FFFFFF",
}


def parse_year(paper_id: object) -> int | float:
    parts = str(paper_id).split(".")
    if parts and parts[0].isdigit():
        return int(parts[0])
    return np.nan


def parse_venue(paper_id: object) -> str:
    parts = str(paper_id).split(".")
    if len(parts) < 3:
        return "unknown"
    venue = ".".join(parts[1:-1]).removeprefix("findings-")
    return venue.split("-")[0].upper()


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.09,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="black",
    )


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def build_summary(all_df: pd.DataFrame, gpu_df: pd.DataFrame, strict_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for frame in (all_df, gpu_df, strict_df):
        frame["year"] = frame["paper_id"].map(parse_year)
        frame["venue"] = frame["paper_id"].map(parse_venue)

    gpu_ids = set(gpu_df["paper_id"].astype(str))
    strict_ids = set(strict_df.loc[strict_df["paper_total_compute_capability"].notna(), "paper_id"].astype(str))

    work = all_df[["paper_id", "year", "venue"]].copy()
    work["gpu_only"] = work["paper_id"].astype(str).isin(gpu_ids)
    work["strict_gpu"] = work["paper_id"].astype(str).isin(strict_ids)

    def summarize(group_cols: list[str]) -> pd.DataFrame:
        out = (
            work.groupby(group_cols, dropna=False)
            .agg(
                total_papers=("paper_id", "nunique"),
                gpu_papers=("gpu_only", "sum"),
                strict_gpu_papers=("strict_gpu", "sum"),
            )
            .reset_index()
        )
        out["gpu_share_pct"] = out["gpu_papers"] / out["total_papers"] * 100
        out["strict_share_pct"] = out["strict_gpu_papers"] / out["total_papers"] * 100
        out["gpu_missing_quantity_papers"] = out["gpu_papers"] - out["strict_gpu_papers"]
        out["missing_qty_within_gpu_pct"] = out["gpu_missing_quantity_papers"] / out["gpu_papers"] * 100
        return out

    by_year = summarize(["year"]).sort_values("year")
    by_venue = summarize(["venue"]).sort_values("venue")
    return by_year, by_venue


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", labelsize=6.5, width=0.6, length=3)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.55, zorder=0)


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    path = output_stem.with_suffix(".png")
    try:
        fig.savefig(path, bbox_inches="tight", dpi=300)
    except OSError:
        if path.exists():
            return
        raise


def plot_panel_a(ax_a: plt.Axes, by_year: pd.DataFrame, label: str | None = "a", legend_ncols: int = 2) -> None:
    years = by_year["year"].astype(int).to_numpy()
    gpu_share = by_year["gpu_share_pct"].to_numpy()
    strict_share = by_year["strict_share_pct"].to_numpy()

    ax_a.plot(years, gpu_share, color=PALETTE["blue_dark"], marker="o", markersize=4.2, linewidth=1.8, label="GPU model reported")
    ax_a.plot(years, strict_share, color=PALETTE["teal"], marker="s", markersize=3.8, linewidth=1.55, label="GPU model + count")
    ax_a.fill_between(years, strict_share, gpu_share, color=PALETTE["blue_light"], alpha=0.28, linewidth=0)
    for x, y, count in zip(years, gpu_share, by_year["gpu_papers"]):
        ax_a.text(x, y + 2.2, f"{int(count):,}", ha="center", va="bottom", fontsize=5.8, color=PALETTE["blue_dark"])
    ax_a.set_ylim(0, 66)
    ax_a.set_xlim(years.min() - 0.25, years.max() + 0.25)
    ax_a.set_xticks(years)
    ax_a.set_ylabel("Share of all papers (%)", fontsize=7)
    ax_a.set_xlabel("Publication year", fontsize=7)
    ax_a.legend(loc="upper left", ncols=legend_ncols, fontsize=6.2, handlelength=2.4, columnspacing=1.4)
    style_axes(ax_a)
    if label:
        add_panel_label(ax_a, label)


def plot_panel_b(ax_b: plt.Axes, by_venue: pd.DataFrame, label: str | None = "b", show_legend: bool = True) -> None:
    venue_order = ["ACL", "EMNLP", "NAACL"]
    venue = by_venue.set_index("venue").loc[venue_order].reset_index()
    y = np.arange(len(venue_order))
    h = 0.32
    ax_b.barh(y + h / 2, venue["gpu_share_pct"], height=h, color=PALETTE["blue_dark"], label="GPU model reported", zorder=3)
    ax_b.barh(y - h / 2, venue["strict_share_pct"], height=h, color=PALETTE["teal"], label="GPU model + count", zorder=3)
    for i, row in venue.iterrows():
        ax_b.text(row["gpu_share_pct"] + 1.0, i + h / 2, f"{row['gpu_papers']:,} ({fmt_pct(row['gpu_share_pct'])})", va="center", fontsize=5.8, color=PALETTE["blue_dark"])
        ax_b.text(row["strict_share_pct"] + 1.0, i - h / 2, f"{row['strict_gpu_papers']:,} ({fmt_pct(row['strict_share_pct'])})", va="center", fontsize=5.8, color=PALETTE["teal"])
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(venue_order, fontsize=6.7)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 68)
    ax_b.set_xlabel("Share of all papers (%)", fontsize=7)
    if show_legend:
        ax_b.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncols=2,
            fontsize=5.8,
            handlelength=1.5,
            frameon=False,
        )
    style_axes(ax_b)
    if label:
        add_panel_label(ax_b, label)


def plot_ab_horizontal(by_year: pd.DataFrame, by_venue: pd.DataFrame, output_stem: Path) -> None:
    fig = plt.figure(figsize=(7.20, 2.70), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.16, 1.0], wspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    plot_panel_a(ax_a, by_year, "a", legend_ncols=1)
    plot_panel_b(ax_b, by_venue, "b", show_legend=False)

    fig.subplots_adjust(left=0.075, right=0.98, top=0.88, bottom=0.18)
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_a_only(by_year: pd.DataFrame, output_stem: Path) -> None:
    fig = plt.figure(figsize=(4.20, 2.75), facecolor="white")
    ax_a = fig.add_subplot(111)

    plot_panel_a(ax_a, by_year, "a", legend_ncols=1)

    fig.subplots_adjust(left=0.145, right=0.96, top=0.86, bottom=0.21)
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_b_only(by_venue: pd.DataFrame, output_stem: Path) -> None:
    fig = plt.figure(figsize=(4.10, 2.75), facecolor="white")
    ax_b = fig.add_subplot(111)

    plot_panel_b(ax_b, by_venue, "b", show_legend=True)

    fig.subplots_adjust(left=0.145, right=0.94, top=0.80, bottom=0.19)
    save_figure(fig, output_stem)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    reprot_dir = script_dir.parent
    workspace_dir = reprot_dir.parent
    summary_dir = reprot_dir / "data"
    fig_dir = reprot_dir / "fig"
    report_dir = reprot_dir / "report"

    year_summary = summary_dir / "gpu_only_year_summary.csv"
    venue_summary = summary_dir / "gpu_only_venue_summary.csv"

    if year_summary.exists() and venue_summary.exists():
        by_year = pd.read_csv(year_summary)
        by_venue = pd.read_csv(venue_summary)
    else:
        data_dir = workspace_dir / "data"
        all_df = pd.read_excel(data_dir / "soft_compute_paper_level.xlsx")
        gpu_df = pd.read_excel(data_dir / "compute_paper_level_gpu_only.xlsx")
        strict_df = pd.read_excel(data_dir / "strict_compute_paper_level.xlsx")

        by_year, by_venue = build_summary(all_df, gpu_df, strict_df)

        summary_dir.mkdir(parents=True, exist_ok=True)
        by_year.to_csv(year_summary, index=False)
        by_venue.to_csv(venue_summary, index=False)

    plot_ab_horizontal(by_year, by_venue, fig_dir / "gpu_only_reporting_time_venue")
    plot_a_only(by_year, fig_dir / "gpu_only_reporting_time_venue_a")
    plot_b_only(by_venue, fig_dir / "gpu_only_reporting_time_venue_b")

    notes = [
        "Figure contract",
        "Core conclusion: GPU reporting rises sharply from 2020 to 2025, with EMNLP and ACL showing higher GPU-only coverage than NAACL.",
        "Archetype: quantitative grid.",
        "Backend: Python/matplotlib only.",
        "Exports: PNG only.",
        "GPU-only reporting layout: panels a and b are exported as standalone figures.",
        "GPU-only reporting combined export: panels a and b are also retained as a horizontal figure.",
        "Source data: gpu_only_year_summary.csv, gpu_only_venue_summary.csv.",
        f"Total papers: {int(by_year['total_papers'].sum()):,}.",
        f"GPU-only papers: {int(by_year['gpu_papers'].sum()):,}.",
        f"Strict GPU papers: {int(by_year['strict_gpu_papers'].sum()):,}.",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "gpu_only_reporting_time_venue_qa.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

