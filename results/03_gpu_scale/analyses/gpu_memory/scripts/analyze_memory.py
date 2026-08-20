from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "teal": "#42949E",
    "red_strong": "#B64342",
    "gold": "#D9A441",
    "violet": "#9A4D8E",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}

BIN_ORDER = [
    "Unknown",
    "<=12 GB",
    "16 GB",
    "24 GB",
    "32 GB",
    "40-48 GB",
    "64-80 GB",
    ">80 GB",
]

BIN_COLORS = {
    "Unknown": "#D8D8D8",
    "<=12 GB": "#767676",
    "16 GB": "#9FB6C8",
    "24 GB": "#6F96BF",
    "32 GB": "#3775BA",
    "40-48 GB": "#42949E",
    "64-80 GB": "#D9A441",
    ">80 GB": "#B64342",
}

VENUE_COLORS = {
    "ACL": "#0F4D92",
    "EMNLP": "#42949E",
    "NAACL": "#B64342",
}


mpl.rcParams.update(
    {
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


def find_analysis_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (
            parent / "data" / "analysis_ready" / "paper_compute_rows.csv"
        ).exists():
            return parent
    raise FileNotFoundError("Could not find analysis root containing data.")


ROOT = find_analysis_root(Path(__file__).resolve())
INPUT_ROWS = ROOT / "data" / "analysis_ready" / "paper_compute_rows.csv"
INPUT_PAPERS = ROOT / "data" / "analysis_ready" / "compute_papers.csv"
INPUT_HARDWARE = ROOT / "data" / "analysis_ready" / "hardware_catalog.csv"
BUNDLE = Path(os.environ.get("REPRO_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "reproduced"))
OUT_DATA = BUNDLE / "source_data"
OUT_FIG = BUNDLE / "figures"
OUT_REPORT = BUNDLE / "reports"


def ensure_dirs() -> None:
    for path in (OUT_DATA, OUT_FIG, OUT_REPORT):
        path.mkdir(parents=True, exist_ok=True)


def parse_venue(paper_id: str) -> str:
    token = str(paper_id).split(".")[1] if "." in str(paper_id) else ""
    if token.startswith("emnlp"):
        return "EMNLP"
    if token.startswith("naacl"):
        return "NAACL"
    if token.startswith("acl"):
        return "ACL"
    return token.upper() or "UNKNOWN"


def infer_memory_bytes_from_text(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value)
    matches = list(
        re.finditer(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*G(?:B|iB)?\b", text, flags=re.I)
    )
    if not matches:
        return np.nan
    return float(matches[-1].group(1)) * 1e9


def memory_bin(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    if value <= 12:
        return "<=12 GB"
    if value <= 16:
        return "16 GB"
    if value <= 24:
        return "24 GB"
    if value <= 32:
        return "32 GB"
    if value <= 48:
        return "40-48 GB"
    if value <= 80:
        return "64-80 GB"
    return ">80 GB"


def quantile(q: float):
    def _inner(values: pd.Series) -> float:
        return values.quantile(q)

    _inner.__name__ = f"q{int(q * 100):02d}"
    return _inner


def load_row_level() -> pd.DataFrame:
    rows = pd.read_csv(INPUT_ROWS)
    hardware = pd.read_csv(INPUT_HARDWARE)
    hardware = hardware[
        [
            "Hardware name",
            "Memory (bytes)",
            "Memory bandwidth (byte/s)",
            "Generation",
            "Family",
            "Manufacturer",
        ]
    ].rename(
        columns={
            "Hardware name": "hardware_name_catalog",
            "Memory (bytes)": "catalog_memory_bytes",
            "Memory bandwidth (byte/s)": "catalog_memory_bandwidth_bytes_s",
            "Generation": "catalog_generation",
            "Family": "catalog_family",
            "Manufacturer": "catalog_manufacturer",
        }
    )
    df = rows.merge(
        hardware,
        left_on="benchmark_gpu_name",
        right_on="hardware_name_catalog",
        how="left",
        validate="many_to_one",
    )

    df["year"] = df["paper_id"].astype(str).str.extract(r"^(\d{4})")[0].astype(int)
    df["venue"] = df["paper_id"].map(parse_venue)
    df["gpu_num"] = pd.to_numeric(df["gpu_num"], errors="coerce")
    df["gpu_num_filled"] = pd.to_numeric(df["gpu_num_filled"], errors="coerce")
    df["memory_bytes"] = pd.to_numeric(df["catalog_memory_bytes"], errors="coerce")
    df["memory_source"] = np.where(df["memory_bytes"].notna(), "catalog", "missing")

    for col in ("benchmark_gpu_name", "gpu_name", "gpu_group"):
        inferred = df[col].map(infer_memory_bytes_from_text)
        mask = df["memory_bytes"].isna() & inferred.notna()
        df.loc[mask, "memory_bytes"] = inferred[mask]
        df.loc[mask, "memory_source"] = "name_explicit_gb"

    df["memory_gb"] = df["memory_bytes"] / 1e9
    df["memory_bandwidth_tb_s"] = df["catalog_memory_bandwidth_bytes_s"] / 1e12
    df["row_total_vram_gb_filled"] = df["memory_gb"] * df["gpu_num_filled"]
    df["row_total_vram_gb_reported_qty"] = df["memory_gb"] * df["gpu_num"]
    df["has_memory"] = df["memory_gb"].notna()
    df["has_reported_gpu_qty"] = df["gpu_num"].notna()
    df["memory_source"] = pd.Categorical(
        df["memory_source"],
        categories=["catalog", "name_explicit_gb", "missing"],
        ordered=True,
    )
    return df


def build_paper_level(rows: pd.DataFrame) -> pd.DataFrame:
    known = rows[rows["has_memory"]].copy()
    total_known_gpu_count = (
        known.groupby("paper_id")["gpu_num_filled"].sum().rename("known_gpu_count_filled")
    )
    base = (
        rows.groupby("paper_id")
        .agg(
            year=("year", "first"),
            venue=("venue", "first"),
            n_gpu_rows=("benchmark_gpu_name", "size"),
            n_memory_known_rows=("has_memory", "sum"),
            n_reported_qty_rows=("has_reported_gpu_qty", "sum"),
            has_missing_gpu_qty=("has_reported_gpu_qty", lambda s: not bool(s.all())),
            has_missing_memory=("has_memory", lambda s: not bool(s.all())),
            paper_max_gpu_memory_gb=("memory_gb", "max"),
            paper_total_vram_gb_filled=("row_total_vram_gb_filled", "sum"),
            paper_total_vram_gb_reported_qty=(
                "row_total_vram_gb_reported_qty",
                "sum",
            ),
        )
        .reset_index()
        .merge(total_known_gpu_count.reset_index(), on="paper_id", how="left")
    )
    base.loc[base["n_memory_known_rows"].eq(0), "paper_total_vram_gb_filled"] = np.nan
    incomplete_reported = base["has_missing_gpu_qty"] | base["has_missing_memory"]
    base.loc[incomplete_reported, "paper_total_vram_gb_reported_qty"] = np.nan
    base["known_gpu_count_filled"] = base["known_gpu_count_filled"].fillna(0)
    base["paper_quantity_weighted_memory_gb"] = (
        base["paper_total_vram_gb_filled"] / base["known_gpu_count_filled"]
    )
    base.loc[
        base["known_gpu_count_filled"].eq(0), "paper_quantity_weighted_memory_gb"
    ] = np.nan
    base["paper_memory_bin"] = pd.Categorical(
        base["paper_max_gpu_memory_gb"].map(memory_bin),
        categories=BIN_ORDER,
        ordered=True,
    )

    main_idx = (
        rows[rows["has_memory"]]
        .sort_values(
            ["paper_id", "row_total_vram_gb_filled", "memory_gb"],
            ascending=[True, False, False],
        )
        .groupby("paper_id")
        .head(1)
        .set_index("paper_id")
    )
    base["paper_main_gpu_name"] = base["paper_id"].map(main_idx["benchmark_gpu_name"])
    base["paper_main_gpu_family"] = base["paper_id"].map(main_idx["benchmark_family"])
    base["paper_main_gpu_generation"] = base["paper_id"].map(
        main_idx["benchmark_generation"]
    )
    return base


def summarize_by_year(papers: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    annual = (
        papers.groupby("year")
        .agg(
            gpu_papers=("paper_id", "nunique"),
            papers_with_memory=("paper_max_gpu_memory_gb", "count"),
            median_max_gpu_memory_gb=("paper_max_gpu_memory_gb", "median"),
            p25_max_gpu_memory_gb=("paper_max_gpu_memory_gb", quantile(0.25)),
            p75_max_gpu_memory_gb=("paper_max_gpu_memory_gb", quantile(0.75)),
            p90_max_gpu_memory_gb=("paper_max_gpu_memory_gb", quantile(0.90)),
            median_weighted_gpu_memory_gb=(
                "paper_quantity_weighted_memory_gb",
                "median",
            ),
            median_total_vram_gb_filled=("paper_total_vram_gb_filled", "median"),
            p75_total_vram_gb_filled=("paper_total_vram_gb_filled", quantile(0.75)),
            p90_total_vram_gb_filled=("paper_total_vram_gb_filled", quantile(0.90)),
            max_total_vram_gb_filled=("paper_total_vram_gb_filled", "max"),
        )
        .reset_index()
    )
    row_annual = (
        rows.groupby("year")
        .agg(
            gpu_rows=("paper_id", "size"),
            rows_with_memory=("has_memory", "sum"),
            gpu_instances_filled=("gpu_num_filled", "sum"),
        )
        .reset_index()
    )
    annual = annual.merge(row_annual, on="year", how="left")
    annual["paper_memory_coverage_pct"] = (
        annual["papers_with_memory"] / annual["gpu_papers"] * 100
    )
    annual["row_memory_coverage_pct"] = (
        annual["rows_with_memory"] / annual["gpu_rows"] * 100
    )
    for threshold in (24, 40, 80):
        name = f"share_papers_max_ge_{threshold}gb"
        annual[name] = (
            papers.assign(flag=papers["paper_max_gpu_memory_gb"].ge(threshold))
            .groupby("year")["flag"]
            .mean()
            .mul(100)
            .reindex(annual["year"])
            .to_numpy()
        )
    return annual.round(3)


def summarize_bins(papers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    totals = (
        papers.groupby("year")["paper_id"].nunique().rename("gpu_papers").reset_index()
    )
    bins = (
        papers.groupby(["year", "paper_memory_bin"], observed=False)["paper_id"]
        .nunique()
        .rename("papers")
        .reset_index()
        .merge(totals, on="year", how="left")
    )
    bins["share_pct"] = bins["papers"] / bins["gpu_papers"] * 100
    bins["paper_memory_bin"] = bins["paper_memory_bin"].astype(str)
    bins = bins.sort_values(
        ["year", "paper_memory_bin"],
        key=lambda col: col.map({name: i for i, name in enumerate(BIN_ORDER)})
        if col.name == "paper_memory_bin"
        else col,
    )
    matrix = (
        bins.pivot(index="year", columns="paper_memory_bin", values="share_pct")
        .reindex(columns=BIN_ORDER)
        .fillna(0)
    )
    return bins.round(3), matrix


def summarize_by_venue_year(papers: pd.DataFrame) -> pd.DataFrame:
    return (
        papers.groupby(["year", "venue"])
        .agg(
            gpu_papers=("paper_id", "nunique"),
            papers_with_memory=("paper_max_gpu_memory_gb", "count"),
            median_max_gpu_memory_gb=("paper_max_gpu_memory_gb", "median"),
            p75_max_gpu_memory_gb=("paper_max_gpu_memory_gb", quantile(0.75)),
            p90_max_gpu_memory_gb=("paper_max_gpu_memory_gb", quantile(0.90)),
            median_total_vram_gb_filled=("paper_total_vram_gb_filled", "median"),
            p90_total_vram_gb_filled=("paper_total_vram_gb_filled", quantile(0.90)),
        )
        .reset_index()
        .round(3)
    )


def summarize_top_models(rows: pd.DataFrame) -> pd.DataFrame:
    top = (
        rows[rows["has_memory"]]
        .groupby(["year", "benchmark_gpu_name"])
        .agg(
            gpu_rows=("paper_id", "size"),
            unique_papers=("paper_id", "nunique"),
            memory_gb=("memory_gb", "median"),
            total_gpu_instances_filled=("gpu_num_filled", "sum"),
            family=("benchmark_family", "first"),
            generation=("benchmark_generation", "first"),
        )
        .reset_index()
    )
    top["rank_in_year"] = top.groupby("year")["gpu_rows"].rank(
        method="first", ascending=False
    )
    return top.sort_values(["year", "rank_in_year"]).round(3)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_memory_trends(
    annual: pd.DataFrame,
    bin_matrix: pd.DataFrame,
    venue_year: pd.DataFrame,
) -> plt.Figure:
    years = annual["year"].to_numpy()
    x = np.arange(len(years))
    fig = plt.figure(figsize=(183 / 25.4, 126 / 25.4), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        bottom=0.105,
        top=0.94,
        hspace=0.52,
        wspace=0.34,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(
        years,
        annual["p25_max_gpu_memory_gb"],
        annual["p75_max_gpu_memory_gb"],
        color=PALETTE["blue_secondary"],
        alpha=0.18,
        linewidth=0,
        label="IQR",
    )
    ax.plot(
        years,
        annual["median_max_gpu_memory_gb"],
        color=PALETTE["blue_main"],
        marker="o",
        markersize=3.2,
        linewidth=1.5,
        label="Median",
    )
    ax.plot(
        years,
        annual["p90_max_gpu_memory_gb"],
        color=PALETTE["red_strong"],
        marker="s",
        markersize=2.8,
        linewidth=1.1,
        label="P90",
    )
    ax.set_ylabel("Paper max GPU memory (GB)")
    ax.set_xticks(years)
    ax.set_ylim(0, max(90, annual["p90_max_gpu_memory_gb"].max() + 10))
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.02))
    add_panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    bottom = np.zeros(len(bin_matrix.index))
    for name in BIN_ORDER:
        values = bin_matrix[name].to_numpy()
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
    ax.set_ylabel("Share of GPU papers (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_matrix.index.astype(int))
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.35,
        fontsize=6.0,
    )
    add_panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(
        years,
        annual["median_total_vram_gb_filled"],
        color=PALETTE["teal"],
        marker="o",
        markersize=3.2,
        linewidth=1.5,
        label="Median",
    )
    ax.plot(
        years,
        annual["p90_total_vram_gb_filled"],
        color=PALETTE["gold"],
        marker="s",
        markersize=2.8,
        linewidth=1.1,
        label="P90",
    )
    ax.set_yscale("log", base=2)
    ax.set_yticks([16, 32, 64, 128, 256, 512, 1024])
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_ylabel("Total VRAM per paper (GB)")
    ax.set_xticks(years)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.02))
    add_panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    for venue in ["ACL", "EMNLP", "NAACL"]:
        sub = venue_year[venue_year["venue"].eq(venue)].sort_values("year")
        ax.plot(
            sub["year"],
            sub["median_max_gpu_memory_gb"],
            color=VENUE_COLORS[venue],
            marker="o",
            markersize=3.0,
            linewidth=1.25,
            label=venue,
        )
    ax.set_ylabel("Median max GPU memory (GB)")
    ax.set_xticks(years)
    ax.set_ylim(0, max(90, venue_year["median_max_gpu_memory_gb"].max() + 12))
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.02))
    add_panel_label(ax, "d")

    first = annual.iloc[0]
    last = annual.iloc[-1]
    fig.text(
        0.075,
        0.025,
        (
            "Memory is expressed as catalog bytes / 1e9. "
            "Paper-level total VRAM uses upstream filled GPU counts, treating missing quantities as one GPU. "
            f"Known-memory coverage: {first.paper_memory_coverage_pct:.1f}% in {int(first.year)} and "
            f"{last.paper_memory_coverage_pct:.1f}% in {int(last.year)}."
        ),
        ha="left",
        va="bottom",
        fontsize=6.0,
        color="#555555",
    )
    return fig


def plot_memory_distribution(papers: pd.DataFrame, annual: pd.DataFrame) -> plt.Figure:
    years = sorted(papers["year"].unique())
    data = [
        papers.loc[papers["year"].eq(year), "paper_max_gpu_memory_gb"].dropna()
        for year in years
    ]
    fig = plt.figure(figsize=(183 / 25.4, 84 / 25.4), constrained_layout=False)
    ax = fig.add_axes([0.075, 0.20, 0.91, 0.64])
    parts = ax.violinplot(
        data,
        positions=np.arange(len(years)),
        widths=0.76,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(PALETTE["blue_secondary"])
        body.set_alpha(0.18)
        body.set_edgecolor(PALETTE["blue_main"])
        body.set_linewidth(0.7)
    box = ax.boxplot(
        data,
        positions=np.arange(len(years)),
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["red_strong"], "linewidth": 1.2},
        boxprops={
            "facecolor": "white",
            "edgecolor": PALETTE["neutral_dark"],
            "linewidth": 0.7,
        },
        whiskerprops={"color": PALETTE["neutral_dark"], "linewidth": 0.7},
        capprops={"color": PALETTE["neutral_dark"], "linewidth": 0.7},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.85)
    ax.set_ylabel("Paper max GPU memory (GB)")
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years)
    ax.set_ylim(0, 205)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)
    for i, row in enumerate(annual.itertuples(index=False)):
        ax.text(
            i,
            -0.17,
            f"n={int(row.papers_with_memory)}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.0,
            color="#555555",
        )
    fig.text(
        0.075,
        0.04,
        "Violins are clipped at 205 GB for readability; source tables retain larger catalog/inferred memory values.",
        ha="left",
        va="bottom",
        fontsize=6.0,
        color="#555555",
    )
    return fig


def write_report(
    annual: pd.DataFrame,
    bins: pd.DataFrame,
    venue_year: pd.DataFrame,
    rows: pd.DataFrame,
    papers: pd.DataFrame,
) -> None:
    first = annual.iloc[0]
    last = annual.iloc[-1]
    med_ratio = (
        last["median_max_gpu_memory_gb"] / first["median_max_gpu_memory_gb"]
        if first["median_max_gpu_memory_gb"]
        else np.nan
    )
    total_ratio = (
        last["median_total_vram_gb_filled"] / first["median_total_vram_gb_filled"]
        if first["median_total_vram_gb_filled"]
        else np.nan
    )
    known_papers = int(papers["paper_max_gpu_memory_gb"].notna().sum())
    total_papers = int(papers["paper_id"].nunique())
    known_rows = int(rows["has_memory"].sum())
    total_rows = int(len(rows))
    source_counts = rows["memory_source"].value_counts(dropna=False).to_dict()
    share_40_first = float(first["share_papers_max_ge_40gb"])
    share_40_last = float(last["share_papers_max_ge_40gb"])
    share_80_first = float(first["share_papers_max_ge_80gb"])
    share_80_last = float(last["share_papers_max_ge_80gb"])
    bin_2025 = bins[bins["year"].eq(int(last["year"]))].set_index("paper_memory_bin")
    bin_2020 = bins[bins["year"].eq(int(first["year"]))].set_index("paper_memory_bin")
    high_2025 = (
        bin_2025.loc[["64-80 GB", ">80 GB"], "share_pct"].sum()
        if not bin_2025.empty
        else np.nan
    )
    high_2020 = (
        bin_2020.loc[["64-80 GB", ">80 GB"], "share_pct"].sum()
        if not bin_2020.empty
        else np.nan
    )

    annual_denominators = ", ".join(
        f"{int(row.year)} n={int(row.gpu_papers)}"
        for row in annual.itertuples(index=False)
    )
    venue_lines = []
    for venue in ["ACL", "EMNLP", "NAACL"]:
        sub = venue_year[venue_year["venue"].eq(venue)].sort_values("year")
        if sub.empty:
            continue
        venue_lines.append(
            f"- {venue}: median max GPU memory {sub.iloc[0].median_max_gpu_memory_gb:g} GB "
            f"in {int(sub.iloc[0].year)} to {sub.iloc[-1].median_max_gpu_memory_gb:g} GB "
            f"in {int(sub.iloc[-1].year)}."
        )

    report = f"""# RQ1: GPU Memory over Time

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
Annual denominators: {annual_denominators}.
Known-memory coverage: {known_papers}/{total_papers} papers and {known_rows}/{total_rows} GPU rows.
Memory source counts: catalog={int(source_counts.get("catalog", 0))}, name-explicit GB={int(source_counts.get("name_explicit_gb", 0))}, missing={int(source_counts.get("missing", 0))}.

## Main Result
The median maximum per-GPU memory increased from {first.median_max_gpu_memory_gb:g} GB in {int(first.year)} to {last.median_max_gpu_memory_gb:g} GB in {int(last.year)}, a {med_ratio:.1f}x increase.
The P90 maximum per-GPU memory increased from {first.p90_max_gpu_memory_gb:g} GB to {last.p90_max_gpu_memory_gb:g} GB.
Papers using at least 40 GB-class GPUs rose from {share_40_first:.1f}% to {share_40_last:.1f}%.
Papers using at least 80 GB-class GPUs rose from {share_80_first:.1f}% to {share_80_last:.1f}%.
The combined `64-80 GB` plus `>80 GB` memory-bin share changed from {high_2020:.1f}% in {int(first.year)} to {high_2025:.1f}% in {int(last.year)}.
The median lower-bound total VRAM per paper increased from {first.median_total_vram_gb_filled:g} GB to {last.median_total_vram_gb_filled:g} GB, a {total_ratio:.1f}x increase.

## Venue Pattern
{chr(10).join(venue_lines)}

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
"""
    (OUT_REPORT / "memory_over_time.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    rows = load_row_level()
    papers = build_paper_level(rows)
    annual = summarize_by_year(papers, rows)
    bins, bin_matrix = summarize_bins(papers)
    venue_year = summarize_by_venue_year(papers)
    top_models = summarize_top_models(rows)
    source_coverage = (
        rows.groupby(["memory_source"], observed=False)
        .agg(gpu_rows=("paper_id", "size"), unique_papers=("paper_id", "nunique"))
        .reset_index()
    )

    rows.to_csv(OUT_DATA / "memory_row_level_enriched.csv", index=False)
    papers.to_csv(OUT_DATA / "memory_paper_level.csv", index=False)
    annual.to_csv(OUT_DATA / "memory_summary_by_year.csv", index=False)
    bins.to_csv(OUT_DATA / "memory_bins_by_year.csv", index=False)
    bin_matrix.to_csv(OUT_DATA / "memory_bin_share_matrix.csv")
    venue_year.to_csv(OUT_DATA / "memory_summary_by_venue_year.csv", index=False)
    top_models.to_csv(OUT_DATA / "top_memory_gpu_models_by_year.csv", index=False)
    source_coverage.to_csv(OUT_DATA / "memory_source_coverage.csv", index=False)

    fig = plot_memory_trends(annual, bin_matrix, venue_year)
    save_figure(fig, OUT_FIG / "memory_trends_over_time")
    plt.close(fig)

    fig = plot_memory_distribution(papers, annual)
    save_figure(fig, OUT_FIG / "memory_distribution_by_year")
    plt.close(fig)

    write_report(annual, bins, venue_year, rows, papers)


if __name__ == "__main__":
    main()



