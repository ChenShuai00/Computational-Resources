from __future__ import annotations

import os
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        "axes.edgecolor": "#222222",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
    }
)


TOP_COMPUTE_QUANTILE = 0.80
TOP_COMPUTE_BASELINE = 0.20
TOP_COMPUTE_LABEL = "top 20%"
MAIN_THRESHOLD = 50
APPENDIX_THRESHOLD = 30
COMPUTE_COL = "paper_max_row_compute_capability_gfimp_lb1"
COUNTRY_CODE_ALIASES = {
    "HK": "CN",
    "TW": "CN",
}
COUNTRY_DISPLAY_NAMES = {
    "AE": "United Arab Emirates",
    "AU": "Australia",
    "CA": "Canada",
    "CH": "Switzerland",
    "CN": "China",
    "DE": "Germany",
    "ES": "Spain",
    "FR": "France",
    "GB": "United Kingdom",
    "IL": "Israel",
    "IN": "India",
    "IT": "Italy",
    "JP": "Japan",
    "KR": "South Korea",
    "SG": "Singapore",
    "US": "United States",
}

PALETTE = {
    "neutral_dark": "#303030",
    "neutral_mid": "#767676",
    "neutral_light": "#D8D8D8",
    "blue": "#2F6DA3",
    "teal": "#3E8C87",
    "gold": "#C89932",
    "red": "#B54A4A",
}


def find_analysis_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (
            parent
            / "data"
            / "analysis_ready"
            / "compute_papers_with_contributions.csv"
        ).exists():
            return parent
    raise FileNotFoundError("Could not find analysis root containing GPU-only inputs.")


ROOT = find_analysis_root(Path(__file__).resolve())
COMPUTE_INPUT = ROOT / "data" / "analysis_ready" / "compute_papers_with_contributions.csv"
ORG_INPUT = ROOT / "data" / "analysis_ready" / "paper_organizations.csv"
BUNDLE = Path(os.environ.get("REPRO_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "reproduced"))
OUT_DATA = BUNDLE / "source_data"
OUT_FIG = BUNDLE / "figures"
OUT_REPORT = BUNDLE / "reports"


def ensure_dirs() -> None:
    for path in (OUT_DATA, OUT_FIG, OUT_REPORT):
        path.mkdir(parents=True, exist_ok=True)


def save_pub_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing {name} columns: {missing}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    compute = pd.read_csv(COMPUTE_INPUT)
    org_long = pd.read_csv(ORG_INPUT)
    return compute, org_long


def normalize_country_code(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    code = str(value).strip().upper()
    if code in {"", "NAN", "NONE", "NULL"}:
        return pd.NA
    return COUNTRY_CODE_ALIASES.get(code, code)


def country_display_name(code: object) -> str:
    text = str(code)
    return COUNTRY_DISPLAY_NAMES.get(text, text)


def add_year_top20_flags(papers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = (
        papers.groupby("year", as_index=False)["compute_tflops"]
        .quantile(TOP_COMPUTE_QUANTILE)
        .rename(columns={"compute_tflops": "year_p80_cutoff_tflops"})
    )
    out = papers.merge(thresholds, on="year", how="left", validate="many_to_one")
    out["is_year_top20_compute"] = (
        out["compute_tflops"] >= out["year_p80_cutoff_tflops"]
    ).astype(int)
    yearly = (
        out.groupby("year", as_index=False)
        .agg(
            quantifiable_papers=("paper_id", "nunique"),
            year_p80_cutoff_tflops=("year_p80_cutoff_tflops", "first"),
            top20_papers=("is_year_top20_compute", "sum"),
        )
        .sort_values("year")
    )
    yearly["top20_share"] = yearly["top20_papers"] / yearly["quantifiable_papers"]
    return out, yearly


def build_country_paper_panel(
    compute: pd.DataFrame, org_long: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    require_columns(
        compute,
        {"paper_id", "is_lb1_gfimp", "paper_year", "paper_venue", COMPUTE_COL},
        "compute",
    )
    require_columns(org_long, {"paper_id", "org_country_code"}, "organization long")

    papers = compute[
        ["paper_id", "is_lb1_gfimp", "paper_year", "paper_venue", COMPUTE_COL]
    ].copy()
    papers["year"] = pd.to_numeric(papers["paper_year"], errors="coerce")
    papers["compute_tflops"] = pd.to_numeric(papers[COMPUTE_COL], errors="coerce") / 1e12
    papers["is_lb1_gfimp"] = pd.to_numeric(papers["is_lb1_gfimp"], errors="coerce")
    papers = papers[
        papers["paper_id"].notna()
        & papers["year"].notna()
        & papers["compute_tflops"].gt(0)
        & papers["is_lb1_gfimp"].eq(1)
    ].copy()
    papers["year"] = papers["year"].round().astype(int)
    papers["paper_venue"] = papers["paper_venue"].astype("object")
    papers["log10_compute_tflops"] = np.log10(papers["compute_tflops"])
    papers, yearly_thresholds = add_year_top20_flags(papers)

    countries = org_long[["paper_id", "org_country_code"]].copy()
    countries["country_code"] = countries["org_country_code"].map(normalize_country_code)
    countries = countries[countries["paper_id"].notna() & countries["country_code"].notna()]
    countries = countries[["paper_id", "country_code"]].drop_duplicates()

    panel = countries.merge(
        papers[
            [
                "paper_id",
                "year",
                "paper_venue",
                "compute_tflops",
                "log10_compute_tflops",
                "year_p80_cutoff_tflops",
                "is_year_top20_compute",
            ]
        ],
        on="paper_id",
        how="inner",
        validate="many_to_one",
    )
    panel["country_count_per_paper"] = panel.groupby("paper_id")["country_code"].transform("nunique")
    panel = panel.sort_values(["year", "country_code", "paper_id"]).reset_index(drop=True)

    audit = {
        "compute_input_papers": int(compute["paper_id"].nunique()),
        "quantifiable_papers": int(papers["paper_id"].nunique()),
        "org_long_rows": int(len(org_long)),
        "unique_paper_country_pairs": int(len(countries)),
        "country_paper_rows": int(len(panel)),
        "country_paper_papers": int(panel["paper_id"].nunique()),
        "countries_observed": int(panel["country_code"].nunique()),
        "years_observed": int(panel["year"].nunique()),
        "overall_full_count_top20_share": float(panel["is_year_top20_compute"].mean()),
    }
    audit["yearly_thresholds"] = yearly_thresholds.to_dict(orient="records")
    return panel, audit


def summarize_countries(panel: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        panel,
        {
            "country_code",
            "paper_id",
            "compute_tflops",
            "log10_compute_tflops",
            "is_year_top20_compute",
        },
        "country-paper panel",
    )
    summary = (
        panel.groupby("country_code", as_index=False)
        .agg(
            full_count_n=("paper_id", "nunique"),
            full_count_top20_n=("is_year_top20_compute", "sum"),
            median_compute_tflops=("compute_tflops", "median"),
            p80_compute_tflops=("compute_tflops", lambda x: x.quantile(0.80)),
            median_log10_compute=("log10_compute_tflops", "median"),
            p80_log10_compute=("log10_compute_tflops", lambda x: x.quantile(0.80)),
        )
        .sort_values(["full_count_n", "full_count_top20_n"], ascending=False)
        .reset_index(drop=True)
    )
    summary["top20_compute_share"] = summary["full_count_top20_n"] / summary["full_count_n"]
    summary["top20_compute_intensity"] = summary["top20_compute_share"] / TOP_COMPUTE_BASELINE
    summary["full_count_n_rank"] = summary["full_count_n"].rank(method="min", ascending=False).astype(int)
    summary["top20_compute_share_rank"] = (
        summary["top20_compute_share"].rank(method="min", ascending=False).astype(int)
    )
    return summary


def filter_countries_for_main(summary: pd.DataFrame, threshold: int = MAIN_THRESHOLD) -> pd.DataFrame:
    return (
        summary.loc[summary["full_count_n"] >= threshold]
        .sort_values(["top20_compute_share", "full_count_n"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _point_sizes(values: pd.Series) -> np.ndarray:
    values = pd.to_numeric(values, errors="coerce").fillna(0)
    max_value = max(float(values.max()), 1.0)
    return 20 + 125 * np.sqrt(values / max_value)


def _point_size_for_value(value: float, max_value: float) -> float:
    return float(20 + 125 * np.sqrt(max(value, 0) / max(max_value, 1.0)))


def _label_countries(plot_df: pd.DataFrame, n_volume: int = 8, n_tail: int = 6) -> set[str]:
    volume = plot_df.nlargest(n_volume, "full_count_n")["country_code"]
    tail = plot_df.nlargest(n_tail, "top20_compute_share")["country_code"]
    return set(pd.concat([volume, tail]).drop_duplicates())


LABEL_OFFSETS = {
    "CN": (-25, 8),
    "US": (8, -4),
    "JP": (7, 7),
    "CA": (7, 7),
    "GB": (8, 5),
    "DE": (8, -9),
    "KR": (8, 8),
    "SG": (8, 5),
    "AE": (8, 6),
    "CH": (8, 8),
    "AU": (8, -8),
    "IN": (8, -8),
}


def annotate_country_labels(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    y_col: str,
    label_set: set[str],
    default_offset: tuple[int, int] = (6, 5),
) -> None:
    for _, row in plot_df.loc[plot_df["country_code"].isin(label_set)].iterrows():
        code = row["country_code"]
        ax.annotate(
            code,
            xy=(row["full_count_n"], row[y_col]),
            xytext=LABEL_OFFSETS.get(code, default_offset),
            textcoords="offset points",
            fontsize=6.2,
            color=PALETTE["neutral_dark"],
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
            zorder=4,
        )


def add_size_legend(ax: plt.Axes, max_top20: float) -> None:
    values = [25, 100, 400]
    values = [value for value in values if value <= max_top20]
    if len(values) < 2:
        values = [max(5, max_top20 / 3), max_top20]
    handles = [
        ax.scatter(
            [],
            [],
            s=_point_size_for_value(value, max_top20),
            facecolor=PALETTE["neutral_light"],
            edgecolor="white",
            linewidth=0.5,
        )
        for value in values
    ]
    ax.legend(
        handles,
        [f"{int(value)}" for value in values],
        title="Top-20% papers",
        loc="lower right",
        fontsize=5.8,
        title_fontsize=5.8,
        handletextpad=0.8,
        labelspacing=0.7,
        borderpad=0.2,
    )


def plot_volume_vs_top20_compute_share(summary: pd.DataFrame) -> None:
    plot_df = filter_countries_for_main(summary)
    fig, ax = plt.subplots(figsize=(3.55, 2.65))

    colors = np.where(
        plot_df["top20_compute_share"] >= TOP_COMPUTE_BASELINE,
        PALETTE["teal"],
        PALETTE["gold"],
    )
    ax.scatter(
        plot_df["full_count_n"],
        plot_df["top20_compute_share"],
        s=_point_sizes(plot_df["full_count_top20_n"]),
        c=colors,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.axhline(
        TOP_COMPUTE_BASELINE,
        color=PALETTE["neutral_mid"],
        linestyle="--",
        linewidth=0.8,
        zorder=1,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Papers (log)")
    ax.set_ylabel("Top-20% compute share")
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Country volume and top-20% compute concentration", loc="left", fontsize=7.6, pad=5)
    ax.grid(axis="both", color="#ECECEC", linewidth=0.45, alpha=0.9, zorder=0)

    label_set = _label_countries(plot_df)
    annotate_country_labels(ax, plot_df, "top20_compute_share", label_set)

    ymax = max(0.24, float(plot_df["top20_compute_share"].max()) * 1.18)
    ax.set_ylim(0, ymax)
    ax.margins(x=0.16)
    ax.text(
        0.02,
        TOP_COMPUTE_BASELINE - 0.006,
        "20% baseline",
        transform=ax.get_yaxis_transform(),
        color=PALETTE["neutral_mid"],
        fontsize=5.8,
        va="top",
    )
    add_size_legend(ax, float(plot_df["full_count_top20_n"].max()))
    save_pub_figure(fig, OUT_FIG / "country_compute_volume_vs_top20_share")
    plt.close(fig)


def plot_top20_compute_lollipop(summary: pd.DataFrame) -> None:
    plot_df = (
        filter_countries_for_main(summary)
        .nlargest(12, "top20_compute_share")
        .sort_values("top20_compute_share", ascending=True)
        .copy()
    )
    plot_df["country_label"] = plot_df["country_code"].map(country_display_name)
    height = max(2.55, 0.22 * len(plot_df) + 0.45)
    fig, ax = plt.subplots(figsize=(3.75, height))

    y = np.arange(len(plot_df))
    colors = np.where(
        plot_df["top20_compute_share"] >= TOP_COMPUTE_BASELINE,
        PALETTE["teal"],
        PALETTE["gold"],
    )
    ax.hlines(y, 0, plot_df["top20_compute_share"], color=PALETTE["neutral_light"], linewidth=0.95)
    ax.scatter(plot_df["top20_compute_share"], y, s=24, c=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(TOP_COMPUTE_BASELINE, color=PALETTE["neutral_mid"], linestyle="--", linewidth=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["country_label"])
    ax.set_xlabel("Share of Papers in Yearly Top-20% Compute (%)")
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color="#ECECEC", linewidth=0.45, alpha=0.9)
    ax.set_xlim(0, max(0.24, float(plot_df["top20_compute_share"].max()) * 1.16))
    baseline_transform = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        TOP_COMPUTE_BASELINE,
        1.015,
        "20% baseline",
        transform=baseline_transform,
        color=PALETTE["neutral_mid"],
        fontsize=5.8,
        va="bottom",
        ha="center",
        clip_on=False,
    )
    for yi, value in zip(y, plot_df["top20_compute_share"]):
        near_baseline = abs(value - TOP_COMPUTE_BASELINE) < 0.018
        label_x = value - 0.006 if near_baseline and value < TOP_COMPUTE_BASELINE else value + 0.003
        label_ha = "right" if near_baseline and value < TOP_COMPUTE_BASELINE else "left"
        ax.text(
            label_x,
            yi,
            f"{value * 100:.1f}",
            va="center",
            ha=label_ha,
            fontsize=5.5,
            color=PALETTE["neutral_mid"],
            bbox=dict(facecolor="white", edgecolor="none", pad=0.35),
            zorder=4,
        )

    save_pub_figure(fig, OUT_FIG / "country_top20_compute_share_lollipop")
    plt.close(fig)


def write_outputs(panel: pd.DataFrame, summary: pd.DataFrame, audit: dict[str, int | float]) -> None:
    ensure_dirs()
    main = filter_countries_for_main(summary, MAIN_THRESHOLD)
    appendix = filter_countries_for_main(summary, APPENDIX_THRESHOLD)
    yearly_thresholds = pd.DataFrame(audit["yearly_thresholds"])

    panel.to_csv(OUT_DATA / "country_paper_compute_full_count.csv", index=False)
    summary.to_csv(OUT_DATA / "country_compute_summary.csv", index=False)
    main.to_csv(OUT_DATA / "country_compute_summary_main_threshold50.csv", index=False)
    appendix.to_csv(OUT_DATA / "country_compute_summary_appendix_threshold30.csv", index=False)
    yearly_thresholds.to_csv(OUT_DATA / "yearly_top20_compute_thresholds.csv", index=False)
    main.to_csv(OUT_DATA / "source_data_country_compute_volume_scale.csv", index=False)
    main.sort_values("top20_compute_share", ascending=False).to_csv(
        OUT_DATA / "source_data_country_top20_compute_share.csv", index=False
    )
    (OUT_DATA / "country_compute_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plot_volume_vs_top20_compute_share(summary)
    plot_top20_compute_lollipop(summary)
    write_report(summary, audit)


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_report(summary: pd.DataFrame, audit: dict[str, int | float]) -> None:
    main = filter_countries_for_main(summary, MAIN_THRESHOLD)
    top_share = main.sort_values("top20_compute_share", ascending=False).head(8)
    top_volume = main.sort_values("full_count_n", ascending=False).head(8)

    share_lines = "\n".join(
        f"- {row.country_code}: {format_pct(row.top20_compute_share)} "
        f"({int(row.full_count_top20_n)}/{int(row.full_count_n)} full-count country-paper observations)"
        for row in top_share.itertuples()
    )
    volume_lines = "\n".join(
        f"- {row.country_code}: {int(row.full_count_n)} full-count quantifiable papers"
        for row in top_volume.itertuples()
    )

    report = f"""# Country-Level Compute Association

Core framing: this analysis describes compute associated with papers involving organizations from a country. It does not estimate national compute ownership or country-level compute capacity.

## Method

Input data: GPU-only paper-level compute and the GPU-only paper-organization long table. Quantifiable papers require positive `{COMPUTE_COL}`, `is_lb1_gfimp == 1`, and a valid publication year. The compute measure is paper-level maximum GPU-row normalized compute capacity in TFLOP/s.

Country assignment uses full counting. Hong Kong (HK) and Taiwan (TW) organization country codes are folded into China (CN) before de-duplication. For each paper, duplicate organization rows from the same country are collapsed to one paper-country observation. If a paper involves organizations from multiple countries, it contributes one full observation to each associated country. Therefore, the sum of country counts can exceed the number of papers.

The top-20% compute flag is computed within each publication year using the inclusive P80 cutoff. Ties at the cutoff can make the observed top-20% share differ from exactly 20%.

## Audit

- Quantifiable papers: {audit["quantifiable_papers"]:,}
- Full-count paper-country observations: {audit["country_paper_rows"]:,}
- Countries observed: {audit["countries_observed"]:,}
- Main-text country threshold: full_count_n >= {MAIN_THRESHOLD}
- Countries meeting main threshold: {len(main):,}

## Main-Threshold Countries With Highest Top-20% Compute Share

{share_lines}

## Largest Main-Threshold Country-Associated Paper Volumes

{volume_lines}

## Figure Captions

**Country compute volume and top-20% compute concentration.** Each point represents a country associated with at least {MAIN_THRESHOLD} quantifiable papers. Hong Kong and Taiwan are included under China. Multi-country papers are counted once for each associated country, after collapsing duplicate countries within the same paper. The x-axis shows full-count paper volume, the y-axis shows the share of papers in the within-year top 20% compute group, and point size is proportional to the full-count number of top-20% papers. The dashed line marks the 0.20 nominal top-20% baseline.

**Country-level concentration of top-20% compute papers.** The lollipop plot displays the 12 countries with the highest top-20% compute share among countries meeting the main sample threshold. The vertical reference line at 0.20 corresponds to the nominal expected share under no country-level concentration. Countries above this line are overrepresented among within-year top-20% compute papers among their country-associated quantifiable papers.
"""
    (OUT_REPORT / "country_compute_association.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    compute, org_long = load_inputs()
    panel, audit = build_country_paper_panel(compute, org_long)
    summary = summarize_countries(panel)
    write_outputs(panel, summary, audit)
    print(
        "Country compute association outputs written: "
        f"{len(summary)} countries, {audit['country_paper_rows']} full-count paper-country rows."
    )


if __name__ == "__main__":
    main()

