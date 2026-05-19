from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def find_analysis_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (
            parent / "data" / "compute_paper_level_gpu_only.xlsx"
        ).exists() and (
            parent
            / "data"
            / "acl_arr_topics_all_acl_metadata_desirouter_complete_gpu_only.xlsx"
        ).exists():
            return parent
    raise FileNotFoundError("Could not find analysis root containing GPU-only inputs.")


ROOT = find_analysis_root(Path(__file__).resolve())
COMPUTE_INPUT = ROOT / "data" / "compute_paper_level_gpu_only.xlsx"
TOPIC_INPUT = (
    ROOT
    / "data"
    / "acl_arr_topics_all_acl_metadata_desirouter_complete_gpu_only.xlsx"
)
BUNDLE = Path(__file__).resolve().parents[1]
OUT_DATA = BUNDLE / "data"
OUT_FIG = BUNDLE / "fig"
OUT_REPORT = BUNDLE / "report"

FIG_BASENAME = "nlp_topic_compute_scale"
APPENDIX_FIG_BASENAME = "nlp_topic_compute_appendix_full_topics_p99_trimmed"
COMPACT_RANKED_FIG_BASENAME = "nlp_topic_compute_emnlp_p99_trimmed"
COMPUTE_COL = "paper_max_row_compute_capability_gfimp_lb1"
COMPUTE_LABEL = "maximum GPU-row compute capacity"
TRIM_QUANTILE = 0.99
COMPACT_TAIL_TOPICS = 6


PALETTE = {
    "ink": "#222222",
    "muted": "#6F6F6F",
    "grid": "#E7E7E7",
    "blue": "#305F9F",
    "blue_light": "#AFC4DE",
    "teal": "#3B8F8C",
    "amber": "#D99035",
    "red": "#C85250",
}


SHORT_LABELS = {
    "Clinical and Biomedical Applications": "Clinical/biomedical",
    "Computational Social Science and Cultural Analytics": "Computational social sci.",
    "Dialogue and Interactive Systems": "Dialogue systems",
    "Discourse and Pragmatics": "Discourse/pragmatics",
    "Efficient Methods for NLP": "Efficient methods",
    "Ethics, Bias, and Fairness": "Ethics/bias/fairness",
    "Generation": "Generation",
    "Human-Centered NLP and Human-AI Interaction": "Human-centered NLP",
    "Information Extraction": "Information extraction",
    "Information Retrieval and Text Mining": "IR/text mining",
    "Interpretability and Analysis of Models for NLP": "Interpretability",
    "LLM agents": "LLM agents",
    "Language Modeling": "Language modeling",
    "Linguistic Theories, Cognitive Modeling, and Psycholinguistics": "Linguistic theory",
    "Machine Learning for NLP": "ML for NLP",
    "Machine Translation": "Machine translation",
    "Multilingualism and Cross-Lingual NLP": "Multilingual NLP",
    "Multimodality and Language Grounding to Vision, Robotics and Beyond": "Multimodality",
    "NLP Applications": "NLP applications",
    "NLP and Code Models": "Code models",
    "NLP and Symbolic Reasoning": "Symbolic reasoning",
    "Phonology, Morphology, and Word Segmentation": "Phonology/morphology",
    "Question Answering": "Question answering",
    "Resources and Evaluation": "Resources/evaluation",
    "Semantics: Lexical and Sentence-Level": "Semantics",
    "Sentiment Analysis, Stylistic Analysis, and Argument Mining": "Sentiment/argument",
    "Speech Recognition, Text-to-Speech and Spoken Language Understanding": "Speech/spoken language",
    "Summarization": "Summarization",
    "Syntax: Tagging, Chunking and Parsing": "Syntax/parsing",
}


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


def format_tflops(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


def topic_labels_with_n(df: pd.DataFrame) -> list[str]:
    return [f"{row.topic_short} (N={int(row.papers)})" for row in df.itertuples(index=False)]


def load_topic_compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    compute = pd.read_excel(COMPUTE_INPUT)
    topics = pd.read_excel(TOPIC_INPUT, sheet_name="topics")

    compute_required = {"paper_id", COMPUTE_COL, "is_lb1_gfimp"}
    topic_required = {"paper_id", "title", "topic", "confidence", "needs_review"}
    missing_compute = sorted(compute_required - set(compute.columns))
    missing_topics = sorted(topic_required - set(topics.columns))
    if missing_compute:
        raise ValueError(f"Missing compute columns: {missing_compute}")
    if missing_topics:
        raise ValueError(f"Missing topic columns: {missing_topics}")

    compute = compute.copy()
    compute["year"] = compute["paper_id"].map(parse_year)
    compute["venue"] = compute["paper_id"].map(parse_venue)
    compute["compute_tflops"] = pd.to_numeric(compute[COMPUTE_COL], errors="coerce") / 1e12
    compute = compute[
        compute["compute_tflops"].gt(0) & compute["is_lb1_gfimp"].eq(1)
    ].copy()

    merged = topics.merge(
        compute[
            [
                "paper_id",
                "year",
                "venue",
                "compute_tflops",
            ]
        ],
        on="paper_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged[merged["topic"].notna()].copy()
    merged["topic_short"] = merged["topic"].map(SHORT_LABELS).fillna(merged["topic"])
    merged, year_top10 = recompute_year_context(merged)
    return merged, year_top10


def recompute_year_context(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    drop_cols = [
        "year_median_tflops",
        "year_normalized_compute_ratio",
        "log10_compute_tflops",
        "year_centered_log10_compute",
        "year_p90_cutoff_tflops",
        "is_year_top10_compute",
    ]
    out = out.drop(columns=[col for col in drop_cols if col in out.columns])
    out["year_median_tflops"] = out.groupby("year")["compute_tflops"].transform("median")
    out["year_normalized_compute_ratio"] = out["compute_tflops"] / out["year_median_tflops"]
    out["log10_compute_tflops"] = np.log10(out["compute_tflops"])
    out["year_centered_log10_compute"] = out["log10_compute_tflops"] - out.groupby(
        "year"
    )["log10_compute_tflops"].transform("median")

    thresholds = (
        out.groupby("year", as_index=False)["compute_tflops"]
        .quantile(0.90)
        .rename(columns={"compute_tflops": "year_p90_cutoff_tflops"})
    )
    out = out.merge(thresholds, on="year", how="left")
    out["is_year_top10_compute"] = (
        out["compute_tflops"] >= out["year_p90_cutoff_tflops"]
    )
    year_top10 = (
        out.groupby("year", as_index=False)
        .agg(
            papers=("paper_id", "nunique"),
            year_p90_cutoff_tflops=("year_p90_cutoff_tflops", "first"),
            top10_papers=("is_year_top10_compute", "sum"),
        )
        .sort_values("year")
    )
    year_top10["top10_share_pct"] = (
        year_top10["top10_papers"] / year_top10["papers"] * 100
    )
    return out, year_top10


def trim_extreme_compute(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    cutoff = float(df["compute_tflops"].quantile(TRIM_QUANTILE))
    trimmed = df[df["compute_tflops"].le(cutoff)].copy()
    trimmed, year_top10 = recompute_year_context(trimmed)
    trim_info = {
        "trim_quantile": TRIM_QUANTILE,
        "trim_cutoff_tflops": cutoff,
        "removed_papers": int(df["paper_id"].nunique() - trimmed["paper_id"].nunique()),
        "retained_papers": int(trimmed["paper_id"].nunique()),
        "original_papers": int(df["paper_id"].nunique()),
    }
    return trimmed, year_top10, trim_info


def summarize_topics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    topic_summary = (
        df.groupby(["topic", "topic_short"], as_index=False)
        .agg(
            papers=("paper_id", "nunique"),
            years_covered=("year", "nunique"),
            median_tflops=("compute_tflops", "median"),
            q25_tflops=("compute_tflops", lambda x: x.quantile(0.25)),
            q75_tflops=("compute_tflops", lambda x: x.quantile(0.75)),
            p90_tflops=("compute_tflops", lambda x: x.quantile(0.90)),
            p95_tflops=("compute_tflops", lambda x: x.quantile(0.95)),
            max_tflops=("compute_tflops", "max"),
            mean_tflops=("compute_tflops", "mean"),
            median_year_normalized_ratio=("year_normalized_compute_ratio", "median"),
            p90_year_normalized_ratio=("year_normalized_compute_ratio", lambda x: x.quantile(0.90)),
            top10_papers=("is_year_top10_compute", "sum"),
            mean_topic_confidence=("confidence", "mean"),
            review_flagged_papers=("needs_review", "sum"),
        )
        .sort_values(["median_tflops", "papers"], ascending=[False, False])
        .reset_index(drop=True)
    )
    topic_summary["iqr_tflops"] = topic_summary["q75_tflops"] - topic_summary["q25_tflops"]
    topic_summary["top10_share_pct"] = (
        topic_summary["top10_papers"] / topic_summary["papers"] * 100
    )
    topic_summary["median_rank"] = (
        topic_summary["median_tflops"].rank(method="min", ascending=False).astype(int)
    )
    topic_summary["top10_share_rank"] = (
        topic_summary["top10_share_pct"].rank(method="min", ascending=False).astype(int)
    )

    topic_year = (
        df.groupby(["topic", "topic_short", "year"], as_index=False)
        .agg(
            papers=("paper_id", "nunique"),
            median_tflops=("compute_tflops", "median"),
            q25_tflops=("compute_tflops", lambda x: x.quantile(0.25)),
            q75_tflops=("compute_tflops", lambda x: x.quantile(0.75)),
            p90_tflops=("compute_tflops", lambda x: x.quantile(0.90)),
            top10_papers=("is_year_top10_compute", "sum"),
        )
        .sort_values(["topic", "year"])
    )
    topic_year["iqr_tflops"] = topic_year["q75_tflops"] - topic_year["q25_tflops"]
    topic_year["top10_share_pct"] = topic_year["top10_papers"] / topic_year["papers"] * 100
    return topic_summary, topic_year


def compute_tests(df: pd.DataFrame) -> dict[str, float | int]:
    groups_raw = [
        sub["log10_compute_tflops"].dropna().to_numpy()
        for _, sub in df.groupby("topic")
        if len(sub) >= 2
    ]
    groups_centered = [
        sub["year_centered_log10_compute"].dropna().to_numpy()
        for _, sub in df.groupby("topic")
        if len(sub) >= 2
    ]
    raw_kw = stats.kruskal(*groups_raw)
    centered_kw = stats.kruskal(*groups_centered)

    k = len(groups_raw)
    n = int(sum(len(group) for group in groups_raw))
    raw_epsilon_sq = max(0.0, (float(raw_kw.statistic) - k + 1) / (n - k))
    centered_epsilon_sq = max(0.0, (float(centered_kw.statistic) - k + 1) / (n - k))

    contingency = pd.crosstab(df["topic"], df["is_year_top10_compute"])
    chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)
    cramers_v = float(np.sqrt(chi2 / (len(df) * (min(contingency.shape) - 1))))

    return {
        "n_papers": int(df["paper_id"].nunique()),
        "n_topics": int(df["topic"].nunique()),
        "n_years": int(df["year"].nunique()),
        "kruskal_log10_compute_h": float(raw_kw.statistic),
        "kruskal_log10_compute_p": float(raw_kw.pvalue),
        "kruskal_log10_compute_epsilon_sq": raw_epsilon_sq,
        "kruskal_year_centered_log10_compute_h": float(centered_kw.statistic),
        "kruskal_year_centered_log10_compute_p": float(centered_kw.pvalue),
        "kruskal_year_centered_log10_compute_epsilon_sq": centered_epsilon_sq,
        "top10_chi2": float(chi2),
        "top10_chi2_p": float(chi2_p),
        "top10_cramers_v": cramers_v,
        "overall_median_tflops": float(df["compute_tflops"].median()),
        "overall_p90_tflops": float(df["compute_tflops"].quantile(0.90)),
        "overall_top10_share_pct": float(df["is_year_top10_compute"].mean() * 100),
    }


def save_figure(fig: plt.Figure, basename: str) -> None:
    target = OUT_FIG / f"{basename}.png"
    fig.savefig(target, bbox_inches="tight", dpi=600)


def plot_topic_summary(
    topic_summary: pd.DataFrame,
    tests: dict[str, float | int],
    basename: str = FIG_BASENAME,
    title: str | None = "NLP topic differences in GPU compute scale, 2020-2025",
    note: str | None = None,
) -> None:
    plot_df = topic_summary.sort_values(
        ["median_tflops", "papers"], ascending=[True, True]
    ).reset_index(drop=True)
    y = np.arange(len(plot_df))
    baseline = float(tests["overall_top10_share_pct"])
    overall_median = float(tests["overall_median_tflops"])

    has_title = bool(title)
    has_note = note is not None
    fig = plt.figure(figsize=(7.4, 8.45), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.28, 0.72],
        left=0.34,
        right=0.97,
        top=0.94 if not has_title else 0.89,
        bottom=0.075 if not has_note else 0.10,
        wspace=0.12,
    )
    ax_dist = fig.add_subplot(gs[0, 0])
    ax_share = fig.add_subplot(gs[0, 1], sharey=ax_dist)

    ax_dist.hlines(
        y=y,
        xmin=plot_df["q25_tflops"],
        xmax=plot_df["q75_tflops"],
        color=PALETTE["blue_light"],
        linewidth=3.0,
        label="IQR",
        zorder=1,
    )
    ax_dist.scatter(
        plot_df["median_tflops"],
        y,
        s=20,
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.35,
        label="Median",
        zorder=3,
    )
    ax_dist.scatter(
        plot_df["p90_tflops"],
        y,
        s=18,
        marker="^",
        color=PALETTE["red"],
        edgecolor="white",
        linewidth=0.35,
        label="P90",
        zorder=4,
    )

    xmin = max(0.8, plot_df["q25_tflops"].min() / 2.2)
    xmax = plot_df["p90_tflops"].max() * 1.9
    ax_dist.set_xscale("log")
    ax_dist.set_xlim(xmin, xmax)
    ax_dist.set_yticks(y)
    ax_dist.set_yticklabels(topic_labels_with_n(plot_df), fontsize=5.6)
    ax_dist.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax_dist.set_xlabel("Reported peak GPU configuration capacity (TFLOP/s, log10 scale)")
    ax_dist.set_title("a  Topic compute distribution", loc="left", fontsize=8, pad=6)
    ax_dist.axvline(
        overall_median,
        color=PALETTE["ink"],
        linestyle=(0, (2, 2)),
        linewidth=0.75,
        zorder=0,
    )
    ax_dist.text(
        overall_median * 1.04,
        len(plot_df) - 0.25,
        "overall median",
        fontsize=5.8,
        color=PALETTE["muted"],
        va="top",
    )
    ax_dist.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.025),
        ncol=3,
        columnspacing=0.9,
        handletextpad=0.35,
        fontsize=6.3,
    )

    bar_colors = np.where(
        plot_df["top10_share_pct"] >= baseline, PALETTE["teal"], PALETTE["amber"]
    )
    ax_share.barh(
        y,
        plot_df["top10_share_pct"],
        height=0.58,
        color=bar_colors,
        alpha=0.88,
    )
    ax_share.axvline(
        baseline,
        color=PALETTE["ink"],
        linestyle=(0, (2, 2)),
        linewidth=0.8,
    )
    ax_share.text(
        baseline + 0.3,
        len(plot_df) - 0.35,
        f"overall {baseline:.1f}%",
        fontsize=6.1,
        color=PALETTE["ink"],
        va="top",
    )
    ax_share.set_xlim(0, max(25, plot_df["top10_share_pct"].max() * 1.22))
    ax_share.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax_share.tick_params(axis="y", left=False, labelleft=False)
    ax_share.set_xlabel("Share of papers at/above yearly P90 (%)")
    ax_share.set_title("b  Within-year upper tail", loc="left", fontsize=8, pad=6)

    if has_title:
        fig.suptitle(
            title,
            x=0.34,
            y=0.975,
            ha="left",
            fontsize=9,
            fontweight="bold",
        )
    if has_note:
        fig.text(
            0.34,
            0.036,
            note,
            ha="left",
            va="bottom",
            fontsize=6.2,
            color=PALETTE["muted"],
        )

    save_figure(fig, basename)
    plt.close(fig)


def select_ranked_topics(topic_summary: pd.DataFrame) -> pd.DataFrame:
    high = topic_summary.nlargest(COMPACT_TAIL_TOPICS, "median_tflops")
    low = topic_summary.nsmallest(COMPACT_TAIL_TOPICS, "median_tflops")
    selected = pd.concat([high, low], ignore_index=True).drop_duplicates("topic")
    return selected.sort_values(["median_tflops", "papers"], ascending=[True, True])


def plot_compact_ranked_panel(
    topic_summary: pd.DataFrame,
    tests: dict[str, float | int],
    trim_info: dict[str, float | int],
) -> None:
    plot_df = select_ranked_topics(topic_summary).reset_index(drop=True)
    y = np.arange(len(plot_df))
    baseline = float(tests["overall_top10_share_pct"])
    overall_median = float(tests["overall_median_tflops"])

    fig = plt.figure(figsize=(7.0, 2.9), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.30, 0.70],
        left=0.23,
        right=0.98,
        top=0.84,
        bottom=0.18,
        wspace=0.13,
    )
    ax_dist = fig.add_subplot(gs[0, 0])
    ax_share = fig.add_subplot(gs[0, 1], sharey=ax_dist)

    ax_dist.hlines(
        y=y,
        xmin=plot_df["q25_tflops"],
        xmax=plot_df["q75_tflops"],
        color=PALETTE["blue_light"],
        linewidth=2.5,
        zorder=1,
        label="IQR",
    )
    ax_dist.scatter(
        plot_df["median_tflops"],
        y,
        s=19,
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.35,
        label="Median",
        zorder=3,
    )
    ax_dist.scatter(
        plot_df["p90_tflops"],
        y,
        s=18,
        marker="^",
        color=PALETTE["red"],
        edgecolor="white",
        linewidth=0.35,
        label="P90",
        zorder=4,
    )
    ax_dist.set_xscale("log")
    ax_dist.set_xlim(max(0.8, plot_df["q25_tflops"].min() / 2.0), plot_df["p90_tflops"].max() * 1.55)
    ax_dist.set_yticks(y)
    ax_dist.set_yticklabels(topic_labels_with_n(plot_df), fontsize=5.3)
    ax_dist.grid(axis="x", color=PALETTE["grid"], linewidth=0.45)
    ax_dist.set_xlabel(
        "Reported peak GPU configuration capacity (TFLOP/s, log10 scale)",
        fontsize=6.2,
    )
    ax_dist.set_title("a  Median, IQR and P90", loc="left", fontsize=6.8, pad=3)
    ax_dist.tick_params(axis="x", labelsize=5.8)
    ax_dist.axvline(
        overall_median,
        color=PALETTE["ink"],
        linestyle=(0, (2, 2)),
        linewidth=0.7,
        zorder=0,
    )
    ax_dist.text(
        overall_median * 1.04,
        len(plot_df) - 0.10,
        "overall median",
        fontsize=5.2,
        color=PALETTE["muted"],
        va="top",
    )
    ax_dist.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.025),
        ncol=3,
        columnspacing=0.8,
        handletextpad=0.35,
        fontsize=5.6,
    )

    bar_colors = np.where(
        plot_df["top10_share_pct"] >= baseline, PALETTE["teal"], PALETTE["amber"]
    )
    ax_share.barh(y, plot_df["top10_share_pct"], height=0.55, color=bar_colors, alpha=0.9)
    ax_share.axvline(baseline, color=PALETTE["ink"], linestyle=(0, (2, 2)), linewidth=0.75)
    ax_share.text(
        baseline + 0.35,
        len(plot_df) - 0.15,
        f"overall {baseline:.1f}%",
        fontsize=5.8,
        color=PALETTE["ink"],
        va="top",
    )
    ax_share.set_xlim(0, max(26, plot_df["top10_share_pct"].max() * 1.20))
    ax_share.grid(axis="x", color=PALETTE["grid"], linewidth=0.45)
    ax_share.tick_params(axis="y", left=False, labelleft=False)
    ax_share.tick_params(axis="x", labelsize=5.8)
    ax_share.set_xlabel("Share at/above yearly P90 (%)", fontsize=6.2)
    ax_share.set_title("b  Upper-tail enrichment", loc="left", fontsize=6.8, pad=3)
    save_figure(fig, COMPACT_RANKED_FIG_BASENAME)
    plt.close(fig)


def write_outputs(
    df: pd.DataFrame,
    topic_summary: pd.DataFrame,
    topic_year: pd.DataFrame,
    year_top10: pd.DataFrame,
    tests: dict[str, float | int],
    suffix: str = "",
) -> None:
    topic_summary.to_csv(OUT_DATA / f"nlp_topic_compute_summary{suffix}.csv", index=False)
    topic_year.to_csv(OUT_DATA / f"nlp_topic_compute_by_year{suffix}.csv", index=False)
    year_top10.to_csv(OUT_DATA / f"yearly_top10_compute_thresholds{suffix}.csv", index=False)
    df[
        [
            "paper_id",
            "title",
            "year",
            "venue",
            "topic",
            "topic_short",
            "compute_tflops",
            "year_median_tflops",
            "year_normalized_compute_ratio",
            "year_p90_cutoff_tflops",
            "is_year_top10_compute",
            "confidence",
            "needs_review",
        ]
    ].sort_values(["year", "topic", "compute_tflops"], ascending=[True, True, False]).to_csv(
        OUT_DATA / f"paper_topic_compute_with_top10_flag{suffix}.csv", index=False
    )
    topic_summary.to_csv(
        OUT_DATA / f"source_data_nlp_topic_compute_scale{suffix}.csv", index=False
    )
    topic_year.assign(
        log10_median_tflops=lambda data: np.log10(data["median_tflops"])
    ).rename(columns={"topic": "primary_topic"}).to_csv(
        OUT_DATA / f"source_data_nlp_topic_year_median_capacity{suffix}.csv",
        index=False,
    )

    with pd.ExcelWriter(
        OUT_DATA / f"nlp_topic_compute_summary{suffix}.xlsx", engine="openpyxl"
    ) as writer:
        topic_summary.to_excel(writer, sheet_name="topic_summary", index=False)
        topic_year.to_excel(writer, sheet_name="topic_year", index=False)
        year_top10.to_excel(writer, sheet_name="year_top10_thresholds", index=False)
        pd.DataFrame([tests]).to_excel(writer, sheet_name="tests", index=False)

    (OUT_DATA / f"nlp_topic_compute_tests{suffix}.json").write_text(
        json.dumps(tests, indent=2), encoding="utf-8"
    )


def write_report(
    topic_summary: pd.DataFrame,
    year_top10: pd.DataFrame,
    tests: dict[str, float | int],
    report_name: str = "nlp_topic_compute_scale.md",
    figure_basename: str = FIG_BASENAME,
    suffix: str = "",
    trim_info: dict[str, float | int] | None = None,
) -> None:
    top_median = topic_summary.sort_values("median_tflops", ascending=False).head(6)
    top_p90 = topic_summary.sort_values("p90_tflops", ascending=False).head(6)
    top_tail = topic_summary.sort_values("top10_share_pct", ascending=False).head(6)
    low_tail = topic_summary.sort_values("top10_share_pct", ascending=True).head(4)

    def bullet_rows(df: pd.DataFrame, metric: str) -> str:
        rows = []
        for row in df.itertuples(index=False):
            if metric == "median":
                rows.append(
                    f"- {row.topic}: median {format_tflops(row.median_tflops)} TFLOP/s, "
                    f"IQR {format_tflops(row.q25_tflops)}-{format_tflops(row.q75_tflops)}, "
                    f"P90 {format_tflops(row.p90_tflops)} (n={int(row.papers)})."
                )
            elif metric == "p90":
                rows.append(
                    f"- {row.topic}: P90 {format_tflops(row.p90_tflops)} TFLOP/s, "
                    f"median {format_tflops(row.median_tflops)} (n={int(row.papers)})."
                )
            else:
                rows.append(
                    f"- {row.topic}: {row.top10_share_pct:.1f}% in yearly P90-or-above compute "
                    f"({int(row.top10_papers)}/{int(row.papers)} papers)."
                )
        return "\n".join(rows)

    years = ", ".join(
        f"{int(row.year)}: cutoff {format_tflops(row.year_p90_cutoff_tflops)} TFLOP/s"
        for row in year_top10.itertuples(index=False)
    )
    trim_sentence = ""
    if trim_info is not None:
        trim_sentence = (
            f" Extreme values are removed using a global P99 cutoff: papers above "
            f"{format_tflops(float(trim_info['trim_cutoff_tflops']))} TFLOP/s are excluded "
            f"({int(trim_info['removed_papers'])} of {int(trim_info['original_papers'])} papers removed)."
        )

    report = f"""# RQ2 NLP Topic Compute Scale

## Figure contract

Core conclusion: NLP topics differ in reported maximum GPU-row compute scale, both in their central tendency and in how often their papers enter the within-year high-compute tail.

Evidence chain: panel a reports each topic's median, IQR, and P90 paper-level maximum GPU-row compute capacity; panel b reports the share of each topic's papers that exceed the same-year P90 max-row compute cutoff. Archetype: quantitative grid. Backend: Python/matplotlib only. Export contract: PNG figures with source-data tables.

## Methods

Input data: GPU-only paper-level compute and ACL topic metadata. The compute measure is `{COMPUTE_COL} / 1e12`, interpreted as the paper-level maximum GPU-row normalized compute capacity in TFLOP/s, rather than total paper-level summed compute.{trim_sentence} The upper-tail flag is computed within each publication year using that year's inclusive P90 cutoff across all retained valid GPU-only papers, then summarized by NLP topic. Because common GPU configurations can tie at the cutoff, the overall flagged share is {tests["overall_top10_share_pct"]:.1f}% rather than exactly 10%.

Yearly P90 cutoffs: {years}.

Statistical checks are descriptive-supporting, not causal. Across {int(tests["n_papers"]):,} papers and {int(tests["n_topics"])} topics, a Kruskal-Wallis test on log10 compute gives H={tests["kruskal_log10_compute_h"]:.1f}, p={tests["kruskal_log10_compute_p"]:.2e}, epsilon-squared={tests["kruskal_log10_compute_epsilon_sq"]:.3f}. After centering log10 compute by year, H={tests["kruskal_year_centered_log10_compute_h"]:.1f}, p={tests["kruskal_year_centered_log10_compute_p"]:.2e}, epsilon-squared={tests["kruskal_year_centered_log10_compute_epsilon_sq"]:.3f}. Topic and top-decile membership are associated by chi-square p={tests["top10_chi2_p"]:.2e}, Cramer's V={tests["top10_cramers_v"]:.3f}.

## Key results

Highest median compute topics:

{bullet_rows(top_median, "median")}

Largest upper tails by P90:

{bullet_rows(top_p90, "p90")}

Topics most enriched in yearly P90-or-above compute papers:

{bullet_rows(top_tail, "tail")}

Topics least represented in yearly P90-or-above compute papers:

{bullet_rows(low_tail, "tail")}

## Interpretation

The answer to the RQ is yes: NLP topic is associated with maximum GPU-row compute scale. The difference is clearest for topics tied to large models and multimodal systems, where both median max-row compute and yearly P90-or-above representation are higher. Smaller-resource and more analysis-oriented topics usually sit lower in the compute distribution, though raw medians should be read alongside the within-year upper-tail metric because topic composition changes across 2020-2025.

## Outputs

- `4.3/NLP topic/fig/{figure_basename}.png`
- `4.3/NLP topic/data/nlp_topic_compute_summary{suffix}.csv`
- `4.3/NLP topic/data/nlp_topic_compute_by_year{suffix}.csv`
- `4.3/NLP topic/data/yearly_top10_compute_thresholds{suffix}.csv`
- `4.3/NLP topic/data/paper_topic_compute_with_top10_flag{suffix}.csv`
- `4.3/NLP topic/data/nlp_topic_compute_summary{suffix}.xlsx`

## Review risks

Topic labels come from the existing topic-classification table and inherit its uncertainty. The compute metric captures the maximum reported or imputed GPU-row capacity per paper, not total training FLOPs or wall-clock compute. The P90-or-above share reduces year-composition confounding but does not control for venue, paper type, or model-family covariates.
"""
    (OUT_REPORT / report_name).write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df, _ = load_topic_compute()
    trimmed_df, trimmed_year_top10, trim_info = trim_extreme_compute(df)
    trimmed_summary, trimmed_topic_year = summarize_topics(trimmed_df)
    trimmed_tests = compute_tests(trimmed_df)
    trimmed_tests.update(trim_info)
    write_outputs(
        trimmed_df,
        trimmed_summary,
        trimmed_topic_year,
        trimmed_year_top10,
        trimmed_tests,
        suffix="_p99_trimmed",
    )
    plot_topic_summary(
        trimmed_summary,
        trimmed_tests,
        basename=APPENDIX_FIG_BASENAME,
        title=None,
        note=None,
    )
    plot_compact_ranked_panel(trimmed_summary, trimmed_tests, trim_info)
    write_report(
        trimmed_summary,
        trimmed_year_top10,
        trimmed_tests,
        report_name="nlp_topic_compute_scale_p99_trimmed.md",
        figure_basename=COMPACT_RANKED_FIG_BASENAME,
        suffix="_p99_trimmed",
        trim_info=trim_info,
    )
    print(
        f"Completed P99-trimmed NLP topic compute analysis: {len(trimmed_df):,} papers retained, "
        f"{int(trim_info['removed_papers'])} extreme papers removed."
    )
    print(f"EMNLP figure: {OUT_FIG / (COMPACT_RANKED_FIG_BASENAME + '.png')}")
    print(f"Report: {OUT_REPORT / 'nlp_topic_compute_scale_p99_trimmed.md'}")


if __name__ == "__main__":
    main()



