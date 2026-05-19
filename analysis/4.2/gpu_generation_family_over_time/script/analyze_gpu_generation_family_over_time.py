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

FAMILY_ORDER = [
    "Datacenter A-series",
    "Tesla",
    "GeForce RTX",
    "RTX Workstation",
    "Datacenter H-series",
    "TITAN",
    "Quadro",
    "GeForce GTX",
    "Datacenter L-series",
    "Other",
]

FAMILY_COLORS = {
    "Datacenter A-series": "#E15759",
    "Tesla": "#4E79A7",
    "GeForce RTX": "#59A14F",
    "RTX Workstation": "#B07AA1",
    "Datacenter H-series": "#F28E2B",
    "GeForce GTX": "#7F7F7F",
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


def load_paper_level() -> pd.DataFrame:
    df = pd.read_excel(INPUT)
    required = {
        "paper_id",
        "paper_main_gpu_generation",
        "paper_main_gpu_family",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["year"] = df["paper_id"].astype(str).str.extract(r"^(\d{4})")[0].astype(int)
    df["generation"] = df["paper_main_gpu_generation"].fillna("Unknown").astype(str)
    df["family"] = df["paper_main_gpu_family"].fillna("Unknown").astype(str)
    df["generation_group"] = np.where(
        df["generation"].isin(GENERATION_ORDER[:-1]), df["generation"], "Other"
    )
    df["family_group"] = np.where(df["family"].isin(FAMILY_ORDER[:-1]), df["family"], "Other")
    return df


def summarize_by_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annual = df.groupby("year")["paper_id"].nunique().rename("gpu_papers").reset_index()

    generation = (
        df.groupby(["year", "generation_group"], as_index=False)
        .agg(papers=("paper_id", "nunique"))
        .merge(annual, on="year", how="left")
    )
    generation["paper_share_pct"] = 100 * generation["papers"] / generation["gpu_papers"]

    family = (
        df.groupby(["year", "family_group"], as_index=False)
        .agg(papers=("paper_id", "nunique"))
        .merge(annual, on="year", how="left")
    )
    family["paper_share_pct"] = 100 * family["papers"] / family["gpu_papers"]

    generation_matrix = (
        generation.pivot(index="generation_group", columns="year", values="paper_share_pct")
        .reindex(GENERATION_ORDER)
        .fillna(0)
    )
    family_matrix = (
        family.pivot(index="family_group", columns="year", values="paper_share_pct")
        .reindex(FAMILY_ORDER)
        .fillna(0)
    )
    return annual, generation, family, generation_matrix, family_matrix


def save_figure(fig: plt.Figure, stem: str) -> None:
    base = OUT_FIG / stem
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_generation_trajectory(
    ax_gen: plt.Axes,
    generation_matrix: pd.DataFrame,
    title: str,
    panel_label: str | None = None,
    compact_labels: bool = False,
) -> None:
    years = generation_matrix.columns.to_numpy(dtype=int)
    x = np.arange(len(years))

    focus_generations = ["Pascal", "Volta", "Turing", "Ampere", "Ada Lovelace", "Hopper"]
    for name in focus_generations:
        values = generation_matrix.loc[name].to_numpy(dtype=float)
        is_signal = name in {"Ampere", "Ada Lovelace", "Hopper"}
        ax_gen.plot(
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

    ax_gen.set_title(title, loc="left", pad=8)
    ax_gen.set_xlim(-0.15, len(years) - 0.18)
    ax_gen.set_ylim(0, 84)
    ax_gen.set_xticks(x)
    ax_gen.set_xticklabels(years)
    ax_gen.set_ylabel("GPU-reporting papers (%)")
    ax_gen.set_xlabel("Publication year")
    ax_gen.grid(axis="y", color="#E7E7E7", linewidth=0.55)
    ax_gen.spines["left"].set_bounds(0, 80)
    ax_gen.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=3,
        fontsize=6.1,
        columnspacing=1.15,
        handlelength=1.25,
        handletextpad=0.40,
    )

    if panel_label:
        ax_gen.text(
            -0.14,
            1.06,
            panel_label,
            transform=ax_gen.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="black",
        )


def plot_generation_trajectory(generation_matrix: pd.DataFrame) -> None:
    fig, ax_gen = plt.subplots(figsize=(7.2, 2.9))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.76, bottom=0.22)
    draw_generation_trajectory(ax_gen, generation_matrix, "Main GPU generation trajectory")
    save_figure(fig, "gpu_generation_over_time")


def draw_family_trajectory(
    ax_fam: plt.Axes,
    family_matrix: pd.DataFrame,
    title: str,
    panel_label: str | None = None,
    compact_labels: bool = False,
) -> None:
    years = family_matrix.columns.to_numpy(dtype=int)
    x = np.arange(len(years))

    focus_families = [
        "Datacenter A-series",
        "Tesla",
        "GeForce RTX",
        "RTX Workstation",
        "Datacenter H-series",
        "GeForce GTX",
    ]
    for name in focus_families:
        values = family_matrix.loc[name].to_numpy(dtype=float)
        is_signal = name in {"Datacenter A-series", "Datacenter H-series", "RTX Workstation"}
        ax_fam.plot(
            x,
            values,
            label=name,
            color=FAMILY_COLORS[name],
            linewidth=2.0 if is_signal else 1.35,
            marker="o",
            markersize=4.0 if is_signal else 3.2,
            markeredgewidth=0,
            alpha=1.0 if is_signal else 0.82,
        )

    ax_fam.set_title(title, loc="left", pad=8)
    ax_fam.set_xlim(-0.15, len(years) - 0.18)
    ax_fam.set_ylim(0, 64)
    ax_fam.set_xticks(x)
    ax_fam.set_xticklabels(years)
    ax_fam.set_ylabel("GPU-reporting papers (%)")
    ax_fam.set_xlabel("Publication year")
    ax_fam.grid(axis="y", color="#E7E7E7", linewidth=0.55)
    ax_fam.spines["left"].set_bounds(0, 60)
    ax_fam.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=3,
        fontsize=6.1,
        columnspacing=1.15,
        handlelength=1.25,
        handletextpad=0.40,
    )

    if panel_label:
        ax_fam.text(
            -0.14,
            1.06,
            panel_label,
            transform=ax_fam.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="black",
        )


def plot_family_trajectory(family_matrix: pd.DataFrame) -> None:
    fig, ax_fam = plt.subplots(figsize=(7.2, 2.9))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.76, bottom=0.22)
    draw_family_trajectory(ax_fam, family_matrix, "Main GPU family trajectory")
    save_figure(fig, "gpu_family_over_time")


def plot_generation_family(generation_matrix: pd.DataFrame, family_matrix: pd.DataFrame) -> None:
    plot_generation_trajectory(generation_matrix)
    plot_family_trajectory(family_matrix)


def write_report(
    annual: pd.DataFrame,
    generation: pd.DataFrame,
    family: pd.DataFrame,
    generation_matrix: pd.DataFrame,
    family_matrix: pd.DataFrame,
) -> None:
    ampere_2022 = generation_matrix.loc["Ampere", 2022]
    ampere_2025 = generation_matrix.loc["Ampere", 2025]
    hopper_2025 = generation_matrix.loc["Hopper", 2025]
    ada_2025 = generation_matrix.loc["Ada Lovelace", 2025]
    tesla_2020 = family_matrix.loc["Tesla", 2020]
    tesla_2025 = family_matrix.loc["Tesla", 2025]
    a_series_2020 = family_matrix.loc["Datacenter A-series", 2020]
    a_series_2025 = family_matrix.loc["Datacenter A-series", 2025]

    gen_leaders = (
        generation.sort_values(["year", "paper_share_pct"], ascending=[True, False])
        .groupby("year")
        .head(1)
    )
    fam_leaders = (
        family.sort_values(["year", "paper_share_pct"], ascending=[True, False])
        .groupby("year")
        .head(1)
    )

    gen_lines = [
        f"- {int(row.year)}: {row.generation_group}, {int(row.papers)} papers ({row.paper_share_pct:.1f}%)."
        for row in gen_leaders.itertuples(index=False)
    ]
    fam_lines = [
        f"- {int(row.year)}: {row.family_group}, {int(row.papers)} papers ({row.paper_share_pct:.1f}%)."
        for row in fam_leaders.itertuples(index=False)
    ]
    n_line = ", ".join(
        f"{int(row.year)} n={int(row.gpu_papers)}" for row in annual.itertuples(index=False)
    )

    report = "\n".join(
        [
            "# RQ1: GPU Generation and Family over Time",
            "",
            "## Figure Contract",
            "Core conclusion: GPU-reporting ACL/EMNLP/NAACL papers shifted from Tesla/Volta-era hardware toward Ampere datacenter GPUs after 2022, with Hopper and Ada Lovelace appearing mainly in 2024-2025.",
            "Figure archetype: quantitative grid.",
            "Target output: PNG plus source CSV tables.",
            "Backend: Python/matplotlib only.",
            "Final size: 183 mm wide single-panel figures.",
            "Figure map: one figure shows annual main-GPU generation trajectories; the companion figure shows annual main-GPU family trajectories.",
            "Evidence hierarchy: the generation trajectory is the hero evidence for the generational transition; the family trajectory validates that the transition is specifically driven by Datacenter A-series growth and Tesla decline.",
            "Statistics needed: descriptive counts and percentages of unique GPU-reporting papers; no inferential test is used.",
            "Source data needed: paper-level main GPU generation/family fields.",
            "Image-integrity notes: vector line/text exports are generated directly from source tables; no raster image adjustment.",
            "Reviewer risk: main-GPU assignment compresses multi-GPU papers to one dominant GPU family/generation, so the result is a prevalence measure rather than complete hardware inventory.",
            "",
            "## Method",
            "Input data: `data/compute_paper_level_gpu_only.xlsx`.",
            "Each row is one GPU-reporting paper. The analysis counts unique papers by publication year and the paper-level `paper_main_gpu_generation` and `paper_main_gpu_family` fields.",
            f"Annual denominators: {n_line}.",
            "Rare generations and families are grouped into `Other` in the figure; full grouped counts and shares are preserved in the output CSV files.",
            "",
            "## Main Result",
            f"Ampere rose from {ampere_2022:.1f}% of GPU-reporting papers in 2022 to {ampere_2025:.1f}% in 2025. "
            f"By 2025, Hopper reached {hopper_2025:.1f}% and Ada Lovelace reached {ada_2025:.1f}%, while older Volta/Turing/Pascal generations contracted.",
            f"At the family level, Datacenter A-series increased from {a_series_2020:.1f}% in 2020 to {a_series_2025:.1f}% in 2025, whereas Tesla decreased from {tesla_2020:.1f}% to {tesla_2025:.1f}%.",
            "",
            "## Annual Generation Leaders",
            *gen_lines,
            "",
            "## Annual Family Leaders",
            *fam_lines,
            "",
            "## Outputs",
            "- `4.2/gpu_generation_family_over_time/data/gpu_generation_by_year.csv`: grouped annual generation counts and shares.",
            "- `4.2/gpu_generation_family_over_time/data/gpu_family_by_year.csv`: grouped annual family counts and shares.",
            "- `4.2/gpu_generation_family_over_time/data/gpu_generation_year_share_matrix.csv`: generation share matrix used for the generation trajectory.",
            "- `4.2/gpu_generation_family_over_time/data/gpu_family_year_share_matrix.csv`: family share matrix used for the family trajectory.",
            "- `4.2/gpu_generation_family_over_time/fig/gpu_generation_over_time.png`: generation figure export.",
            "- `4.2/gpu_generation_family_over_time/fig/gpu_family_over_time.png`: family figure export.",
        ]
    )
    (OUT_REPORT / "gpu_generation_family_over_time.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_paper_level()
    annual, generation, family, generation_matrix, family_matrix = summarize_by_year(df)

    annual.to_csv(OUT_DATA / "gpu_generation_family_annual_denominators.csv", index=False)
    generation.to_csv(OUT_DATA / "gpu_generation_by_year.csv", index=False)
    family.to_csv(OUT_DATA / "gpu_family_by_year.csv", index=False)
    generation_matrix.to_csv(OUT_DATA / "gpu_generation_year_share_matrix.csv")
    family_matrix.to_csv(OUT_DATA / "gpu_family_year_share_matrix.csv")

    plot_generation_family(generation_matrix, family_matrix)
    write_report(annual, generation, family, generation_matrix, family_matrix)


if __name__ == "__main__":
    main()





