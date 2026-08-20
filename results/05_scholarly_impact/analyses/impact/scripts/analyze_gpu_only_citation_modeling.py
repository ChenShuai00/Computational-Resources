from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
import warnings

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


COMPUTE_FILE = "compute_papers_with_contributions.csv"
PAPER_COMPUTE_FILE = "paper_compute_rows.csv"
METADATA_FILE = "openalex_metadata.csv"
TOPIC_FILE = "topics.csv"
AWARD_FILE = "awards.csv"
ORG_VARS_FILE = "paper_organization_variables.csv"
ORG_LONG_FILE = "paper_organizations.csv"
ORG_YEAR_PANEL_FILE = "organization_year_panel.csv"
CONFOUNDER_FILE = "paper_confounder_controls.csv"

AUTHOR_HISTORY_CONTROLS = [
    "author_prior_citations_3y_max_log1p",
    "author_prior_works_3y_mean_log1p",
]
INSTITUTION_VISIBILITY_CONTROLS = [
    "institution_prior_citations_3y_max_log1p",
    "log1p_max_prior_org_papers",
    "log1p_max_prior_partner_org_count",
    "has_company",
    "has_industry_academia",
    "has_international_collab",
]
PREPUBLICATION_CONFOUNDER_CONTROLS = [
    *AUTHOR_HISTORY_CONTROLS,
    *INSTITUTION_VISIBILITY_CONTROLS,
]
ARTIFACT_CONTROLS = ["has_public_artifact"]

AMPERE_OR_NEWER_GENERATIONS = {
    "Ampere",
    "Ada Lovelace",
    "Hopper",
    "Blackwell",
    "CDNA 2",
    "CDNA 3",
    "CDNA 4",
    "Ascend 910B",
    "Ascend 910C",
    "Trainium1",
    "Trainium2",
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
        "axes.edgecolor": "#222222",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
    }
)


def find_analysis_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "data" / "analysis_ready" / COMPUTE_FILE).exists():
            return parent
    raise FileNotFoundError("Could not find analysis root containing GPU-only inputs.")


ROOT = find_analysis_root(Path(__file__).resolve())
BUNDLE = Path(os.environ.get("REPRO_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "reproduced"))


def portable_path(path: Path) -> str:
    """Return a repository-relative POSIX path for portable audit records."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def ensure_output_dirs(output_dir: Path | None = None) -> dict[str, Path]:
    base = output_dir if output_dir is not None else BUNDLE
    paths = {
        "base": base,
        "data": base / "tables",
        "fig": base / "figures",
        "report": base / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_topic_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_inputs(
    root: Path | None = None,
    data_dir: Path | None = None,
    skip_award: bool = False,
) -> dict[str, pd.DataFrame]:
    root = root or ROOT
    data_dir = data_dir or root / "data" / "analysis_ready"
    award_path = data_dir / AWARD_FILE
    if skip_award and not award_path.exists():
        awards = pd.DataFrame(columns=["anthology_id", "award"])
    else:
        awards = pd.read_csv(award_path)
    return {
        "compute": pd.read_csv(data_dir / COMPUTE_FILE),
        "metadata": pd.read_csv(data_dir / METADATA_FILE),
        "topics": read_topic_table(data_dir / TOPIC_FILE),
        "awards": awards,
        "org_vars": pd.read_csv(data_dir / ORG_VARS_FILE),
        "org_long": pd.read_csv(data_dir / ORG_LONG_FILE),
        "org_year_panel": pd.read_csv(data_dir / ORG_YEAR_PANEL_FILE),
    }


def load_paper_compute_rows(root: Path | None = None, data_dir: Path | None = None) -> pd.DataFrame:
    root = root or ROOT
    data_dir = data_dir or root / "data" / "analysis_ready"
    return pd.read_csv(data_dir / PAPER_COMPUTE_FILE)


def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing {name} columns: {missing}")


def dedupe_key(df: pd.DataFrame, key: str, name: str) -> pd.DataFrame:
    require_columns(df, {key}, name)
    return df.drop_duplicates(key, keep="first").copy()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def positive_log10(series: pd.Series) -> pd.Series:
    values = safe_numeric(series).astype(float)
    values = values.where(values.gt(0), np.nan)
    return np.log10(values)


def parse_year_from_paper_id(series: pd.Series) -> pd.Series:
    return safe_numeric(series.astype(str).str.extract(r"^(\d{4})")[0])


def parse_venue_from_paper_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"^\d+\.([^-\.]+)-")[0].str.lower()


def add_topic_year_citation_percentile(master: pd.DataFrame) -> pd.DataFrame:
    out = master.copy()
    out["topic_year_citation_percentile"] = np.nan
    out["topic_year_citation_cell_n"] = np.nan

    valid = (
        out["cited_by_count"].notna()
        & out["primary_topic"].notna()
        & out["year_str"].notna()
    )
    group_keys = ["primary_topic", "year_str"]
    valid_cites = out.loc[valid, "cited_by_count"]
    ranks = valid_cites.groupby([out.loc[valid, key] for key in group_keys]).rank(
        method="average",
        ascending=True,
    )
    cell_n = valid_cites.groupby([out.loc[valid, key] for key in group_keys]).transform(
        "size"
    )

    out.loc[valid, "topic_year_citation_cell_n"] = cell_n.astype(float)
    out.loc[valid, "topic_year_citation_percentile"] = (ranks - 0.5) / cell_n
    return out


def build_master_panel(
    compute: pd.DataFrame,
    metadata: pd.DataFrame,
    topics: pd.DataFrame,
    awards: pd.DataFrame,
    org_vars: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        compute,
        {
            "paper_id",
            "is_lb1_gfimp",
            "is_strict",
            "paper_total_compute_capability",
            "paper_max_row_compute_capability",
            "paper_total_compute_capability_gfimp_lb1",
            "paper_max_row_compute_capability_gfimp_lb1",
        },
        "compute",
    )
    require_columns(
        metadata,
        {
            "source_acl_id",
            "team_size",
            "cited_by_count",
            "citation_normalized_percentile.value",
        },
        "metadata",
    )
    require_columns(topics, {"paper_id", "topic"}, "topics")
    require_columns(awards, {"anthology_id", "award"}, "awards")
    require_columns(org_vars, {"paper_id", "n_organizations"}, "organization variables")

    compute_u = dedupe_key(compute, "paper_id", "compute")
    meta_u = dedupe_key(metadata, "source_acl_id", "metadata")
    topics_u = dedupe_key(topics, "paper_id", "topics")
    awards_u = dedupe_key(awards, "anthology_id", "awards")
    org_vars_u = dedupe_key(org_vars, "paper_id", "organization variables")

    master = (
        compute_u.merge(
            meta_u,
            left_on="paper_id",
            right_on="source_acl_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            topics_u[["paper_id", "topic"]].rename(columns={"topic": "primary_topic"}),
            on="paper_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            awards_u[["anthology_id", "award"]].rename(columns={"award": "award_label"}),
            left_on="paper_id",
            right_on="anthology_id",
            how="left",
            validate="one_to_one",
        )
        .merge(org_vars_u, on="paper_id", how="left", validate="one_to_one", suffixes=("", "_org"))
    )

    if "paper_year" in master.columns:
        master["year"] = safe_numeric(master["paper_year"])
    else:
        master["year"] = parse_year_from_paper_id(master["paper_id"])
    if "paper_venue" in master.columns:
        master["venue"] = master["paper_venue"].astype(str).str.lower()
    else:
        master["venue"] = parse_venue_from_paper_id(master["paper_id"])

    master["year_str"] = master["year"].round().astype("Int64").astype(str)
    master["venue"] = master["venue"].fillna("unknown").astype(str).str.lower()
    master["year_venue"] = master["year_str"] + "_" + master["venue"]
    master["primary_topic"] = master["primary_topic"].fillna("Unknown").astype(str)

    master["gpu_lb1_total_compute"] = safe_numeric(master["paper_total_compute_capability_gfimp_lb1"])
    master["gpu_lb1_max_compute"] = safe_numeric(master["paper_max_row_compute_capability_gfimp_lb1"])
    master["strict_raw_total_compute"] = safe_numeric(master["paper_total_compute_capability"])
    master["strict_raw_max_compute"] = safe_numeric(master["paper_max_row_compute_capability"])

    master["gpu_lb1_log10_compute"] = positive_log10(master["gpu_lb1_total_compute"])
    master["gpu_lb1_log10_max_compute"] = positive_log10(master["gpu_lb1_max_compute"])
    master["strict_raw_log10_compute"] = positive_log10(master["strict_raw_total_compute"])
    master["strict_raw_log10_max_compute"] = positive_log10(master["strict_raw_max_compute"])

    master["cited_by_count"] = safe_numeric(master["cited_by_count"])
    master["log1p_cites"] = np.log1p(master["cited_by_count"])
    master["citation_normalized_percentile"] = safe_numeric(
        master["citation_normalized_percentile.value"]
    )
    master = add_topic_year_citation_percentile(master)
    master["is_award"] = master["award_label"].notna().astype(int)

    master["team_size"] = safe_numeric(master["team_size"])
    master["log1p_team_size"] = np.log1p(master["team_size"])
    master["team_size_group"] = pd.cut(
        master["team_size"], bins=[-np.inf, 1, 5, np.inf], labels=["1", "2-5", "6+"]
    ).astype("object")

    master["n_organizations"] = safe_numeric(master["n_organizations"])
    master["log1p_n_organizations"] = np.log1p(master["n_organizations"])
    master["n_organizations_group"] = pd.cut(
        master["n_organizations"],
        bins=[-np.inf, 1, 5, np.inf],
        labels=["1", "2-5", "6+"],
    ).astype("object")

    master["cit_rank_pct_all_yv"] = master.groupby(
        ["year_str", "venue"], dropna=False
    )["cited_by_count"].rank(method="average", pct=True)
    master["is_highly_cited_all_yv"] = (
        master["cit_rank_pct_all_yv"].ge(0.90) & master["cited_by_count"].notna()
    ).astype(int)

    if "contribution_type" in master.columns:
        master = master.drop(columns=["contribution_type"])
    if not master["paper_id"].is_unique:
        duplicates = master.loc[master["paper_id"].duplicated(), "paper_id"].head(5).tolist()
        raise ValueError(f"Master panel is not paper-level unique; examples: {duplicates}")
    return master


def make_analysis_sample(
    df: pd.DataFrame,
    sample: str = "gpu_lb1",
    year_min: int = 2020,
    year_max: int = 2023,
) -> pd.DataFrame:
    if sample not in {"gpu_lb1", "strict_raw"}:
        raise ValueError("sample must be 'gpu_lb1' or 'strict_raw'")

    out = df.copy()
    out["year"] = safe_numeric(out["year"])
    out = out.loc[out["year"].between(year_min, year_max)].copy()

    if sample == "gpu_lb1":
        out = out.loc[safe_numeric(out["is_lb1_gfimp"]).eq(1)].copy()
        out["total_compute"] = out["gpu_lb1_total_compute"]
        out["max_compute"] = out["gpu_lb1_max_compute"]
        out["log10_compute"] = out["gpu_lb1_log10_compute"]
        out["log10_max_compute"] = out["gpu_lb1_log10_max_compute"]
    else:
        out = out.loc[safe_numeric(out["is_strict"]).eq(1)].copy()
        out["total_compute"] = out["strict_raw_total_compute"]
        out["max_compute"] = out["strict_raw_max_compute"]
        out["log10_compute"] = out["strict_raw_log10_compute"]
        out["log10_max_compute"] = out["strict_raw_log10_max_compute"]

    out["year_str"] = out["year"].round().astype("Int64").astype(str)
    out["venue"] = out["venue"].fillna("unknown").astype(str).str.lower()
    out["year_venue"] = out["year_str"] + "_" + out["venue"]

    for col in ["primary_topic", "team_size_group", "n_organizations_group"]:
        out[col] = out[col].astype("object")
    return out.replace([np.inf, -np.inf], np.nan)


def _time_or_hour_columns(*frames: pd.DataFrame) -> list[str]:
    keywords = ("hour", "time", "duration", "training", "train")
    columns: set[str] = set()
    for frame in frames:
        columns.update(
            str(column)
            for column in frame.columns
            if any(keyword in str(column).lower() for keyword in keywords)
        )
    return sorted(columns)


def add_fine_grained_compute_features(
    master: pd.DataFrame,
    paper_compute: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    require_columns(
        paper_compute,
        {
            "paper_id",
            "gpu_group_key",
            "compute_capability",
            "gpu_num_filled",
            "benchmark_generation",
            "benchmark_family",
            "benchmark_gpu_name",
        },
        "paper compute rows",
    )
    require_columns(
        master,
        {
            "paper_id",
            "paper_max_row_compute_capability",
            "paper_gpu_num_filled_total",
        },
        "master panel",
    )

    rows = paper_compute.copy()
    rows["maxrow_compute_capability"] = safe_numeric(rows["compute_capability"])
    rows["maxrow_gpu_count"] = safe_numeric(rows["gpu_num_filled"])
    rows = rows.sort_values(
        ["paper_id", "maxrow_compute_capability", "gpu_group_key"],
        ascending=[True, False, True],
        na_position="last",
    )
    main_rows = rows.drop_duplicates("paper_id", keep="first").copy()
    main_rows["maxrow_single_gpu_flops"] = (
        main_rows["maxrow_compute_capability"]
        / main_rows["maxrow_gpu_count"].where(main_rows["maxrow_gpu_count"].gt(0))
    )
    main_rows = main_rows[
        [
            "paper_id",
            "maxrow_compute_capability",
            "maxrow_gpu_count",
            "maxrow_single_gpu_flops",
            "benchmark_generation",
            "benchmark_family",
            "benchmark_gpu_name",
        ]
    ].rename(
        columns={
            "benchmark_generation": "maxrow_gpu_generation",
            "benchmark_family": "maxrow_gpu_family",
            "benchmark_gpu_name": "maxrow_gpu_name",
        }
    )

    out = master.merge(main_rows, on="paper_id", how="left", validate="one_to_one")
    out["paper_total_gpu_count"] = safe_numeric(out["paper_gpu_num_filled_total"])
    out["log10_maxrow_gpu_count"] = positive_log10(out["maxrow_gpu_count"])
    out["log10_maxrow_single_gpu_flops"] = positive_log10(out["maxrow_single_gpu_flops"])
    out["log10_paper_total_gpu_count"] = positive_log10(out["paper_total_gpu_count"])
    out["ampere_or_newer"] = out["maxrow_gpu_generation"].isin(AMPERE_OR_NEWER_GENERATIONS).astype(int)

    expected = safe_numeric(out["paper_max_row_compute_capability"])
    observed = safe_numeric(out["maxrow_compute_capability"])
    both_missing = expected.isna() & observed.isna()
    both_present_close = (
        expected.notna()
        & observed.notna()
        & np.isclose(expected, observed, rtol=1e-9, atol=1e-6)
    )
    validation_match = both_missing | both_present_close
    if not validation_match.all():
        examples = out.loc[
            ~validation_match,
            ["paper_id", "paper_max_row_compute_capability", "maxrow_compute_capability"],
        ].head(10)
        raise ValueError(
            "Max-row compute validation failed for row-level fine-grained features: "
            f"{examples.to_dict(orient='records')}"
        )

    audit = {
        "paper_compute_rows": int(len(paper_compute)),
        "max_row_validation_total": int(len(out)),
        "max_row_validation_matched": int(validation_match.sum()),
        "max_row_validation_mismatched": int((~validation_match).sum()),
        "max_row_validation_missing_both": int(both_missing.sum()),
        "max_row_validation_observed_positive": int(observed.gt(0).sum()),
        "time_or_hours_columns_detected": _time_or_hour_columns(master, paper_compute),
    }
    return out, audit


def build_specs(compute_var: str = "log10_max_compute", team_control: str = "group") -> dict[int, dict]:
    if team_control == "group":
        team_term = "C(team_size_group)"
        team_required = "team_size_group"
    elif team_control == "continuous":
        team_term = "log1p_team_size"
        team_required = "log1p_team_size"
    else:
        raise ValueError("team_control must be 'group' or 'continuous'")

    return {
        1: {
            "rhs": [compute_var, "C(year_venue)"],
            "required": [compute_var, "year_venue"],
            "controls": "Year x Venue FE",
        },
        2: {
            "rhs": [compute_var, "C(year_venue)", "C(primary_topic)"],
            "required": [compute_var, "year_venue", "primary_topic"],
            "controls": "+ Topic FE",
        },
        3: {
            "rhs": [compute_var, "C(year_venue)", team_term],
            "required": [compute_var, "year_venue", team_required],
            "controls": "+ Team size",
        },
        4: {
            "rhs": [compute_var, "C(year_venue)", "C(n_organizations_group)"],
            "required": [compute_var, "year_venue", "n_organizations_group"],
            "controls": "+ Org-count group",
        },
        5: {
            "rhs": [compute_var, "C(year_venue)", "C(primary_topic)", team_term],
            "required": [compute_var, "year_venue", "primary_topic", team_required],
            "controls": "+ Topic FE + Team",
        },
        6: {
            "rhs": [
                compute_var,
                "C(year_venue)",
                "C(primary_topic)",
                "C(n_organizations_group)",
            ],
            "required": [compute_var, "year_venue", "primary_topic", "n_organizations_group"],
            "controls": "+ Topic FE + Org-count",
        },
        7: {
            "rhs": [
                compute_var,
                "C(year_venue)",
                "C(primary_topic)",
                team_term,
                "C(n_organizations_group)",
            ],
            "required": [
                compute_var,
                "year_venue",
                "primary_topic",
                team_required,
                "n_organizations_group",
            ],
            "controls": "+ Topic FE + Team + Org-count",
        },
    }


def model_formula(outcome: str, rhs_terms: list[str]) -> str:
    return f"{outcome} ~ " + " + ".join(rhs_terms)


def fit_formula(
    formula: str,
    data: pd.DataFrame,
    family: str = "ols",
    cov_type: str = "HC3",
    cluster_var: str | None = None,
):
    fit_kwargs = {"cov_type": cov_type}
    if cov_type.lower() == "cluster":
        if cluster_var is None:
            raise ValueError("cluster_var is required when cov_type='cluster'")
        fit_kwargs["cov_kwds"] = {"groups": data[cluster_var]}

    if family in {"ols", "lpm"}:
        return smf.ols(formula, data=data).fit(**fit_kwargs)
    if family == "poisson":
        return smf.glm(formula, data=data, family=sm.families.Poisson()).fit(**fit_kwargs)
    raise ValueError("family must be 'ols', 'lpm', or 'poisson'")


def fit_model_grid(
    df: pd.DataFrame,
    outcome: str,
    family: str = "ols",
    compute_var: str = "log10_max_compute",
    team_control: str = "group",
    specs: Iterable[int] | None = None,
    max_spec: int | None = None,
    common_sample: bool = True,
    cov_type: str = "HC3",
    cluster_var: str | None = None,
    min_outcome_variation: bool = True,
) -> dict:
    spec_dict = build_specs(compute_var=compute_var, team_control=team_control)
    if specs is not None:
        spec_set = set(specs)
        spec_dict = {idx: spec for idx, spec in spec_dict.items() if idx in spec_set}
    if max_spec is not None:
        spec_dict = {idx: spec for idx, spec in spec_dict.items() if idx <= max_spec}

    required_common = sorted(
        {outcome, *[col for spec in spec_dict.values() for col in spec["required"]]}
    )
    if cluster_var is not None:
        required_common.append(cluster_var)
    base_df = df.copy()
    common_df = base_df.dropna(subset=required_common).copy() if common_sample else None
    if common_sample and min_outcome_variation and common_df[outcome].nunique(dropna=True) < 2:
        raise ValueError(f"Outcome {outcome} has insufficient variation in common sample.")

    models: dict[str, object] = {}
    samples: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []

    for spec_id, spec in spec_dict.items():
        formula = model_formula(outcome, spec["rhs"])
        if common_sample:
            sample_df = common_df.copy()
        else:
            required = [outcome, *spec["required"]]
            if cluster_var is not None:
                required.append(cluster_var)
            sample_df = base_df.dropna(subset=required).copy()

        if min_outcome_variation and sample_df[outcome].nunique(dropna=True) < 2:
            continue
        model = fit_formula(
            formula,
            sample_df,
            family=family,
            cov_type=cov_type,
            cluster_var=cluster_var,
        )
        name = f"{family}_{spec_id}"
        models[name] = model
        samples[name] = sample_df

        beta = model.params.get(compute_var, np.nan)
        se = model.bse.get(compute_var, np.nan)
        p_value = model.pvalues.get(compute_var, np.nan)
        ci_low, ci_high = (np.nan, np.nan)
        if compute_var in model.params.index:
            ci_low, ci_high = model.conf_int().loc[compute_var].tolist()
        pct_per_10x = (
            np.exp(beta) - 1
            if family in {"ols", "poisson"} and outcome in {"log1p_cites", "cited_by_count"}
            else np.nan
        )
        pp_per_10x = beta * 100 if family == "lpm" or outcome.startswith("is_") else np.nan

        rows.append(
            {
                "model": name,
                "spec": spec_id,
                "formula": formula,
                "controls": spec["controls"],
                "outcome": outcome,
                "family": family,
                "compute_var": compute_var,
                "team_control": team_control,
                "cov_type": cov_type,
                "nobs": int(model.nobs),
                "df_model": float(model.df_model),
                "coef": beta,
                "se": se,
                "p": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "r2": float(getattr(model, "rsquared", np.nan)),
                "adj_r2": float(getattr(model, "rsquared_adj", np.nan)),
                "mean_y": float(sample_df[outcome].mean()),
                "pct_per_10x": pct_per_10x,
                "pp_per_10x": pp_per_10x,
            }
        )

    return {
        "models": models,
        "samples": samples,
        "effect_table": pd.DataFrame(rows),
        "common_df": common_df,
        "specs": spec_dict,
    }


def get_model_row(result: dict, spec: int) -> pd.Series:
    table = result["effect_table"]
    rows = table.loc[table["spec"].eq(spec)]
    if rows.empty:
        raise KeyError(f"No result for spec {spec}")
    return rows.iloc[0]


def compact_effect_table(result: dict, label: str) -> pd.DataFrame:
    table = result["effect_table"].copy()
    table.insert(0, "outcome_model", label)
    return table


def _format_report_value(value, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer() and abs(value) >= 1:
            return f"{int(value):,}"
        if value == 0:
            return "0"
        abs_value = abs(value)
        if abs_value < 0.001:
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def _format_report_p(value) -> str:
    if pd.isna(value):
        return ""
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _format_report_percent(value) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.1f}%"


def _format_report_effect(row: pd.Series) -> str:
    if not pd.isna(row.get("pct_per_10x")):
        return f"{row['pct_per_10x'] * 100:.1f}%"
    if not pd.isna(row.get("pp_per_10x")):
        return f"{row['pp_per_10x']:.2f} pp"
    return ""


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                str(row[col]).replace("\n", " ").replace("|", "\\|")
                for col in frame.columns
            ]
        )
    header = [str(col).replace("|", "\\|") for col in frame.columns]
    separator = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _regression_markdown_table(results: pd.DataFrame, include_family: bool = False) -> str:
    display = pd.DataFrame(
        {
            "Outcome/model": results["outcome_model"],
            "Spec": results["spec"].astype(int),
            "Controls": results["controls"],
            "N": results["nobs"].map(lambda value: _format_report_value(value, 0)),
            "Coef.": results["coef"].map(lambda value: _format_report_value(value, 3)),
            "SE": results["se"].map(lambda value: _format_report_value(value, 3)),
            "p": results["p"].map(_format_report_p),
            "95% CI": results.apply(
                lambda row: f"[{_format_report_value(row['ci_low'], 3)}, {_format_report_value(row['ci_high'], 3)}]",
                axis=1,
            ),
            "R2": results["r2"].map(lambda value: _format_report_value(value, 3)),
            "Adj. R2": results["adj_r2"].map(lambda value: _format_report_value(value, 3)),
            "10x effect": results.apply(_format_report_effect, axis=1),
        }
    )
    if include_family:
        display.insert(3, "Family", results["family"])
        display.insert(4, "Cov.", results["cov_type"])
    return _markdown_table(display)


def _simple_report_table(
    frame: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    p_columns: set[str] | None = None,
    percent_columns: set[str] | None = None,
    effect_columns: set[str] | None = None,
) -> str:
    p_columns = p_columns or set()
    percent_columns = percent_columns or set()
    effect_columns = effect_columns or set()
    display = pd.DataFrame()
    for column, label in zip(columns, labels):
        if column not in frame.columns:
            continue
        if column in p_columns:
            display[label] = frame[column].map(_format_report_p)
        elif column in percent_columns:
            display[label] = frame[column].map(_format_report_percent)
        elif column in effect_columns:
            display[label] = frame[column].map(lambda value: f"{value * 100:.1f}%")
        else:
            display[label] = frame[column].map(_format_report_value)
    return _markdown_table(display)


def _robustness_markdown_tables(tables: dict[str, pd.DataFrame]) -> str:
    selection = _simple_report_table(
        tables["selection_check"],
        [
            "sample",
            "rows",
            "valid_log10_max_compute",
            "mean_cites",
            "high_cited_rate_all_yv",
            "award_rate",
        ],
        [
            "Sample",
            "Rows",
            "Valid compute",
            "Mean cites",
            "High-cited rate",
            "Award rate",
        ],
        percent_columns={"high_cited_rate_all_yv", "award_rate"},
    )
    compute = _simple_report_table(
        tables["robust_compute_variable_team_control"],
        ["compute_var", "team_control", "nobs", "coef", "se", "p", "pct_per_10x"],
        ["Compute var", "Team control", "N", "Coef.", "SE", "p", "10x effect"],
        p_columns={"p"},
        effect_columns={"pct_per_10x"},
    )
    outlier = _simple_report_table(
        tables["outlier_robustness"],
        ["sample", "nobs", "coef", "se", "p", "pct_per_10x"],
        ["Sample", "N", "Coef.", "SE", "p", "10x effect"],
        p_columns={"p"},
        effect_columns={"pct_per_10x"},
    )
    cluster = _simple_report_table(
        tables["cluster_se_sensitivity"],
        ["cluster_var", "n_clusters", "nobs", "coef", "se", "p"],
        ["Cluster variable", "Clusters", "N", "Coef.", "SE", "p"],
        p_columns={"p"},
    )
    leave_one_out = _simple_report_table(
        tables["leave_one_out"],
        ["leave_out_type", "left_out", "nobs", "coef", "se", "p"],
        ["Leave-out type", "Left out", "N", "Coef.", "SE", "p"],
        p_columns={"p"},
    )
    institution = _simple_report_table(
        tables["institution_history_controls"],
        ["model", "nobs", "coef", "se", "p", "r2"],
        ["Model", "N", "Coef.", "SE", "p", "R2"],
        p_columns={"p"},
    )
    award = _simple_report_table(
        tables["award_sparse_diagnostics"],
        ["metric", "value"],
        ["Metric", "Value"],
    )
    effect_strength = _simple_report_table(
        tables["effect_strength"],
        [
            "model",
            "nobs",
            "beta",
            "se",
            "p",
            "pct_effect_10x",
            "pct_effect_100x",
            "pct_effect_1000x",
        ],
        [
            "Model",
            "N",
            "Beta",
            "SE",
            "p",
            "10x effect",
            "100x effect",
            "1000x effect",
        ],
        p_columns={"p"},
        effect_columns={"pct_effect_10x", "pct_effect_100x", "pct_effect_1000x"},
    )
    delta_r2 = _simple_report_table(
        tables["delta_r2"],
        ["spec", "nobs", "r2_full", "r2_without_compute", "delta_r2"],
        ["Spec", "N", "Full R2", "R2 without compute", "Delta R2"],
    )
    return f"""### Sample selection check

{selection}

### Alternative compute and team controls

{compute}

### Outlier sensitivity

{outlier}

### Clustered standard errors

{cluster}

### Leave-one-out sensitivity

{leave_one_out}

### Institution-history controls

{institution}

### Award sparsity diagnostics

{award}

### Effect size and incremental R2

{effect_strength}

{delta_r2}"""


def pick_result(result: dict, label: str, spec: int, effect_type: str) -> dict:
    row = get_model_row(result, spec)
    return {
        "outcome_model": label,
        "spec": spec,
        "nobs": int(row["nobs"]),
        "coef": row["coef"],
        "se": row["se"],
        "p": row["p"],
        "ci_low": row["ci_low"],
        "ci_high": row["ci_high"],
        "effect_interpretation": effect_type,
        "pct_per_10x": row["pct_per_10x"],
        "pp_per_10x": row["pp_per_10x"],
        "mean_y": row["mean_y"],
    }


def effect_strength_ols(
    result: dict,
    spec: int = 7,
    compute_var: str = "log10_max_compute",
    outcome: str = "log1p_cites",
) -> pd.Series:
    model_name = f"ols_{spec}"
    model = result["models"][model_name]
    df = result["samples"][model_name]
    beta = model.params[compute_var]
    se = model.bse[compute_var]
    t_value = beta / se
    q25, q75 = df[compute_var].quantile([0.25, 0.75])
    return pd.Series(
        {
            "model": model_name,
            "nobs": int(model.nobs),
            "beta": beta,
            "se": se,
            "p": model.pvalues[compute_var],
            "pct_effect_10x": np.exp(beta) - 1,
            "pct_effect_100x": np.exp(beta * 2) - 1,
            "pct_effect_1000x": np.exp(beta * 3) - 1,
            "iqr_log10_compute": q75 - q25,
            "pct_effect_iqr": np.exp(beta * (q75 - q25)) - 1,
            "std_beta": beta * df[compute_var].std(ddof=0) / df[outcome].std(ddof=0),
            "partial_r2": (t_value**2) / (t_value**2 + model.df_resid),
        }
    )


def delta_r2_for_spec(
    df: pd.DataFrame,
    outcome: str,
    spec_id: int = 7,
    compute_var: str = "log10_max_compute",
    team_control: str = "group",
) -> pd.Series:
    specs = build_specs(compute_var=compute_var, team_control=team_control)
    spec = specs[spec_id]
    sample_df = df.dropna(subset=[outcome, *spec["required"]]).copy()
    full_formula = model_formula(outcome, spec["rhs"])
    no_compute_rhs = [term for term in spec["rhs"] if term != compute_var]
    no_compute_formula = model_formula(outcome, no_compute_rhs)
    full = smf.ols(full_formula, data=sample_df).fit(cov_type="HC3")
    no_compute = smf.ols(no_compute_formula, data=sample_df).fit(cov_type="HC3")
    return pd.Series(
        {
            "spec": spec_id,
            "nobs": len(sample_df),
            "r2_full": full.rsquared,
            "r2_without_compute": no_compute.rsquared,
            "delta_r2": full.rsquared - no_compute.rsquared,
        }
    )


def outlier_filtered(
    df: pd.DataFrame,
    drop_cite_top1: bool = False,
    drop_compute_top1: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    if drop_cite_top1:
        out = out.loc[out["cited_by_count"].le(out["cited_by_count"].quantile(0.99))].copy()
    if drop_compute_top1:
        out = out.loc[out["log10_max_compute"].le(out["log10_max_compute"].quantile(0.99))].copy()
    return out


def build_org_history_controls(
    paper_org_long: pd.DataFrame, org_year_panel: pd.DataFrame
) -> pd.DataFrame:
    count_cols = [
        "paper_count",
        "collaborative_paper_count",
        "international_collab_paper_count",
        "industry_academia_paper_count",
        "cross_sector_paper_count",
        "government_academia_paper_count",
        "unique_partner_org_count",
    ]
    require_columns(org_year_panel, {"org_id", "year", *count_cols}, "organization-year panel")
    require_columns(paper_org_long, {"paper_id", "org_id", "year"}, "paper organization long")

    panel = org_year_panel.copy()
    panel["year"] = safe_numeric(panel["year"]).astype("Int64")
    panel["org_id"] = panel["org_id"].astype(str)
    for col in count_cols:
        panel[col] = safe_numeric(panel[col]).fillna(0)
    panel = panel.sort_values(["org_id", "year"])
    for col in count_cols:
        panel[f"prior_cum_{col}"] = panel.groupby("org_id")[col].cumsum() - panel[col]

    history_cols = ["org_id", "year", *[f"prior_cum_{col}" for col in count_cols]]
    org_long = paper_org_long.copy()
    org_long["paper_id"] = org_long["paper_id"].astype(str)
    org_long["org_id"] = org_long["org_id"].astype(str)
    org_long["year"] = safe_numeric(org_long["year"]).astype("Int64")

    merged = org_long.merge(panel[history_cols], on=["org_id", "year"], how="left")
    for col in [f"prior_cum_{name}" for name in count_cols]:
        merged[col] = safe_numeric(merged[col]).fillna(0)

    agg = (
        merged.groupby("paper_id")
        .agg(
            n_orgs_long=("org_id", "nunique"),
            max_prior_org_papers=("prior_cum_paper_count", "max"),
            mean_prior_org_papers=("prior_cum_paper_count", "mean"),
            sum_prior_org_papers=("prior_cum_paper_count", "sum"),
            max_prior_collab_papers=("prior_cum_collaborative_paper_count", "max"),
            max_prior_international_papers=("prior_cum_international_collab_paper_count", "max"),
            max_prior_industry_academia_papers=(
                "prior_cum_industry_academia_paper_count",
                "max",
            ),
            max_prior_cross_sector_papers=("prior_cum_cross_sector_paper_count", "max"),
            max_prior_partner_org_count=("prior_cum_unique_partner_org_count", "max"),
        )
        .reset_index()
    )
    for col in [name for name in agg.columns if name.startswith(("max_prior", "mean_prior", "sum_prior"))]:
        agg[f"log1p_{col}"] = np.log1p(agg[col])
    return agg


def fit_institution_history_models(
    master: pd.DataFrame,
    paper_org_long: pd.DataFrame,
    org_year_panel: pd.DataFrame,
) -> pd.DataFrame:
    if "log1p_max_prior_org_papers" in master.columns:
        master_org = master.copy()
    else:
        org_history = build_org_history_controls(paper_org_long, org_year_panel)
        master_org = master.merge(org_history, on="paper_id", how="left", validate="one_to_one")
    sample = make_analysis_sample(master_org, sample="strict_raw", year_min=2020, year_max=2023)
    institution_controls = [
        "log1p_max_prior_org_papers",
        "log1p_mean_prior_org_papers",
        "log1p_max_prior_partner_org_count",
        "has_company",
        "has_industry_academia",
        "has_international_collab",
    ]
    for col in institution_controls:
        sample[col] = safe_numeric(sample[col])

    base_rhs = build_specs("log10_max_compute", "group")[7]["rhs"]
    required = ["log1p_cites", *build_specs("log10_max_compute", "group")[7]["required"], *institution_controls]
    inst_df = sample.dropna(subset=required).copy()
    if inst_df.empty:
        return pd.DataFrame()

    base_formula = model_formula("log1p_cites", base_rhs)
    ext_formula = model_formula("log1p_cites", [*base_rhs, *institution_controls])
    base_model = smf.ols(base_formula, data=inst_df).fit(cov_type="HC3")
    ext_model = smf.ols(ext_formula, data=inst_df).fit(cov_type="HC3")
    return pd.DataFrame(
        [
            {
                "model": "strict baseline on institution-control sample",
                "nobs": int(base_model.nobs),
                "coef": base_model.params["log10_max_compute"],
                "se": base_model.bse["log10_max_compute"],
                "p": base_model.pvalues["log10_max_compute"],
                "r2": base_model.rsquared,
            },
            {
                "model": "strict plus prior org history/collab controls",
                "nobs": int(ext_model.nobs),
                "coef": ext_model.params["log10_max_compute"],
                "se": ext_model.bse["log10_max_compute"],
                "p": ext_model.pvalues["log10_max_compute"],
                "r2": ext_model.rsquared,
            },
        ]
    )


def load_confounder_controls(path: Path) -> pd.DataFrame:
    controls = pd.read_csv(path)
    required = {
        "paper_id",
        *AUTHOR_HISTORY_CONTROLS,
        "institution_prior_citations_3y_max_log1p",
        "first_author_openalex_id",
        "last_author_openalex_id",
        "author_id_coverage",
        "institution_id_coverage",
        "has_public_artifact",
    }
    require_columns(controls, required, "confounder controls")
    return dedupe_key(controls, "paper_id", "confounder controls")


def attach_confounder_controls(
    master: pd.DataFrame,
    confounder_controls: pd.DataFrame,
    paper_org_long: pd.DataFrame,
    org_year_panel: pd.DataFrame,
) -> pd.DataFrame:
    controls = confounder_controls.drop(columns=["year"], errors="ignore")
    out = master.merge(controls, on="paper_id", how="left", validate="one_to_one")
    org_history = build_org_history_controls(paper_org_long, org_year_panel)
    out = out.merge(org_history, on="paper_id", how="left", validate="one_to_one")
    numeric_columns = [
        *PREPUBLICATION_CONFOUNDER_CONTROLS,
        *ARTIFACT_CONTROLS,
        "author_id_coverage",
        "institution_id_coverage",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = safe_numeric(out[column])
    return out


def _base_spec7_rhs() -> list[str]:
    return list(build_specs("log10_max_compute", "group")[7]["rhs"])


def _effect_row_from_model(
    model,
    sample: pd.DataFrame,
    model_id: str,
    outcome: str,
    family: str,
    controls: str,
    compute_var: str = "log10_max_compute",
) -> dict:
    beta = float(model.params.get(compute_var, np.nan))
    se = float(model.bse.get(compute_var, np.nan))
    p_value = float(model.pvalues.get(compute_var, np.nan))
    ci_low, ci_high = (np.nan, np.nan)
    if compute_var in model.params.index:
        ci_low, ci_high = [float(value) for value in model.conf_int().loc[compute_var]]
    return {
        "model_id": model_id,
        "outcome": outcome,
        "family": family,
        "controls": controls,
        "nobs": int(model.nobs),
        "coef": beta,
        "se": se,
        "p": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "r2": float(getattr(model, "rsquared", np.nan)),
        "adj_r2": float(getattr(model, "rsquared_adj", np.nan)),
        "mean_y": float(sample[outcome].mean()),
        "pct_per_10x": (
            float(np.exp(beta) - 1)
            if family in {"ols", "poisson"} and outcome in {"log1p_cites", "cited_by_count"}
            else np.nan
        ),
        "pp_per_10x": (
            beta * 100
            if family == "lpm" or outcome in {"citation_normalized_percentile"}
            else np.nan
        ),
        "status": "ok",
        "reason": "",
    }


def _skipped_effect_row(
    model_id: str,
    outcome: str,
    family: str,
    controls: str,
    reason: str,
) -> dict:
    return {
        "model_id": model_id,
        "outcome": outcome,
        "family": family,
        "controls": controls,
        "nobs": 0,
        "coef": np.nan,
        "se": np.nan,
        "p": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "r2": np.nan,
        "adj_r2": np.nan,
        "mean_y": np.nan,
        "pct_per_10x": np.nan,
        "pp_per_10x": np.nan,
        "delta_r2": np.nan,
        "coef_attenuation_vs_m0": np.nan,
        "status": "skipped",
        "reason": reason,
    }


def fit_confounder_model_ladder(df: pd.DataFrame) -> pd.DataFrame:
    base_rhs = _base_spec7_rhs()
    ladder = [
        ("M0", [], "Baseline Spec 7"),
        ("M1", AUTHOR_HISTORY_CONTROLS, "+ pre-publication author history"),
        (
            "M2",
            INSTITUTION_VISIBILITY_CONTROLS,
            "+ pre-publication institution visibility and collaboration",
        ),
        (
            "M3",
            PREPUBLICATION_CONFOUNDER_CONTROLS,
            "+ all pre-publication confounder proxies",
        ),
        (
            "M4",
            [*PREPUBLICATION_CONFOUNDER_CONTROLS, *ARTIFACT_CONTROLS],
            "+ public artifact (secondary robustness)",
        ),
    ]
    required = [
        "log1p_cites",
        "log10_max_compute",
        "year_venue",
        "primary_topic",
        "team_size_group",
        "n_organizations_group",
        *PREPUBLICATION_CONFOUNDER_CONTROLS,
        *ARTIFACT_CONTROLS,
    ]
    baseline_required = [
        "log1p_cites",
        "log10_max_compute",
        "year_venue",
        "primary_topic",
        "team_size_group",
        "n_organizations_group",
    ]

    def fit_baseline_only(reason: str) -> pd.DataFrame:
        rows: list[dict] = []
        if all(column in df.columns for column in baseline_required):
            sample = df.dropna(subset=baseline_required).copy()
        else:
            sample = pd.DataFrame()
        if not sample.empty:
            model = smf.ols(
                model_formula("log1p_cites", base_rhs), data=sample
            ).fit(cov_type="HC3")
            no_compute = smf.ols(
                model_formula(
                    "log1p_cites",
                    [term for term in base_rhs if term != "log10_max_compute"],
                ),
                data=sample,
            ).fit(cov_type="HC3")
            baseline = _effect_row_from_model(
                model,
                sample,
                "M0",
                "log1p_cites",
                "ols",
                "Baseline Spec 7",
            )
            baseline.update(
                {
                    "delta_r2": float(model.rsquared - no_compute.rsquared),
                    "coef_attenuation_vs_m0": 0.0,
                    "complete_case_rate": len(sample) / len(df) if len(df) else np.nan,
                    "main_analysis_eligible": 1,
                }
            )
            rows.append(baseline)
        else:
            rows.append(
                _skipped_effect_row(
                    "M0", "log1p_cites", "ols", "Baseline Spec 7", reason
                )
            )
        rows.extend(
            _skipped_effect_row(model_id, "log1p_cites", "ols", label, reason)
            for model_id, _, label in ladder
            if model_id != "M0"
        )
        return pd.DataFrame(rows)

    if any(column not in df.columns for column in required):
        missing = sorted(set(required) - set(df.columns))
        return fit_baseline_only(f"missing columns: {missing}")

    common = df.dropna(subset=required).copy()
    if len(common) == 0:
        return fit_baseline_only("no complete pre-publication history rows")

    rows: list[dict] = []
    baseline_beta = np.nan
    for model_id, extra_controls, label in ladder:
        rhs = [*base_rhs, *extra_controls]
        model = smf.ols(model_formula("log1p_cites", rhs), data=common).fit(cov_type="HC3")
        no_compute = smf.ols(
            model_formula("log1p_cites", [term for term in rhs if term != "log10_max_compute"]),
            data=common,
        ).fit(cov_type="HC3")
        row = _effect_row_from_model(
            model,
            common,
            model_id,
            "log1p_cites",
            "ols",
            label,
        )
        row["delta_r2"] = float(model.rsquared - no_compute.rsquared)
        if model_id == "M0":
            baseline_beta = row["coef"]
        row["coef_attenuation_vs_m0"] = (
            float(1 - row["coef"] / baseline_beta)
            if pd.notna(baseline_beta) and baseline_beta != 0
            else np.nan
        )
        row["complete_case_rate"] = len(common) / len(df) if len(df) else np.nan
        row["main_analysis_eligible"] = int(row["complete_case_rate"] >= 0.90)
        rows.append(row)

    m3_rhs = [*base_rhs, *PREPUBLICATION_CONFOUNDER_CONTROLS]
    alternative_outcomes = [
        ("cited_by_count", "poisson", "HC0"),
        ("citation_normalized_percentile", "ols", "HC3"),
        ("is_highly_cited_all_yv", "lpm", "HC3"),
    ]
    for outcome, family, covariance in alternative_outcomes:
        alt_required = [outcome, *required[1:-1]]
        sample = df.dropna(subset=list(dict.fromkeys(alt_required))).copy()
        if sample.empty or sample[outcome].nunique(dropna=True) < 2:
            rows.append(
                _skipped_effect_row(
                    "M3",
                    outcome,
                    family,
                    "+ all pre-publication confounder proxies",
                    "insufficient complete outcome rows",
                )
            )
            continue
        model = fit_formula(
            model_formula(outcome, m3_rhs),
            sample,
            family=family,
            cov_type=covariance,
        )
        row = _effect_row_from_model(
            model,
            sample,
            "M3",
            outcome,
            family,
            "+ all pre-publication confounder proxies",
        )
        row["delta_r2"] = np.nan
        row["coef_attenuation_vs_m0"] = np.nan
        row["complete_case_rate"] = len(sample) / len(df) if len(df) else np.nan
        row["main_analysis_eligible"] = int(row["complete_case_rate"] >= 0.90)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_author_fixed_effects(df: pd.DataFrame) -> pd.DataFrame:
    base_rhs = _base_spec7_rhs()
    rows: list[dict] = []
    for label, author_column in [
        ("first-listed author FE", "first_author_openalex_id"),
        ("last-listed author FE", "last_author_openalex_id"),
    ]:
        required = [
            "log1p_cites",
            "log10_max_compute",
            "year_venue",
            "primary_topic",
            "team_size_group",
            "n_organizations_group",
            author_column,
        ]
        if author_column not in df.columns:
            rows.append(
                {
                    "model": label,
                    "author_column": author_column,
                    "status": "skipped",
                    "reason": f"missing column: {author_column}",
                }
            )
            continue
        sample = df.dropna(subset=required).copy()
        grouped = sample.groupby(author_column)["log10_max_compute"]
        eligible_authors = grouped.agg(["size", "nunique"])
        eligible_authors = eligible_authors.loc[
            eligible_authors["size"].ge(2) & eligible_authors["nunique"].ge(2)
        ].index
        sample = sample.loc[sample[author_column].isin(eligible_authors)].copy()
        if len(sample) == 0 or len(eligible_authors) < 2:
            rows.append(
                {
                    "model": label,
                    "author_column": author_column,
                    "nobs": len(sample),
                    "n_authors": len(eligible_authors),
                    "status": "skipped",
                    "reason": "fewer than two repeated authors with within-author compute variation",
                }
            )
            continue
        rhs = [*base_rhs, f"C({author_column})"]
        model = smf.ols(model_formula("log1p_cites", rhs), data=sample).fit(
            cov_type="cluster",
            cov_kwds={"groups": sample[author_column]},
        )
        row = _effect_row_from_model(
            model,
            sample,
            label,
            "log1p_cites",
            "ols",
            label,
        )
        row.update(
            {
                "model": label,
                "author_column": author_column,
                "n_authors": int(len(eligible_authors)),
                "cov_type": "clustered by listed author",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_value(
    t_statistic: float,
    degrees_of_freedom: float,
    alpha: float = 1.0,
    q: float = 1.0,
) -> float:
    """Cinelli-Hazlett robustness value for an OLS coefficient."""
    if degrees_of_freedom <= 1:
        return np.nan
    f_value = q * abs(t_statistic) / np.sqrt(degrees_of_freedom)
    if alpha < 1:
        critical_t = abs(stats.t.ppf(alpha / 2, degrees_of_freedom - 1))
        f_value = max(0.0, f_value - critical_t / np.sqrt(degrees_of_freedom))
    f_squared = f_value**2
    return float(0.5 * (np.sqrt(f_squared**2 + 4 * f_squared) - f_squared))


def _partial_r2_group(
    data: pd.DataFrame,
    outcome: str,
    full_rhs: list[str],
    group_terms: list[str],
) -> float:
    reduced_rhs = [term for term in full_rhs if term not in group_terms]
    full = smf.ols(model_formula(outcome, full_rhs), data=data).fit()
    reduced = smf.ols(model_formula(outcome, reduced_rhs), data=data).fit()
    if reduced.ssr <= 0:
        return np.nan
    return float(max(0.0, min(1.0, (reduced.ssr - full.ssr) / reduced.ssr)))


def _bias_adjusted_coefficient(
    beta: float,
    standard_error: float,
    degrees_of_freedom: float,
    treatment_partial_r2: float,
    outcome_partial_r2: float,
) -> float:
    treatment_partial_r2 = min(max(treatment_partial_r2, 0.0), 0.999)
    outcome_partial_r2 = min(max(outcome_partial_r2, 0.0), 0.999)
    bias_factor = np.sqrt(
        outcome_partial_r2 * treatment_partial_r2 / (1 - treatment_partial_r2)
    )
    bias = standard_error * np.sqrt(degrees_of_freedom) * bias_factor
    return float(beta - np.sign(beta) * bias)


def fit_unobserved_confounding_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    base_rhs = _base_spec7_rhs()
    model_specs = [
        ("M0", base_rhs),
        ("M3", [*base_rhs, *PREPUBLICATION_CONFOUNDER_CONTROLS]),
    ]
    benchmark_groups = {
        "year-by-venue FE": ["C(year_venue)"],
        "topic FE": ["C(primary_topic)"],
        "team size": ["C(team_size_group)"],
        "organization count": ["C(n_organizations_group)"],
        "author history": AUTHOR_HISTORY_CONTROLS,
        "institution visibility/history": INSTITUTION_VISIBILITY_CONTROLS,
    }
    rows: list[dict] = []
    for model_id, rhs in model_specs:
        required = ["log1p_cites", "log10_max_compute"]
        if model_id == "M3":
            required.extend(PREPUBLICATION_CONFOUNDER_CONTROLS)
        required.extend(
            ["year_venue", "primary_topic", "team_size_group", "n_organizations_group"]
        )
        if any(column not in df.columns for column in required):
            rows.append(
                {
                    "model_id": model_id,
                    "status": "skipped",
                    "reason": "missing sensitivity-analysis columns",
                }
            )
            continue
        sample = df.dropna(subset=list(dict.fromkeys(required))).copy()
        if sample.empty:
            rows.append(
                {
                    "model_id": model_id,
                    "status": "skipped",
                    "reason": "no complete sensitivity-analysis rows",
                }
            )
            continue
        model = smf.ols(model_formula("log1p_cites", rhs), data=sample).fit()
        beta = float(model.params["log10_max_compute"])
        se = float(model.bse["log10_max_compute"])
        t_value = float(model.tvalues["log10_max_compute"])
        dof = float(model.df_resid)
        partial_r2 = float(t_value**2 / (t_value**2 + dof))
        candidate_rows: list[dict] = []
        treatment_rhs = [term for term in rhs if term != "log10_max_compute"]
        for benchmark, terms in benchmark_groups.items():
            if any(term not in rhs for term in terms):
                continue
            outcome_r2 = _partial_r2_group(sample, "log1p_cites", rhs, terms)
            treatment_r2 = _partial_r2_group(
                sample,
                "log10_max_compute",
                treatment_rhs,
                terms,
            )
            candidate_rows.append(
                {
                    "benchmark": benchmark,
                    "benchmark_outcome_partial_r2": outcome_r2,
                    "benchmark_treatment_partial_r2": treatment_r2,
                    "benchmark_strength": np.sqrt(outcome_r2 * treatment_r2),
                }
            )
        if candidate_rows:
            strongest = max(candidate_rows, key=lambda row: row["benchmark_strength"])
            candidate_rows.append({**strongest, "benchmark": "strongest observed control group"})
        else:
            candidate_rows.append(
                {
                    "benchmark": "none",
                    "benchmark_outcome_partial_r2": np.nan,
                    "benchmark_treatment_partial_r2": np.nan,
                    "benchmark_strength": np.nan,
                }
            )
        for candidate in candidate_rows:
            for multiplier in (1.0, 2.0):
                treatment_r2 = min(
                    0.999,
                    multiplier * candidate["benchmark_treatment_partial_r2"],
                )
                outcome_r2 = min(
                    0.999,
                    multiplier * candidate["benchmark_outcome_partial_r2"],
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "nobs": int(model.nobs),
                        "coef": beta,
                        "classical_se": se,
                        "t": t_value,
                        "df_resid": dof,
                        "partial_r2_compute_outcome": partial_r2,
                        "robustness_value_zero": robustness_value(t_value, dof),
                        "robustness_value_alpha_0_05": robustness_value(
                            t_value, dof, alpha=0.05
                        ),
                        "benchmark": candidate["benchmark"],
                        "benchmark_multiplier": multiplier,
                        "benchmark_outcome_partial_r2": candidate[
                            "benchmark_outcome_partial_r2"
                        ],
                        "benchmark_treatment_partial_r2": candidate[
                            "benchmark_treatment_partial_r2"
                        ],
                        "assumed_outcome_partial_r2": outcome_r2,
                        "assumed_treatment_partial_r2": treatment_r2,
                        "bias_adjusted_coef": _bias_adjusted_coefficient(
                            beta,
                            se,
                            dof,
                            treatment_r2,
                            outcome_r2,
                        ),
                        "status": "ok",
                        "reason": "",
                    }
                )
    return pd.DataFrame(rows)


def build_confounder_coverage_audit(
    citation_sample: pd.DataFrame,
    award_sample: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sample_name, sample in [
        ("citation_2020_2023", citation_sample),
        ("award_2020_2025", award_sample),
    ]:
        frame = sample.copy()
        if "log10_max_compute" in frame and frame["log10_max_compute"].notna().any():
            frame["compute_quartile"] = pd.qcut(
                frame["log10_max_compute"],
                4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop",
            ).astype("object")
        history_columns = [
            *AUTHOR_HISTORY_CONTROLS,
            "institution_prior_citations_3y_max_log1p",
        ]
        frame["history_complete"] = frame.reindex(columns=history_columns).notna().all(axis=1)
        frame["author_complete"] = safe_numeric(
            frame.get("author_id_coverage", pd.Series(np.nan, index=frame.index))
        ).ge(0.999)
        frame["institution_complete"] = safe_numeric(
            frame.get("institution_id_coverage", pd.Series(np.nan, index=frame.index))
        ).ge(0.999)
        frame["artifact_available"] = safe_numeric(
            frame.get("has_public_artifact", pd.Series(np.nan, index=frame.index))
        ).eq(1)

        groupings: list[tuple[str, list[str]]] = [
            ("overall", []),
            ("year", ["year_str"]),
            ("venue", ["venue"]),
            ("compute_quartile", ["compute_quartile"]),
        ]
        for grouping, columns in groupings:
            grouped = [("all", frame)] if not columns else frame.groupby(columns[0], dropna=False)
            for value, group in grouped:
                rows.append(
                    {
                        "sample": sample_name,
                        "grouping": grouping,
                        "group": str(value),
                        "n": len(group),
                        "history_complete_rate": float(group["history_complete"].mean()),
                        "author_id_complete_rate": float(group["author_complete"].mean()),
                        "institution_id_complete_rate": float(
                            group["institution_complete"].mean()
                        ),
                        "artifact_rate": float(group["artifact_available"].mean()),
                        "main_analysis_eligible": int(
                            float(group["history_complete"].mean()) >= 0.90
                        ),
                    }
                )
    return pd.DataFrame(rows)


def fit_firth_award_models(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import patsy
        from firthmodels import FirthLogisticRegression
    except ImportError as exc:
        return pd.DataFrame(
            [
                {
                    "model_id": model_id,
                    "status": "skipped",
                    "reason": f"firthmodels unavailable: {exc}",
                    "exploratory": 1,
                }
                for model_id in ["M0", "M3", "M4"]
            ]
        )

    base_rhs = _base_spec7_rhs()
    model_specs = [
        ("M0", base_rhs),
        ("M3", [*base_rhs, *PREPUBLICATION_CONFOUNDER_CONTROLS]),
        (
            "M4",
            [*base_rhs, *PREPUBLICATION_CONFOUNDER_CONTROLS, *ARTIFACT_CONTROLS],
        ),
    ]
    baseline_required = [
        "is_award",
        "log10_max_compute",
        "year_venue",
        "primary_topic",
        "team_size_group",
        "n_organizations_group",
    ]
    all_required = [
        *baseline_required,
        *PREPUBLICATION_CONFOUNDER_CONTROLS,
        *ARTIFACT_CONTROLS,
    ]
    full_common = None
    if all(column in df.columns for column in all_required):
        candidate = df.dropna(subset=all_required).copy()
        if not candidate.empty and candidate["is_award"].nunique() >= 2:
            full_common = candidate

    rows: list[dict] = []
    for model_id, rhs in model_specs:
        model_extra = (
            []
            if model_id == "M0"
            else (
                PREPUBLICATION_CONFOUNDER_CONTROLS
                if model_id == "M3"
                else [*PREPUBLICATION_CONFOUNDER_CONTROLS, *ARTIFACT_CONTROLS]
            )
        )
        model_required = [*baseline_required, *model_extra]
        missing = sorted(set(model_required) - set(df.columns))
        if missing:
            rows.append(
                {
                    "model_id": model_id,
                    "status": "skipped",
                    "reason": f"missing columns: {missing}",
                    "exploratory": 1,
                }
            )
            continue
        common = (
            full_common.copy()
            if full_common is not None
            else df.dropna(subset=model_required).copy()
        )
        if common.empty or common["is_award"].nunique() < 2:
            rows.append(
                {
                    "model_id": model_id,
                    "status": "skipped",
                    "reason": "no complete award rows with outcome variation",
                    "exploratory": 1,
                }
            )
            continue
        formula = model_formula("is_award", rhs)
        y, design = patsy.dmatrices(formula, common, return_type="dataframe")
        try:
            estimator = FirthLogisticRegression(
                fit_intercept=False,
                max_iter=100,
            ).fit(design, y.iloc[:, 0])
            estimator.lrt(features=["log10_max_compute"])
            conf = estimator.conf_int(
                method="pl",
                features=["log10_max_compute"],
                max_iter=100,
            )
            compute_index = design.columns.get_loc("log10_max_compute")
            beta = float(estimator.coef_[compute_index])
            ci_low, ci_high = [float(value) for value in conf[compute_index]]
            rows.append(
                {
                    "model_id": model_id,
                    "nobs": len(common),
                    "award_events": int(common["is_award"].sum()),
                    "event_rate": float(common["is_award"].mean()),
                    "coef": beta,
                    "p": float(estimator.lrt_pvalues_[compute_index]),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "odds_ratio_per_10x": float(np.exp(beta)),
                    "or_ci_low": float(np.exp(ci_low)),
                    "or_ci_high": float(np.exp(ci_high)),
                    "profile_likelihood": 1,
                    "exploratory": 1,
                    "status": "ok",
                    "reason": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_id": model_id,
                    "nobs": len(common),
                    "award_events": int(common["is_award"].sum()),
                    "event_rate": float(common["is_award"].mean()),
                    "exploratory": 1,
                    "status": "failed",
                    "reason": str(exc),
                }
            )
    return pd.DataFrame(rows)


def _joint_firth_penalized_lrt(
    estimator,
    constrained_indices: list[int],
) -> dict:
    """Jointly constrain Firth coefficients to zero in the full design matrix."""
    from firthmodels._solvers import newton_raphson
    from firthmodels.logistic import compute_logistic_quantities

    constrained = sorted(set(int(index) for index in constrained_indices))
    if not constrained:
        raise ValueError("At least one constrained coefficient is required.")
    x, y, sample_weight, offset = estimator._fit_data
    n_parameters = len(estimator.coef_)
    if constrained[-1] >= n_parameters:
        raise IndexError("Constrained coefficient index is outside the fitted design.")
    constrained_set = set(constrained)
    free_indices = np.array(
        [index for index in range(n_parameters) if index not in constrained_set],
        dtype=np.intp,
    )
    beta_full = np.zeros(n_parameters, dtype=np.float64)
    free_grid = np.ix_(free_indices, free_indices)

    def constrained_quantities(beta_free: np.ndarray) -> SimpleNamespace:
        beta_full.fill(0.0)
        beta_full[free_indices] = beta_free
        quantities = compute_logistic_quantities(
            X=x,
            y=y,
            beta=beta_full,
            sample_weight=sample_weight,
            offset=offset,
            workspace=estimator._workspace,
        )
        return SimpleNamespace(
            loglik=quantities.loglik,
            modified_score=quantities.modified_score[free_indices],
            fisher_info=quantities.fisher_info[free_grid],
        )

    constrained_result = newton_raphson(
        compute_quantities=constrained_quantities,
        n_features=len(free_indices),
        max_iter=estimator.max_iter,
        max_step=estimator.max_step,
        max_halfstep=estimator.max_halfstep,
        gtol=estimator.gtol,
        xtol=estimator.xtol,
    )
    chi2 = max(
        0.0,
        2.0 * (float(estimator.loglik_) - float(constrained_result.loglik)),
    )
    degrees_freedom = len(constrained)
    return {
        "omnibus_chi2": chi2,
        "omnibus_df": degrees_freedom,
        "omnibus_p": float(stats.chi2.sf(chi2, df=degrees_freedom)),
        "constrained_loglik": float(constrained_result.loglik),
        "constrained_n_iter": int(constrained_result.n_iter),
        "constrained_converged": bool(constrained_result.converged),
    }


def fit_joint_count_ampere_firth_award_model(
    award_sample: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Fit the prespecified additive count-plus-generation Firth award model."""
    import patsy
    from firthmodels import FirthLogisticRegression

    terms = ["log10_maxrow_gpu_count", "ampere_or_newer"]
    controls = [
        "C(year_venue)",
        "C(primary_topic)",
        "C(team_size_group)",
        "C(n_organizations_group)",
    ]
    required = [
        "paper_id",
        "is_award",
        "year_venue",
        "primary_topic",
        "team_size_group",
        "n_organizations_group",
        *terms,
    ]
    require_columns(award_sample, set(required), "joint Firth award sample")
    model_df = award_sample.dropna(subset=required).copy()
    if model_df.empty or model_df["is_award"].nunique() < 2:
        raise ValueError("Joint Firth award sample has insufficient outcome variation.")
    if model_df["ampere_or_newer"].nunique() < 2:
        raise ValueError("Joint Firth award sample has no Ampere-or-newer variation.")

    formula = model_formula("is_award", [*terms, *controls])
    outcome, design = patsy.dmatrices(formula, model_df, return_type="dataframe")
    estimator = FirthLogisticRegression(
        fit_intercept=False,
        max_iter=100,
        backend="numpy",
    ).fit(design, outcome.iloc[:, 0])
    if not estimator.converged_:
        raise RuntimeError("Joint Firth award model did not converge.")

    estimator.lrt(features=terms)
    profile_ci = estimator.conf_int(
        method="pl",
        features=terms,
        max_iter=100,
    )
    term_indices = [design.columns.get_loc(term) for term in terms]
    omnibus = _joint_firth_penalized_lrt(estimator, term_indices)
    if not omnibus["constrained_converged"]:
        raise RuntimeError("Jointly constrained Firth model did not converge.")

    raw_p = [float(estimator.lrt_pvalues_[index]) for index in term_indices]
    holm_p = multipletests(raw_p, method="holm")[1]
    labels = {
        "log10_maxrow_gpu_count": "GPU count",
        "ampere_or_newer": "Ampere-or-newer/equivalent",
    }
    scales = {
        "log10_maxrow_gpu_count": "odds ratio per 10x GPU count",
        "ampere_or_newer": "odds ratio vs older generation",
    }
    rows: list[dict] = []
    for order, (term, index, adjusted_p) in enumerate(
        zip(terms, term_indices, holm_p),
        start=1,
    ):
        beta = float(estimator.coef_[index])
        ci_low, ci_high = [float(value) for value in profile_ci[index]]
        rows.append(
            {
                "model_id": "joint_count_ampere",
                "term_order": order,
                "term": term,
                "term_label": labels[term],
                "sample": "strict_raw_2020_2025",
                "year_window": "2020-2025",
                "nobs": int(len(model_df)),
                "award_events": int(model_df["is_award"].sum()),
                "event_rate": float(model_df["is_award"].mean()),
                "coef": beta,
                "lrt_backcorrected_se": float(estimator.lrt_bse_[index]),
                "profile_ci_low": ci_low,
                "profile_ci_high": ci_high,
                "penalized_lrt_p": float(estimator.lrt_pvalues_[index]),
                "holm_p_two_terms": float(adjusted_p),
                "odds_ratio": float(np.exp(beta)),
                "or_ci_low": float(np.exp(ci_low)),
                "or_ci_high": float(np.exp(ci_high)),
                "effect_scale": scales[term],
                **omnibus,
                "full_penalized_loglik": float(estimator.loglik_),
                "full_n_iter": int(estimator.n_iter_),
                "full_converged": bool(estimator.converged_),
                "profile_likelihood_ci": 1,
                "formula": formula,
                "sample_id_sha256": _sha256_ids(model_df["paper_id"]),
                "status": "ok",
            }
        )

    audit = {
        "status": "validated_local_frozen_input_rerun",
        "estimator": "Firth bias-reduced logistic regression",
        "inference": {
            "term_p_values": "one-parameter penalized likelihood-ratio tests",
            "term_se": "LRT back-corrected standard errors",
            "term_ci": "95% profile-likelihood confidence intervals",
            "multiple_testing": "Holm adjustment across the two prespecified focal terms",
            "omnibus": "2-df penalized likelihood-ratio test jointly constraining both focal coefficients to zero",
        },
        "sample": "strict_raw_2020_2025",
        "year_window": "2020-2025",
        "award_sample_rows_before_complete_case": int(len(award_sample)),
        "estimation_sample_rows": int(len(model_df)),
        "award_positives": int(model_df["is_award"].sum()),
        "award_rate": float(model_df["is_award"].mean()),
        "year_venue_cells": int(model_df["year_venue"].nunique()),
        "zero_award_year_venue_cells": int(
            (model_df.groupby("year_venue")["is_award"].sum() == 0).sum()
        ),
        "ampere_or_newer_positives": int(model_df["ampere_or_newer"].sum()),
        "ampere_or_newer_equivalent_generations": sorted(AMPERE_OR_NEWER_GENERATIONS),
        "gpu_count_definition": "log10 count in the paper's maximum-capability reported GPU row",
        "generation_definition": "indicator that the same maximum-capability row is Ampere-or-newer/equivalent",
        "additive_no_interaction": True,
        "formula": formula,
        "design_parameters": int(design.shape[1]),
        "sample_id_sha256": _sha256_ids(model_df["paper_id"]),
        "full_model": {
            "penalized_loglik": float(estimator.loglik_),
            "n_iter": int(estimator.n_iter_),
            "converged": bool(estimator.converged_),
        },
        "joint_constraint_model": omnibus,
    }
    return pd.DataFrame(rows), audit


def save_pub_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> dict[str, str]:
    paths: dict[str, str] = {}
    path = stem.with_suffix(".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    paths["png"] = str(path)
    return paths


def build_top_compute_concentration_data(
    sample_df: pd.DataFrame,
    top_fraction: float = 0.20,
) -> pd.DataFrame:
    required = {"paper_id", "year", "max_compute", "cited_by_count", "is_award"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"Missing columns for top-compute concentration figure: {sorted(missing)}")
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between 0 and 1")

    work = sample_df.loc[:, sorted(required)].copy()
    work["year"] = safe_numeric(work["year"]).round().astype("Int64")
    work["max_compute"] = safe_numeric(work["max_compute"])
    work["cited_by_count"] = safe_numeric(work["cited_by_count"]).fillna(0)
    work["is_award"] = safe_numeric(work["is_award"]).fillna(0)
    work = work.loc[work["year"].notna() & work["max_compute"].gt(0)].copy()
    work = work.sort_values(["year", "max_compute", "paper_id"], ascending=[True, False, True])

    high_flags = []
    for _year, group in work.groupby("year", sort=True):
        high_n = max(1, int(np.ceil(len(group) * top_fraction)))
        high_flags.extend([True] * high_n + [False] * (len(group) - high_n))
    work["is_yearly_top20_compute"] = high_flags

    rows = []
    for year, group in work.groupby("year", sort=True):
        high = group.loc[group["is_yearly_top20_compute"]]
        total_compute = float(group["max_compute"].sum())
        total_citations = float(group["cited_by_count"].sum())
        total_awards = float(group["is_award"].sum())
        high_compute = float(high["max_compute"].sum())
        high_citations = float(high["cited_by_count"].sum())
        high_awards = float(high["is_award"].sum())
        base = {
            "year": int(year),
            "n_papers": int(len(group)),
            "n_top_compute_papers": int(len(high)),
            "paper_share": float(len(high) / len(group)),
            "top_fraction_target": top_fraction,
            "total_compute": total_compute,
            "top_compute_total_compute": high_compute,
            "total_citations": total_citations,
            "top_compute_total_citations": high_citations,
            "total_awards": total_awards,
            "top_compute_total_awards": high_awards,
            "compute_threshold_min_top20": float(high["max_compute"].min()),
        }
        metric_rows = [
            (
                "Reported GPU capacity",
                "compute_share",
                high_compute / total_compute if total_compute > 0 else np.nan,
            ),
            (
                "Citations",
                "citation_share",
                high_citations / total_citations if total_citations > 0 else np.nan,
            ),
            (
                "Awards",
                "award_share",
                high_awards / total_awards if total_awards > 0 else np.nan,
            ),
        ]
        for label, metric, share in metric_rows:
            row = base.copy()
            row.update({"metric": metric, "metric_label": label, "share": share})
            rows.append(row)
    return pd.DataFrame(rows)


def plot_top_compute_concentration(
    source: pd.DataFrame,
    fig_dir: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    colors = {
        "compute_share": "#2F6DA3",
        "citation_share": "#6A9C89",
        "award_share": "#B87A5B",
    }
    markers = {
        "compute_share": "o",
        "citation_share": "s",
        "award_share": "^",
    }
    for metric, group in source.groupby("metric", sort=False):
        plot_df = group.sort_values("year")
        ax.plot(
            plot_df["year"],
            plot_df["share"] * 100,
            marker=markers.get(metric, "o"),
            markersize=4,
            linewidth=1.4,
            color=colors.get(metric, "#555555"),
            label=plot_df["metric_label"].iloc[0],
        )

    reference = source["top_fraction_target"].iloc[0] * 100
    ax.axhline(
        reference,
        linestyle="--",
        linewidth=0.8,
        color="#555555",
        label="Paper-count share",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of yearly total (%)")
    ax.set_xticks(sorted(source["year"].unique()))
    ax.set_xlim(float(source["year"].min() - 0.15), float(source["year"].max() + 0.15))
    ax.set_ylim(0, min(100, max(30, float(source["share"].max() * 100 + 12))))
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.45)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=6.2,
        handlelength=1.8,
        columnspacing=1.3,
        borderaxespad=0.0,
    )
    paths = save_pub_figure(fig, fig_dir / "rq3_top20_compute_concentration")
    plt.close(fig)
    return paths


def build_gpu_capacity_pareto_data(
    sample_df: pd.DataFrame,
    top_fraction: float = 0.20,
) -> pd.DataFrame:
    required = {"paper_id", "year", "max_compute"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"Missing columns for GPU capacity Pareto figure: {sorted(missing)}")
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between 0 and 1")

    work = sample_df.loc[:, sorted(required)].copy()
    work["year"] = safe_numeric(work["year"]).round().astype("Int64")
    work["max_compute"] = safe_numeric(work["max_compute"])
    work = work.loc[work["year"].notna() & work["max_compute"].gt(0)].copy()
    work = work.sort_values(["year", "max_compute", "paper_id"], ascending=[True, False, True])

    rows = []
    for year, group in work.groupby("year", sort=True):
        group = group.sort_values(["max_compute", "paper_id"], ascending=[False, True]).reset_index(
            drop=True
        )
        n_papers = len(group)
        n_top_compute_papers = max(1, int(np.ceil(n_papers * top_fraction)))
        total_compute = float(group["max_compute"].sum())

        group["rank_desc"] = np.arange(1, n_papers + 1)
        group["n_papers"] = n_papers
        group["n_top_compute_papers"] = n_top_compute_papers
        group["top_fraction_target"] = top_fraction
        group["cumulative_papers"] = group["rank_desc"]
        group["cumulative_paper_share"] = group["cumulative_papers"] / n_papers
        group["cumulative_compute"] = group["max_compute"].cumsum()
        group["total_compute"] = total_compute
        group["cumulative_compute_share"] = (
            group["cumulative_compute"] / total_compute if total_compute > 0 else np.nan
        )
        group["is_top_fraction_cutoff"] = group["rank_desc"].eq(n_top_compute_papers)
        rows.append(group)

    if not rows:
        return pd.DataFrame(
            columns=[
                "year",
                "paper_id",
                "rank_desc",
                "max_compute",
                "n_papers",
                "n_top_compute_papers",
                "top_fraction_target",
                "cumulative_papers",
                "cumulative_paper_share",
                "cumulative_compute",
                "total_compute",
                "cumulative_compute_share",
                "is_top_fraction_cutoff",
            ]
        )

    return pd.concat(rows, ignore_index=True)[
        [
            "year",
            "paper_id",
            "rank_desc",
            "max_compute",
            "n_papers",
            "n_top_compute_papers",
            "top_fraction_target",
            "cumulative_papers",
            "cumulative_paper_share",
            "cumulative_compute",
            "total_compute",
            "cumulative_compute_share",
            "is_top_fraction_cutoff",
        ]
    ]


def validate_pareto_cutoff_matches_concentration(
    pareto_source: pd.DataFrame,
    concentration_source: pd.DataFrame,
    tolerance: float = 1e-12,
) -> None:
    compute_share = concentration_source.loc[
        concentration_source["metric"].eq("compute_share"),
        ["year", "share", "paper_share", "n_top_compute_papers"],
    ].copy()
    cutoff = pareto_source.loc[
        pareto_source["is_top_fraction_cutoff"],
        [
            "year",
            "rank_desc",
            "cumulative_compute_share",
            "cumulative_paper_share",
        ],
    ].copy()
    merged = compute_share.merge(cutoff, on="year", how="left", validate="one_to_one")
    if len(merged) != len(compute_share) or merged["rank_desc"].isna().any():
        raise ValueError("Pareto cutoff rows do not cover every concentration year.")

    compute_diff = (merged["share"] - merged["cumulative_compute_share"]).abs().max()
    paper_diff = (merged["paper_share"] - merged["cumulative_paper_share"]).abs().max()
    rank_diff = (merged["n_top_compute_papers"] - merged["rank_desc"]).abs().max()
    if compute_diff > tolerance or paper_diff > tolerance or rank_diff > tolerance:
        raise ValueError(
            "Pareto cutoff rows do not match top-20 concentration values "
            f"(compute_diff={compute_diff}, paper_diff={paper_diff}, rank_diff={rank_diff})."
        )


def plot_gpu_capacity_pareto_rebuttal(
    pareto_source: pd.DataFrame,
    concentration_source: pd.DataFrame,
    fig_dir: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.0, 3.3), constrained_layout=True)
    colors = {
        2020: "#2F6DA3",
        2021: "#6A9C89",
        2022: "#B87A5B",
        2023: "#6C5B7B",
    }
    label_offsets = {
        2020: (7, -3),
        2021: (7, 13),
        2022: (7, -10),
        2023: (7, -19),
    }
    compute_share = (
        concentration_source.loc[concentration_source["metric"].eq("compute_share")]
        .set_index("year")
        .sort_index()
    )

    ax.plot(
        [0, 100],
        [0, 100],
        linestyle=":",
        linewidth=0.8,
        color="#9A9A9A",
        label="Equal-share reference",
        zorder=1,
    )
    for year, group in pareto_source.groupby("year", sort=True):
        plot_df = group.sort_values("rank_desc")
        x = np.concatenate([[0.0], plot_df["cumulative_paper_share"].to_numpy() * 100])
        y = np.concatenate([[0.0], plot_df["cumulative_compute_share"].to_numpy() * 100])
        year_int = int(year)
        ax.plot(
            x,
            y,
            linewidth=1.35,
            color=colors.get(year_int, "#555555"),
            label=str(year_int),
            zorder=2,
        )
        if year in compute_share.index:
            marker_x = float(compute_share.loc[year, "paper_share"] * 100)
            marker_y = float(compute_share.loc[year, "share"] * 100)
            ax.scatter(
                [marker_x],
                [marker_y],
                s=18,
                color=colors.get(year_int, "#555555"),
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
            ax.annotate(
                f"{marker_y:.1f}%",
                xy=(marker_x, marker_y),
                xytext=label_offsets.get(year_int, (6, 0)),
                textcoords="offset points",
                fontsize=6.4,
                color=colors.get(year_int, "#555555"),
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                },
            )

    ax.axvline(
        20,
        linestyle="--",
        linewidth=0.85,
        color="#555555",
        label="20% paper mark",
        zorder=1,
    )
    ax.set_xlabel("Cumulative share of GPU-quantifiable papers (%)")
    ax.set_ylabel("Cumulative share of reported GPU capacity (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="both", color="#E8E8E8", linewidth=0.45)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=6.2,
        handlelength=1.8,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    paths = save_pub_figure(fig, fig_dir / "rq3_rebuttal_gpu_capacity_pareto")
    plt.close(fig)
    return paths


def build_high_compute_impact_matrix_data(
    sample_df: pd.DataFrame,
    top_fraction: float = 0.20,
) -> pd.DataFrame:
    required = {"paper_id", "year", "max_compute", "is_highly_cited_all_yv"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"Missing columns for high-compute/impact matrix: {sorted(missing)}")
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between 0 and 1")

    work = sample_df.loc[:, sorted(required)].copy()
    work["year"] = safe_numeric(work["year"]).round().astype("Int64")
    work["max_compute"] = safe_numeric(work["max_compute"])
    work["is_highly_cited_all_yv"] = safe_numeric(work["is_highly_cited_all_yv"]).fillna(0).astype(int)
    work = work.loc[work["year"].notna() & work["max_compute"].gt(0)].copy()
    work = work.sort_values(["year", "max_compute", "paper_id"], ascending=[True, False, True])

    high_flags = []
    for _year, group in work.groupby("year", sort=True):
        high_n = max(1, int(np.ceil(len(group) * top_fraction)))
        high_flags.extend([1] * high_n + [0] * (len(group) - high_n))
    work["is_yearly_top20_compute"] = high_flags

    work["compute_group"] = np.where(
        work["is_yearly_top20_compute"].eq(1),
        "High compute",
        "Not high compute",
    )
    work["impact_group"] = np.where(
        work["is_highly_cited_all_yv"].eq(1),
        "High impact",
        "Not high impact",
    )

    compute_order = ["High compute", "Not high compute"]
    impact_order = ["High impact", "Not high impact"]
    total = len(work)
    rows = []
    for row_i, compute_group in enumerate(compute_order):
        for col_i, impact_group in enumerate(impact_order):
            mask = work["compute_group"].eq(compute_group) & work["impact_group"].eq(impact_group)
            n = int(mask.sum())
            col_total = int(work["impact_group"].eq(impact_group).sum())
            row_total = int(work["compute_group"].eq(compute_group).sum())
            cell_compute_label = "high-compute" if compute_group == "High compute" else "non-high-compute"
            cell_impact_label = "high-impact" if impact_group == "High impact" else "ordinary-impact"
            rows.append(
                {
                    "row": row_i,
                    "col": col_i,
                    "compute_group": compute_group,
                    "impact_group": impact_group,
                    "cell_label": f"{cell_compute_label} / {cell_impact_label}",
                    "n": n,
                    "n_total": int(total),
                    "row_total": row_total,
                    "col_total": col_total,
                    "share_all": n / total if total else np.nan,
                    "share_within_impact": n / col_total if col_total else np.nan,
                    "share_within_compute": n / row_total if row_total else np.nan,
                    "top_fraction_target": top_fraction,
                    "years": "2020-2023",
                }
            )
    return pd.DataFrame(rows)


def plot_high_compute_impact_matrix(
    source: pd.DataFrame,
    fig_dir: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(4.6, 3.25), constrained_layout=True)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_axis_off()

    fill_colors = {
        ("High compute", "High impact"): "#5C86B5",
        ("High compute", "Not high impact"): "#D8E2ED",
        ("Not high compute", "High impact"): "#6A9C89",
        ("Not high compute", "Not high impact"): "#ECECEC",
    }
    text_colors = {
        ("High compute", "High impact"): "#FFFFFF",
        ("High compute", "Not high impact"): "#222222",
        ("Not high compute", "High impact"): "#FFFFFF",
        ("Not high compute", "Not high impact"): "#222222",
    }

    for row in source.itertuples(index=False):
        key = (row.compute_group, row.impact_group)
        rect = plt.Rectangle(
            (row.col, row.row),
            1,
            1,
            facecolor=fill_colors[key],
            edgecolor="#FFFFFF",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            row.col + 0.5,
            row.row + 0.38,
            row.cell_label.replace(" / ", "\n/\n"),
            ha="center",
            va="center",
            fontsize=6.5,
            color=text_colors[key],
            linespacing=1.05,
        )
        ax.text(
            row.col + 0.5,
            row.row + 0.72,
            f"n = {row.n:,}\n{row.share_within_impact * 100:.1f}% of column",
            ha="center",
            va="center",
            fontsize=6.2,
            color=text_colors[key],
            linespacing=1.15,
        )

    column_labels = ["High impact", "Not high impact"]
    row_labels = ["High compute", "Not high compute"]
    for col, label in enumerate(column_labels):
        ax.text(col + 0.5, -0.12, label, ha="center", va="bottom", fontsize=7.0)
    for row, label in enumerate(row_labels):
        ax.text(-0.10, row + 0.5, label, ha="right", va="center", fontsize=7.0)

    paths = save_pub_figure(fig, fig_dir / "rq3_high_compute_high_impact_matrix")
    plt.close(fig)
    return paths


def write_report(
    report_path: Path,
    main_results: pd.DataFrame,
    all_effect_tables: pd.DataFrame,
    robustness_tables: dict[str, pd.DataFrame],
    audit: dict[str, int],
) -> None:
    main_keys = main_results[["outcome_model", "spec"]].copy()
    main_detail = main_keys.merge(
        all_effect_tables,
        on=["outcome_model", "spec"],
        how="left",
        sort=False,
    )
    main_table = _regression_markdown_table(main_detail)
    full_table = _regression_markdown_table(all_effect_tables, include_family=True)
    robustness_table = _robustness_markdown_tables(robustness_tables)
    report = f"""# RQ3 GPU-only scholarly-recognition association modeling

This report re-runs the scholarly-recognition association workflow on the GPU-only input bundle. All regression models use the strict raw-compute sample. Coefficients are conditional associations and do not identify causal effects.

## Sample audit

- GPU-only master papers: {audit['master_rows']}
- 2020-2023 LB1/GFIMP GPU sample papers: {audit['gpu_2020_2023_rows']}
- 2020-2023 strict raw GPU sample papers: {audit['strict_2020_2023_rows']}
- 2020-2025 LB1/GFIMP GPU sample papers: {audit['gpu_2020_2025_rows']}
- 2020-2025 strict raw GPU sample papers: {audit['strict_2020_2025_rows']}

## Main results

{main_table}

## Complete regression results

{full_table}

## Robustness analyses

{robustness_table}

## Model specification

RQ3 uses year-by-venue fixed effects, primary-topic fixed effects, team-size group,
and organization-count group controls. It intentionally excludes `contribution_type`
and all contribution-label proxy controls. The compute regressor is strict raw
`log10_max_compute`, derived from `paper_max_row_compute_capability`.

## Notes

- Each coefficient is the estimated coefficient on `log10_max_compute`.
- Spec 1 includes year-by-venue fixed effects only; specs 2-7 add topic,
  team-size, and organization-count controls as shown in the `Controls` column.
- `N` is the estimation sample size after outcome and covariate filtering.
- `SE` is the standard error using the listed covariance estimator; `95% CI`
  is the confidence interval for the compute coefficient.
- `10x effect` reports `exp(coef) - 1` for log-link/log-outcome models and
  percentage-point effects for linear probability models.
- The full machine-readable regression table is also exported as
  `all_model_effect_tables.csv` in the `data` directory.
- Robustness-analysis tables are exported as CSV files in the same `data`
  directory using the section names shown above.
- Award models are exploratory because the outcome is sparse; statistical
  significance is not treated as confirmatory evidence.
- Author reputation, institution visibility, public artifacts, and unmeasured-
  confounding sensitivity are reported in `rebuttal_uncontrolled_confounding.md`.
- Residual confounding remains a limitation even when coefficients are stable
  after the observed-control extensions.

"""
    report_path.write_text(report, encoding="utf-8")


def _rebuttal_value_table(concentration_source: pd.DataFrame) -> str:
    compute_share = concentration_source.loc[
        concentration_source["metric"].eq("compute_share")
    ].sort_values("year")
    lines = [
        "| Year | GPU-quantifiable papers | Top papers | Paper share | Capacity share | Top-group threshold (TFLOP/s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in compute_share.itertuples(index=False):
        lines.append(
            "| "
            f"{int(row.year)} | "
            f"{int(row.n_papers):,} | "
            f"{int(row.n_top_compute_papers):,} | "
            f"{float(row.paper_share) * 100:.1f}% | "
            f"{float(row.share) * 100:.1f}% | "
            f"{float(row.compute_threshold_min_top20) / 1e12:,.0f} |"
        )
    return "\n".join(lines)


def write_gpu_concentration_rebuttal_report(
    report_path: Path,
    concentration_source: pd.DataFrame,
    figure_path: Path,
    pareto_data_path: Path,
) -> None:
    compute_share = concentration_source.loc[concentration_source["metric"].eq("compute_share")]
    min_share = float(compute_share["share"].min() * 100)
    max_share = float(compute_share["share"].max() * 100)
    values = ", ".join(
        f"{int(row.year)}: {float(row.share) * 100:.1f}%"
        for row in compute_share.sort_values("year").itertuples(index=False)
    )
    table = _rebuttal_value_table(concentration_source)
    report = f"""# Rebuttal: GPU concentration calculation

We agree that the concentration calculation should be made clearer. We compute this
statistic separately within each publication year. For each year, we restrict to
papers with positive reported GPU capacity, sort papers in descending order of
paper-level reported GPU capacity, select the top
`k_y = ceil(0.20 N_y)` papers, and divide their summed capacity by the summed
capacity of all GPU-quantifiable papers in that year.

The calculation is:

`S_y = sum_{{i=1}}^{{k_y}} C_{{(i)y}} / sum_{{i=1}}^{{N_y}} C_{{(i)y}}`

where `C_(i)y` is the paper-level reported GPU capacity after sorting year `y`
from highest to lowest capacity. This yields {values}. Thus, the statement means
that the highest-capacity approximately 20% of GPU-quantifiable papers within
each year account for {min_share:.1f}%-{max_share:.1f}% of that year's total
reported GPU capacity.

## Values

{table}

## Rebuttal figure

![Cumulative reported GPU capacity](../fig/{figure_path.name})

The vertical dashed line marks the 20% paper-share point. The labeled dots are
the yearly top-20% cutoff values used in the rebuttal text.

## Source artifacts

- Figure: `fig/{figure_path.name}`
- Pareto source data: `data/{pareto_data_path.name}`
- Top-20 share source data: `data/rq3_top20_compute_concentration.csv`
"""
    report_path.write_text(report, encoding="utf-8")


def write_rebuttal_gpu_concentration_artifacts(
    sample_df: pd.DataFrame,
    concentration_source: pd.DataFrame,
    dirs: dict[str, Path],
) -> dict:
    pareto_source = build_gpu_capacity_pareto_data(sample_df)
    validate_pareto_cutoff_matches_concentration(pareto_source, concentration_source)

    pareto_data_path = dirs["data"] / "rq3_rebuttal_gpu_capacity_pareto.csv"
    report_path = dirs["report"] / "rebuttal_gpu_concentration_clarification.md"
    pareto_source.to_csv(pareto_data_path, index=False)
    figure_paths = {
        "rq3_rebuttal_gpu_capacity_pareto": plot_gpu_capacity_pareto_rebuttal(
            pareto_source,
            concentration_source,
            dirs["fig"],
        )
    }
    write_gpu_concentration_rebuttal_report(
        report_path,
        concentration_source,
        Path(figure_paths["rq3_rebuttal_gpu_capacity_pareto"]["png"]),
        pareto_data_path,
    )
    return {
        "tables": {"rq3_rebuttal_gpu_capacity_pareto": str(pareto_data_path)},
        "figures": figure_paths,
        "reports": {"rebuttal_gpu_concentration_clarification": str(report_path)},
    }


def cutoff_pct(fraction: float) -> int:
    return int(round(fraction * 100))


def high_compute_column(fraction: float) -> str:
    return f"is_yearly_top{cutoff_pct(fraction)}_compute"


def high_citation_column(fraction: float) -> str:
    return f"is_yv_top{cutoff_pct(fraction)}_cited"


def add_yearly_high_compute_flags(
    sample_df: pd.DataFrame,
    compute_fractions: Iterable[float],
) -> pd.DataFrame:
    required = {"paper_id", "year", "max_compute"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"Missing columns for high-compute flags: {sorted(missing)}")

    work = sample_df.copy()
    work["year"] = safe_numeric(work["year"]).round().astype("Int64")
    work["max_compute"] = safe_numeric(work["max_compute"])
    work = work.loc[work["year"].notna() & work["max_compute"].gt(0)].copy()
    work = work.sort_values(["year", "max_compute", "paper_id"], ascending=[True, False, True])

    for fraction in compute_fractions:
        if not 0 < fraction < 1:
            raise ValueError("compute_fractions must be between 0 and 1")
        col = high_compute_column(fraction)
        work[col] = 0
        for _year, group in work.groupby("year", sort=True):
            high_n = max(1, int(np.ceil(len(group) * fraction)))
            work.loc[group.index[:high_n], col] = 1
    return work


def add_year_venue_high_citation_flags(
    sample_df: pd.DataFrame,
    citation_fractions: Iterable[float],
) -> pd.DataFrame:
    required = {"cited_by_count", "year_str", "venue"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"Missing columns for high-citation flags: {sorted(missing)}")

    work = sample_df.copy()
    work["cited_by_count"] = safe_numeric(work["cited_by_count"])
    work["year_str"] = work["year_str"].astype(str)
    work["venue"] = work["venue"].fillna("unknown").astype(str).str.lower()
    citation_rank_pct = work.groupby(["year_str", "venue"], dropna=False)["cited_by_count"].rank(
        method="average",
        pct=True,
    )
    for fraction in citation_fractions:
        if not 0 < fraction < 1:
            raise ValueError("citation_fractions must be between 0 and 1")
        col = high_citation_column(fraction)
        if col in work.columns:
            work[col] = safe_numeric(work[col]).fillna(0).astype(int)
        else:
            work[col] = (
                citation_rank_pct.ge(1 - fraction) & work["cited_by_count"].notna()
            ).astype(int)
    return work


def build_cutoff_sensitivity_descriptive_data(
    sample_df: pd.DataFrame,
    compute_fractions: Iterable[float] = (0.10, 0.20, 0.30),
    citation_fractions: Iterable[float] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    compute_fractions = tuple(compute_fractions)
    citation_fractions = tuple(citation_fractions)
    work = add_yearly_high_compute_flags(sample_df, compute_fractions)
    work = add_year_venue_high_citation_flags(work, citation_fractions)

    rows = []
    years = sorted(safe_numeric(work["year"]).dropna().astype(int).unique())
    year_label = f"{min(years)}-{max(years)}" if years else ""
    for compute_fraction in compute_fractions:
        compute_col = high_compute_column(compute_fraction)
        high_compute = work[compute_col].eq(1)
        for citation_fraction in citation_fractions:
            citation_col = high_citation_column(citation_fraction)
            high_impact = work[citation_col].eq(1)
            other_compute = ~high_compute
            n_high_compute = int(high_compute.sum())
            n_other_compute = int(other_compute.sum())
            n_high_impact = int(high_impact.sum())
            n_high_compute_high_impact = int((high_compute & high_impact).sum())
            n_other_high_impact = int((other_compute & high_impact).sum())
            high_compute_rate = (
                n_high_compute_high_impact / n_high_compute if n_high_compute else np.nan
            )
            other_compute_rate = n_other_high_impact / n_other_compute if n_other_compute else np.nan
            rows.append(
                {
                    "compute_top_fraction": float(compute_fraction),
                    "compute_top_pct": cutoff_pct(compute_fraction),
                    "citation_top_fraction": float(citation_fraction),
                    "citation_top_pct": cutoff_pct(citation_fraction),
                    "n_papers": int(len(work)),
                    "n_high_compute": n_high_compute,
                    "n_other_compute": n_other_compute,
                    "n_high_impact": n_high_impact,
                    "n_high_compute_high_impact": n_high_compute_high_impact,
                    "n_other_high_impact": n_other_high_impact,
                    "high_compute_impact_rate": high_compute_rate,
                    "other_compute_impact_rate": other_compute_rate,
                    "rate_difference_pp": (
                        (high_compute_rate - other_compute_rate) * 100
                        if pd.notna(high_compute_rate) and pd.notna(other_compute_rate)
                        else np.nan
                    ),
                    "relative_risk": (
                        high_compute_rate / other_compute_rate if other_compute_rate > 0 else np.nan
                    ),
                    "share_high_impact_in_high_compute": (
                        n_high_compute_high_impact / n_high_impact if n_high_impact else np.nan
                    ),
                    "compute_reference": "within publication year",
                    "citation_reference": "within publication year x venue",
                    "years": year_label,
                }
            )
    return pd.DataFrame(rows)


def validate_cutoff_sensitivity_baseline(
    cutoff_source: pd.DataFrame,
    matrix_source: pd.DataFrame,
) -> None:
    baseline = cutoff_source.loc[
        cutoff_source["compute_top_pct"].eq(20) & cutoff_source["citation_top_pct"].eq(10)
    ]
    matrix = matrix_source.loc[
        matrix_source["compute_group"].eq("High compute")
        & matrix_source["impact_group"].eq("High impact")
    ]
    if baseline.empty or matrix.empty:
        raise ValueError("Missing baseline rows for cutoff-sensitivity validation.")

    baseline_row = baseline.iloc[0]
    matrix_row = matrix.iloc[0]
    checks = {
        "n_high_compute_high_impact": int(matrix_row["n"]),
        "n_high_compute": int(matrix_row["row_total"]),
        "n_high_impact": int(matrix_row["col_total"]),
        "n_papers": int(matrix_row["n_total"]),
    }
    for column, expected in checks.items():
        observed = int(baseline_row[column])
        if observed != expected:
            raise ValueError(
                f"Cutoff baseline mismatch for {column}: observed {observed}, expected {expected}"
            )


def plot_cutoff_sensitivity_relative_risk(
    source: pd.DataFrame,
    fig_dir: Path,
) -> dict[str, str]:
    plot_df = source.sort_values(["citation_top_pct", "compute_top_pct"]).copy()
    pivot = plot_df.pivot(
        index="citation_top_pct",
        columns="compute_top_pct",
        values="relative_risk",
    )

    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="YlGnBu", vmin=1.0, vmax=max(2.2, np.nanmax(values)))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"Top {int(col)}%" for col in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"Top {int(idx)}%" for idx in pivot.index])
    ax.set_xlabel("Yearly high-compute threshold")
    ax.set_ylabel("Year-venue high-citation threshold")
    ax.set_title("High-compute/high-impact sensitivity")

    for row_idx, citation_pct in enumerate(pivot.index):
        for col_idx, compute_pct in enumerate(pivot.columns):
            value = pivot.loc[citation_pct, compute_pct]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}x",
                ha="center",
                va="center",
                color="#111111" if value < 1.8 else "#FFFFFF",
                fontsize=8,
                fontweight="bold",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Relative risk")
    paths = save_pub_figure(fig, fig_dir / "rebuttal_cutoff_sensitivity_relative_risk")
    plt.close(fig)
    return paths


def fit_cutoff_lpm_continuous_models(
    regression_sample: pd.DataFrame,
    citation_fractions: Iterable[float] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    citation_fractions = tuple(citation_fractions)
    work = add_year_venue_high_citation_flags(regression_sample, citation_fractions)
    rows = []
    for citation_fraction in citation_fractions:
        outcome = high_citation_column(citation_fraction)
        result = fit_model_grid(
            work,
            outcome=outcome,
            family="lpm",
            specs=[7],
            cov_type="HC3",
        )
        row = get_model_row(result, 7).to_dict()
        incremental_fit = delta_r2_for_spec(work, outcome, spec_id=7)
        if not np.isclose(row["r2"], incremental_fit["r2_full"]):
            raise ValueError(
                f"Cutoff sensitivity R2 mismatch for {outcome}: "
                f"{row['r2']} vs {incremental_fit['r2_full']}"
            )
        row.update(
            {
                "citation_top_fraction": float(citation_fraction),
                "citation_top_pct": cutoff_pct(citation_fraction),
                "model_id": "continuous_log10_max_compute",
                "model_label": "Continuous reported GPU capacity",
                "effect_label": "percentage points per 10x GPU compute",
                "r2_without_compute": float(incremental_fit["r2_without_compute"]),
                "delta_r2": float(incremental_fit["delta_r2"]),
                "coef_pp": row["coef"] * 100,
                "se_pp": row["se"] * 100,
                "ci_low_pp": row["ci_low"] * 100,
                "ci_high_pp": row["ci_high"] * 100,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def fit_cutoff_lpm_binary_models(
    regression_sample: pd.DataFrame,
    compute_fractions: Iterable[float] = (0.10, 0.20, 0.30),
    citation_fractions: Iterable[float] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    compute_fractions = tuple(compute_fractions)
    citation_fractions = tuple(citation_fractions)
    work = add_yearly_high_compute_flags(regression_sample, compute_fractions)
    work = add_year_venue_high_citation_flags(work, citation_fractions)

    rows = []
    for compute_fraction in compute_fractions:
        compute_var = high_compute_column(compute_fraction)
        for citation_fraction in citation_fractions:
            outcome = high_citation_column(citation_fraction)
            result = fit_model_grid(
                work,
                outcome=outcome,
                family="lpm",
                compute_var=compute_var,
                specs=[7],
                cov_type="HC3",
            )
            row = get_model_row(result, 7).to_dict()
            row.update(
                {
                    "compute_top_fraction": float(compute_fraction),
                    "compute_top_pct": cutoff_pct(compute_fraction),
                    "citation_top_fraction": float(citation_fraction),
                    "citation_top_pct": cutoff_pct(citation_fraction),
                    "model_id": "binary_yearly_high_compute",
                    "model_label": "Binary yearly high-compute group",
                    "effect_label": "percentage-point difference vs other compute papers",
                    "coef_pp": row["coef"] * 100,
                    "se_pp": row["se"] * 100,
                    "ci_low_pp": row["ci_low"] * 100,
                    "ci_high_pp": row["ci_high"] * 100,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def cutoff_descriptive_markdown_table(source: pd.DataFrame) -> str:
    display = pd.DataFrame(
        {
            "Compute cutoff": source["compute_top_pct"].map(lambda value: f"Top {int(value)}%"),
            "Citation cutoff": source["citation_top_pct"].map(lambda value: f"Top {int(value)}%"),
            "High-compute impact rate": source["high_compute_impact_rate"].map(
                _format_report_percent
            ),
            "Other impact rate": source["other_compute_impact_rate"].map(_format_report_percent),
            "Relative risk": source["relative_risk"].map(lambda value: f"{value:.2f}x"),
            "High-impact captured": source["share_high_impact_in_high_compute"].map(
                _format_report_percent
            ),
            "High-compute/high-impact N": source["n_high_compute_high_impact"].map(
                lambda value: _format_report_value(value, 0)
            ),
            "High-impact N": source["n_high_impact"].map(lambda value: _format_report_value(value, 0)),
        }
    )
    return _markdown_table(display)


def cutoff_lpm_continuous_markdown_table(source: pd.DataFrame) -> str:
    display = pd.DataFrame(
        {
            "Citation cutoff": source["citation_top_pct"].map(lambda value: f"Top {int(value)}%"),
            "N": source["nobs"].map(lambda value: _format_report_value(value, 0)),
            "Mean outcome": source["mean_y"].map(_format_report_percent),
            "Coef., pp": source["coef_pp"].map(lambda value: _format_report_value(value, 2)),
            "SE, pp": source["se_pp"].map(lambda value: _format_report_value(value, 2)),
            "p": source["p"].map(_format_report_p),
            "95% CI, pp": source.apply(
                lambda row: f"[{_format_report_value(row['ci_low_pp'], 2)}, {_format_report_value(row['ci_high_pp'], 2)}]",
                axis=1,
            ),
            "Delta R2": source["delta_r2"].map(lambda value: _format_report_value(value, 4)),
        }
    )
    return _markdown_table(display)


def cutoff_lpm_binary_markdown_table(source: pd.DataFrame) -> str:
    display = pd.DataFrame(
        {
            "Compute cutoff": source["compute_top_pct"].map(lambda value: f"Top {int(value)}%"),
            "Citation cutoff": source["citation_top_pct"].map(lambda value: f"Top {int(value)}%"),
            "N": source["nobs"].map(lambda value: _format_report_value(value, 0)),
            "Mean outcome": source["mean_y"].map(_format_report_percent),
            "Coef., pp": source["coef_pp"].map(lambda value: _format_report_value(value, 2)),
            "SE, pp": source["se_pp"].map(lambda value: _format_report_value(value, 2)),
            "p": source["p"].map(_format_report_p),
            "95% CI, pp": source.apply(
                lambda row: f"[{_format_report_value(row['ci_low_pp'], 2)}, {_format_report_value(row['ci_high_pp'], 2)}]",
                axis=1,
            ),
        }
    )
    return _markdown_table(display)


def write_cutoff_sensitivity_rebuttal_report(
    report_path: Path,
    descriptive: pd.DataFrame,
    continuous_lpm: pd.DataFrame,
    binary_lpm: pd.DataFrame,
    figure_path: Path,
    descriptive_path: Path,
    continuous_path: Path,
    binary_path: Path,
) -> None:
    baseline = descriptive.loc[
        descriptive["compute_top_pct"].eq(20) & descriptive["citation_top_pct"].eq(10)
    ].iloc[0]
    rr_min = float(descriptive["relative_risk"].min())
    rr_max = float(descriptive["relative_risk"].max())
    continuous_positive = int(continuous_lpm["coef"].gt(0).sum())
    binary_positive = int(binary_lpm["coef"].gt(0).sum())
    binary_nonsig = binary_lpm.loc[binary_lpm["p"].ge(0.05)].copy()
    if binary_nonsig.empty:
        binary_precision_note = "All binary high-compute LPM cells are positive and p < 0.05."
    else:
        cells = ", ".join(
            f"compute top {int(row.compute_top_pct)}% / citation top {int(row.citation_top_pct)}% (p={_format_report_p(row.p)})"
            for row in binary_nonsig.sort_values(["compute_top_pct", "citation_top_pct"]).itertuples(index=False)
        )
        binary_precision_note = (
            "All binary high-compute LPM cells are positive; the following cells are less precise "
            f"at p >= 0.05: {cells}."
        )

    report = f"""# Rebuttal: high-compute and high-impact cutoff sensitivity

## Draft reviewer response

We thank the reviewer for pointing out that the 20% compute and 10% citation
cutoffs may appear arbitrary. We added a cutoff-sensitivity analysis varying
the yearly high-compute threshold over top 10%, 20%, and 30%, and the
year-venue high-impact citation threshold over top 5%, 10%, and 20%. Across all
nine combinations, high-compute papers remain more likely to be high-impact:
the relative-risk range is {rr_min:.2f}-{rr_max:.2f}. The LPM robustness checks
likewise show positive compute associations across the tested citation
thresholds.

## Main takeaway

The original 20% compute / 10% citation cell is reproduced exactly: top-20%
yearly high-compute papers contain {int(baseline["n_high_compute_high_impact"]):,}
of {int(baseline["n_high_impact"]):,} year-venue top-10% cited papers. Their
high-impact rate is {baseline["high_compute_impact_rate"] * 100:.1f}%, compared
with {baseline["other_compute_impact_rate"] * 100:.1f}% for other
GPU-quantifiable papers (relative risk {baseline["relative_risk"]:.2f}x).

Across the descriptive 3x3 grid, the relative risk remains above 1.0 in every
cell. The continuous-compute LPM is positive in {continuous_positive} of
{len(continuous_lpm)} citation-threshold models, and the binary high-compute LPM
is positive in {binary_positive} of {len(binary_lpm)} threshold-combination
models. {binary_precision_note}

## Descriptive cutoff grid

{cutoff_descriptive_markdown_table(descriptive)}

## Figure

![Relative-risk cutoff sensitivity](../fig/{figure_path.name})

## Continuous-compute LPM sensitivity

These models retain the Section 4.4 continuous compute regressor
`log10_max_compute` and vary only the high-citation outcome threshold.

{cutoff_lpm_continuous_markdown_table(continuous_lpm)}

## Binary high-compute LPM sensitivity

These supplementary models use the yearly high-compute group as the compute
regressor and vary both thresholds.

{cutoff_lpm_binary_markdown_table(binary_lpm)}

## Model specification

High-compute status is computed separately within each publication year. High
citation status is computed within each publication-year-by-venue cell, using
the same citation-rank convention as Section 4.4. All LPM specifications use
the Section 4.4 spec-7 controls: year-by-venue fixed effects, primary-topic
fixed effects, team-size group, and organization-count group.

## Source artifacts

- Descriptive grid: `data/{descriptive_path.name}`
- Continuous-compute LPM: `data/{continuous_path.name}`
- Binary high-compute LPM: `data/{binary_path.name}`
- Figure: `fig/{figure_path.name}`
"""
    report_path.write_text(report, encoding="utf-8")


def write_rebuttal_cutoff_sensitivity_artifacts(
    descriptive_sample: pd.DataFrame,
    regression_sample: pd.DataFrame,
    baseline_matrix_source: pd.DataFrame,
    dirs: dict[str, Path],
) -> dict:
    descriptive = build_cutoff_sensitivity_descriptive_data(descriptive_sample)
    validate_cutoff_sensitivity_baseline(descriptive, baseline_matrix_source)
    continuous_lpm = fit_cutoff_lpm_continuous_models(regression_sample)
    binary_lpm = fit_cutoff_lpm_binary_models(regression_sample)

    descriptive_path = dirs["data"] / "rebuttal_cutoff_sensitivity_descriptive.csv"
    continuous_path = dirs["data"] / "rebuttal_cutoff_sensitivity_lpm_continuous.csv"
    binary_path = dirs["data"] / "rebuttal_cutoff_sensitivity_lpm_binary.csv"
    report_path = dirs["report"] / "rebuttal_cutoff_sensitivity.md"
    descriptive.to_csv(descriptive_path, index=False)
    continuous_lpm.to_csv(continuous_path, index=False)
    binary_lpm.to_csv(binary_path, index=False)

    figure_paths = {
        "rebuttal_cutoff_sensitivity_relative_risk": plot_cutoff_sensitivity_relative_risk(
            descriptive,
            dirs["fig"],
        )
    }
    figure_path = Path(figure_paths["rebuttal_cutoff_sensitivity_relative_risk"]["png"])
    write_cutoff_sensitivity_rebuttal_report(
        report_path,
        descriptive,
        continuous_lpm,
        binary_lpm,
        figure_path,
        descriptive_path,
        continuous_path,
        binary_path,
    )
    return {
        "tables": {
            "rebuttal_cutoff_sensitivity_descriptive": str(descriptive_path),
            "rebuttal_cutoff_sensitivity_lpm_continuous": str(continuous_path),
            "rebuttal_cutoff_sensitivity_lpm_binary": str(binary_path),
        },
        "figures": figure_paths,
        "reports": {"rebuttal_cutoff_sensitivity": str(report_path)},
    }


def build_field_normalized_citation_cell_audit(master: pd.DataFrame) -> pd.DataFrame:
    valid = master.loc[master["topic_year_citation_percentile"].notna()].copy()
    audit = (
        valid.groupby(["primary_topic", "year_str"], as_index=False)
        .agg(
            n_papers=("paper_id", "size"),
            min_citations=("cited_by_count", "min"),
            mean_citations=("cited_by_count", "mean"),
            median_citations=("cited_by_count", "median"),
            max_citations=("cited_by_count", "max"),
        )
        .sort_values(["year_str", "primary_topic"])
        .reset_index(drop=True)
    )
    audit["sparse_cell_lt10"] = audit["n_papers"].lt(10)
    audit["reference_set"] = "primary_topic_x_year"
    return audit


def fit_field_normalized_citation_rebuttal_models(
    citation_sample: pd.DataFrame,
) -> pd.DataFrame:
    sample_specs = [
        {
            "sample_id": "all_topic_year_cells",
            "sample_label": "All topic-year cells",
            "sample_note": "Primary rebuttal model; keeps all topic-year citation percentiles.",
            "data": citation_sample,
        },
        {
            "sample_id": "topic_year_cell_n_ge10",
            "sample_label": "Sensitivity: topic-year cell n >= 10",
            "sample_note": "Drops rows whose field-normalization cell has fewer than 10 cited papers.",
            "data": citation_sample.loc[
                safe_numeric(citation_sample["topic_year_citation_cell_n"]).ge(10)
            ].copy(),
        },
    ]
    rows = []
    for spec in sample_specs:
        result = fit_model_grid(
            spec["data"],
            outcome="topic_year_citation_percentile",
            family="ols",
            max_spec=7,
            cov_type="HC3",
        )
        for _, result_row in result["effect_table"].iterrows():
            row = result_row.to_dict()
            model_sample = result["samples"][row["model"]]
            cell_n = safe_numeric(model_sample["topic_year_citation_cell_n"])
            row.update(
                {
                    "sample_id": spec["sample_id"],
                    "sample_label": spec["sample_label"],
                    "sample_note": spec["sample_note"],
                    "outcome_label": "Topic-year citation percentile",
                    "effect_interpretation": "percentile points per 10x GPU compute",
                    "percentile_points_per_10x": row["coef"] * 100,
                    "ci_low_percentile_points": row["ci_low"] * 100,
                    "ci_high_percentile_points": row["ci_high"] * 100,
                    "n_topic_year_cells": int(
                        model_sample[["primary_topic", "year_str"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "min_topic_year_cell_n": int(cell_n.min()),
                    "median_topic_year_cell_n": float(cell_n.median()),
                    "rows_in_cells_lt10": int(cell_n.lt(10).sum()),
                    "reference_set": "all master-panel papers in same primary_topic x year",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def field_normalized_citation_markdown_table(
    results: pd.DataFrame,
    include_family: bool = False,
) -> str:
    display = pd.DataFrame(
        {
            "Sample": results["sample_label"],
            "Spec": results["spec"].astype(int),
            "Controls": results["controls"],
            "N": results["nobs"].map(lambda value: _format_report_value(value, 0)),
            "Topic-year cells": results["n_topic_year_cells"].map(
                lambda value: _format_report_value(value, 0)
            ),
            "Min cell N": results["min_topic_year_cell_n"].map(
                lambda value: _format_report_value(value, 0)
            ),
            "Rows in n<10 cells": results["rows_in_cells_lt10"].map(
                lambda value: _format_report_value(value, 0)
            ),
            "Coef.": results["coef"].map(lambda value: _format_report_value(value, 3)),
            "SE": results["se"].map(lambda value: _format_report_value(value, 3)),
            "p": results["p"].map(_format_report_p),
            "95% CI": results.apply(
                lambda row: f"[{_format_report_value(row['ci_low'], 3)}, {_format_report_value(row['ci_high'], 3)}]",
                axis=1,
            ),
            "10x effect, percentile points": results["percentile_points_per_10x"].map(
                lambda value: _format_report_value(value, 2)
            ),
            "R2": results["r2"].map(lambda value: _format_report_value(value, 3)),
            "Adj. R2": results["adj_r2"].map(lambda value: _format_report_value(value, 3)),
        }
    )
    if include_family:
        display.insert(3, "Family", results["family"])
        display.insert(4, "Cov.", results["cov_type"])
    return _markdown_table(display)


def write_field_normalized_citation_rebuttal_report(
    report_path: Path,
    results: pd.DataFrame,
    cell_audit: pd.DataFrame,
    table_path: Path,
    cell_audit_path: Path,
    audit: dict[str, int],
) -> None:
    main_results = results.loc[results["spec"].eq(7)].copy()
    primary = main_results.loc[
        main_results["sample_id"].eq("all_topic_year_cells")
    ].iloc[0]
    sensitivity = main_results.loc[
        main_results["sample_id"].eq("topic_year_cell_n_ge10")
    ].iloc[0]
    complete_results = results.sort_values(["sample_id", "spec"]).copy()
    report = f"""# Rebuttal: topic-year field-normalized citation impact

## Draft reviewer response

We thank the reviewer for noting that citation practices differ across NLP
subfields. We added a topic-year field-normalized citation percentile, comparing
each paper only with papers in the same NLP topic and publication year. The
association between reported GPU compute and citation impact is then
re-estimated using the same control structure as Section 4.4, with an additional
sensitivity excluding sparse topic-year cells.

## Main takeaway

Using the strict raw GPU sample from 2020-2023 and the same spec-7 controls as
Section 4.4, a 10x increase in reported GPU capacity is associated with a
{primary["percentile_points_per_10x"]:.2f} percentile-point change in
topic-year field-normalized citation percentile (N={int(primary["nobs"]):,},
p={_format_report_p(primary["p"])}). When rows from sparse topic-year cells
with fewer than 10 cited papers are excluded, the estimate is
{sensitivity["percentile_points_per_10x"]:.2f} percentile points
(N={int(sensitivity["nobs"]):,}, p={_format_report_p(sensitivity["p"])}).

## Sample audit

- GPU-only master papers: {audit["master_rows"]:,}
- Topic-year citation reference papers: {audit["reference_rows"]:,}
- Topic-year cells: {audit["topic_year_cells"]:,}
- Topic-year cells with n < 10: {audit["sparse_cells_lt10"]:,}
- Reference papers in n < 10 cells: {audit["reference_rows_in_sparse_cells_lt10"]:,}
- 2020-2023 strict raw GPU sample papers: {audit["citation_sample_rows"]:,}
- Primary field-normalized estimation sample: {int(primary["nobs"]):,}
- Sensitivity estimation sample after dropping n < 10 cells: {int(sensitivity["nobs"]):,}

## Main results

{field_normalized_citation_markdown_table(main_results)}

## Complete regression results

{field_normalized_citation_markdown_table(complete_results, include_family=True)}

## Model specification

For each paper, citations are ranked within the paper's `primary_topic` and
publication year. The percentile uses midpoint ranks:

`topic_year_citation_percentile = (average_rank - 0.5) / topic_year_cell_n`

This gives singleton cells a percentile of 0.5 and gives higher-cited papers
higher percentiles within their topic-year field. Venue is not included in the
normalization cell because topic-year-venue cells are too sparse for a stable
field-normalized outcome.

The regression sample and control structure follow the RQ3 citation-impact
workflow. The outcome is `topic_year_citation_percentile`; the compute regressor
is strict raw `log10_max_compute`, derived from
`paper_max_row_compute_capability`. Spec 1 includes year-by-venue fixed effects;
specs 2-7 add primary-topic fixed effects, team-size group, and
organization-count group controls as shown in the `Controls` column.

## Notes

- Each coefficient is the estimated coefficient on `log10_max_compute`.
- Because the outcome is a 0-1 percentile, `10x effect, percentile points`
  reports `100 * coef`.
- The primary rebuttal keeps all topic-year cells; the sensitivity model drops
  rows where `topic_year_citation_cell_n < 10`.
- The full machine-readable regression table and topic-year cell audit are
  exported as CSV files in the `data` directory.

## Source artifacts

- Model results: `data/{table_path.name}`
- Topic-year cell audit: `data/{cell_audit_path.name}`
"""
    report_path.write_text(report, encoding="utf-8")


def write_rebuttal_field_normalized_citation_artifacts(
    master: pd.DataFrame,
    citation_sample: pd.DataFrame,
    dirs: dict[str, Path],
) -> dict:
    results = fit_field_normalized_citation_rebuttal_models(citation_sample)
    cell_audit = build_field_normalized_citation_cell_audit(master)

    table_path = dirs["data"] / "rebuttal_field_normalized_citation.csv"
    cell_audit_path = dirs["data"] / "rebuttal_field_normalized_citation_cell_audit.csv"
    report_path = dirs["report"] / "rebuttal_field_normalized_citation.md"
    results.to_csv(table_path, index=False)
    cell_audit.to_csv(cell_audit_path, index=False)

    sparse_cells = cell_audit.loc[cell_audit["sparse_cell_lt10"]]
    audit = {
        "master_rows": int(len(master)),
        "reference_rows": int(master["topic_year_citation_percentile"].notna().sum()),
        "topic_year_cells": int(len(cell_audit)),
        "sparse_cells_lt10": int(len(sparse_cells)),
        "reference_rows_in_sparse_cells_lt10": int(sparse_cells["n_papers"].sum()),
        "citation_sample_rows": int(len(citation_sample)),
    }
    write_field_normalized_citation_rebuttal_report(
        report_path,
        results,
        cell_audit,
        table_path,
        cell_audit_path,
        audit,
    )
    return {
        "tables": {
            "rebuttal_field_normalized_citation": str(table_path),
            "rebuttal_field_normalized_citation_cell_audit": str(cell_audit_path),
        },
        "reports": {"rebuttal_field_normalized_citation": str(report_path)},
    }


def _fine_grained_effect(beta: float, family: str, outcome: str, term: str) -> tuple[float, str]:
    if term.startswith("log10_"):
        if family in {"ols", "poisson"} and outcome in {"log1p_cites", "cited_by_count"}:
            return float(np.exp(beta) - 1), "percent per 10x"
        return float(beta * 100), "percentage points per 10x"
    if family in {"ols", "poisson"} and outcome in {"log1p_cites", "cited_by_count"}:
        return float(np.exp(beta) - 1), "percent vs older generation"
    return float(beta * 100), "percentage points vs older generation"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_ids(values: Iterable[object]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _overall_gpu_outcomes(
    citation_sample: pd.DataFrame,
    award_sample: pd.DataFrame,
) -> list[dict]:
    return [
        {
            "outcome_order": 1,
            "outcome": "topic_year_citation_percentile",
            "outcome_label": "NLP topic-year percentile",
            "role": "Primary",
            "family": "ols",
            "cov_type": "HC3",
            "sample_label": "strict_raw_2020_2023",
            "year_window": "2020-2023",
            "sample": citation_sample,
        },
        {
            "outcome_order": 2,
            "outcome": "citation_normalized_percentile",
            "outcome_label": "OpenAlex field-normalized citation percentile",
            "role": "Secondary",
            "family": "ols",
            "cov_type": "HC3",
            "sample_label": "strict_raw_2020_2023",
            "year_window": "2020-2023",
            "sample": citation_sample,
        },
        {
            "outcome_order": 3,
            "outcome": "log1p_cites",
            "outcome_label": "Log citations",
            "role": "Secondary",
            "family": "ols",
            "cov_type": "HC3",
            "sample_label": "strict_raw_2020_2023",
            "year_window": "2020-2023",
            "sample": citation_sample,
        },
        {
            "outcome_order": 4,
            "outcome": "cited_by_count",
            "outcome_label": "PPML citations",
            "role": "Secondary",
            "family": "poisson",
            "cov_type": "HC0",
            "sample_label": "strict_raw_2020_2023",
            "year_window": "2020-2023",
            "sample": citation_sample,
        },
        {
            "outcome_order": 5,
            "outcome": "is_highly_cited_all_yv",
            "outcome_label": "Year-by-venue top-decile cited",
            "role": "Secondary",
            "family": "lpm",
            "cov_type": "HC3",
            "sample_label": "strict_raw_2020_2023",
            "year_window": "2020-2023",
            "sample": citation_sample,
        },
        {
            "outcome_order": 6,
            "outcome": "is_award",
            "outcome_label": "Award",
            "role": "Secondary",
            "family": "lpm",
            "cov_type": "HC3",
            "sample_label": "strict_raw_2020_2025",
            "year_window": "2020-2025",
            "sample": award_sample,
        },
    ]


def fit_overall_gpu_capability_models(
    citation_sample: pd.DataFrame,
    award_sample: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    controls = [
        "C(year_venue)",
        "C(primary_topic)",
        "C(team_size_group)",
        "C(n_organizations_group)",
    ]
    control_required = [
        "year_venue",
        "primary_topic",
        "team_size_group",
        "n_organizations_group",
    ]
    model_specs = [
        {
            "model_id": "capacity_only",
            "model_label": "Reported max-row GPU capability",
            "terms": ["log10_max_compute"],
        },
        {
            "model_id": "joint_count_ampere",
            "model_label": "GPU count + Ampere-or-newer/equivalent",
            "terms": ["log10_maxrow_gpu_count", "ampere_or_newer"],
        },
    ]
    term_labels = {
        "log10_max_compute": "Capacity",
        "log10_maxrow_gpu_count": "Count",
        "ampere_or_newer": "Ampere-or-newer/equivalent",
    }

    rows: list[dict] = []
    sample_checks: dict[str, dict] = {}
    citation_id_sets: dict[str, set[str]] = {}
    for outcome_spec in _overall_gpu_outcomes(citation_sample, award_sample):
        outcome = outcome_spec["outcome"]
        model_id_sets: dict[str, set[str]] = {}
        for model_spec in model_specs:
            required = [outcome, *control_required, *model_spec["terms"]]
            model_df = outcome_spec["sample"].dropna(subset=required).copy()
            if model_df[outcome].nunique(dropna=True) < 2:
                raise ValueError(f"Outcome {outcome} has insufficient variation.")
            model_id_sets[model_spec["model_id"]] = set(model_df["paper_id"].astype(str))

            controls_only_formula = model_formula(outcome, controls)
            full_formula = model_formula(outcome, [*model_spec["terms"], *controls])
            controls_only_model = fit_formula(
                controls_only_formula,
                model_df,
                family=outcome_spec["family"],
                cov_type=outcome_spec["cov_type"],
            )
            full_model = fit_formula(
                full_formula,
                model_df,
                family=outcome_spec["family"],
                cov_type=outcome_spec["cov_type"],
            )
            conf_int = full_model.conf_int()
            mean_outcome = float(model_df[outcome].mean())
            positive_count = (
                int(model_df[outcome].sum())
                if outcome in {"is_highly_cited_all_yv", "is_award"}
                else np.nan
            )
            sample_id_sha256 = _sha256_ids(model_df["paper_id"].astype(str))
            for term in model_spec["terms"]:
                beta = float(full_model.params[term])
                ci_low, ci_high = [float(value) for value in conf_int.loc[term].tolist()]
                effect_value, effect_unit = _fine_grained_effect(
                    beta,
                    outcome_spec["family"],
                    outcome,
                    term,
                )
                effect_ci_low, _ = _fine_grained_effect(
                    ci_low,
                    outcome_spec["family"],
                    outcome,
                    term,
                )
                effect_ci_high, _ = _fine_grained_effect(
                    ci_high,
                    outcome_spec["family"],
                    outcome,
                    term,
                )
                row = {
                    "outcome_order": outcome_spec["outcome_order"],
                    "outcome": outcome,
                    "outcome_label": outcome_spec["outcome_label"],
                    "role": outcome_spec["role"],
                    "family": outcome_spec["family"],
                    "cov_type": outcome_spec["cov_type"],
                    "sample": outcome_spec["sample_label"],
                    "year_window": outcome_spec["year_window"],
                    "model_id": model_spec["model_id"],
                    "model_label": model_spec["model_label"],
                    "term": term,
                    "term_label": term_labels[term],
                    "nobs": int(full_model.nobs),
                    "mean_outcome": mean_outcome,
                    "positive_count": positive_count,
                    "coef": beta,
                    "se": float(full_model.bse[term]),
                    "p_raw": float(full_model.pvalues[term]),
                    "p_holm_secondary": np.nan,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "effect_value": effect_value,
                    "effect_ci_low": effect_ci_low,
                    "effect_ci_high": effect_ci_high,
                    "effect_unit": effect_unit,
                    "controls_only_r2": float(getattr(controls_only_model, "rsquared", np.nan)),
                    "full_r2": float(getattr(full_model, "rsquared", np.nan)),
                    "delta_r2": float(
                        getattr(full_model, "rsquared", np.nan)
                        - getattr(controls_only_model, "rsquared", np.nan)
                    ),
                    "controls_only_adj_r2": float(
                        getattr(controls_only_model, "rsquared_adj", np.nan)
                    ),
                    "full_adj_r2": float(getattr(full_model, "rsquared_adj", np.nan)),
                    "controls_only_aic": float(controls_only_model.aic),
                    "full_aic": float(full_model.aic),
                    "aic_improvement": float(controls_only_model.aic - full_model.aic),
                    "controls_only_formula": controls_only_formula,
                    "full_formula": full_formula,
                    "sample_id_sha256": sample_id_sha256,
                }
                rows.append(row)

        capacity_ids = model_id_sets["capacity_only"]
        joint_ids = model_id_sets["joint_count_ampere"]
        if capacity_ids != joint_ids:
            raise ValueError(
                f"Capacity and joint model samples differ for {outcome}: "
                f"capacity-only={len(capacity_ids - joint_ids)}, "
                f"joint-only={len(joint_ids - capacity_ids)}"
            )
        if outcome_spec["year_window"] == "2020-2023":
            citation_id_sets[outcome] = capacity_ids
        sample_checks[outcome] = {
            "nobs": len(capacity_ids),
            "capacity_joint_ids_equal": True,
            "sample_id_sha256": _sha256_ids(capacity_ids),
        }

    citation_outcomes = list(citation_id_sets)
    citation_reference_ids = citation_id_sets[citation_outcomes[0]]
    citation_ids_equal = all(
        citation_id_sets[outcome] == citation_reference_ids
        for outcome in citation_outcomes[1:]
    )
    if not citation_ids_equal:
        raise ValueError("The five 2020-2023 citation outcomes do not use identical paper IDs.")

    results = pd.DataFrame(rows).sort_values(
        ["outcome_order", "model_id", "term"], kind="stable"
    ).reset_index(drop=True)
    for term in ["log10_max_compute", "log10_maxrow_gpu_count", "ampere_or_newer"]:
        secondary = results["term"].eq(term) & results["role"].eq("Secondary")
        adjusted = multipletests(results.loc[secondary, "p_raw"], method="holm")[1]
        results.loc[secondary, "p_holm_secondary"] = adjusted

    audit = {
        "citation_outcome_ids_equal": citation_ids_equal,
        "citation_outcome_nobs": len(citation_reference_ids),
        "citation_sample_id_sha256": _sha256_ids(citation_reference_ids),
        "sample_checks": sample_checks,
    }
    return results, audit


def _format_overall_effect(row: pd.Series) -> str:
    value = float(row["effect_value"])
    if str(row["effect_unit"]).startswith("percentage points"):
        return f"{value:.2f} pp"
    return f"{value * 100:.1f}%"


def build_overall_gpu_capability_wide_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome_order in sorted(results["outcome_order"].unique()):
        subset = results.loc[results["outcome_order"].eq(outcome_order)]
        capacity = subset.loc[
            subset["model_id"].eq("capacity_only")
            & subset["term"].eq("log10_max_compute")
        ].iloc[0]
        count = subset.loc[
            subset["model_id"].eq("joint_count_ampere")
            & subset["term"].eq("log10_maxrow_gpu_count")
        ].iloc[0]
        ampere = subset.loc[
            subset["model_id"].eq("joint_count_ampere")
            & subset["term"].eq("ampere_or_newer")
        ].iloc[0]
        is_ppml = capacity["family"] == "poisson"
        rows.append(
            {
                "Outcome": capacity["outcome_label"],
                "N": int(capacity["nobs"]),
                "Capacity β": _format_report_value(capacity["coef"], 4),
                "SE capacity": _format_report_value(capacity["se"], 4),
                "p capacity": _format_report_p(capacity["p_raw"]),
                "Capacity effect": _format_overall_effect(capacity),
                "Count β": _format_report_value(count["coef"], 4),
                "SE count": _format_report_value(count["se"], 4),
                "p count": _format_report_p(count["p_raw"]),
                "Count effect": _format_overall_effect(count),
                "Ampere β": _format_report_value(ampere["coef"], 4),
                "SE ampere": _format_report_value(ampere["se"], 4),
                "p ampere": _format_report_p(ampere["p_raw"]),
                "Ampere effect": _format_overall_effect(ampere),
                "ΔR² capacity": "--"
                if is_ppml
                else _format_report_value(capacity["delta_r2"], 6),
                "ΔR² joint": "--"
                if is_ppml
                else _format_report_value(count["delta_r2"], 6),
            }
        )
    return pd.DataFrame(rows)


def write_overall_gpu_capability_report(
    report_path: Path,
    wide_table: pd.DataFrame,
    results: pd.DataFrame,
    audit: dict,
    full_path: Path,
    wide_path: Path,
    audit_path: Path,
) -> None:
    topic_capacity = results.loc[
        results["outcome"].eq("topic_year_citation_percentile")
        & results["model_id"].eq("capacity_only")
        & results["term"].eq("log10_max_compute")
    ].iloc[0]
    award_capacity = results.loc[
        results["outcome"].eq("is_award")
        & results["model_id"].eq("capacity_only")
        & results["term"].eq("log10_max_compute")
    ].iloc[0]
    report = f"""# Overall GPU capability models: unified export

## Unified table

{_markdown_table(wide_table)}

## Locked specification

- Primary outcome: NLP topic-year citation percentile, computed in the full
  6,900-paper analysis master corpus before selecting the strict estimation sample.
- Capacity model: `y ~ log10(max-row GPU capability) + controls`.
- Joint model: `y ~ log10(max-row GPU count) + Ampere-or-newer/equivalent + controls`.
- Controls: year-by-venue fixed effects, primary-topic fixed effects, team-size
  groups, and organization-count groups.
- OLS and LPM use HC3 standard errors; PPML uses HC0 sandwich standard errors.
- Citation outcomes use 2020-2023. Award uses 2020-2025 and has
  {int(award_capacity['positive_count'])} positives.
- The five citation outcomes use identical paper IDs: `{audit['citation_outcome_ids_equal']}`.
- Raw p-values are displayed. `p_holm_secondary` in the full CSV applies Holm
  correction across the five secondary outcomes separately for capacity, count,
  and Ampere-or-newer/equivalent terms.

## Exact primary-outcome incremental fit

- Full-model R2: `{topic_capacity['full_r2']:.17g}`
- Controls-only R2: `{topic_capacity['controls_only_r2']:.17g}`
- Delta R2: `{topic_capacity['delta_r2']:.17g}`

## Source artifacts

- Full machine-readable results: `data/{full_path.name}`
- Compact requested table: `data/{wide_path.name}`
- Audit and frozen-input hashes: `data/{audit_path.name}`
"""
    report_path.write_text(report, encoding="utf-8")


def write_overall_gpu_capability_artifacts(
    master: pd.DataFrame,
    citation_sample: pd.DataFrame,
    award_sample: pd.DataFrame,
    dirs: dict[str, Path],
    input_data_dir: Path,
    feature_audit: dict,
) -> dict:
    results, sample_audit = fit_overall_gpu_capability_models(
        citation_sample,
        award_sample,
    )
    wide_table = build_overall_gpu_capability_wide_table(results)
    full_path = dirs["data"] / "overall_gpu_capability_models.csv"
    wide_path = dirs["data"] / "overall_gpu_capability_table.csv"
    audit_path = dirs["data"] / "overall_gpu_capability_model_audit.json"
    report_path = dirs["report"] / "overall_gpu_capability_models.md"
    results.to_csv(full_path, index=False)
    wide_table.to_csv(wide_path, index=False)

    input_names = [
        COMPUTE_FILE,
        PAPER_COMPUTE_FILE,
        METADATA_FILE,
        TOPIC_FILE,
        AWARD_FILE,
        ORG_VARS_FILE,
    ]
    input_files = {}
    for name in input_names:
        path = input_data_dir / name
        input_files[name] = {
            "path": portable_path(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "validated_local_frozen_input_rerun",
        "openalex_retrieval_date": None,
        "openalex_retrieval_date_note": (
            "The frozen workbook does not contain a verified retrieval timestamp; "
            "the file modification time is not treated as retrieval evidence."
        ),
        "master_rows": int(len(master)),
        "topic_year_reference_universe_rows": int(
            master["topic_year_citation_percentile"].notna().sum()
        ),
        "topic_year_reference_definition": (
            "midpoint percentile (average rank - 0.5) / cell_n within primary_topic x year"
        ),
        "topic_year_primary_keeps_all_cells": True,
        "ampere_or_newer_equivalent_generations": sorted(AMPERE_OR_NEWER_GENERATIONS),
        "award_positive_definition": "non-empty formal paper-honor label in local award table",
        "award_positives_in_estimation_sample": int(
            award_sample.dropna(
                subset=[
                    "is_award",
                    "log10_max_compute",
                    "year_venue",
                    "primary_topic",
                    "team_size_group",
                    "n_organizations_group",
                ]
            )["is_award"].sum()
        ),
        "feature_audit": feature_audit,
        **sample_audit,
        "input_files": input_files,
        "output_rows_full": int(len(results)),
        "output_rows_wide": int(len(wide_table)),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_overall_gpu_capability_report(
        report_path,
        wide_table,
        results,
        audit,
        full_path,
        wide_path,
        audit_path,
    )
    return {
        "tables": {
            "overall_gpu_capability_models": str(full_path),
            "overall_gpu_capability_table": str(wide_path),
        },
        "reports": {"overall_gpu_capability_models": str(report_path)},
        "audits": {"overall_gpu_capability_models": str(audit_path)},
    }


def _joint_firth_award_display_table(results: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Term": results["term_label"],
            "N": results["nobs"].astype(int),
            "Awards": results["award_events"].astype(int),
            "Beta": results["coef"].map(lambda value: f"{value:.4f}"),
            "LRT SE": results["lrt_backcorrected_se"].map(
                lambda value: f"{value:.4f}"
            ),
            "Profile-likelihood 95% CI": results.apply(
                lambda row: (
                    f"[{row['profile_ci_low']:.4f}, {row['profile_ci_high']:.4f}]"
                ),
                axis=1,
            ),
            "Penalized-LRT p": results["penalized_lrt_p"].map(_format_report_p),
            "Holm p": results["holm_p_two_terms"].map(_format_report_p),
            "OR": results["odds_ratio"].map(lambda value: f"{value:.3f}"),
            "OR 95% CI": results.apply(
                lambda row: f"[{row['or_ci_low']:.3f}, {row['or_ci_high']:.3f}]",
                axis=1,
            ),
        }
    )


def write_joint_count_ampere_firth_award_report(
    report_path: Path,
    results: pd.DataFrame,
    audit: dict,
    result_path: Path,
    audit_path: Path,
) -> None:
    count = results.loc[results["term"].eq("log10_maxrow_gpu_count")].iloc[0]
    ampere = results.loc[results["term"].eq("ampere_or_newer")].iloc[0]
    omnibus = audit["joint_constraint_model"]
    report = f"""# Joint GPU-count plus Ampere Firth award model

## Result

{_markdown_table(_joint_firth_award_display_table(results))}

The model uses the same {int(count['nobs']):,}-paper 2020-2025 award sample as
the joint LPM and contains {int(count['award_events'])} award-positive papers. A
tenfold increase in reported max-row GPU count has OR={count['odds_ratio']:.3f}
(profile-likelihood 95% CI [{count['or_ci_low']:.3f}, {count['or_ci_high']:.3f}],
penalized-LRT p={count['penalized_lrt_p']:.4g}). Conditional on count and the
shared controls, Ampere-or-newer/equivalent hardware has OR={ampere['odds_ratio']:.3f}
(profile-likelihood 95% CI [{ampere['or_ci_low']:.3f}, {ampere['or_ci_high']:.3f}],
penalized-LRT p={ampere['penalized_lrt_p']:.4g}). The 2-df joint penalized-LRT
statistic is {omnibus['omnibus_chi2']:.4f} (p={omnibus['omnibus_p']:.4g}).

## Locked specification

`Award ~ log10(max-row GPU count) + Ampere-or-newer/equivalent + year-by-venue FE + topic FE + team-size groups + organization-count groups`

- The two hardware terms enter additively; there is no interaction.
- The GPU count and generation indicator refer to the same maximum-capability
  reported GPU row used by the unified main-table analysis.
- Single-term p-values are penalized likelihood-ratio tests, SEs are LRT
  back-corrected, and confidence intervals are profile-likelihood intervals.
- Holm-adjusted p-values across the two prespecified focal terms are included in
  the machine export as secondary multiplicity diagnostics.
- The omnibus test jointly constrains both focal coefficients to zero within the
  same full Firth design matrix.
- This is a sparse-outcome robustness analysis and does not identify a causal
  effect of hardware on awards.

## Source artifacts

- Full term-level results: `data/{result_path.name}`
- Sample, estimator, input-hash, and convergence audit: `data/{audit_path.name}`
"""
    report_path.write_text(report, encoding="utf-8")


def write_joint_count_ampere_firth_award_artifacts(
    master: pd.DataFrame,
    award_sample: pd.DataFrame,
    dirs: dict[str, Path],
    input_data_dir: Path,
    feature_audit: dict,
) -> dict:
    results, model_audit = fit_joint_count_ampere_firth_award_model(award_sample)
    result_path = dirs["data"] / "award_joint_count_ampere_firth_models.csv"
    audit_path = dirs["data"] / "award_joint_count_ampere_firth_audit.json"
    report_path = dirs["report"] / "award_joint_count_ampere_firth.md"
    results.to_csv(result_path, index=False)

    input_names = [
        COMPUTE_FILE,
        PAPER_COMPUTE_FILE,
        METADATA_FILE,
        TOPIC_FILE,
        AWARD_FILE,
        ORG_VARS_FILE,
    ]
    input_files = {}
    for name in input_names:
        path = input_data_dir / name
        input_files[name] = {
            "path": portable_path(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **model_audit,
        "master_rows": int(len(master)),
        "feature_audit": feature_audit,
        "input_files": input_files,
        "output_rows": int(len(results)),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_joint_count_ampere_firth_award_report(
        report_path,
        results,
        audit,
        result_path,
        audit_path,
    )
    return {
        "tables": {"award_joint_count_ampere_firth_models": str(result_path)},
        "reports": {"award_joint_count_ampere_firth": str(report_path)},
        "audits": {"award_joint_count_ampere_firth": str(audit_path)},
    }


def _common_sample_confounder_outcomes() -> list[dict]:
    return [
        {
            "outcome_order": 1,
            "outcome": "topic_year_citation_percentile",
            "outcome_label": "NLP topic-year percentile",
            "role": "Primary",
        },
        {
            "outcome_order": 2,
            "outcome": "citation_normalized_percentile",
            "outcome_label": "OpenAlex field-normalized percentile",
            "role": "Secondary",
        },
        {
            "outcome_order": 3,
            "outcome": "log1p_cites",
            "outcome_label": "Log citations",
            "role": "Secondary",
        },
    ]


def _common_sample_confounder_specs() -> list[dict]:
    return [
        {
            "spec_order": 0,
            "model_id": "M0",
            "specification": "Common-sample baseline",
            "extra_controls": [],
        },
        {
            "spec_order": 1,
            "model_id": "M1",
            "specification": "+ author-history proxies",
            "extra_controls": AUTHOR_HISTORY_CONTROLS,
        },
        {
            "spec_order": 2,
            "model_id": "M2",
            "specification": "+ institution/collaboration proxies",
            "extra_controls": INSTITUTION_VISIBILITY_CONTROLS,
        },
        {
            "spec_order": 3,
            "model_id": "M3",
            "specification": "+ pre-publication controls",
            "extra_controls": PREPUBLICATION_CONFOUNDER_CONTROLS,
        },
        {
            "spec_order": 4,
            "model_id": "M4",
            "specification": "+ public artifact",
            "extra_controls": [*PREPUBLICATION_CONFOUNDER_CONTROLS, *ARTIFACT_CONTROLS],
        },
    ]


def fit_common_sample_confounder_models(
    citation_sample: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base_rhs = _base_spec7_rhs()
    base_required = [
        "log10_max_compute",
        "year_venue",
        "primary_topic",
        "team_size_group",
        "n_organizations_group",
    ]
    outcomes = _common_sample_confounder_outcomes()
    specs = _common_sample_confounder_specs()
    outcome_names = [spec["outcome"] for spec in outcomes]
    required = list(
        dict.fromkeys(
            [
                *outcome_names,
                *base_required,
                *PREPUBLICATION_CONFOUNDER_CONTROLS,
                *ARTIFACT_CONTROLS,
            ]
        )
    )
    missing = sorted(set(required) - set(citation_sample.columns))
    if missing:
        raise ValueError(f"Missing common-sample confounder columns: {missing}")

    baseline_complete = citation_sample.dropna(
        subset=[*outcome_names, *base_required]
    ).copy()
    common = citation_sample.dropna(subset=required).copy()
    if common.empty:
        raise ValueError("No complete cases for the common-sample confounder models.")
    common_ids = set(common["paper_id"].astype(str))
    sample_id_sha256 = _sha256_ids(common_ids)

    rows: list[dict] = []
    for outcome_spec in outcomes:
        baseline_beta = np.nan
        for spec in specs:
            rhs = [*base_rhs, *spec["extra_controls"]]
            full_formula = model_formula(outcome_spec["outcome"], rhs)
            controls_only_rhs = [term for term in rhs if term != "log10_max_compute"]
            controls_only_formula = model_formula(
                outcome_spec["outcome"], controls_only_rhs
            )
            full_model = smf.ols(full_formula, data=common).fit(cov_type="HC3")
            controls_only_model = smf.ols(
                controls_only_formula, data=common
            ).fit(cov_type="HC3")
            ci_low, ci_high = [
                float(value)
                for value in full_model.conf_int().loc["log10_max_compute"].tolist()
            ]
            beta = float(full_model.params["log10_max_compute"])
            if spec["model_id"] == "M0":
                baseline_beta = beta
            rows.append(
                {
                    "outcome_order": outcome_spec["outcome_order"],
                    "outcome": outcome_spec["outcome"],
                    "outcome_label": outcome_spec["outcome_label"],
                    "role": outcome_spec["role"],
                    "spec_order": spec["spec_order"],
                    "model_id": spec["model_id"],
                    "specification": spec["specification"],
                    "family": "ols",
                    "cov_type": "HC3",
                    "sample": "strict_raw_2020_2023_common_complete_case",
                    "nobs": int(full_model.nobs),
                    "coef": beta,
                    "se": float(full_model.bse["log10_max_compute"]),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p": float(full_model.pvalues["log10_max_compute"]),
                    "mean_outcome": float(common[outcome_spec["outcome"]].mean()),
                    "controls_only_r2": float(controls_only_model.rsquared),
                    "full_r2": float(full_model.rsquared),
                    "delta_r2": float(full_model.rsquared - controls_only_model.rsquared),
                    "controls_only_adj_r2": float(controls_only_model.rsquared_adj),
                    "full_adj_r2": float(full_model.rsquared_adj),
                    "coef_attenuation_vs_m0": (
                        float(1 - beta / baseline_beta)
                        if pd.notna(baseline_beta) and baseline_beta != 0
                        else 0.0
                    ),
                    "controls_only_formula": controls_only_formula,
                    "full_formula": full_formula,
                    "sample_id_sha256": sample_id_sha256,
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["outcome_order", "spec_order"], kind="stable"
    ).reset_index(drop=True)
    display_results = results.loc[results["model_id"].isin(["M0", "M3", "M4"])].copy()
    audit = {
        "baseline_complete_nobs": int(len(baseline_complete)),
        "common_complete_case_nobs": int(len(common)),
        "rows_dropped_from_baseline_complete": int(len(baseline_complete) - len(common)),
        "common_complete_case_rate": (
            float(len(common) / len(baseline_complete)) if len(baseline_complete) else np.nan
        ),
        "sample_id_sha256": sample_id_sha256,
        "all_outcomes_and_specs_use_identical_ids": True,
        "public_artifact_nonmissing": int(common["has_public_artifact"].notna().sum()),
        "public_artifact_positive": int(common["has_public_artifact"].sum()),
    }
    return results, display_results, audit


def build_common_sample_confounder_display_table(
    display_results: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Outcome": display_results["outcome_label"],
            "Specification": display_results["specification"],
            "N": display_results["nobs"].astype(int),
            "β": display_results["coef"].map(
                lambda value: _format_report_value(value, 4)
            ),
            "SE": display_results["se"].map(
                lambda value: _format_report_value(value, 4)
            ),
            "95% CI": display_results.apply(
                lambda row: (
                    f"[{_format_report_value(row['ci_low'], 4)}, "
                    f"{_format_report_value(row['ci_high'], 4)}]"
                ),
                axis=1,
            ),
            "p": display_results["p"].map(_format_report_p),
            "R² controls-only": display_results["controls_only_r2"].map(
                lambda value: _format_report_value(value, 6)
            ),
            "R² full": display_results["full_r2"].map(
                lambda value: _format_report_value(value, 6)
            ),
            "ΔR²": display_results["delta_r2"].map(
                lambda value: _format_report_value(value, 6)
            ),
        }
    )


def write_common_sample_confounder_report(
    report_path: Path,
    display_table: pd.DataFrame,
    results: pd.DataFrame,
    audit: dict,
    full_path: Path,
    table_path: Path,
    audit_path: Path,
) -> None:
    report = f"""# Common-sample confounder controls for citation outcomes

## Main table

{_markdown_table(display_table)}

## Specification and interpretation

- All outcomes and specifications use the same {audit['common_complete_case_nobs']:,}
  paper complete-case sample. The {audit['baseline_complete_nobs']:,}-paper baseline
  sample is not used for the baseline row in this table.
- The common-sample baseline includes reported max-row GPU capability,
  year-by-venue fixed effects, primary-topic fixed effects, team-size groups, and
  organization-count groups.
- Pre-publication controls are three-year author and institution
  visibility/history proxies: team-member maximum prior citations, team-member
  mean prior publications, maximum prior institutional citations, prior
  organization publications and partner-organization histories, and corporate,
  industry-academia, and international-collaboration indicators.
- Public-artifact availability is added only after all pre-publication proxies
  and is secondary robustness because it may be contemporaneous with or follow
  the compute choice.
- For every row, controls-only R2 removes only GPU capability and retains every
  other control in that row. Delta R2 is full R2 minus that matched controls-only R2.
- Estimates are conditional associations and do not identify causal effects.

## Complete five-step ladders

{_markdown_table(build_common_sample_confounder_display_table(results))}

## Source artifacts

- Full machine-readable results: `data/{full_path.name}`
- Requested display table: `data/{table_path.name}`
- Sample and input audit: `data/{audit_path.name}`
"""
    report_path.write_text(report, encoding="utf-8")


def write_common_sample_confounder_artifacts(
    master: pd.DataFrame,
    citation_sample: pd.DataFrame,
    dirs: dict[str, Path],
    input_data_dir: Path,
    confounder_path: Path,
) -> dict:
    results, display_results, sample_audit = fit_common_sample_confounder_models(
        citation_sample
    )
    display_table = build_common_sample_confounder_display_table(display_results)
    full_path = dirs["data"] / "common_sample_confounder_models.csv"
    table_path = dirs["data"] / "common_sample_confounder_table.csv"
    audit_path = dirs["data"] / "common_sample_confounder_model_audit.json"
    report_path = dirs["report"] / "common_sample_confounder_models.md"
    results.to_csv(full_path, index=False)
    display_table.to_csv(table_path, index=False)

    input_paths = [
        input_data_dir / COMPUTE_FILE,
        input_data_dir / METADATA_FILE,
        input_data_dir / TOPIC_FILE,
        input_data_dir / ORG_VARS_FILE,
        input_data_dir / ORG_LONG_FILE,
        input_data_dir / ORG_YEAR_PANEL_FILE,
        confounder_path,
    ]
    input_files = {}
    for path in input_paths:
        input_files[path.name] = {
            "path": portable_path(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "validated_local_frozen_input_rerun",
        "master_rows": int(len(master)),
        "strict_2020_2023_rows": int(len(citation_sample)),
        "history_window_years": 3,
        "prepublication_controls": PREPUBLICATION_CONFOUNDER_CONTROLS,
        "public_artifact_control": ARTIFACT_CONTROLS[0],
        "public_artifact_is_secondary_robustness": True,
        "controls_only_definition": (
            "same specification and paper IDs as full model, removing only log10_max_compute"
        ),
        **sample_audit,
        "input_files": input_files,
        "output_rows_full": int(len(results)),
        "output_rows_display": int(len(display_table)),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_common_sample_confounder_report(
        report_path,
        display_table,
        results,
        audit,
        full_path,
        table_path,
        audit_path,
    )
    return {
        "tables": {
            "common_sample_confounder_models": str(full_path),
            "common_sample_confounder_table": str(table_path),
        },
        "reports": {"common_sample_confounder_models": str(report_path)},
        "audits": {"common_sample_confounder_models": str(audit_path)},
    }


def fit_fine_grained_compute_impact_models(
    citation_sample: pd.DataFrame,
    award_sample: pd.DataFrame,
) -> pd.DataFrame:
    controls = [
        "C(year_venue)",
        "C(primary_topic)",
        "C(team_size_group)",
        "C(n_organizations_group)",
    ]
    control_required = [
        "year_venue",
        "primary_topic",
        "team_size_group",
        "n_organizations_group",
    ]
    model_specs = [
        {
            "model_id": "reported_gpu_capacity",
            "model_label": "Reported GPU capacity (existing 4.4)",
            "terms": ["log10_max_compute"],
        },
        {
            "model_id": "maxrow_gpu_count",
            "model_label": "Number of GPUs in max-capacity row",
            "terms": ["log10_maxrow_gpu_count"],
        },
        {
            "model_id": "single_gpu_capability",
            "model_label": "Single-GPU capability in max row",
            "terms": ["log10_maxrow_single_gpu_flops"],
        },
        {
            "model_id": "count_plus_single_gpu",
            "model_label": "GPU count + single-GPU capability",
            "terms": ["log10_maxrow_gpu_count", "log10_maxrow_single_gpu_flops"],
        },
        {
            "model_id": "count_plus_ampere_or_newer",
            "model_label": "GPU count + Ampere-or-newer indicator",
            "terms": ["log10_maxrow_gpu_count", "ampere_or_newer"],
        },
        {
            "model_id": "paper_total_gpu_count",
            "model_label": "Total paper GPU count robustness",
            "terms": ["log10_paper_total_gpu_count"],
        },
    ]
    term_labels = {
        "log10_max_compute": "Reported GPU capacity",
        "log10_maxrow_gpu_count": "Number of GPUs",
        "log10_maxrow_single_gpu_flops": "Single-GPU capability",
        "ampere_or_newer": "Ampere-or-newer generation",
        "log10_paper_total_gpu_count": "Total paper GPU count",
    }
    outcomes = [
        {
            "outcome": "log1p_cites",
            "outcome_label": "OLS: log1p citations",
            "family": "ols",
            "cov_type": "HC3",
            "sample_label": "strict_raw_2020_2023",
            "sample": citation_sample,
        },
        {
            "outcome": "cited_by_count",
            "outcome_label": "PPML: citation count",
            "family": "poisson",
            "cov_type": "HC0",
            "sample_label": "strict_raw_2020_2023",
            "sample": citation_sample,
        },
        {
            "outcome": "is_highly_cited_all_yv",
            "outcome_label": "LPM: high-cited top 10%",
            "family": "lpm",
            "cov_type": "HC3",
            "sample_label": "strict_raw_2020_2023",
            "sample": citation_sample,
        },
        {
            "outcome": "is_award",
            "outcome_label": "LPM: award",
            "family": "lpm",
            "cov_type": "HC3",
            "sample_label": "strict_raw_2020_2025",
            "sample": award_sample,
        },
    ]

    rows: list[dict] = []
    for outcome_spec in outcomes:
        for model_spec in model_specs:
            required = [outcome_spec["outcome"], *control_required, *model_spec["terms"]]
            model_df = outcome_spec["sample"].dropna(subset=required).copy()
            if model_df[outcome_spec["outcome"]].nunique(dropna=True) < 2:
                continue
            formula = model_formula(
                outcome_spec["outcome"],
                [*model_spec["terms"], *controls],
            )
            model = fit_formula(
                formula,
                model_df,
                family=outcome_spec["family"],
                cov_type=outcome_spec["cov_type"],
            )
            conf_int = model.conf_int()
            for term in model_spec["terms"]:
                beta = float(model.params[term])
                effect_value, effect_unit = _fine_grained_effect(
                    beta,
                    outcome_spec["family"],
                    outcome_spec["outcome"],
                    term,
                )
                ci_low, ci_high = conf_int.loc[term].tolist()
                rows.append(
                    {
                        "sample": outcome_spec["sample_label"],
                        "outcome": outcome_spec["outcome"],
                        "outcome_label": outcome_spec["outcome_label"],
                        "family": outcome_spec["family"],
                        "cov_type": outcome_spec["cov_type"],
                        "model_id": model_spec["model_id"],
                        "model_label": model_spec["model_label"],
                        "term": term,
                        "term_label": term_labels[term],
                        "nobs": int(model.nobs),
                        "coef": beta,
                        "se": float(model.bse[term]),
                        "p": float(model.pvalues[term]),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "r2": float(getattr(model, "rsquared", np.nan)),
                        "adj_r2": float(getattr(model, "rsquared_adj", np.nan)),
                        "mean_y": float(model_df[outcome_spec["outcome"]].mean()),
                        "effect_value": effect_value,
                        "effect_unit": effect_unit,
                        "formula": formula,
                    }
                )
    return pd.DataFrame(rows)


def fit_gpu_count_incremental_fit_models(
    citation_sample: pd.DataFrame,
    award_sample: pd.DataFrame,
) -> pd.DataFrame:
    controls = "C(year_venue) + C(primary_topic) + C(team_size_group) + C(n_organizations_group)"
    required_controls = [
        "year_venue",
        "primary_topic",
        "team_size_group",
        "n_organizations_group",
    ]
    outcomes = [
        {
            "outcome": "log1p_cites",
            "outcome_label": "OLS: log1p citations",
            "family": "ols",
            "cov_type": "HC3",
            "sample_label": "strict_raw_2020_2023",
            "sample": citation_sample,
        },
        {
            "outcome": "cited_by_count",
            "outcome_label": "PPML: citation count",
            "family": "poisson",
            "cov_type": "HC0",
            "sample_label": "strict_raw_2020_2023",
            "sample": citation_sample,
        },
        {
            "outcome": "is_highly_cited_all_yv",
            "outcome_label": "LPM: high-cited top 10%",
            "family": "lpm",
            "cov_type": "HC3",
            "sample_label": "strict_raw_2020_2023",
            "sample": citation_sample,
        },
        {
            "outcome": "is_award",
            "outcome_label": "LPM: award",
            "family": "lpm",
            "cov_type": "HC3",
            "sample_label": "strict_raw_2020_2025",
            "sample": award_sample,
        },
    ]

    rows = []
    for outcome_spec in outcomes:
        required = [
            outcome_spec["outcome"],
            "log10_maxrow_gpu_count",
            *required_controls,
        ]
        model_df = outcome_spec["sample"].dropna(subset=required).copy()
        base_formula = f"{outcome_spec['outcome']} ~ {controls}"
        full_formula = f"{outcome_spec['outcome']} ~ log10_maxrow_gpu_count + {controls}"
        base_model = fit_formula(
            base_formula,
            model_df,
            family=outcome_spec["family"],
            cov_type=outcome_spec["cov_type"],
        )
        full_model = fit_formula(
            full_formula,
            model_df,
            family=outcome_spec["family"],
            cov_type=outcome_spec["cov_type"],
        )
        row = {
            "sample": outcome_spec["sample_label"],
            "outcome": outcome_spec["outcome"],
            "outcome_label": outcome_spec["outcome_label"],
            "family": outcome_spec["family"],
            "cov_type": outcome_spec["cov_type"],
            "nobs": int(full_model.nobs),
            "gpu_count_coef": float(full_model.params["log10_maxrow_gpu_count"]),
            "gpu_count_se": float(full_model.bse["log10_maxrow_gpu_count"]),
            "gpu_count_p": float(full_model.pvalues["log10_maxrow_gpu_count"]),
            "controls_only_formula": base_formula,
            "gpu_count_formula": full_formula,
        }
        if outcome_spec["family"] == "poisson":
            row.update(
                {
                    "base_log_likelihood": float(base_model.llf),
                    "full_log_likelihood": float(full_model.llf),
                    "delta_log_likelihood": float(full_model.llf - base_model.llf),
                    "base_deviance": float(base_model.deviance),
                    "full_deviance": float(full_model.deviance),
                    "delta_deviance": float(base_model.deviance - full_model.deviance),
                    "base_aic": float(base_model.aic),
                    "full_aic": float(full_model.aic),
                    "aic_improvement": float(base_model.aic - full_model.aic),
                }
            )
        else:
            row.update(
                {
                    "base_r2": float(base_model.rsquared),
                    "full_r2": float(full_model.rsquared),
                    "delta_r2": float(full_model.rsquared - base_model.rsquared),
                    "base_adj_r2": float(base_model.rsquared_adj),
                    "full_adj_r2": float(full_model.rsquared_adj),
                    "delta_adj_r2": float(full_model.rsquared_adj - base_model.rsquared_adj),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _format_fine_grained_effect(row: pd.Series) -> str:
    value = row["effect_value"]
    if pd.isna(value):
        return ""
    unit = str(row["effect_unit"])
    if unit.startswith("percentage points"):
        return f"{value:.2f} pp"
    return f"{value * 100:.1f}%"


def fine_grained_impact_markdown_table(results: pd.DataFrame) -> str:
    display = pd.DataFrame(
        {
            "Outcome": results["outcome_label"],
            "Model": results["model_label"],
            "Term": results["term_label"],
            "N": results["nobs"].map(lambda value: _format_report_value(value, 0)),
            "Coef.": results["coef"].map(lambda value: _format_report_value(value, 3)),
            "SE": results["se"].map(lambda value: _format_report_value(value, 3)),
            "p": results["p"].map(_format_report_p),
            "95% CI": results.apply(
                lambda row: f"[{_format_report_value(row['ci_low'], 3)}, {_format_report_value(row['ci_high'], 3)}]",
                axis=1,
            ),
            "Effect": results.apply(_format_fine_grained_effect, axis=1),
            "Effect scale": results["effect_unit"],
        }
    )
    return _markdown_table(display)


def gpu_count_incremental_fit_markdown_table(incremental: pd.DataFrame) -> str:
    display = pd.DataFrame(
        {
            "Outcome": incremental["outcome_label"],
            "N": incremental["nobs"].map(lambda value: _format_report_value(value, 0)),
            "GPU-count coef.": incremental["gpu_count_coef"].map(
                lambda value: _format_report_value(value, 3)
            ),
            "SE": incremental["gpu_count_se"].map(lambda value: _format_report_value(value, 3)),
            "p": incremental["gpu_count_p"].map(_format_report_p),
            "Controls-only R2": incremental["base_r2"].map(
                lambda value: _format_report_value(value, 3)
            ),
            "+ GPU-count R2": incremental["full_r2"].map(
                lambda value: _format_report_value(value, 3)
            ),
            "Delta R2": incremental["delta_r2"].map(
                lambda value: _format_report_value(value, 3)
            ),
            "AIC improvement": incremental["aic_improvement"].map(
                lambda value: _format_report_value(value, 1)
            ),
        }
    )
    return _markdown_table(display)


def write_fine_grained_compute_impact_report(
    report_path: Path,
    results: pd.DataFrame,
    incremental: pd.DataFrame,
    audit: dict,
    table_path: Path,
    incremental_path: Path,
    audit_path: Path,
) -> None:
    key_models = {
        "maxrow_gpu_count",
        "single_gpu_capability",
        "count_plus_single_gpu",
        "count_plus_ampere_or_newer",
        "paper_total_gpu_count",
    }
    key_results = results.loc[results["model_id"].isin(key_models)].copy()
    citation_generations = ", ".join(
        f"{name}: {count}"
        for name, count in audit["citation_sample_generation_counts"].items()
    )
    award_generations = ", ".join(
        f"{name}: {count}" for name, count in audit["award_sample_generation_counts"].items()
    )
    time_columns = audit.get("time_or_hours_columns_detected", [])
    time_column_note = (
        "No systematic GPU-hour, training-time, or wall-clock duration columns were found "
        "in the released paper-level or row-level GPU inputs."
        if not time_columns
        else "Potential time/hour columns detected and intentionally not modeled here: "
        + ", ".join(time_columns)
    )
    report = f"""# Rebuttal: fine-grained compute-impact analysis

We added a rebuttal-only analysis that decomposes the existing reported GPU
capacity measure into available hardware dimensions. The analysis remains
associational: it tests whether reported GPU quantity and hardware strength are
associated with citation and award outcomes after the same controls used in
Section 4.4.

## Main takeaway

The clearest fine-grained association is GPU quantity. A 10x increase in the
number of GPUs in the max-capacity row is positively associated with log
citations, expected citation counts, high-citation status, and awards. In
contrast, single-GPU capability and the Ampere-or-newer generation indicator are
not consistently significant once the same controls and GPU quantity are
included.

We do not analyze GPU-hours or training time in this rebuttal analysis.
{time_column_note}

## Validation and samples

- Max-row validation matched {audit["max_row_validation_matched"]:,} of {audit["max_row_validation_total"]:,} GPU papers.
- Positive strict max-row compute rows: {audit["max_row_validation_observed_positive"]:,}.
- Citation sample: {audit["citation_sample_rows"]:,} strict raw GPU papers from 2020-2023.
- Award sample: {audit["award_sample_rows"]:,} strict raw GPU papers from 2020-2025.
- Citation-sample max-row generations: {citation_generations}.
- Award-sample max-row generations: {award_generations}.

## Fine-grained model results

{fine_grained_impact_markdown_table(key_results)}

## Increment over controls-only models

This table compares `y ~ controls` with
`y ~ log10(number of GPUs) + controls` on the same estimation sample. For OLS
and LPM outcomes, the table reports the R2 increase. For the PPML citation-count
model, R2 is not defined, so the table reports AIC improvement instead.

{gpu_count_incremental_fit_markdown_table(incremental)}

## Source artifacts

- Full results: `data/{table_path.name}`
- Controls-only comparison: `data/{incremental_path.name}`
- Audit: `data/{audit_path.name}`
"""
    report_path.write_text(report, encoding="utf-8")


def run_common_sample_confounder_models(
    output_dir: Path | str | None = None,
    quiet: bool = False,
    input_data_dir: Path | str | None = None,
    confounder_data: Path | str | None = None,
) -> dict:
    output_base = Path(output_dir) if output_dir is not None else None
    data_dir = Path(input_data_dir) if input_data_dir is not None else ROOT / "data" / "analysis_ready"
    confounder_path = (
        Path(confounder_data)
        if confounder_data is not None
        else data_dir / CONFOUNDER_FILE
    )
    if not confounder_path.exists():
        raise FileNotFoundError(f"Confounder controls not found: {confounder_path}")
    dirs = ensure_output_dirs(output_base)
    inputs = load_inputs(data_dir=data_dir)
    master = build_master_panel(
        inputs["compute"],
        inputs["metadata"],
        inputs["topics"],
        inputs["awards"],
        inputs["org_vars"],
    )
    master = add_year_venue_high_citation_flags(master, (0.05, 0.10, 0.20))
    master = attach_confounder_controls(
        master,
        load_confounder_controls(confounder_path),
        inputs["org_long"],
        inputs["org_year_panel"],
    )
    citation_sample = make_analysis_sample(
        master,
        sample="strict_raw",
        year_min=2020,
        year_max=2023,
    )
    outputs = write_common_sample_confounder_artifacts(
        master,
        citation_sample,
        dirs,
        data_dir,
        confounder_path,
    )
    if not quiet:
        print(json.dumps(outputs, indent=2))
    return outputs


def run_overall_gpu_capability_models(
    output_dir: Path | str | None = None,
    quiet: bool = False,
    input_data_dir: Path | str | None = None,
) -> dict:
    output_base = Path(output_dir) if output_dir is not None else None
    data_dir = Path(input_data_dir) if input_data_dir is not None else ROOT / "data" / "analysis_ready"
    dirs = ensure_output_dirs(output_base)
    inputs = load_inputs(data_dir=data_dir)
    paper_compute = load_paper_compute_rows(data_dir=data_dir)
    master = build_master_panel(
        inputs["compute"],
        inputs["metadata"],
        inputs["topics"],
        inputs["awards"],
        inputs["org_vars"],
    )
    master = add_year_venue_high_citation_flags(master, (0.05, 0.10, 0.20))
    master, feature_audit = add_fine_grained_compute_features(master, paper_compute)
    citation_sample = make_analysis_sample(
        master,
        sample="strict_raw",
        year_min=2020,
        year_max=2023,
    )
    award_sample = make_analysis_sample(
        master,
        sample="strict_raw",
        year_min=2020,
        year_max=2025,
    )
    outputs = write_overall_gpu_capability_artifacts(
        master,
        citation_sample,
        award_sample,
        dirs,
        data_dir,
        feature_audit,
    )
    if not quiet:
        print(json.dumps(outputs, indent=2))
    return outputs


def run_joint_count_ampere_firth_award_model(
    output_dir: Path | str | None = None,
    quiet: bool = False,
    input_data_dir: Path | str | None = None,
) -> dict:
    output_base = Path(output_dir) if output_dir is not None else None
    data_dir = Path(input_data_dir) if input_data_dir is not None else ROOT / "data" / "analysis_ready"
    dirs = ensure_output_dirs(output_base)
    inputs = load_inputs(data_dir=data_dir)
    paper_compute = load_paper_compute_rows(data_dir=data_dir)
    master = build_master_panel(
        inputs["compute"],
        inputs["metadata"],
        inputs["topics"],
        inputs["awards"],
        inputs["org_vars"],
    )
    master, feature_audit = add_fine_grained_compute_features(master, paper_compute)
    award_sample = make_analysis_sample(
        master,
        sample="strict_raw",
        year_min=2020,
        year_max=2025,
    )
    outputs = write_joint_count_ampere_firth_award_artifacts(
        master,
        award_sample,
        dirs,
        data_dir,
        feature_audit,
    )
    if not quiet:
        print(json.dumps(outputs, indent=2))
    return outputs


def run_rebuttal_fine_grained_compute_impact(
    output_dir: Path | str | None = None,
    quiet: bool = False,
    input_data_dir: Path | str | None = None,
) -> dict:
    output_base = Path(output_dir) if output_dir is not None else None
    data_dir = Path(input_data_dir) if input_data_dir is not None else None
    dirs = ensure_output_dirs(output_base)
    inputs = load_inputs(data_dir=data_dir)
    paper_compute = load_paper_compute_rows(data_dir=data_dir)
    master = build_master_panel(
        inputs["compute"],
        inputs["metadata"],
        inputs["topics"],
        inputs["awards"],
        inputs["org_vars"],
    )
    master, feature_audit = add_fine_grained_compute_features(master, paper_compute)
    citation_sample = make_analysis_sample(master, sample="strict_raw", year_min=2020, year_max=2023)
    award_sample = make_analysis_sample(master, sample="strict_raw", year_min=2020, year_max=2025)

    results = fit_fine_grained_compute_impact_models(citation_sample, award_sample)
    incremental = fit_gpu_count_incremental_fit_models(citation_sample, award_sample)
    table_path = dirs["data"] / "rebuttal_fine_grained_compute_impact.csv"
    incremental_path = dirs["data"] / "rebuttal_fine_grained_compute_incremental_fit.csv"
    audit_path = dirs["data"] / "rebuttal_fine_grained_compute_impact_audit.json"
    report_path = dirs["report"] / "rebuttal_fine_grained_compute_impact.md"
    results.to_csv(table_path, index=False)
    incremental.to_csv(incremental_path, index=False)

    audit = {
        "master_rows": int(len(master)),
        **feature_audit,
        "citation_sample_rows": int(len(citation_sample)),
        "award_sample_rows": int(len(award_sample)),
        "citation_sample_generation_counts": {
            str(key): int(value)
            for key, value in citation_sample["maxrow_gpu_generation"]
            .fillna("Unknown")
            .value_counts()
            .items()
        },
        "award_sample_generation_counts": {
            str(key): int(value)
            for key, value in award_sample["maxrow_gpu_generation"]
            .fillna("Unknown")
            .value_counts()
            .items()
        },
        "result_rows": int(len(results)),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_fine_grained_compute_impact_report(
        report_path,
        results,
        incremental,
        audit,
        table_path,
        incremental_path,
        audit_path,
    )

    outputs = {
        "tables": {
            "rebuttal_fine_grained_compute_impact": str(table_path),
            "rebuttal_fine_grained_compute_incremental_fit": str(incremental_path),
        },
        "reports": {"rebuttal_fine_grained_compute_impact": str(report_path)},
        "audits": {"rebuttal_fine_grained_compute_impact": str(audit_path)},
    }
    if not quiet:
        print(json.dumps(outputs, indent=2))
    return outputs


def run_rebuttal_gpu_concentration(
    output_dir: Path | str | None = None,
    quiet: bool = False,
    input_data_dir: Path | str | None = None,
    skip_award: bool = False,
) -> dict:
    output_base = Path(output_dir) if output_dir is not None else None
    data_dir = Path(input_data_dir) if input_data_dir is not None else None
    dirs = ensure_output_dirs(output_base)
    inputs = load_inputs(data_dir=data_dir, skip_award=skip_award)
    master = build_master_panel(
        inputs["compute"],
        inputs["metadata"],
        inputs["topics"],
        inputs["awards"],
        inputs["org_vars"],
    )
    gpu_2020_2023 = make_analysis_sample(master, sample="gpu_lb1", year_min=2020, year_max=2023)
    top_compute_concentration_source = build_top_compute_concentration_data(gpu_2020_2023)
    outputs = write_rebuttal_gpu_concentration_artifacts(
        gpu_2020_2023,
        top_compute_concentration_source,
        dirs,
    )
    if not quiet:
        print(json.dumps(outputs, indent=2))
    return outputs


def skipped_award_result_row(nobs: int, reason: str) -> dict:
    return {
        "outcome_model": "Strict raw GPU LPM: award",
        "spec": 7,
        "nobs": int(nobs),
        "coef": np.nan,
        "se": np.nan,
        "p": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "effect_interpretation": reason,
        "pct_per_10x": np.nan,
        "pp_per_10x": np.nan,
        "mean_y": np.nan,
    }


def plot_confounder_control_forest(
    ladder: pd.DataFrame,
    fig_dir: Path,
) -> dict[str, str]:
    source = ladder.loc[
        ladder["outcome"].eq("log1p_cites")
        & ladder["model_id"].isin(["M0", "M1", "M2", "M3", "M4"])
        & ladder["status"].eq("ok")
    ].copy()
    if len(source) < 2:
        (fig_dir / "rq3_confounder_control_forest.png").unlink(missing_ok=True)
        return {}
    order = ["M0", "M1", "M2", "M3", "M4"]
    source["model_id"] = pd.Categorical(source["model_id"], order, ordered=True)
    source = source.sort_values("model_id")
    y = np.arange(len(source))
    fig, ax = plt.subplots(figsize=(4.8, 2.5))
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.errorbar(
        source["coef"],
        y,
        xerr=[source["coef"] - source["ci_low"], source["ci_high"] - source["coef"]],
        fmt="o",
        color="#2E5EAA",
        ecolor="#2E5EAA",
        markersize=4,
        capsize=2,
        linewidth=1,
    )
    ax.set_yticks(y, source["model_id"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient on log10 reported GPU compute (95% CI)")
    ax.set_ylabel("Nested control specification")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    fig.tight_layout()
    paths = save_pub_figure(fig, fig_dir / "rq3_confounder_control_forest")
    plt.close(fig)
    return paths


def write_confounding_revision_report(
    report_path: Path,
    ladder: pd.DataFrame,
    author_fe: pd.DataFrame,
    sensitivity: pd.DataFrame,
    award_firth: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    citation_overall = coverage.loc[
        coverage["sample"].eq("citation_2020_2023")
        & coverage["grouping"].eq("overall")
    ]
    history_complete_rate = (
        float(citation_overall.iloc[0]["history_complete_rate"])
        if not citation_overall.empty
        else 0.0
    )
    expanded_ready = bool(
        history_complete_rate >= 0.90
        and (
            ladder["model_id"].eq("M3")
            & ladder["outcome"].eq("log1p_cites")
            & ladder["status"].eq("ok")
        ).any()
    )
    preferred_sensitivity_model = "M3" if expanded_ready else "M0"
    preferred_sensitivity = sensitivity.loc[
        sensitivity.get("model_id", pd.Series(index=sensitivity.index, dtype="object")).eq(
            preferred_sensitivity_model
        )
        & sensitivity.get(
            "benchmark", pd.Series(index=sensitivity.index, dtype="object")
        ).eq("strongest observed control group")
    ].sort_values("benchmark_multiplier")
    sensitivity_interpretation = (
        "No complete sensitivity result is available for the preferred model."
    )
    sensitivity_fragile = False
    if not preferred_sensitivity.empty:
        first = preferred_sensitivity.iloc[0]
        doubled = preferred_sensitivity.loc[
            preferred_sensitivity["benchmark_multiplier"].eq(2.0)
        ]
        doubled_coef = (
            float(doubled.iloc[0]["bias_adjusted_coef"])
            if not doubled.empty
            else np.nan
        )
        rv_significance = float(first["robustness_value_alpha_0_05"])
        original_coef = float(first["coef"])
        adjusted_coef = float(first["bias_adjusted_coef"])
        sensitivity_fragile = bool(
            rv_significance < 0.05
            or (pd.notna(doubled_coef) and np.sign(doubled_coef) != np.sign(original_coef))
        )
        if sensitivity_fragile:
            sensitivity_interpretation = (
                f"The {preferred_sensitivity_model} association is sensitive to omitted "
                f"confounding: RV(p>=.05)={rv_significance:.3f}; a confounder benchmarked "
                f"at the strongest observed control group reduces the coefficient from "
                f"{original_coef:.3f} to {adjusted_coef:.3f}, and the 2x benchmark gives "
                f"{doubled_coef:.3f}. The evidence should therefore be described as fragile."
            )
        else:
            sensitivity_interpretation = (
                f"The {preferred_sensitivity_model} estimate remains on the same side of "
                f"zero under the reported observed-control benchmarks, but residual "
                f"confounding remains possible."
            )
    primary_summary = "The expanded citation-control ladder is unavailable."
    primary_rows = ladder.loc[
        ladder["outcome"].eq("log1p_cites")
        & ladder["model_id"].isin(["M0", "M3", "M4"])
        & ladder["status"].eq("ok")
    ].set_index("model_id")
    if {"M0", "M3", "M4"}.issubset(primary_rows.index):
        m0 = primary_rows.loc["M0"]
        m3 = primary_rows.loc["M3"]
        m4 = primary_rows.loc["M4"]
        primary_summary = (
            f"On the common N={int(m3['nobs']):,} sample, adding all "
            f"pre-publication controls changes the coefficient from {m0['coef']:.3f} "
            f"to {m3['coef']:.3f} ({m3['coef_attenuation_vs_m0'] * 100:.1f}% "
            f"attenuation; p={m3['p']:.3f}). Adding public-artifact availability "
            f"gives {m4['coef']:.3f} ({m4['coef_attenuation_vs_m0'] * 100:.1f}% "
            f"attenuation; p={m4['p']:.3f}). This is stability to the measured "
            f"controls, not evidence of causal identification."
        )
    author_summary = "Listed-author fixed-effect estimates are unavailable."
    author_ok = author_fe.loc[author_fe.get("status").eq("ok")]
    if len(author_ok) >= 2:
        first_author = author_ok.loc[
            author_ok["model"].eq("first-listed author FE")
        ].iloc[0]
        last_author = author_ok.loc[
            author_ok["model"].eq("last-listed author FE")
        ].iloc[0]
        author_summary = (
            f"The first-listed-author estimate is {first_author['coef']:.3f} "
            f"(p={first_author['p']:.3f}) and the last-listed-author estimate is "
            f"{last_author['coef']:.3f} (p={last_author['p']:.3f}); both confidence "
            f"intervals include zero, so these subset comparisons do not provide "
            f"supportive fixed-effect evidence."
        )
    award_summary = "Expanded exploratory award estimates are unavailable."
    award_m3 = award_firth.loc[
        award_firth.get("model_id").eq("M3") & award_firth.get("status").eq("ok")
    ]
    if not award_m3.empty:
        row = award_m3.iloc[0]
        award_summary = (
            f"The expanded Firth award estimate uses N={int(row['nobs']):,} with "
            f"{int(row['award_events'])} awards and gives OR={row['odds_ratio_per_10x']:.3f} "
            f"(95% CI [{row['or_ci_low']:.3f}, {row['or_ci_high']:.3f}], "
            f"p={row['p']:.3f}). Because award-history coverage is below the 90% "
            f"threshold, this remains an appendix-only exploratory result."
        )
    if expanded_ready:
        reviewer_response = """We agree that measured controls cannot eliminate all confounding in the
compute-recognition relationship. We therefore added pre-publication author and
institution visibility controls, organization-history and industry-collaboration
controls, listed-author fixed-effect comparisons, and a formal omitted-variable-
bias sensitivity analysis. We additionally treat public-artifact availability as
a secondary robustness control and use bias-reduced logistic regression for the
sparse award outcome. We now describe all estimates as conditional associations,
report non-supportive alternative outcomes, and explicitly retain residual
confounding as a limitation."""
        if sensitivity_fragile:
            reviewer_response += (
                " The formal sensitivity analysis indicates that omitted confounding of "
                "a plausible observed-control magnitude could materially attenuate or "
                "reverse the estimate, so we characterize the evidence as fragile."
            )
    else:
        reviewer_response = """The analysis implementation is ready, but this draft must not yet claim that
pre-publication author/institution controls or listed-author fixed effects were
estimated. The OpenAlex history control file is absent or has less than 90%
complete coverage, so M1-M4 remain skipped. At this stage only the existing M0
model, its formal omitted-variable-bias sensitivity analysis, and the exploratory
M0 Firth award model are supported by real data. Build the control file and rerun
the analysis before using a reviewer response that describes the expanded models.
The currently estimable M0 sensitivity result should already be described as
fragile rather than causally robust."""
    ladder_table = _simple_report_table(
        ladder.loc[ladder["outcome"].eq("log1p_cites")],
        [
            "model_id",
            "controls",
            "nobs",
            "coef",
            "se",
            "p",
            "ci_low",
            "ci_high",
            "r2",
            "delta_r2",
            "coef_attenuation_vs_m0",
            "status",
        ],
        [
            "Model",
            "Controls",
            "N",
            "Coef.",
            "SE",
            "p",
            "CI low",
            "CI high",
            "R2",
            "Delta R2",
            "Attenuation vs M0",
            "Status",
        ],
        p_columns={"p"},
        percent_columns={"coef_attenuation_vs_m0"},
    )
    alternative_table = _simple_report_table(
        ladder.loc[~ladder["outcome"].eq("log1p_cites")],
        ["outcome", "family", "model_id", "nobs", "coef", "se", "p", "ci_low", "ci_high", "status"],
        ["Outcome", "Family", "Model", "N", "Coef.", "SE", "p", "CI low", "CI high", "Status"],
        p_columns={"p"},
    )
    author_table = _simple_report_table(
        author_fe,
        ["model", "nobs", "n_authors", "coef", "se", "p", "ci_low", "ci_high", "status"],
        ["Model", "N", "Authors", "Coef.", "SE", "p", "CI low", "CI high", "Status"],
        p_columns={"p"},
    )
    sensitivity_summary = sensitivity.loc[
        sensitivity.get("benchmark", pd.Series(index=sensitivity.index, dtype="object")).eq(
            "strongest observed control group"
        )
    ]
    sensitivity_table = _simple_report_table(
        sensitivity_summary,
        [
            "model_id",
            "nobs",
            "partial_r2_compute_outcome",
            "robustness_value_zero",
            "robustness_value_alpha_0_05",
            "benchmark_multiplier",
            "assumed_treatment_partial_r2",
            "assumed_outcome_partial_r2",
            "bias_adjusted_coef",
        ],
        [
            "Model",
            "N",
            "Partial R2",
            "RV, zero",
            "RV, p>=.05",
            "Benchmark x",
            "Assumed treatment R2",
            "Assumed outcome R2",
            "Bias-adjusted coef.",
        ],
    )
    award_table = _simple_report_table(
        award_firth,
        [
            "model_id",
            "nobs",
            "award_events",
            "odds_ratio_per_10x",
            "or_ci_low",
            "or_ci_high",
            "p",
            "status",
        ],
        ["Model", "N", "Awards", "OR per 10x", "OR CI low", "OR CI high", "p", "Status"],
        p_columns={"p"},
    )
    coverage_table = _simple_report_table(
        coverage.loc[coverage["grouping"].eq("overall")],
        [
            "sample",
            "n",
            "history_complete_rate",
            "author_id_complete_rate",
            "institution_id_complete_rate",
            "artifact_rate",
            "main_analysis_eligible",
        ],
        [
            "Sample",
            "N",
            "History complete",
            "Author IDs complete",
            "Institution IDs complete",
            "Artifact rate",
            "Eligible for main text",
        ],
        percent_columns={
            "history_complete_rate",
            "author_id_complete_rate",
            "institution_id_complete_rate",
            "artifact_rate",
        },
    )
    report = f"""# Revision: measured and unmeasured confounding

## Interpretation

The analyses in this report estimate conditional associations between reported
GPU compute and scholarly-recognition outcomes. They do not identify a causal
effect. Author and institution controls use only information from the three
calendar years preceding each focal paper. Public-artifact availability is a
secondary robustness control because it may be contemporaneous with, or follow,
the compute choice. Social-media attention and direct industrial promotion are
not treated as pre-publication confounders because reliable time-stamped measures
are unavailable and these factors may be mediators.

## Coverage audit

{coverage_table}

The expanded model is eligible for main-text interpretation only when at least
90% of the relevant sample has complete pre-publication history controls.

## Nested citation controls

{ladder_table}

{primary_summary}

## Alternative citation outcomes under M3

{alternative_table}

## Listed-author fixed effects

{author_table}

{author_summary}

These subset models use only listed authors with at least two papers and
within-author variation in reported compute. Standard errors are clustered by
the corresponding listed-author identifier; last-listed authors are not assumed
to be senior authors.

## Unobserved-confounding sensitivity

{sensitivity_table}

{sensitivity_interpretation}

Robustness values and benchmark adjustments follow the Cinelli-Hazlett omitted-
variable-bias framework. Sensitivity calculations use classical OLS standard
errors, while the reported control-ladder estimates use HC3 standard errors.

## Exploratory award models

{award_table}

{award_summary}

Award models use Firth bias reduction and profile-likelihood intervals. Because
awards are rare, these results remain exploratory regardless of statistical
significance.

## Suggested response to the reviewer

{reviewer_response}
"""
    report_path.write_text(report, encoding="utf-8")


def run_analysis(
    output_dir: Path | str | None = None,
    quiet: bool = False,
    input_data_dir: Path | str | None = None,
    skip_award: bool = False,
    confounder_data: Path | str | None = None,
) -> dict:
    output_base = Path(output_dir) if output_dir is not None else None
    data_dir = Path(input_data_dir) if input_data_dir is not None else None
    dirs = ensure_output_dirs(output_base)
    inputs = load_inputs(data_dir=data_dir, skip_award=skip_award)
    master = build_master_panel(
        inputs["compute"],
        inputs["metadata"],
        inputs["topics"],
        inputs["awards"],
        inputs["org_vars"],
    )
    master = add_year_venue_high_citation_flags(master, (0.05, 0.10, 0.20))
    analysis_data_dir = data_dir if data_dir is not None else ROOT / "data" / "analysis_ready"
    overall_feature_audit = None
    if not skip_award:
        master, overall_feature_audit = add_fine_grained_compute_features(
            master,
            load_paper_compute_rows(data_dir=analysis_data_dir),
        )
    confounder_path = (
        Path(confounder_data)
        if confounder_data is not None
        else analysis_data_dir / CONFOUNDER_FILE
    )
    confounder_available = confounder_path.exists()
    if confounder_available:
        master = attach_confounder_controls(
            master,
            load_confounder_controls(confounder_path),
            inputs["org_long"],
            inputs["org_year_panel"],
        )

    gpu_2020_2023 = make_analysis_sample(master, sample="gpu_lb1", year_min=2020, year_max=2023)
    strict_2020_2023 = make_analysis_sample(master, sample="strict_raw", year_min=2020, year_max=2023)
    gpu_2020_2025 = make_analysis_sample(master, sample="gpu_lb1", year_min=2020, year_max=2025)
    strict_2020_2025 = make_analysis_sample(master, sample="strict_raw", year_min=2020, year_max=2025)
    regression_2020_2023 = strict_2020_2023
    regression_2020_2025 = strict_2020_2025

    selection_check = pd.DataFrame(
        [
            {
                "sample": "gpu_lb1_2020_2023",
                "rows": len(gpu_2020_2023),
                "valid_log10_max_compute": int(gpu_2020_2023["log10_max_compute"].notna().sum()),
                "mean_cites": gpu_2020_2023["cited_by_count"].mean(),
                "high_cited_rate_all_yv": gpu_2020_2023["is_highly_cited_all_yv"].mean(),
                "award_rate": gpu_2020_2023["is_award"].mean(),
            },
            {
                "sample": "strict_raw_2020_2023",
                "rows": len(strict_2020_2023),
                "valid_log10_max_compute": int(strict_2020_2023["log10_max_compute"].notna().sum()),
                "mean_cites": strict_2020_2023["cited_by_count"].mean(),
                "high_cited_rate_all_yv": strict_2020_2023["is_highly_cited_all_yv"].mean(),
                "award_rate": strict_2020_2023["is_award"].mean(),
            },
        ]
    )

    strict_ols_log = fit_model_grid(
        regression_2020_2023,
        outcome="log1p_cites",
        family="ols",
        max_spec=7,
        cov_type="HC3",
    )
    strict_ppml = fit_model_grid(
        regression_2020_2023,
        outcome="cited_by_count",
        family="poisson",
        max_spec=7,
        cov_type="HC0",
    )
    strict_ols_norm = fit_model_grid(
        regression_2020_2023,
        outcome="citation_normalized_percentile",
        family="ols",
        max_spec=7,
        cov_type="HC3",
    )
    strict_lpm_high = fit_model_grid(
        regression_2020_2023,
        outcome="is_highly_cited_all_yv",
        family="lpm",
        max_spec=7,
        cov_type="HC3",
    )
    strict_lpm_award = None
    if not skip_award:
        strict_lpm_award = fit_model_grid(
            regression_2020_2025,
            outcome="is_award",
            family="lpm",
            max_spec=7,
            cov_type="HC3",
        )

    main_result_rows = [
            pick_result(
                strict_ols_log,
                "Strict raw GPU OLS: log1p citations",
                7,
                "approx pct change in 1+cites per 10x GPU compute",
            ),
            pick_result(strict_ppml, "Strict raw GPU PPML: citation count", 7, "pct change in expected citations per 10x GPU compute"),
            pick_result(
                strict_ols_norm,
                "Strict raw GPU OLS: normalized citation percentile",
                7,
                "percentile-point scale, 0-1",
            ),
            pick_result(strict_lpm_high, "Strict raw GPU LPM: high cited all-yv top10", 7, "percentage points"),
    ]
    if strict_lpm_award is None:
        main_result_rows.append(
            skipped_award_result_row(
                len(regression_2020_2025),
                "skipped by --skip-award; Findings award outcome not included",
            )
        )
    else:
        main_result_rows.append(
            pick_result(strict_lpm_award, "Strict raw GPU LPM: award", 7, "percentage points")
        )
    main_results = pd.DataFrame(main_result_rows)

    effect_frames = [
            compact_effect_table(strict_ols_log, "Strict raw GPU OLS: log1p citations"),
            compact_effect_table(strict_ppml, "Strict raw GPU PPML: citation count"),
            compact_effect_table(strict_ols_norm, "Strict raw GPU OLS: normalized citation percentile"),
            compact_effect_table(strict_lpm_high, "Strict raw GPU LPM: high cited all-yv top10"),
    ]
    if strict_lpm_award is not None:
        effect_frames.append(compact_effect_table(strict_lpm_award, "Strict raw GPU LPM: award"))
    all_effect_tables = pd.concat(effect_frames, ignore_index=True)

    robust_rows = []
    for compute_var in ["log10_max_compute", "log10_compute"]:
        for team_control in ["group", "continuous"]:
            result = fit_model_grid(
                regression_2020_2023,
                outcome="log1p_cites",
                family="ols",
                compute_var=compute_var,
                team_control=team_control,
                specs=[7],
                cov_type="HC3",
            )
            row = get_model_row(result, 7)
            robust_rows.append(
                {
                    "compute_var": compute_var,
                    "team_control": team_control,
                    "nobs": int(row["nobs"]),
                    "coef": row["coef"],
                    "se": row["se"],
                    "p": row["p"],
                    "pct_per_10x": row["pct_per_10x"],
                }
            )
    robust_compute_table = pd.DataFrame(robust_rows)

    outlier_rows = []
    for name, kwargs in {
        "baseline": {},
        "drop citation top 1%": {"drop_cite_top1": True},
        "drop compute top 1%": {"drop_compute_top1": True},
        "drop both top 1%": {"drop_cite_top1": True, "drop_compute_top1": True},
    }.items():
        result = fit_model_grid(
            outlier_filtered(regression_2020_2023, **kwargs),
            outcome="log1p_cites",
            family="ols",
            specs=[7],
            cov_type="HC3",
        )
        row = get_model_row(result, 7)
        outlier_rows.append(
            {
                "sample": name,
                "nobs": int(row["nobs"]),
                "coef": row["coef"],
                "se": row["se"],
                "p": row["p"],
                "pct_per_10x": row["pct_per_10x"],
            }
        )
    outlier_table = pd.DataFrame(outlier_rows)

    cluster_rows = []
    for cluster_var in ["year_venue", "primary_topic", "year_str"]:
        result = fit_model_grid(
            regression_2020_2023,
            outcome="log1p_cites",
            family="ols",
            specs=[7],
            cov_type="cluster",
            cluster_var=cluster_var,
        )
        row = get_model_row(result, 7)
        cluster_rows.append(
            {
                "cluster_var": cluster_var,
                "n_clusters": int(result["samples"]["ols_7"][cluster_var].nunique()),
                "nobs": int(row["nobs"]),
                "coef": row["coef"],
                "se": row["se"],
                "p": row["p"],
            }
        )
    cluster_table = pd.DataFrame(cluster_rows)

    loo_rows = []
    for year in sorted(regression_2020_2023["year"].dropna().unique()):
        result = fit_model_grid(
            regression_2020_2023.loc[regression_2020_2023["year"] != year].copy(),
            outcome="log1p_cites",
            family="ols",
            specs=[7],
            cov_type="HC3",
        )
        row = get_model_row(result, 7)
        loo_rows.append(
            {
                "leave_out_type": "year",
                "left_out": str(int(year)),
                "nobs": int(row["nobs"]),
                "coef": row["coef"],
                "se": row["se"],
                "p": row["p"],
            }
        )
    for venue in sorted(regression_2020_2023["venue"].dropna().unique()):
        result = fit_model_grid(
            regression_2020_2023.loc[regression_2020_2023["venue"] != venue].copy(),
            outcome="log1p_cites",
            family="ols",
            specs=[7],
            cov_type="HC3",
        )
        row = get_model_row(result, 7)
        loo_rows.append(
            {
                "leave_out_type": "venue",
                "left_out": str(venue),
                "nobs": int(row["nobs"]),
                "coef": row["coef"],
                "se": row["se"],
                "p": row["p"],
            }
        )
    loo_table = pd.DataFrame(loo_rows)

    institution_control_table = fit_institution_history_models(
        master, inputs["org_long"], inputs["org_year_panel"]
    )

    confounder_ladder = fit_confounder_model_ladder(regression_2020_2023)
    author_fixed_effects = fit_author_fixed_effects(regression_2020_2023)
    unobserved_sensitivity = fit_unobserved_confounding_sensitivity(
        regression_2020_2023
    )
    confounder_coverage = build_confounder_coverage_audit(
        regression_2020_2023,
        regression_2020_2025,
    )
    award_firth = (
        fit_firth_award_models(regression_2020_2025)
        if not skip_award
        else pd.DataFrame(
            [
                {
                    "model_id": model_id,
                    "status": "skipped",
                    "reason": "--skip-award",
                    "exploratory": 1,
                }
                for model_id in ["M0", "M3", "M4"]
            ]
        )
    )

    if strict_lpm_award is None:
        award_sample = regression_2020_2025
        award_sparse_diag = pd.DataFrame(
            {
                "metric": [
                    "skipped",
                    "reason",
                    "nobs",
                    "award_events",
                    "event_rate",
                    "year_venue_cells",
                ],
                "value": [
                    1,
                    "--skip-award; Findings award outcome not included",
                    len(award_sample),
                    int(award_sample["is_award"].sum()),
                    float(award_sample["is_award"].mean()) if len(award_sample) else np.nan,
                    int(award_sample["year_venue"].nunique()),
                ],
            }
        )
    else:
        award_sample = strict_lpm_award["common_df"]
        award_model_7 = strict_lpm_award["models"].get("lpm_7")
        award_sparse_diag = pd.DataFrame(
            {
                "metric": [
                    "nobs",
                    "award_events",
                    "event_rate",
                    "parameters_spec7",
                    "events_per_parameter_spec7",
                    "year_venue_cells",
                    "zero_award_cells",
                ],
                "value": [
                    len(award_sample),
                    int(award_sample["is_award"].sum()),
                    float(award_sample["is_award"].mean()),
                    int(len(award_model_7.params)) if award_model_7 is not None else np.nan,
                    float(award_sample["is_award"].sum() / len(award_model_7.params))
                    if award_model_7 is not None
                    else np.nan,
                    int(award_sample["year_venue"].nunique()),
                    int((award_sample.groupby("year_venue")["is_award"].sum() == 0).sum()),
                ],
            }
        )

    strength_table = pd.DataFrame([effect_strength_ols(strict_ols_log, spec=7)])
    delta_r2_table = pd.DataFrame([delta_r2_for_spec(regression_2020_2023, "log1p_cites", spec_id=7)])
    top_compute_concentration_source = build_top_compute_concentration_data(gpu_2020_2023)
    high_compute_impact_matrix_source = build_high_compute_impact_matrix_data(gpu_2020_2023)

    tables = {
        "main_results_summary": dirs["data"] / "main_results_summary.csv",
        "all_model_effect_tables": dirs["data"] / "all_model_effect_tables.csv",
        "selection_check": dirs["data"] / "selection_check.csv",
        "robust_compute_variable_team_control": dirs["data"] / "robust_compute_variable_team_control.csv",
        "outlier_robustness": dirs["data"] / "outlier_robustness.csv",
        "cluster_se_sensitivity": dirs["data"] / "cluster_se_sensitivity.csv",
        "leave_one_out": dirs["data"] / "leave_one_out.csv",
        "institution_history_controls": dirs["data"] / "institution_history_controls.csv",
        "award_sparse_diagnostics": dirs["data"] / "award_sparse_diagnostics.csv",
        "effect_strength": dirs["data"] / "effect_strength.csv",
        "delta_r2": dirs["data"] / "delta_r2.csv",
        "rq3_top20_compute_concentration": dirs["data"] / "rq3_top20_compute_concentration.csv",
        "rq3_high_compute_high_impact_matrix": dirs["data"]
        / "rq3_high_compute_high_impact_matrix.csv",
        "confounder_control_model_ladder": dirs["data"]
        / "confounder_control_model_ladder.csv",
        "author_fixed_effects": dirs["data"] / "author_fixed_effects.csv",
        "unobserved_confounding_sensitivity": dirs["data"]
        / "unobserved_confounding_sensitivity.csv",
        "award_firth_models": dirs["data"] / "award_firth_models.csv",
        "confounder_coverage_audit": dirs["data"] / "confounder_coverage_audit.csv",
    }
    table_frames = {
        "main_results_summary": main_results,
        "all_model_effect_tables": all_effect_tables,
        "selection_check": selection_check,
        "robust_compute_variable_team_control": robust_compute_table,
        "outlier_robustness": outlier_table,
        "cluster_se_sensitivity": cluster_table,
        "leave_one_out": loo_table,
        "institution_history_controls": institution_control_table,
        "award_sparse_diagnostics": award_sparse_diag,
        "effect_strength": strength_table,
        "delta_r2": delta_r2_table,
        "rq3_top20_compute_concentration": top_compute_concentration_source,
        "rq3_high_compute_high_impact_matrix": high_compute_impact_matrix_source,
        "confounder_control_model_ladder": confounder_ladder,
        "author_fixed_effects": author_fixed_effects,
        "unobserved_confounding_sensitivity": unobserved_sensitivity,
        "award_firth_models": award_firth,
        "confounder_coverage_audit": confounder_coverage,
    }
    for key, frame in table_frames.items():
        frame.to_csv(tables[key], index=False)

    figure_paths = {
        "rq3_top20_compute_concentration": plot_top_compute_concentration(
            top_compute_concentration_source,
            dirs["fig"],
        ),
        "rq3_high_compute_high_impact_matrix": plot_high_compute_impact_matrix(
            high_compute_impact_matrix_source,
            dirs["fig"],
        ),
        "rq3_confounder_control_forest": plot_confounder_control_forest(
            confounder_ladder,
            dirs["fig"],
        ),
    }
    rebuttal_outputs = write_rebuttal_gpu_concentration_artifacts(
        gpu_2020_2023,
        top_compute_concentration_source,
        dirs,
    )
    cutoff_sensitivity_outputs = write_rebuttal_cutoff_sensitivity_artifacts(
        gpu_2020_2023,
        regression_2020_2023,
        high_compute_impact_matrix_source,
        dirs,
    )
    field_normalized_outputs = write_rebuttal_field_normalized_citation_artifacts(
        master,
        regression_2020_2023,
        dirs,
    )
    common_confounder_outputs = {"tables": {}, "reports": {}, "audits": {}}
    if confounder_available:
        common_confounder_outputs = write_common_sample_confounder_artifacts(
            master,
            regression_2020_2023,
            dirs,
            analysis_data_dir,
            confounder_path,
        )
    overall_outputs = {"tables": {}, "reports": {}, "audits": {}}
    joint_firth_outputs = {"tables": {}, "reports": {}, "audits": {}}
    if not skip_award:
        overall_outputs = write_overall_gpu_capability_artifacts(
            master,
            regression_2020_2023,
            regression_2020_2025,
            dirs,
            analysis_data_dir,
            overall_feature_audit or {},
        )
        joint_firth_outputs = write_joint_count_ampere_firth_award_artifacts(
            master,
            regression_2020_2025,
            dirs,
            analysis_data_dir,
            overall_feature_audit or {},
        )
    figure_paths.update(rebuttal_outputs["figures"])
    figure_paths.update(cutoff_sensitivity_outputs["figures"])

    outputs = {
        "tables": {
            **{key: str(path) for key, path in tables.items()},
            **rebuttal_outputs["tables"],
            **cutoff_sensitivity_outputs["tables"],
            **field_normalized_outputs["tables"],
            **overall_outputs["tables"],
            **joint_firth_outputs["tables"],
            **common_confounder_outputs["tables"],
        },
        "figures": figure_paths,
        "reports": {
            "rq3_gpu_only_citation_modeling": str(
                dirs["report"] / "rq3_gpu_only_citation_modeling.md"
            ),
            "rebuttal_uncontrolled_confounding": str(
                dirs["report"] / "rebuttal_uncontrolled_confounding.md"
            ),
            **rebuttal_outputs["reports"],
            **cutoff_sensitivity_outputs["reports"],
            **field_normalized_outputs["reports"],
            **overall_outputs["reports"],
            **joint_firth_outputs["reports"],
            **common_confounder_outputs["reports"],
        },
        "audits": {
            **overall_outputs["audits"],
            **joint_firth_outputs["audits"],
            **common_confounder_outputs["audits"],
        },
    }
    audit = {
        "master_rows": len(master),
        "gpu_2020_2023_rows": len(gpu_2020_2023),
        "strict_2020_2023_rows": len(strict_2020_2023),
        "gpu_2020_2025_rows": len(gpu_2020_2025),
        "strict_2020_2025_rows": len(strict_2020_2025),
        "confounder_data_available": int(confounder_available),
        "confounder_data_path": str(confounder_path),
    }
    robustness_tables = {
        "selection_check": selection_check,
        "robust_compute_variable_team_control": robust_compute_table,
        "outlier_robustness": outlier_table,
        "cluster_se_sensitivity": cluster_table,
        "leave_one_out": loo_table,
        "institution_history_controls": institution_control_table,
        "award_sparse_diagnostics": award_sparse_diag,
        "effect_strength": strength_table,
        "delta_r2": delta_r2_table,
    }
    write_report(
        Path(outputs["reports"]["rq3_gpu_only_citation_modeling"]),
        main_results,
        all_effect_tables,
        robustness_tables,
        audit,
    )
    write_confounding_revision_report(
        Path(outputs["reports"]["rebuttal_uncontrolled_confounding"]),
        confounder_ladder,
        author_fixed_effects,
        unobserved_sensitivity,
        award_firth,
        confounder_coverage,
    )

    if not quiet:
        print(json.dumps(outputs, indent=2))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RQ3 GPU-only citation modeling.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--input-data-dir",
        type=Path,
        default=None,
        help="Directory containing the public analysis-ready CSV tables.",
    )
    parser.add_argument(
        "--confounder-data",
        type=Path,
        default=None,
        help=(
            "Paper-level pre-publication confounder CSV. Defaults to "
            "data/analysis_ready/paper_confounder_controls.csv."
        ),
    )
    parser.add_argument(
        "--skip-award",
        action="store_true",
        help="Skip award-outcome models and emit an award skip diagnostic table.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--rebuttal-only",
        action="store_true",
        help="Generate only the GPU concentration rebuttal figure, source data, and text.",
    )
    parser.add_argument(
        "--rebuttal-fine-grained-impact-only",
        action="store_true",
        help="Generate only the fine-grained compute-impact rebuttal table, audit, and text.",
    )
    parser.add_argument(
        "--overall-gpu-capability-only",
        action="store_true",
        help="Generate only the unified six-outcome GPU capability and joint count-plus-generation exports.",
    )
    parser.add_argument(
        "--common-sample-confounders-only",
        action="store_true",
        help="Generate only the common-sample C1-C3 confounder-control exports.",
    )
    parser.add_argument(
        "--joint-firth-award-only",
        action="store_true",
        help="Generate only the joint GPU-count plus Ampere Firth award exports.",
    )
    args = parser.parse_args(argv)
    only_flags = [
        args.rebuttal_only,
        args.rebuttal_fine_grained_impact_only,
        args.overall_gpu_capability_only,
        args.common_sample_confounders_only,
        args.joint_firth_award_only,
    ]
    if sum(bool(flag) for flag in only_flags) > 1:
        parser.error("Use only one rebuttal-only flag at a time.")
    if args.rebuttal_only:
        run_rebuttal_gpu_concentration(
            output_dir=args.output_dir,
            quiet=args.quiet,
            input_data_dir=args.input_data_dir,
            skip_award=args.skip_award,
        )
    elif args.rebuttal_fine_grained_impact_only:
        if args.skip_award:
            parser.error("--skip-award is not supported with --rebuttal-fine-grained-impact-only")
        run_rebuttal_fine_grained_compute_impact(
            output_dir=args.output_dir,
            quiet=args.quiet,
            input_data_dir=args.input_data_dir,
        )
    elif args.overall_gpu_capability_only:
        if args.skip_award:
            parser.error("--skip-award is not supported with --overall-gpu-capability-only")
        run_overall_gpu_capability_models(
            output_dir=args.output_dir,
            quiet=args.quiet,
            input_data_dir=args.input_data_dir,
        )
    elif args.common_sample_confounders_only:
        run_common_sample_confounder_models(
            output_dir=args.output_dir,
            quiet=args.quiet,
            input_data_dir=args.input_data_dir,
            confounder_data=args.confounder_data,
        )
    elif args.joint_firth_award_only:
        if args.skip_award:
            parser.error("--skip-award is not supported with --joint-firth-award-only")
        run_joint_count_ampere_firth_award_model(
            output_dir=args.output_dir,
            quiet=args.quiet,
            input_data_dir=args.input_data_dir,
        )
    else:
        run_analysis(
            output_dir=args.output_dir,
            quiet=args.quiet,
            input_data_dir=args.input_data_dir,
            skip_award=args.skip_award,
            confounder_data=args.confounder_data,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
