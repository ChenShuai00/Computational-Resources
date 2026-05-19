from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable
import warnings

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


COMPUTE_FILE = "compute_paper_level_with_contributions_gpu_only.xlsx"
METADATA_FILE = "openalex_paper_metadata_gpu_only.xlsx"
TOPIC_FILE = "acl_arr_topics_all_acl_metadata_desirouter_complete_gpu_only.xlsx"
AWARD_FILE = "acl_award_papers_2020_2025_gpu_only.xlsx"
ORG_VARS_FILE = "paper_level_org_variables_gpu_only.csv"
ORG_LONG_FILE = "paper_organization_long_gpu_only.csv"
ORG_YEAR_PANEL_FILE = "organization_year_panel_gpu_only.csv"


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
        if (parent / "data" / COMPUTE_FILE).exists():
            return parent
    raise FileNotFoundError("Could not find analysis root containing GPU-only inputs.")


ROOT = find_analysis_root(Path(__file__).resolve())
BUNDLE = Path(__file__).resolve().parents[1]
AMPLIFIER_BUNDLE = Path(__file__).resolve().parents[2] / "Amplifier interaction modeling"


def ensure_output_dirs(output_dir: Path | None = None) -> dict[str, Path]:
    base = output_dir if output_dir is not None else BUNDLE
    paths = {
        "base": base,
        "data": base / "data",
        "fig": base / "fig",
        "report": base / "report",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_topic_table(path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    sheet_name = "topics" if "topics" in workbook.sheet_names else workbook.sheet_names[0]
    return pd.read_excel(workbook, sheet_name=sheet_name)


def load_inputs(root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = root or ROOT
    data_dir = root / "data"
    return {
        "compute": pd.read_excel(data_dir / COMPUTE_FILE),
        "metadata": pd.read_excel(data_dir / METADATA_FILE),
        "topics": read_topic_table(data_dir / TOPIC_FILE),
        "awards": pd.read_excel(data_dir / AWARD_FILE),
        "org_vars": pd.read_csv(data_dir / ORG_VARS_FILE),
        "org_long": pd.read_csv(data_dir / ORG_LONG_FILE),
        "org_year_panel": pd.read_csv(data_dir / ORG_YEAR_PANEL_FILE),
    }


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


AMPLIFIER_TOPIC_SPECS = [
    (
        "is_language_modeling",
        "Language Modeling",
        ["Language Modeling"],
    ),
    (
        "is_efficient_methods",
        "Efficient Methods for NLP",
        ["Efficient Methods for NLP"],
    ),
    (
        "is_multimodality",
        "Multimodality and Language Grounding",
        ["Multimodality and Language Grounding to Vision, Robotics and Beyond"],
    ),
    (
        "is_generation",
        "Generation",
        ["Generation"],
    ),
    (
        "is_dialogue_interactive",
        "Dialogue and Interactive Systems",
        ["Dialogue and Interactive Systems"],
    ),
    (
        "is_llm_agents",
        "LLM agents",
        ["LLM agents"],
    ),
    (
        "is_information_extraction",
        "Information Extraction",
        ["Information Extraction"],
    ),
    (
        "is_machine_translation",
        "Machine Translation",
        ["Machine Translation"],
    ),
]


AMPLIFIER_MODERATOR_SPECS = [
    *[
        {
            "moderator": col,
            "label": label,
            "absorbed": True,
            "org_control": True,
        }
        for col, label, _topics in AMPLIFIER_TOPIC_SPECS
    ],
    {
        "moderator": "has_company",
        "label": "Company participation",
        "absorbed": False,
        "org_control": True,
    },
    {
        "moderator": "has_industry_academia",
        "label": "Industry-academia collaboration",
        "absorbed": False,
        "org_control": True,
    },
    {
        "moderator": "is_cross_sector",
        "label": "Cross-sector collaboration",
        "absorbed": False,
        "org_control": True,
    },
    {
        "moderator": "is_multi_institution",
        "label": "Multi-institution collaboration",
        "absorbed": False,
        "org_control": True,
    },
    {
        "moderator": "is_international_collab",
        "label": "International collaboration",
        "absorbed": False,
        "org_control": True,
    },
    {
        "moderator": "has_frontier_gpu",
        "label": "A100/H100-class GPU",
        "absorbed": False,
        "org_control": True,
    },
    {
        "moderator": "has_high_generation_hardware",
        "label": "High-generation hardware",
        "absorbed": False,
        "org_control": True,
    },
]


def add_amplifier_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log10_max_compute_c"] = (
        out["log10_max_compute"] - out["log10_max_compute"].mean(skipna=True)
    )
    if "log10_compute" in out.columns:
        out["log10_compute_c"] = out["log10_compute"] - out["log10_compute"].mean(skipna=True)

    for col in [
        "has_company",
        "has_industry_academia",
        "has_cross_sector",
        "has_multi_organization",
        "has_international_collab",
        "n_organizations",
        "n_org_countries",
    ]:
        if col in out.columns:
            out[col] = safe_numeric(out[col]).fillna(0)
        else:
            out[col] = 0

    topics = out["primary_topic"].fillna("Unknown").astype(str)
    for col, _label, topic_values in AMPLIFIER_TOPIC_SPECS:
        out[col] = topics.isin(topic_values).astype(int)

    out["is_multi_institution"] = (
        (out["n_organizations"].fillna(0) >= 2)
        | (out["has_multi_organization"].fillna(0).eq(1))
    ).astype(int)
    out["is_international_collab"] = (
        (out["n_org_countries"].fillna(0) >= 2)
        | (out["has_international_collab"].fillna(0).eq(1))
    ).astype(int)
    out["is_cross_sector"] = out["has_cross_sector"].fillna(0).astype(int)

    gpu_names = out.get(
        "paper_main_gpu_name",
        pd.Series("", index=out.index, dtype=object),
    ).astype(str)
    frontier_gpu_pattern = r"\b(?:A100|H100|H200|A800|H800|B100|B200|GB200)\b"
    high_generation_pattern = (
        r"\b(?:A100|H100|H200|A800|H800|B100|B200|GB200|RTX\s*4090|L40|L40S)\b"
    )
    out["has_frontier_gpu"] = gpu_names.str.contains(
        frontier_gpu_pattern, case=False, na=False
    ).astype(int)
    out["has_high_generation_hardware"] = gpu_names.str.contains(
        high_generation_pattern, case=False, na=False
    ).astype(int)
    return out.replace([np.inf, -np.inf], np.nan)


def amplifier_control_terms(
    include_org_count_control: bool = True,
    team_control: str = "group",
) -> tuple[list[str], list[str]]:
    if team_control == "group":
        team_term = "C(team_size_group)"
        team_required = "team_size_group"
    elif team_control == "continuous":
        team_term = "log1p_team_size"
        team_required = "log1p_team_size"
    else:
        raise ValueError("team_control must be 'group' or 'continuous'")

    controls = ["C(year_venue)", "C(primary_topic)", team_term]
    required = ["year_venue", "primary_topic", team_required]
    if include_org_count_control:
        controls.append("C(n_organizations_group)")
        required.append("n_organizations_group")
    return controls, required


def make_amplifier_formula(
    outcome: str,
    moderator: str,
    compute_var: str = "log10_max_compute_c",
    moderator_main_absorbed: bool = False,
    include_org_count_control: bool = True,
    include_company_controls: bool = True,
    team_control: str = "group",
) -> tuple[str, list[str]]:
    controls, required = amplifier_control_terms(include_org_count_control, team_control)
    if include_company_controls:
        if moderator != "has_company":
            controls.append("has_company")
            required.append("has_company")
        if moderator != "has_industry_academia":
            controls.append("has_industry_academia")
            required.append("has_industry_academia")

    rhs = (
        [compute_var, f"{compute_var}:{moderator}"]
        if moderator_main_absorbed
        else [f"{compute_var}*{moderator}"]
    )
    rhs = [*rhs, *controls]
    required = sorted(set([outcome, compute_var, moderator, *required]))
    return model_formula(outcome, rhs), required


def scalar_from_test(value) -> float:
    return float(np.asarray(value).ravel()[0])


def fit_amplifier_model(
    df: pd.DataFrame,
    outcome: str,
    moderator: str,
    sample_label: str = "",
    model_family: str = "ols",
    compute_var: str = "log10_max_compute_c",
    moderator_main_absorbed: bool = False,
    include_org_count_control: bool = True,
    include_company_controls: bool = True,
    team_control: str = "group",
    cov_type: str = "HC3",
    min_n: int = 50,
) -> tuple[dict, object, pd.DataFrame]:
    formula, required = make_amplifier_formula(
        outcome=outcome,
        moderator=moderator,
        compute_var=compute_var,
        moderator_main_absorbed=moderator_main_absorbed,
        include_org_count_control=include_org_count_control,
        include_company_controls=include_company_controls,
        team_control=team_control,
    )
    model_df = df.dropna(subset=required).copy()
    if len(model_df) < min_n or model_df[moderator].nunique(dropna=True) < 2:
        raise ValueError(
            f"Insufficient sample or moderator variation for {moderator}: "
            f"n={len(model_df)}, levels={model_df[moderator].nunique(dropna=True)}"
        )

    fit_cov_type = "HC0" if model_family == "poisson" and cov_type == "HC3" else cov_type
    model = fit_formula(formula, model_df, family=model_family, cov_type=fit_cov_type)
    inter_candidates = [f"{compute_var}:{moderator}", f"{moderator}:{compute_var}"]
    inter_name = next((name for name in inter_candidates if name in model.params.index), None)
    if inter_name is None:
        matches = [
            name for name in model.params.index if compute_var in name and moderator in name
        ]
        if len(matches) != 1:
            raise ValueError(f"Cannot find interaction term for {moderator}: {matches}")
        inter_name = matches[0]

    params = list(model.params.index)
    linear_constraint = np.zeros(len(params))
    if compute_var in params:
        linear_constraint[params.index(compute_var)] = 1
    linear_constraint[params.index(inter_name)] += 1
    z1_test = model.t_test(linear_constraint)

    ci = model.conf_int()
    ci_base = ci.loc[compute_var].tolist() if compute_var in model.params.index else [np.nan, np.nan]
    ci_inter = ci.loc[inter_name].tolist()
    ci_z1 = np.asarray(z1_test.conf_int()).ravel().tolist()
    binary_moderator = set(model_df[moderator].dropna().unique()).issubset({0, 1})
    row = {
        "sample": sample_label,
        "outcome": outcome,
        "family": model_family,
        "moderator": moderator,
        "nobs": int(model.nobs),
        "n_Z1": int(model_df[moderator].sum()) if binary_moderator else np.nan,
        "share_Z1": float(model_df[moderator].mean()) if binary_moderator else np.nan,
        "formula": formula,
        "slope_Z0_beta1": float(model.params.get(compute_var, np.nan)),
        "se_slope_Z0": float(model.bse.get(compute_var, np.nan)),
        "p_slope_Z0": float(model.pvalues.get(compute_var, np.nan)),
        "ci_low_slope_Z0": float(ci_base[0]),
        "ci_high_slope_Z0": float(ci_base[1]),
        "beta3_interaction": float(model.params[inter_name]),
        "se_beta3": float(model.bse[inter_name]),
        "p_beta3": float(model.pvalues[inter_name]),
        "ci_low_beta3": float(ci_inter[0]),
        "ci_high_beta3": float(ci_inter[1]),
        "slope_Z1_beta1_plus_beta3": scalar_from_test(z1_test.effect),
        "se_slope_Z1": scalar_from_test(z1_test.sd),
        "p_slope_Z1": scalar_from_test(z1_test.pvalue),
        "ci_low_slope_Z1": float(ci_z1[0]),
        "ci_high_slope_Z1": float(ci_z1[1]),
        "r2": float(getattr(model, "rsquared", np.nan)),
        "interaction_term": inter_name,
    }
    if outcome == "log1p_cites":
        row["beta3_pct_extra_return"] = float(np.exp(row["beta3_interaction"]) - 1)
        row["slope_Z1_pct_return"] = float(np.exp(row["slope_Z1_beta1_plus_beta3"]) - 1)
    if outcome in {"is_highly_cited_all_yv", "is_award"}:
        row["beta3_percentage_points"] = row["beta3_interaction"] * 100
        row["slope_Z1_percentage_points"] = row["slope_Z1_beta1_plus_beta3"] * 100
    return row, model, model_df


def run_amplifier_suite(
    df: pd.DataFrame,
    sample_label: str,
    outcome: str = "log1p_cites",
    family: str | None = None,
    specs: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, pd.DataFrame]]:
    family = family or ("lpm" if outcome in {"is_highly_cited_all_yv", "is_award"} else "ols")
    rows = []
    models = {}
    samples = {}
    for spec in specs or AMPLIFIER_MODERATOR_SPECS:
        moderator = spec["moderator"]
        try:
            row, model, model_df = fit_amplifier_model(
                df,
                outcome=outcome,
                moderator=moderator,
                sample_label=sample_label,
                model_family=family,
                moderator_main_absorbed=spec["absorbed"],
                include_org_count_control=spec["org_control"],
                cov_type="HC3",
            )
            row["moderator_label"] = spec["label"]
            row["moderator_main_absorbed_by_FE"] = spec["absorbed"]
            rows.append(row)
            models[moderator] = model
            samples[moderator] = model_df
        except Exception as exc:
            rows.append(
                {
                    "sample": sample_label,
                    "outcome": outcome,
                    "family": family,
                    "moderator": moderator,
                    "moderator_label": spec["label"],
                    "moderator_main_absorbed_by_FE": spec["absorbed"],
                    "error": repr(exc),
                }
            )

    table = pd.DataFrame(rows)
    if "p_beta3" in table.columns:
        mask = table["p_beta3"].notna()
        if mask.any():
            table.loc[mask, "q_beta3_fdr_bh"] = multipletests(
                table.loc[mask, "p_beta3"],
                method="fdr_bh",
            )[1]
    if "beta3_interaction" in table.columns:
        return table.sort_values("beta3_interaction", ascending=False), models, samples
    return table, models, samples


def make_topic_dummies(
    df: pd.DataFrame,
    topic_col: str = "primary_topic",
    min_n: int = 50,
) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    specs = []
    for topic_name, n_topic in out[topic_col].value_counts(dropna=False).items():
        if topic_name == "Unknown" or pd.isna(topic_name) or n_topic < min_n:
            continue
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(topic_name)).strip("_").lower()
        col = f"topic__{safe[:70]}"
        base = col
        suffix = 1
        while col in out.columns:
            col = f"{base}_{suffix}"
            suffix += 1
        out[col] = out[topic_col].eq(topic_name).astype(int)
        specs.append(
            {
                "moderator": col,
                "label": str(topic_name),
                "absorbed": True,
                "org_control": True,
                "topic": str(topic_name),
                "n_topic_raw": int(n_topic),
            }
        )
    return out, specs


def run_topic_amplifier_suite(
    df: pd.DataFrame,
    sample_label: str,
    outcome: str = "log1p_cites",
    min_n: int = 50,
) -> pd.DataFrame:
    df2, specs = make_topic_dummies(df, min_n=min_n)
    table, _models, _samples = run_amplifier_suite(
        df2,
        sample_label=sample_label,
        outcome=outcome,
        specs=specs,
    )
    if table.empty:
        return table
    topic_lookup = {spec["moderator"]: spec for spec in specs}
    table["topic"] = table["moderator"].map(
        lambda moderator: topic_lookup.get(moderator, {}).get("topic", np.nan)
    )
    table["n_topic_raw"] = table["moderator"].map(
        lambda moderator: topic_lookup.get(moderator, {}).get("n_topic_raw", np.nan)
    )
    return table


def fit_combined_amplifier_model(
    df: pd.DataFrame,
    sample_label: str,
    outcome: str = "log1p_cites",
    compute_var: str = "log10_max_compute_c",
) -> tuple[pd.DataFrame, object]:
    absorbed_mods = [spec["moderator"] for spec in AMPLIFIER_MODERATOR_SPECS if spec["absorbed"]]
    explicit_mods = [
        "has_company",
        "has_industry_academia",
        "is_cross_sector",
        "is_multi_institution",
        "is_international_collab",
        "has_frontier_gpu",
    ]
    interactions = [f"{compute_var}:{moderator}" for moderator in absorbed_mods + explicit_mods]
    rhs = [
        compute_var,
        *explicit_mods,
        *interactions,
        "C(year_venue)",
        "C(primary_topic)",
        "C(team_size_group)",
        "C(n_organizations_group)",
    ]
    required = sorted(
        set(
            [
                outcome,
                compute_var,
                *absorbed_mods,
                *explicit_mods,
                "year_venue",
                "primary_topic",
                "team_size_group",
                "n_organizations_group",
            ]
        )
    )
    model_df = df.dropna(subset=required).copy()
    family = "lpm" if outcome in {"is_highly_cited_all_yv", "is_award"} else "ols"
    model = fit_formula(model_formula(outcome, rhs), model_df, family=family, cov_type="HC3")
    label_lookup = {
        spec["moderator"]: spec["label"] for spec in AMPLIFIER_MODERATOR_SPECS
    }
    rows = []
    for moderator in absorbed_mods + explicit_mods:
        inter_name = f"{compute_var}:{moderator}"
        if inter_name not in model.params.index:
            inter_name = f"{moderator}:{compute_var}"
        if inter_name not in model.params.index:
            continue
        ci_low, ci_high = model.conf_int().loc[inter_name].tolist()
        rows.append(
            {
                "sample": sample_label,
                "outcome": outcome,
                "family": family,
                "moderator": moderator,
                "moderator_label": label_lookup.get(moderator, moderator),
                "nobs": int(model.nobs),
                "beta3_interaction": float(model.params[inter_name]),
                "se_beta3": float(model.bse[inter_name]),
                "p_beta3": float(model.pvalues[inter_name]),
                "ci_low_beta3": float(ci_low),
                "ci_high_beta3": float(ci_high),
                "interaction_term": inter_name,
                "formula": model_formula(outcome, rhs),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["q_beta3_fdr_bh"] = multipletests(table["p_beta3"], method="fdr_bh")[1]
    return table, model


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
    org_history = build_org_history_controls(paper_org_long, org_year_panel)
    master_org = master.merge(org_history, on="paper_id", how="left", validate="one_to_one")
    sample = make_analysis_sample(master_org, sample="gpu_lb1", year_min=2020, year_max=2023)
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
                "model": "baseline on institution-control sample",
                "nobs": int(base_model.nobs),
                "coef": base_model.params["log10_max_compute"],
                "se": base_model.bse["log10_max_compute"],
                "p": base_model.pvalues["log10_max_compute"],
                "r2": base_model.rsquared,
            },
            {
                "model": "plus prior org history/collab controls",
                "nobs": int(ext_model.nobs),
                "coef": ext_model.params["log10_max_compute"],
                "se": ext_model.bse["log10_max_compute"],
                "p": ext_model.pvalues["log10_max_compute"],
                "r2": ext_model.rsquared,
            },
        ]
    )


def save_pub_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> dict[str, str]:
    paths: dict[str, str] = {}
    path = stem.with_suffix(".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    paths["png"] = str(path)
    return paths


def plot_amplifier_forest(
    table: pd.DataFrame,
    fig_dir: Path,
    stem: str,
    label_col: str = "moderator_label",
) -> dict[str, str]:
    plot_df = table.dropna(
        subset=["beta3_interaction", "ci_low_beta3", "ci_high_beta3"]
    ).copy()
    plot_df = plot_df.sort_values("beta3_interaction")

    fig_height = max(3.6, 0.28 * max(len(plot_df), 1) + 1.1)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No estimable amplifier interactions", ha="center", va="center")
        ax.set_axis_off()
    else:
        y_pos = np.arange(len(plot_df))
        ax.errorbar(
            plot_df["beta3_interaction"],
            y_pos,
            xerr=[
                plot_df["beta3_interaction"] - plot_df["ci_low_beta3"],
                plot_df["ci_high_beta3"] - plot_df["beta3_interaction"],
            ],
            fmt="o",
            capsize=2.5,
            color="#2F6DA3",
            ecolor="#777777",
            markersize=3.5,
        )
        ax.axvline(0, linestyle="--", linewidth=0.8, color="#555555")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df[label_col].astype(str))
        ax.set_xlabel("beta3: extra slope per 10x GPU compute")
        ax.grid(axis="x", color="#E8E8E8", linewidth=0.45)
    plt.tight_layout()
    paths = save_pub_figure(fig, fig_dir / stem)
    plt.close(fig)
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
    report = f"""# RQ3 GPU-only citation modeling

This report re-runs the citation-impact modeling workflow on the GPU-only input bundle.

## Sample audit

- GPU-only master papers: {audit['master_rows']}
- 2020-2023 LB1/GFIMP GPU sample papers: {audit['gpu_2020_2023_rows']}
- 2020-2023 strict raw GPU sample papers: {audit['strict_2020_2023_rows']}
- 2020-2025 LB1/GFIMP GPU sample papers: {audit['gpu_2020_2025_rows']}

## Main results

{main_table}

## Complete regression results

{full_table}

## Robustness analyses

{robustness_table}

## Model specification

RQ3 uses year-by-venue fixed effects, primary-topic fixed effects, team-size group,
and organization-count group controls. It intentionally excludes `contribution_type`
and all contribution-label proxy controls.

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

Amplifier interaction models are exported separately under `4.4/Amplifier interaction modeling`.
"""
    report_path.write_text(report, encoding="utf-8")


def write_amplifier_report(
    report_path: Path,
    amplifier_results: pd.DataFrame,
    audit: dict[str, int],
) -> None:
    amplifier_cols = [
        "sample",
        "outcome",
        "moderator_label",
        "nobs",
        "n_Z1",
        "beta3_interaction",
        "p_beta3",
        "q_beta3_fdr_bh",
        "ci_low_beta3",
        "ci_high_beta3",
    ]
    amplifier_csv = amplifier_results[
        [col for col in amplifier_cols if col in amplifier_results.columns]
    ].to_csv(index=False)
    report = f"""# RQ3 GPU-only amplifier interaction modeling

This report contains the GPU-only amplifier analysis separated from the main
`Citation modeling` outputs.

## Sample audit

- GPU-only master papers: {audit['master_rows']}
- 2020-2023 LB1/GFIMP GPU sample papers: {audit['gpu_2020_2023_rows']}
- 2020-2023 strict raw GPU sample papers: {audit['strict_2020_2023_rows']}
- 2020-2025 LB1/GFIMP GPU sample papers: {audit['gpu_2020_2025_rows']}

## Amplifier interaction models

The GPU-only amplifier analysis estimates interaction models where the slope on
`log10_max_compute` varies by topic, organization, and hardware moderators. The
core parameter is the interaction coefficient beta3: positive beta3 values indicate
higher marginal citation returns to a 10x GPU-compute increase when the moderator is
present. These are descriptive associations, not causal estimates.

```csv
{amplifier_csv.strip()}
```
"""
    report_path.write_text(report, encoding="utf-8")


def run_analysis(output_dir: Path | str | None = None, quiet: bool = False) -> dict:
    output_base = Path(output_dir) if output_dir is not None else None
    dirs = ensure_output_dirs(output_base)
    amplifier_dirs = ensure_output_dirs(
        (output_base / "Amplifier interaction modeling") if output_base is not None else AMPLIFIER_BUNDLE
    )
    inputs = load_inputs()
    master = build_master_panel(
        inputs["compute"],
        inputs["metadata"],
        inputs["topics"],
        inputs["awards"],
        inputs["org_vars"],
    )

    gpu_2020_2023 = make_analysis_sample(master, sample="gpu_lb1", year_min=2020, year_max=2023)
    strict_2020_2023 = make_analysis_sample(master, sample="strict_raw", year_min=2020, year_max=2023)
    gpu_2020_2025 = make_analysis_sample(master, sample="gpu_lb1", year_min=2020, year_max=2025)
    strict_2020_2025 = make_analysis_sample(master, sample="strict_raw", year_min=2020, year_max=2025)

    gpu_2020_2023_amp = add_amplifier_variables(gpu_2020_2023)
    strict_2020_2023_amp = add_amplifier_variables(strict_2020_2023)
    gpu_2020_2025_amp = add_amplifier_variables(gpu_2020_2025)
    strict_2020_2025_amp = add_amplifier_variables(strict_2020_2025)

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

    gpu_ols_log = fit_model_grid(
        gpu_2020_2023,
        outcome="log1p_cites",
        family="ols",
        max_spec=7,
        cov_type="HC3",
    )
    strict_raw_ols_log = fit_model_grid(
        strict_2020_2023,
        outcome="log1p_cites",
        family="ols",
        max_spec=7,
        cov_type="HC3",
    )
    gpu_ppml = fit_model_grid(
        gpu_2020_2023,
        outcome="cited_by_count",
        family="poisson",
        max_spec=7,
        cov_type="HC0",
    )
    gpu_ols_norm = fit_model_grid(
        gpu_2020_2023,
        outcome="citation_normalized_percentile",
        family="ols",
        max_spec=7,
        cov_type="HC3",
    )
    gpu_lpm_high = fit_model_grid(
        gpu_2020_2023,
        outcome="is_highly_cited_all_yv",
        family="lpm",
        max_spec=7,
        cov_type="HC3",
    )
    gpu_lpm_award = fit_model_grid(
        gpu_2020_2025,
        outcome="is_award",
        family="lpm",
        max_spec=7,
        cov_type="HC3",
    )

    main_results = pd.DataFrame(
        [
            pick_result(gpu_ols_log, "GPU-only OLS: log1p citations", 7, "approx pct change in 1+cites per 10x GPU compute"),
            pick_result(
                strict_raw_ols_log,
                "Strict raw GPU OLS: log1p citations",
                7,
                "approx pct change in 1+cites per 10x GPU compute",
            ),
            pick_result(gpu_ppml, "GPU-only PPML: citation count", 7, "pct change in expected citations per 10x GPU compute"),
            pick_result(
                gpu_ols_norm,
                "GPU-only OLS: normalized citation percentile",
                7,
                "percentile-point scale, 0-1",
            ),
            pick_result(gpu_lpm_high, "GPU-only LPM: high cited all-yv top10", 7, "percentage points"),
            pick_result(gpu_lpm_award, "GPU-only LPM: award", 7, "percentage points"),
        ]
    )

    all_effect_tables = pd.concat(
        [
            compact_effect_table(gpu_ols_log, "GPU-only OLS: log1p citations"),
            compact_effect_table(strict_raw_ols_log, "Strict raw GPU OLS: log1p citations"),
            compact_effect_table(gpu_ppml, "GPU-only PPML: citation count"),
            compact_effect_table(gpu_ols_norm, "GPU-only OLS: normalized citation percentile"),
            compact_effect_table(gpu_lpm_high, "GPU-only LPM: high cited all-yv top10"),
            compact_effect_table(gpu_lpm_award, "GPU-only LPM: award"),
        ],
        ignore_index=True,
    )

    gpu_amp_log, _gpu_amp_log_models, _gpu_amp_log_samples = run_amplifier_suite(
        gpu_2020_2023_amp,
        sample_label="gpu_lb1_2020_2023",
        outcome="log1p_cites",
    )
    strict_amp_log, _strict_amp_log_models, _strict_amp_log_samples = run_amplifier_suite(
        strict_2020_2023_amp,
        sample_label="strict_raw_2020_2023",
        outcome="log1p_cites",
    )
    gpu_amp_high, _gpu_amp_high_models, _gpu_amp_high_samples = run_amplifier_suite(
        gpu_2020_2023_amp,
        sample_label="gpu_lb1_2020_2023",
        outcome="is_highly_cited_all_yv",
    )
    gpu_amp_norm, _gpu_amp_norm_models, _gpu_amp_norm_samples = run_amplifier_suite(
        gpu_2020_2023_amp,
        sample_label="gpu_lb1_2020_2023",
        outcome="citation_normalized_percentile",
    )
    gpu_amp_award, _gpu_amp_award_models, _gpu_amp_award_samples = run_amplifier_suite(
        gpu_2020_2025_amp,
        sample_label="gpu_lb1_2020_2025",
        outcome="is_award",
    )
    strict_amp_award, _strict_amp_award_models, _strict_amp_award_samples = run_amplifier_suite(
        strict_2020_2025_amp,
        sample_label="strict_raw_2020_2025",
        outcome="is_award",
    )
    all_amplifier_results = pd.concat(
        [
            gpu_amp_log,
            gpu_amp_high,
            gpu_amp_norm,
            gpu_amp_award,
            strict_amp_log,
            strict_amp_award,
        ],
        ignore_index=True,
    )
    gpu_topic_amp_log = run_topic_amplifier_suite(
        gpu_2020_2023_amp,
        sample_label="gpu_lb1_2020_2023",
        outcome="log1p_cites",
        min_n=50,
    )
    gpu_combined_amp_log, _gpu_combined_amp_model = fit_combined_amplifier_model(
        gpu_2020_2023_amp,
        sample_label="gpu_lb1_2020_2023",
        outcome="log1p_cites",
    )

    robust_rows = []
    for compute_var in ["log10_max_compute", "log10_compute"]:
        for team_control in ["group", "continuous"]:
            result = fit_model_grid(
                gpu_2020_2023,
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
            outlier_filtered(gpu_2020_2023, **kwargs),
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
            gpu_2020_2023,
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
    for year in sorted(gpu_2020_2023["year"].dropna().unique()):
        result = fit_model_grid(
            gpu_2020_2023.loc[gpu_2020_2023["year"] != year].copy(),
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
    for venue in sorted(gpu_2020_2023["venue"].dropna().unique()):
        result = fit_model_grid(
            gpu_2020_2023.loc[gpu_2020_2023["venue"] != venue].copy(),
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

    award_sample = gpu_lpm_award["common_df"]
    award_model_7 = gpu_lpm_award["models"].get("lpm_7")
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

    strength_table = pd.DataFrame([effect_strength_ols(gpu_ols_log, spec=7)])
    delta_r2_table = pd.DataFrame([delta_r2_for_spec(gpu_2020_2023, "log1p_cites", spec_id=7)])
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
        "gpu_lb1_amplifier_log1p_cites": amplifier_dirs["data"] / "gpu_lb1_amplifier_log1p_cites.csv",
        "strict_raw_amplifier_log1p_cites": amplifier_dirs["data"] / "strict_raw_amplifier_log1p_cites.csv",
        "gpu_only_all_amplifier_interaction_results": amplifier_dirs["data"]
        / "gpu_only_all_amplifier_interaction_results.csv",
        "gpu_lb1_topic_amplifier_log1p_cites": amplifier_dirs["data"]
        / "gpu_lb1_topic_amplifier_log1p_cites.csv",
        "gpu_lb1_combined_interaction_log1p_cites": amplifier_dirs["data"]
        / "gpu_lb1_combined_interaction_log1p_cites.csv",
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
        "gpu_lb1_amplifier_log1p_cites": gpu_amp_log,
        "strict_raw_amplifier_log1p_cites": strict_amp_log,
        "gpu_only_all_amplifier_interaction_results": all_amplifier_results,
        "gpu_lb1_topic_amplifier_log1p_cites": gpu_topic_amp_log,
        "gpu_lb1_combined_interaction_log1p_cites": gpu_combined_amp_log,
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
        "gpu_lb1_amplifier_beta3_forest": plot_amplifier_forest(
            gpu_amp_log,
            amplifier_dirs["fig"],
            "gpu_lb1_amplifier_beta3_forest",
        ),
        "gpu_lb1_topic_amplifier_beta3_forest": plot_amplifier_forest(
            gpu_topic_amp_log,
            amplifier_dirs["fig"],
            "gpu_lb1_topic_amplifier_beta3_forest",
            label_col="topic",
        ),
    }

    outputs = {
        "tables": {key: str(path) for key, path in tables.items()},
        "figures": figure_paths,
        "reports": {
            "rq3_gpu_only_citation_modeling": str(
                dirs["report"] / "rq3_gpu_only_citation_modeling.md"
            ),
            "rq3_gpu_only_amplifier_interactions": str(
                amplifier_dirs["report"] / "rq3_gpu_only_amplifier_interactions.md"
            ),
        },
    }
    audit = {
        "master_rows": len(master),
        "gpu_2020_2023_rows": len(gpu_2020_2023),
        "strict_2020_2023_rows": len(strict_2020_2023),
        "gpu_2020_2025_rows": len(gpu_2020_2025),
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
    write_amplifier_report(
        Path(outputs["reports"]["rq3_gpu_only_amplifier_interactions"]),
        gpu_amp_log,
        audit,
    )

    if not quiet:
        print(json.dumps(outputs, indent=2))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RQ3 GPU-only citation modeling.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    run_analysis(output_dir=args.output_dir, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



