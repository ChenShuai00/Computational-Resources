from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


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
        if (
            parent / "data" / "compute_paper_level_with_contributions_gpu_only.xlsx"
        ).exists():
            return parent
    raise FileNotFoundError("Could not find analysis root containing GPU-only inputs.")


ROOT = find_analysis_root(Path(__file__).resolve())
COMPUTE_INPUT = ROOT / "data" / "compute_paper_level_with_contributions_gpu_only.xlsx"
ORG_INPUT = ROOT / "data" / "paper_level_org_variables_gpu_only.csv"
ORG_LONG_INPUT = ROOT / "data" / "paper_organization_long_gpu_only.csv"
TOPIC_INPUT = (
    ROOT
    / "data"
    / "acl_arr_topics_all_acl_metadata_desirouter_complete_gpu_only.xlsx"
)
BUNDLE = Path(__file__).resolve().parents[1]
OUT_DATA = BUNDLE / "data"
OUT_FIG = BUNDLE / "fig"
OUT_REPORT = BUNDLE / "report"

FE_TERMS = "C(year_fe) + C(topic_fe) + C(venue_fe)"
STRICT_OUTCOME = "log10_max_tflops_raw"
MAIN_OUTCOME = STRICT_OUTCOME
REGRESSION_SAMPLE = "strict_raw_max_row"
TOP_COMPUTE_QUANTILE = 0.80
CORE_TERMS = [
    "Company_i",
    "IndustryAcademia_i",
    "CrossSector_i",
    "International_i",
    "log1p_nOrganizations_i",
]
ACCESS_REGIME_ORDER = [
    "Academic-only",
    "Industry-only",
    "Industry-academia collaboration",
    "Other cross-sector collaboration",
    "Other / mixed",
]


class FocalSpec(NamedTuple):
    model: str
    term: str
    formula_term: str
    label: str


FOCAL_SPECS = [
    FocalSpec("M1", "Company_i", "Company_i", "Company"),
    FocalSpec("M2", "IndustryAcademia_i", "IndustryAcademia_i", "Industry-academia"),
    FocalSpec("M3", "CrossSector_i", "CrossSector_i", "Cross-sector"),
    FocalSpec("M4", "International_i", "International_i", "International"),
    FocalSpec("M5", "log1p_nOrganizations_i", "log1p_nOrganizations_i", "log(1+n organizations)"),
]

COUNTRY_CODE_ALIASES = {"HK": "CN", "TW": "CN"}

COUNTRY_REGION = {
    "AE": "Middle East and North Africa",
    "AM": "Europe and Central Asia",
    "AR": "Latin America and Caribbean",
    "AT": "Europe and Central Asia",
    "AU": "East Asia and Pacific",
    "BD": "South Asia",
    "BE": "Europe and Central Asia",
    "BG": "Europe and Central Asia",
    "BI": "Sub-Saharan Africa",
    "BR": "Latin America and Caribbean",
    "CA": "North America",
    "CH": "Europe and Central Asia",
    "CL": "Latin America and Caribbean",
    "CM": "Sub-Saharan Africa",
    "CN": "East Asia and Pacific",
    "CU": "Latin America and Caribbean",
    "CZ": "Europe and Central Asia",
    "DE": "Europe and Central Asia",
    "DK": "Europe and Central Asia",
    "EE": "Europe and Central Asia",
    "EG": "Middle East and North Africa",
    "ES": "Europe and Central Asia",
    "FI": "Europe and Central Asia",
    "FR": "Europe and Central Asia",
    "GB": "Europe and Central Asia",
    "GE": "Europe and Central Asia",
    "GH": "Sub-Saharan Africa",
    "GR": "Europe and Central Asia",
    "GT": "Latin America and Caribbean",
    "HR": "Europe and Central Asia",
    "HU": "Europe and Central Asia",
    "ID": "East Asia and Pacific",
    "IE": "Europe and Central Asia",
    "IL": "Middle East and North Africa",
    "IN": "South Asia",
    "IQ": "Middle East and North Africa",
    "IR": "Middle East and North Africa",
    "IT": "Europe and Central Asia",
    "JO": "Middle East and North Africa",
    "JP": "East Asia and Pacific",
    "KE": "Sub-Saharan Africa",
    "KH": "East Asia and Pacific",
    "KR": "East Asia and Pacific",
    "KZ": "Europe and Central Asia",
    "LB": "Middle East and North Africa",
    "LK": "South Asia",
    "LT": "Europe and Central Asia",
    "LU": "Europe and Central Asia",
    "MA": "Middle East and North Africa",
    "MK": "Europe and Central Asia",
    "ML": "Sub-Saharan Africa",
    "MO": "East Asia and Pacific",
    "MR": "Sub-Saharan Africa",
    "MW": "Sub-Saharan Africa",
    "MX": "Latin America and Caribbean",
    "MY": "East Asia and Pacific",
    "MZ": "Sub-Saharan Africa",
    "NG": "Sub-Saharan Africa",
    "NL": "Europe and Central Asia",
    "NO": "Europe and Central Asia",
    "NZ": "East Asia and Pacific",
    "PH": "East Asia and Pacific",
    "PK": "South Asia",
    "PL": "Europe and Central Asia",
    "PR": "Latin America and Caribbean",
    "PS": "Middle East and North Africa",
    "PT": "Europe and Central Asia",
    "QA": "Middle East and North Africa",
    "RO": "Europe and Central Asia",
    "RU": "Europe and Central Asia",
    "SA": "Middle East and North Africa",
    "SE": "Europe and Central Asia",
    "SG": "East Asia and Pacific",
    "SI": "Europe and Central Asia",
    "SK": "Europe and Central Asia",
    "SN": "Sub-Saharan Africa",
    "SO": "Sub-Saharan Africa",
    "TG": "Sub-Saharan Africa",
    "TH": "East Asia and Pacific",
    "TR": "Europe and Central Asia",
    "UG": "Sub-Saharan Africa",
    "US": "North America",
    "UY": "Latin America and Caribbean",
    "VN": "East Asia and Pacific",
    "ZA": "Sub-Saharan Africa",
    "ZM": "Sub-Saharan Africa",
    "ZW": "Sub-Saharan Africa",
}


def ensure_dirs() -> None:
    for path in (OUT_DATA, OUT_FIG, OUT_REPORT):
        path.mkdir(parents=True, exist_ok=True)


def save_pub_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing {name} columns: {missing}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compute = pd.read_excel(COMPUTE_INPUT)
    org_vars = pd.read_csv(ORG_INPUT)
    topics = pd.read_excel(TOPIC_INPUT, sheet_name="topics")

    _require_columns(
        compute,
        {
            "paper_id",
            "is_strict",
            "is_lb1_gfimp",
            "paper_year",
            "paper_venue",
            "paper_max_row_compute_capability",
            "paper_max_row_compute_capability_gfimp_lb1",
        },
        "compute",
    )
    _require_columns(
        org_vars,
        {
            "paper_id",
            "n_organizations",
            "has_company",
            "has_education",
            "has_industry_academia",
            "has_cross_sector",
            "has_international_collab",
        },
        "organization variable",
    )
    _require_columns(topics, {"paper_id", "topic"}, "topic")
    return compute, org_vars, topics


def build_panel() -> pd.DataFrame:
    compute, org_vars, topics = load_inputs()

    compute_keep = [
        "paper_id",
        "is_strict",
        "is_lb1_gfimp",
        "paper_year",
        "paper_venue",
        "paper_max_row_compute_capability",
        "paper_max_row_compute_capability_gfimp_lb1",
    ]
    org_keep = [
        "paper_id",
        "n_organizations",
        "has_company",
        "has_education",
        "has_industry_academia",
        "has_cross_sector",
        "has_international_collab",
    ]
    topic_keep = ["paper_id", "topic"]

    panel = (
        compute[compute_keep]
        .merge(org_vars[org_keep], on="paper_id", how="inner", validate="one_to_one")
        .merge(topics[topic_keep], on="paper_id", how="inner", validate="one_to_one")
    )

    numeric_cols = [
        "is_strict",
        "is_lb1_gfimp",
        "paper_year",
        "paper_max_row_compute_capability",
        "paper_max_row_compute_capability_gfimp_lb1",
        "n_organizations",
        "has_company",
        "has_education",
        "has_industry_academia",
        "has_cross_sector",
        "has_international_collab",
    ]
    for col in numeric_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel["max_tflops_raw"] = panel["paper_max_row_compute_capability"] / 1e12
    panel["max_tflops_lb1"] = panel["paper_max_row_compute_capability_gfimp_lb1"] / 1e12
    panel["log10_max_tflops_raw"] = np.log10(panel["max_tflops_raw"].where(panel["max_tflops_raw"].gt(0)))
    panel["log10_max_tflops_lb1"] = np.log10(panel["max_tflops_lb1"].where(panel["max_tflops_lb1"].gt(0)))
    panel["log1p_n_organizations"] = np.log1p(panel["n_organizations"])

    panel["Company_i"] = panel["has_company"]
    panel["IndustryAcademia_i"] = panel["has_industry_academia"]
    panel["CrossSector_i"] = panel["has_cross_sector"]
    panel["International_i"] = panel["has_international_collab"]
    panel["nOrganizations_i"] = panel["n_organizations"]
    panel["log1p_nOrganizations_i"] = panel["log1p_n_organizations"]

    year_values = panel["paper_year"].round().astype("Int64")
    panel["year_fe"] = year_values.astype(str).replace("<NA>", np.nan)
    panel["topic_fe"] = panel["topic"].astype("object")
    panel["venue_fe"] = panel["paper_venue"].astype("object")

    panel = panel.replace([np.inf, -np.inf], np.nan)
    if not panel["paper_id"].is_unique:
        duplicates = panel.loc[panel["paper_id"].duplicated(), "paper_id"].head(5).tolist()
        raise ValueError(f"Panel is not paper-level unique; duplicate paper_id examples: {duplicates}")
    return panel


def _model_sample(df: pd.DataFrame, outcome: str, terms: list[str], sample: str) -> pd.DataFrame:
    required = [outcome, *terms, "year_fe", "topic_fe", "venue_fe"]
    work = df.copy()
    if sample == "lb1_gfimp_max_row":
        work = work[work["is_lb1_gfimp"].eq(1)]
    elif sample == "strict_raw_max_row":
        work = work[work["is_strict"].eq(1)]
    else:
        raise ValueError(f"Unknown model sample: {sample}")
    return work.dropna(subset=required).copy()


def _summarize_result(
    *,
    result,
    model: str,
    model_family: str,
    outcome: str,
    sample: str,
    formula: str,
    terms: list[str],
    labels: dict[str, str],
) -> pd.DataFrame:
    rows = []
    conf = result.conf_int()
    for term in terms:
        coef = float(result.params[term])
        ci_low = float(conf.loc[term, 0])
        ci_high = float(conf.loc[term, 1])
        rows.append(
            {
                "model": model,
                "model_family": model_family,
                "term": term,
                "term_label": labels.get(term, term),
                "outcome": outcome,
                "sample": sample,
                "nobs": int(result.nobs),
                "r_squared": float(result.rsquared),
                "adj_r_squared": float(result.rsquared_adj),
                "coef": coef,
                "std_err": float(result.bse[term]),
                "p_value": float(result.pvalues[term]),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "percent_change": float((10**coef - 1) * 100),
                "percent_change_ci_low": float((10**ci_low - 1) * 100),
                "percent_change_ci_high": float((10**ci_high - 1) * 100),
                "formula": formula,
                "cov_type": "HC3",
            }
        )
    return pd.DataFrame(rows)


def fit_single_model(
    df: pd.DataFrame,
    *,
    model: str,
    model_family: str,
    outcome: str,
    terms: list[str],
    sample: str,
    labels: dict[str, str],
) -> pd.DataFrame:
    work = _model_sample(df, outcome, terms, sample)
    formula = f"{outcome} ~ {' + '.join(terms)} + {FE_TERMS}"
    result = smf.ols(formula, data=work).fit(cov_type="HC3")
    return _summarize_result(
        result=result,
        model=model,
        model_family=model_family,
        outcome=outcome,
        sample=sample,
        formula=formula,
        terms=terms,
        labels=labels,
    )


def fit_main_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in FOCAL_SPECS:
        rows.append(
            fit_single_model(
                df,
                model=spec.model,
                model_family="main_one_focal_fe",
                outcome=MAIN_OUTCOME,
                terms=[spec.formula_term],
                sample=REGRESSION_SAMPLE,
                labels={spec.formula_term: spec.label},
            )
        )
    return pd.concat(rows, ignore_index=True)


def fit_full_model(df: pd.DataFrame) -> pd.DataFrame:
    labels = {spec.formula_term: spec.label for spec in FOCAL_SPECS}
    return fit_single_model(
        df,
        model="A1",
        model_family="appendix_full_model",
        outcome=MAIN_OUTCOME,
        terms=[spec.formula_term for spec in FOCAL_SPECS],
        sample=REGRESSION_SAMPLE,
        labels=labels,
    )


def fit_strict_robustness_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, spec in enumerate(FOCAL_SPECS, start=1):
        rows.append(
            fit_single_model(
                df,
                model=f"S{idx}",
                model_family="strict_raw_robustness",
                outcome=STRICT_OUTCOME,
                terms=[spec.formula_term],
                sample="strict_raw_max_row",
                labels={spec.formula_term: spec.label},
            )
        )
    return pd.concat(rows, ignore_index=True)


def normalize_country_code(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    code = str(value).strip().upper()
    if code in {"", "NAN", "NONE", "NULL"}:
        return pd.NA
    return COUNTRY_CODE_ALIASES.get(code, code)


def country_to_region(code: object) -> str:
    normalized = normalize_country_code(code)
    if pd.isna(normalized):
        return "Unknown"
    return COUNTRY_REGION.get(str(normalized), "Other")


def add_high_compute_tail(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = panel.copy()
    thresholds = (
        work.groupby("year_fe", as_index=False)["max_tflops_lb1"]
        .quantile(TOP_COMPUTE_QUANTILE)
        .rename(columns={"max_tflops_lb1": "year_p80_cutoff_tflops"})
    )
    work = work.merge(thresholds, on="year_fe", how="left", validate="many_to_one")
    work["is_high_compute_tail"] = work["max_tflops_lb1"].ge(work["year_p80_cutoff_tflops"]).astype(int)
    return work, thresholds


def classify_access_regime(row: pd.Series) -> str:
    has_company = bool(row["has_company"] == 1)
    has_education = bool(row["has_education"] == 1)
    has_industry_academia = bool(row["has_industry_academia"] == 1)
    has_cross_sector = bool(row["has_cross_sector"] == 1)

    if has_industry_academia:
        return "Industry-academia collaboration"
    if has_education and not has_company and not has_cross_sector:
        return "Academic-only"
    if has_company and not has_education and not has_cross_sector:
        return "Industry-only"
    if has_cross_sector:
        return "Other cross-sector collaboration"
    return "Other / mixed"


def add_access_regime(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work["access_regime"] = work.apply(classify_access_regime, axis=1)
    work["access_regime"] = pd.Categorical(
        work["access_regime"],
        categories=ACCESS_REGIME_ORDER,
        ordered=True,
    )
    return work


def build_access_regime_summary(panel: pd.DataFrame) -> pd.DataFrame:
    work, _ = add_high_compute_tail(add_access_regime(panel))
    summary = (
        work.groupby("access_regime", observed=False)
        .agg(
            papers=("paper_id", "nunique"),
            median_reported_gpu_capacity_tflops=("max_tflops_lb1", "median"),
            q25_reported_gpu_capacity_tflops=("max_tflops_lb1", lambda x: x.quantile(0.25)),
            q75_reported_gpu_capacity_tflops=("max_tflops_lb1", lambda x: x.quantile(0.75)),
            top20_compute_tail_papers=("is_high_compute_tail", "sum"),
        )
        .reset_index()
    )
    summary["top20_compute_tail_share"] = summary["top20_compute_tail_papers"] / summary["papers"]
    summary["paper_share"] = summary["papers"] / summary["papers"].sum()
    return summary


def gini(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[arr >= 0]
    if arr.size == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1) / n)


def _metric_row(
    level: str,
    metric: str,
    value: float,
    *,
    numerator: float | int | None = None,
    denominator: float | int | None = None,
    detail: str = "",
    method: str = "",
) -> dict:
    return {
        "level": level,
        "metric": metric,
        "detail": detail,
        "value": float(value),
        "numerator": np.nan if numerator is None else float(numerator),
        "denominator": np.nan if denominator is None else float(denominator),
        "method": method,
    }


def build_concentration_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    org_long = pd.read_csv(ORG_LONG_INPUT)
    _require_columns(org_long, {"paper_id", "org_country_code"}, "organization long")

    work, _ = add_high_compute_tail(panel)
    rows = []

    countries = org_long[["paper_id", "org_country_code"]].copy()
    countries["country_code"] = countries["org_country_code"].map(normalize_country_code)
    countries = countries[countries["paper_id"].notna() & countries["country_code"].notna()]
    countries = countries[["paper_id", "country_code"]].drop_duplicates()
    country_panel = countries.merge(
        work[["paper_id", "max_tflops_lb1", "is_high_compute_tail"]],
        on="paper_id",
        how="inner",
        validate="many_to_one",
    )
    country_capacity = (
        country_panel.groupby("country_code", as_index=False)["max_tflops_lb1"]
        .sum()
        .rename(columns={"max_tflops_lb1": "reported_gpu_capacity_tflops"})
        .sort_values("reported_gpu_capacity_tflops", ascending=False)
        .reset_index(drop=True)
    )
    total_capacity = float(country_capacity["reported_gpu_capacity_tflops"].sum())
    top1_capacity = float(country_capacity["reported_gpu_capacity_tflops"].head(1).sum())
    top5_capacity = float(country_capacity["reported_gpu_capacity_tflops"].head(5).sum())
    shares = country_capacity["reported_gpu_capacity_tflops"] / total_capacity
    rows.append(
        _metric_row(
            "Country",
            "Top-1 share of reported GPU capacity",
            top1_capacity / total_capacity,
            numerator=top1_capacity,
            denominator=total_capacity,
            detail=str(country_capacity.iloc[0]["country_code"]),
            method="Full-count paper-country capacity; HK/TW folded into CN.",
        )
    )
    rows.append(
        _metric_row(
            "Country",
            "Top-5 share of reported GPU capacity",
            top5_capacity / total_capacity,
            numerator=top5_capacity,
            denominator=total_capacity,
            detail=", ".join(country_capacity["country_code"].head(5).astype(str)),
            method="Full-count paper-country capacity; HK/TW folded into CN.",
        )
    )
    rows.append(
        _metric_row(
            "Country",
            "HHI of reported GPU capacity",
            float(np.square(shares).sum()),
            method="Sum of squared country capacity shares.",
        )
    )
    rows.append(
        _metric_row(
            "Country",
            "Gini of reported GPU capacity",
            gini(country_capacity["reported_gpu_capacity_tflops"]),
            method="Gini across country-level reported capacity totals.",
        )
    )

    region_panel = country_panel.copy()
    region_panel["region"] = region_panel["country_code"].map(country_to_region)
    region_panel = region_panel[["paper_id", "region", "is_high_compute_tail"]].drop_duplicates()
    region_high = (
        region_panel.loc[region_panel["is_high_compute_tail"].eq(1)]
        .groupby("region", as_index=False)
        .agg(high_compute_tail_rows=("paper_id", "size"))
        .sort_values("high_compute_tail_rows", ascending=False)
    )
    region_denominator = int(region_high["high_compute_tail_rows"].sum())
    for row in region_high.itertuples(index=False):
        rows.append(
            _metric_row(
                "Region",
                "Share of high-compute tail",
                row.high_compute_tail_rows / region_denominator,
                numerator=row.high_compute_tail_rows,
                denominator=region_denominator,
                detail=row.region,
                method="Full-count paper-region rows among yearly top-20% high-compute papers.",
            )
        )

    high_papers = work.loc[work["is_high_compute_tail"].eq(1)].copy()
    high_denominator = int(len(high_papers))
    org_contexts = [
        ("Company high-compute share", high_papers["has_company"].eq(1), "Papers with at least one company."),
        (
            "Academia-only high-compute share",
            high_papers["has_education"].eq(1) & high_papers["has_company"].eq(0),
            "Papers with education participation and without company participation.",
        ),
        (
            "Industry-academia high-compute share",
            high_papers["has_industry_academia"].eq(1),
            "Papers with both company and education participation.",
        ),
    ]
    for metric, mask, method in org_contexts:
        numerator = int(mask.sum())
        rows.append(
            _metric_row(
                "Organization type",
                metric,
                numerator / high_denominator,
                numerator=numerator,
                denominator=high_denominator,
                method=f"Paper-level share among yearly top-20% high-compute papers. {method}",
            )
        )

    topic_high = (
        high_papers.groupby("topic", as_index=False)
        .agg(high_compute_papers=("paper_id", "nunique"))
        .sort_values("high_compute_papers", ascending=False)
    )
    top5_topics = topic_high.head(5)
    top5_n = int(top5_topics["high_compute_papers"].sum())
    rows.append(
        _metric_row(
            "Topic",
            "Top-5 topic share of high-compute papers",
            top5_n / high_denominator,
            numerator=top5_n,
            denominator=high_denominator,
            detail="; ".join(top5_topics["topic"].astype(str)),
            method="Paper-level topics among yearly top-20% high-compute papers.",
        )
    )
    return pd.DataFrame(rows)


def build_audit(panel: pd.DataFrame, main: pd.DataFrame, full: pd.DataFrame, strict: pd.DataFrame) -> dict:
    return {
        "input_compute": str(COMPUTE_INPUT.relative_to(ROOT)),
        "input_org_variables": str(ORG_INPUT.relative_to(ROOT)),
        "input_topics": str(TOPIC_INPUT.relative_to(ROOT)),
        "compute_rows": int(pd.read_excel(COMPUTE_INPUT, usecols=["paper_id"]).shape[0]),
        "paper_level_panel_rows": int(len(panel)),
        "unique_papers": int(panel["paper_id"].nunique()),
        "lb1_gfimp_valid_papers": int(panel["is_lb1_gfimp"].eq(1).sum()),
        "strict_valid_papers": int(panel["is_strict"].eq(1).sum()),
        "main_model_nobs_min": int(main["nobs"].min()),
        "main_model_nobs_max": int(main["nobs"].max()),
        "full_model_nobs": int(full["nobs"].iloc[0]),
        "strict_model_nobs_min": int(strict["nobs"].min()),
        "strict_model_nobs_max": int(strict["nobs"].max()),
        "n_year_fe": int(panel["year_fe"].nunique(dropna=True)),
        "n_topic_fe": int(panel["topic_fe"].nunique(dropna=True)),
        "n_venue_fe": int(panel["venue_fe"].nunique(dropna=True)),
    }


def save_access_regime_figure(summary: pd.DataFrame) -> None:
    plot_df = summary.copy().iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))
    colors = {
        "Academic-only": "#5B83B4",
        "Industry-only": "#C89932",
        "Industry-academia collaboration": "#3E8C87",
        "Other cross-sector collaboration": "#8B6FAE",
        "Other / mixed": "#8A8A8A",
    }
    bar_colors = [colors[str(value)] for value in plot_df["access_regime"]]

    fig = plt.figure(figsize=(7.2, 3.35), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.04, 0.96], wspace=0.12)
    ax_median = fig.add_subplot(gs[0, 0])
    ax_tail = fig.add_subplot(gs[0, 1], sharey=ax_median)

    xerr_left = (
        plot_df["median_reported_gpu_capacity_tflops"]
        - plot_df["q25_reported_gpu_capacity_tflops"]
    )
    xerr_right = (
        plot_df["q75_reported_gpu_capacity_tflops"]
        - plot_df["median_reported_gpu_capacity_tflops"]
    )
    ax_median.errorbar(
        plot_df["median_reported_gpu_capacity_tflops"],
        y,
        xerr=[xerr_left, xerr_right],
        fmt="none",
        ecolor="#555555",
        elinewidth=0.85,
        capsize=2.2,
        zorder=1,
    )
    ax_median.scatter(
        plot_df["median_reported_gpu_capacity_tflops"],
        y,
        s=30,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )
    ax_median.set_xscale("log")
    ax_median.set_yticks(y)
    ax_median.set_yticklabels(plot_df["access_regime"])
    ax_median.set_xlabel("Median reported peak GPU configuration capacity (TFLOP/s, log)")
    ax_median.set_title("a  Median reported GPU capacity", loc="left", fontsize=8, fontweight="bold")
    ax_median.grid(axis="x", color="#E7E7E7", lw=0.5)
    for ypos, row in zip(y, plot_df.itertuples(index=False)):
        ax_median.text(
            row.q75_reported_gpu_capacity_tflops * 1.08,
            ypos,
            f"n={int(row.papers):,}",
            va="center",
            ha="left",
            fontsize=5.8,
            color="#555555",
        )

    tail_pct = plot_df["top20_compute_tail_share"] * 100
    ax_tail.barh(y, tail_pct, color=bar_colors, height=0.58, alpha=0.92)
    ax_tail.axvline(20, color="#222222", lw=0.8, linestyle=(0, (2, 2)))
    ax_tail.tick_params(axis="y", left=False, labelleft=False)
    ax_tail.set_xlabel("Share in yearly top-20% compute tail (%)")
    ax_tail.set_title("b  High-compute tail share", loc="left", fontsize=8, fontweight="bold")
    ax_tail.grid(axis="x", color="#E7E7E7", lw=0.5)
    ax_tail.set_xlim(0, max(40, float(tail_pct.max()) * 1.22))
    for xpos, ypos in zip(tail_pct, y):
        ax_tail.text(
            xpos + 0.8,
            ypos,
            f"{xpos:.1f}%",
            va="center",
            ha="left",
            fontsize=5.8,
            color="#222222",
        )

    save_pub_figure(fig, OUT_FIG / "institution_access_regime_compute")
    plt.close(fig)


def _format_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _results_markdown(df: pd.DataFrame) -> str:
    view = df[
        [
            "model",
            "term_label",
            "nobs",
            "coef",
            "std_err",
            "p_value",
            "percent_change",
            "r_squared",
        ]
    ].copy()
    view["coef"] = view["coef"].map(lambda x: f"{x:.3f}")
    view["std_err"] = view["std_err"].map(lambda x: f"{x:.3f}")
    view["p_value"] = view["p_value"].map(_format_p)
    view["percent_change"] = view["percent_change"].map(lambda x: f"{x:.1f}%")
    view["r_squared"] = view["r_squared"].map(lambda x: f"{x:.3f}")
    return view.to_markdown(index=False)


def _concentration_markdown(df: pd.DataFrame) -> str:
    view = df[["level", "metric", "detail", "value", "numerator", "denominator"]].copy()
    view["value"] = view["value"].map(lambda x: f"{x:.3f}")
    view["numerator"] = view["numerator"].map(lambda x: "" if pd.isna(x) else f"{x:,.0f}")
    view["denominator"] = view["denominator"].map(lambda x: "" if pd.isna(x) else f"{x:,.0f}")
    return view.to_markdown(index=False)


def _access_regime_markdown(df: pd.DataFrame) -> str:
    view = df[
        [
            "access_regime",
            "papers",
            "paper_share",
            "median_reported_gpu_capacity_tflops",
            "q25_reported_gpu_capacity_tflops",
            "q75_reported_gpu_capacity_tflops",
            "top20_compute_tail_papers",
            "top20_compute_tail_share",
        ]
    ].copy()
    view["paper_share"] = view["paper_share"].map(lambda x: f"{x:.1%}")
    for col in [
        "median_reported_gpu_capacity_tflops",
        "q25_reported_gpu_capacity_tflops",
        "q75_reported_gpu_capacity_tflops",
    ]:
        view[col] = view[col].map(lambda x: f"{x:,.1f}")
    view["top20_compute_tail_share"] = view["top20_compute_tail_share"].map(lambda x: f"{x:.1%}")
    return view.to_markdown(index=False)


def write_report(
    audit: dict,
    main: pd.DataFrame,
    full: pd.DataFrame,
    strict: pd.DataFrame,
    concentration: pd.DataFrame,
    access_regime: pd.DataFrame,
) -> None:
    strongest = main.reindex(main["coef"].abs().sort_values(ascending=False).index).iloc[0]
    report = f"""# RQ2 Institution Fixed-Effects Models

## Method

This analysis uses a paper-level panel, not full-count organization-paper rows. The regression outcome is `log10(paper_max_row_compute_capability / 1e12)`, interpreted as strict raw max-row GPU compute in TFLOP/s.

The main table estimates five one-focal-variable-at-a-time OLS models with HC3 robust standard errors. Each model controls year fixed effects, topic fixed effects, and venue fixed effects. The separated specification is intentional because company participation, industry-academia collaboration, cross-sector collaboration, international collaboration, and organization count overlap conceptually and empirically.

Regression sample: `is_strict == 1`; paper-level rows: {audit["paper_level_panel_rows"]:,}; strict-valid papers: {audit["strict_valid_papers"]:,}; main-model observations: {audit["main_model_nobs_min"]:,}-{audit["main_model_nobs_max"]:,}; fixed effects: {audit["n_year_fe"]} years, {audit["n_topic_fe"]} topics, {audit["n_venue_fe"]} venues.

## Main Models

{_results_markdown(main)}

Coefficient interpretation is on a log10 outcome scale. The `percent_change` column reports `(10^coef - 1) * 100`. The largest absolute main-model coefficient is {strongest["term_label"]}: coef={strongest["coef"]:.3f}, corresponding to {strongest["percent_change"]:.1f}% difference in max-row compute under the one-focal FE specification.

## Appendix: Full Conditional Model

The full model includes all five institutional terms simultaneously and should be read as a conditional association check, not the primary estimand.

{_results_markdown(full)}

## Strict Single-Focal Compatibility Table

This table preserves the previous strict-output file shape. It uses the same strict raw outcome and `is_strict == 1` sample as the main regression table.

{_results_markdown(strict)}

## Concentration of Reported GPU Capacity

Country concentration uses full-count paper-country reported GPU capacity, following the country analysis convention that folds HK and TW into CN before country de-duplication. Region, organization-type, and topic concentration metrics use the yearly top-20% high-compute tail.

{_concentration_markdown(concentration)}

## Access-Regime Summary

Access regimes are mutually exclusive paper-level categories: academic-only, industry-only, industry-academia collaboration, other cross-sector collaboration, and other/mixed. Reported GPU capacity is lower-bound imputed maximum GPU-row TFLOP/s; the high-compute tail is the yearly top-20% group.

{_access_regime_markdown(access_regime)}

## Outputs

- `4.3/Institution/data/institution_fe_panel_lb1.csv`
- `4.3/Institution/data/institution_fe_main_models.csv`
- `4.3/Institution/data/institution_fe_full_model.csv`
- `4.3/Institution/data/institution_fe_strict_robustness.csv`
- `4.3/Institution/data/institution_concentration_metrics.csv`
- `4.3/Institution/data/institution_access_regime_summary.csv`
- `4.3/Institution/data/institution_fe_audit.json`
- `4.3/Institution/fig/institution_access_regime_compute.png`

## Review Risks

These models are descriptive fixed-effects associations, not causal estimates. All regression tables are now estimated on the stricter raw-compute sample, so coverage is narrower than the lower-bound imputed descriptive summaries.
"""
    (OUT_REPORT / "institution_fe_models.md").write_text(report, encoding="utf-8")


def run_analysis() -> dict[str, str]:
    ensure_dirs()
    panel = build_panel()
    main = fit_main_models(panel)
    full = fit_full_model(panel)
    strict = fit_strict_robustness_models(panel)
    concentration = build_concentration_metrics(panel)
    access_regime = build_access_regime_summary(panel)
    audit = build_audit(panel, main, full, strict)

    panel.to_csv(OUT_DATA / "institution_fe_panel_lb1.csv", index=False)
    main.to_csv(OUT_DATA / "institution_fe_main_models.csv", index=False)
    full.to_csv(OUT_DATA / "institution_fe_full_model.csv", index=False)
    strict.to_csv(OUT_DATA / "institution_fe_strict_robustness.csv", index=False)
    concentration.to_csv(OUT_DATA / "institution_concentration_metrics.csv", index=False)
    access_regime.to_csv(OUT_DATA / "institution_access_regime_summary.csv", index=False)
    (OUT_DATA / "institution_fe_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    save_access_regime_figure(access_regime)
    write_report(audit, main, full, strict, concentration, access_regime)

    return {
        "panel": str(OUT_DATA / "institution_fe_panel_lb1.csv"),
        "main_models": str(OUT_DATA / "institution_fe_main_models.csv"),
        "full_model": str(OUT_DATA / "institution_fe_full_model.csv"),
        "strict_robustness": str(OUT_DATA / "institution_fe_strict_robustness.csv"),
        "concentration": str(OUT_DATA / "institution_concentration_metrics.csv"),
        "access_regime": str(OUT_DATA / "institution_access_regime_summary.csv"),
        "audit": str(OUT_DATA / "institution_fe_audit.json"),
        "report": str(OUT_REPORT / "institution_fe_models.md"),
        "access_regime_figure": str(OUT_FIG / "institution_access_regime_compute.png"),
    }


def main() -> None:
    outputs = run_analysis()
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()



