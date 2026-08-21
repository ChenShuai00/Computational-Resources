"""Reproduce the Main Conference, Findings, and pooled appendix analyses."""

from __future__ import annotations

import argparse
import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


PAPERS_FILE = "track_extension_papers.csv"
MEMBERSHIP_FILE = "track_extension_membership.csv"
TRACK_ORDER = ["main", "findings", "pooled"]
TRACK_LABELS = {
    "main": "Main Conference",
    "findings": "Findings",
    "pooled": "Main + Findings",
}
OUTCOMES = [
    {
        "outcome_id": "topic_year_percentile",
        "outcome": "topic_year_citation_percentile",
        "label": "NLP topic-year citation percentile",
        "family": "ols",
        "cov_type": "HC3",
        "effect_unit": "percentage points per 10x",
    },
    {
        "outcome_id": "openalex_percentile",
        "outcome": "openalex_field_normalized_percentile",
        "label": "OpenAlex field-normalized percentile",
        "family": "ols",
        "cov_type": "HC3",
        "effect_unit": "percentage points per 10x",
    },
    {
        "outcome_id": "log_citations",
        "outcome": "log1p_citations",
        "label": "log(1+citations)",
        "family": "ols",
        "cov_type": "HC3",
        "effect_unit": "percent per 10x",
    },
    {
        "outcome_id": "citation_count_ppml",
        "outcome": "cited_by_count",
        "label": "Citation count, PPML",
        "family": "poisson",
        "cov_type": "HC0",
        "effect_unit": "percent per 10x",
    },
    {
        "outcome_id": "top10_cited",
        "outcome": "is_top10_cited",
        "label": "Top-10% cited",
        "family": "lpm",
        "cov_type": "HC3",
        "effect_unit": "percentage points per 10x",
    },
]


def find_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "data" / "analysis_ready" / "track_extension" / PAPERS_FILE).is_file():
            return parent
    raise FileNotFoundError("Could not find the track-extension analysis-ready inputs")


ROOT = find_root(Path(__file__).resolve())
BUNDLE = Path(os.environ.get("REPRO_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "reproduced"))


def ensure_output_dirs(output_dir: Path | None) -> dict[str, Path]:
    base = output_dir or BUNDLE
    paths = {"base": base, "tables": base / "tables", "reports": base / "reports"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def add_topic_year_percentile(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    valid = out["cited_by_count"].notna() & out["primary_topic"].notna() & out["year"].notna()
    values = out.loc[valid, "cited_by_count"]
    keys = [out.loc[valid, "primary_topic"], out.loc[valid, "year"]]
    ranks = values.groupby(keys).rank(method="average", ascending=True)
    counts = values.groupby(keys).transform("size")
    out["topic_year_citation_percentile"] = np.nan
    out.loc[valid, "topic_year_citation_percentile"] = (ranks - 0.5) / counts
    return out


def prepare_scope(papers: pd.DataFrame, scope: str) -> dict[str, pd.DataFrame]:
    if scope not in TRACK_ORDER:
        raise ValueError(f"Unknown scope: {scope}")
    if scope == "pooled":
        model = papers.copy()
    else:
        model = papers.loc[papers["track"].eq(scope)].copy()

    model["year"] = safe_numeric(model["year"])
    model["venue"] = model["venue"].astype(str).str.lower()
    model["primary_topic"] = model["primary_topic"].fillna("Unknown").astype(str)
    model["cited_by_count"] = safe_numeric(model["cited_by_count"])
    model["openalex_field_normalized_percentile"] = safe_numeric(
        model["openalex_field_normalized_percentile"]
    )
    model["team_size"] = safe_numeric(model["team_size"])
    model["n_organizations"] = safe_numeric(model["n_organizations"])
    model["paper_gpu_num_total"] = safe_numeric(model["paper_gpu_num_total"])
    model["raw_max_compute"] = safe_numeric(model["paper_max_row_compute_capability"])
    model["lb1_max_compute"] = safe_numeric(
        model["paper_max_row_compute_capability_gfimp_lb1"]
    )
    model["log10_max_compute"] = np.log10(model["raw_max_compute"].where(model["raw_max_compute"].gt(0)))
    model["log1p_citations"] = np.log1p(model["cited_by_count"])
    model["team_size_group"] = pd.cut(
        model["team_size"], [-np.inf, 1, 5, np.inf], labels=["1", "2-5", "6+"]
    ).astype("object")
    model["n_organizations_group"] = pd.cut(
        model["n_organizations"], [-np.inf, 1, 5, np.inf], labels=["1", "2-5", "6+"]
    ).astype("object")
    model["year_venue"] = (
        model["year"].round().astype("Int64").astype(str)
        + "_"
        + model["venue"]
        + "_"
        + model["track"]
    )
    model = add_topic_year_percentile(model)
    model["citation_rank_pct_year_venue"] = model.groupby(
        "year_venue", dropna=False
    )["cited_by_count"].rank(method="average", pct=True)
    model["is_top10_cited"] = (
        model["citation_rank_pct_year_venue"].ge(0.90)
        & model["cited_by_count"].notna()
    ).astype(int)

    strict = model.loc[
        safe_numeric(model["is_strict"]).eq(1) & model["year"].between(2020, 2023)
    ].copy()
    lb1 = model.loc[
        safe_numeric(model["is_lb1_gfimp"]).eq(1)
        & model["year"].between(2020, 2023)
        & model["lb1_max_compute"].gt(0)
    ].copy()
    return {"model": model, "strict": strict, "lb1": lb1}


def spec7_formula(outcome: str, include_compute: bool = True) -> str:
    rhs = [
        "C(year_venue)",
        "C(primary_topic)",
        "C(team_size_group)",
        "C(n_organizations_group)",
    ]
    if include_compute:
        rhs.insert(0, "log10_max_compute")
    return f"{outcome} ~ " + " + ".join(rhs)


def fit_model(sample: pd.DataFrame, outcome_spec: dict[str, str]) -> dict[str, float | int | str]:
    outcome = outcome_spec["outcome"]
    required = [
        outcome,
        "log10_max_compute",
        "year_venue",
        "primary_topic",
        "team_size_group",
        "n_organizations_group",
    ]
    data = sample.dropna(subset=required).copy()
    formula = spec7_formula(outcome, include_compute=True)
    controls_formula = spec7_formula(outcome, include_compute=False)
    if outcome_spec["family"] in {"ols", "lpm"}:
        full = smf.ols(formula, data=data).fit(cov_type=outcome_spec["cov_type"])
        controls = smf.ols(controls_formula, data=data).fit(cov_type=outcome_spec["cov_type"])
        r2_full = float(full.rsquared)
        r2_without_compute = float(controls.rsquared)
        delta_r2 = r2_full - r2_without_compute
    else:
        full = smf.glm(formula, data=data, family=sm.families.Poisson()).fit(
            cov_type=outcome_spec["cov_type"]
        )
        controls = smf.glm(
            controls_formula, data=data, family=sm.families.Poisson()
        ).fit(cov_type=outcome_spec["cov_type"])
        r2_full = np.nan
        r2_without_compute = np.nan
        delta_r2 = np.nan

    beta = float(full.params["log10_max_compute"])
    ci_low, ci_high = [float(value) for value in full.conf_int().loc["log10_max_compute"]]
    if outcome_spec["outcome_id"] in {
        "topic_year_percentile",
        "openalex_percentile",
        "top10_cited",
    }:
        effect_10x = beta * 100
        effect_ci_low = ci_low * 100
        effect_ci_high = ci_high * 100
    else:
        effect_10x = float(np.exp(beta) - 1)
        effect_ci_low = float(np.exp(ci_low) - 1)
        effect_ci_high = float(np.exp(ci_high) - 1)

    return {
        "n": int(full.nobs),
        "beta": beta,
        "se": float(full.bse["log10_max_compute"]),
        "p": float(full.pvalues["log10_max_compute"]),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "r2_without_compute": r2_without_compute,
        "r2_full": r2_full,
        "delta_r2": delta_r2,
        "effect_10x": effect_10x,
        "effect_ci_low": effect_ci_low,
        "effect_ci_high": effect_ci_high,
        "effect_unit": outcome_spec["effect_unit"],
        "formula": formula,
        "controls_formula": controls_formula,
    }


def fit_track_difference(
    main: pd.DataFrame,
    findings: pd.DataFrame,
    outcome_spec: dict[str, str],
) -> dict[str, float | int]:
    outcome = outcome_spec["outcome"]
    stacked = pd.concat([main, findings], ignore_index=True)
    stacked["year_venue_track"] = stacked["year_venue"]
    required = [
        outcome,
        "log10_max_compute",
        "year_venue_track",
        "primary_topic",
        "team_size_group",
        "n_organizations_group",
        "track",
    ]
    data = stacked.dropna(subset=required).copy()
    formula = (
        f"{outcome} ~ 0 + C(track):log10_max_compute + C(year_venue_track) + "
        "C(track):C(primary_topic) + C(track):C(team_size_group) + "
        "C(track):C(n_organizations_group)"
    )
    if outcome_spec["family"] in {"ols", "lpm"}:
        fit = smf.ols(formula, data=data).fit(cov_type=outcome_spec["cov_type"])
    else:
        fit = smf.glm(formula, data=data, family=sm.families.Poisson()).fit(
            cov_type=outcome_spec["cov_type"]
        )

    main_term = "C(track)[main]:log10_max_compute"
    findings_term = "C(track)[findings]:log10_max_compute"
    contrast = np.zeros(len(fit.params))
    contrast[fit.params.index.get_loc(findings_term)] = 1
    contrast[fit.params.index.get_loc(main_term)] = -1
    test = fit.t_test(contrast)
    return {
        "difference_findings_minus_main": float(
            fit.params[findings_term] - fit.params[main_term]
        ),
        "difference_se": float(np.asarray(test.sd).item()),
        "difference_p": float(np.asarray(test.pvalue).item()),
        "difference_test_n": int(fit.nobs),
    }


def build_core_regression_tables(
    scopes: dict[str, dict[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, float | int]] = []

    for outcome_spec in OUTCOMES:
        fits: dict[str, dict[str, float | int | str]] = {}
        for scope in TRACK_ORDER:
            result = fit_model(scopes[scope]["strict"], outcome_spec)
            fits[scope] = result
            long_rows.append(
                {
                    "outcome_id": outcome_spec["outcome_id"],
                    "outcome": outcome_spec["label"],
                    "track": scope,
                    "track_label": TRACK_LABELS[scope],
                    "family": outcome_spec["family"],
                    "cov_type": outcome_spec["cov_type"],
                    **result,
                }
            )

        difference = fit_track_difference(
            scopes["main"]["strict"],
            scopes["findings"]["strict"],
            outcome_spec,
        )
        difference_rows.append(difference)
        wide_rows.append(
            {
                "outcome_id": outcome_spec["outcome_id"],
                "outcome": outcome_spec["label"],
                "family": outcome_spec["family"],
                "cov_type": outcome_spec["cov_type"],
                **{
                    f"{scope}_{field}": fits[scope][field]
                    for scope in TRACK_ORDER
                    for field in [
                        "n",
                        "beta",
                        "se",
                        "p",
                        "ci_low",
                        "ci_high",
                        "r2_without_compute",
                        "r2_full",
                        "delta_r2",
                        "effect_10x",
                        "effect_ci_low",
                        "effect_ci_high",
                    ]
                },
                **difference,
            }
        )

    adjusted = multipletests(
        [float(row["difference_p"]) for row in difference_rows], method="holm"
    )[1]
    for row, p_holm in zip(wide_rows, adjusted, strict=True):
        row["difference_p_holm"] = float(p_holm)
    return pd.DataFrame(long_rows), pd.DataFrame(wide_rows)


def build_sample_comparison(
    papers: pd.DataFrame,
    membership: pd.DataFrame,
    core_wide: pd.DataFrame,
) -> pd.DataFrame:
    values: dict[str, dict[str, float]] = {}
    for scope in TRACK_ORDER:
        if scope == "pooled":
            member = membership
            paper = papers
        else:
            member = membership.loc[membership["track"].eq(scope)]
            paper = papers.loc[papers["track"].eq(scope)]
        strict = paper.loc[safe_numeric(paper["is_strict"]).eq(1)].copy()
        citation_n = int(
            core_wide.loc[
                core_wide["outcome_id"].eq("topic_year_percentile"), f"{scope}_n"
            ].iloc[0]
        )
        values[scope] = {
            "total_papers": len(member),
            "standardized_gpu_model": int(member["model_reported"].sum()),
            "reporting_rate_pct": float(member["model_reported"].mean() * 100),
            "gpu_model_plus_count": int(member["strict_reported"].sum()),
            "strict_reporting_rate_pct": float(member["strict_reported"].mean() * 100),
            "citation_analysis_sample": citation_n,
            "median_reported_gpu_count": float(
                safe_numeric(strict["paper_gpu_num_total"]).median()
            ),
            "median_reported_gpu_capability_tflops": float(
                safe_numeric(strict["paper_max_row_compute_capability"]).median() / 1e12
            ),
        }

    rows = [
        ("total_papers", "Total papers", "papers"),
        ("standardized_gpu_model", "Papers reporting standardized GPU model", "papers"),
        ("reporting_rate_pct", "Reporting rate", "percent"),
        ("gpu_model_plus_count", "Papers reporting GPU model + count", "papers"),
        ("strict_reporting_rate_pct", "Strict reporting rate", "percent"),
        ("citation_analysis_sample", "Citation-analysis sample", "papers"),
        ("median_reported_gpu_count", "Median reported GPU count", "GPUs"),
        (
            "median_reported_gpu_capability_tflops",
            "Median reported GPU capability",
            "TFLOP/s",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "characteristic_id": characteristic_id,
                "characteristic": label,
                "unit": unit,
                "main_conference": values["main"][characteristic_id],
                "findings": values["findings"][characteristic_id],
                "main_plus_findings": values["pooled"][characteristic_id],
            }
            for characteristic_id, label, unit in rows
        ]
    )


def build_high_capability_table(
    scopes: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in TRACK_ORDER:
        work = scopes[scope]["lb1"].copy()
        work = work.sort_values(
            ["year", "lb1_max_compute", "paper_id"], ascending=[True, False, True]
        )
        high_flags = pd.Series(0, index=work.index, dtype=int)
        for _year, indices in work.groupby("year", sort=True).groups.items():
            ordered = list(indices)
            high_n = max(1, int(np.ceil(len(ordered) * 0.20)))
            high_flags.loc[ordered[:high_n]] = 1
        high_capability = high_flags.eq(1)
        high_impact = work["is_top10_cited"].eq(1)
        other = ~high_capability
        n_high_capability = int(high_capability.sum())
        n_other = int(other.sum())
        n_high_impact = int(high_impact.sum())
        n_high_capability_high_impact = int((high_capability & high_impact).sum())
        n_other_high_impact = int((other & high_impact).sum())
        high_rate = n_high_capability_high_impact / n_high_capability
        other_rate = n_other_high_impact / n_other
        rows.append(
            {
                "track": scope,
                "track_label": TRACK_LABELS[scope],
                "n": len(work),
                "n_high_capability": n_high_capability,
                "n_other": n_other,
                "n_high_impact": n_high_impact,
                "n_high_capability_high_impact": n_high_capability_high_impact,
                "n_other_high_impact": n_other_high_impact,
                "high_impact_rate_high_capability": high_rate,
                "high_impact_rate_other": other_rate,
                "risk_ratio": high_rate / other_rate,
                "high_capability_share_high_impact": (
                    n_high_capability_high_impact / n_high_impact
                ),
                "high_capability_not_high_impact": 1 - high_rate,
                "high_capability_definition": "annual top 20% of model-reported GPU papers",
                "high_impact_definition": "top 10% cited within venue-year-track",
                "year_window": "2020-2023",
            }
        )
    return pd.DataFrame(rows)


def fmt_beta_se(beta: float, se: float, delta_r2: float) -> str:
    delta = "—" if pd.isna(delta_r2) else f"{delta_r2:.4f}"
    return f"{beta:.3f} ({se:.3f}) [{delta}]"


def sample_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Characteristic | Main Conference | Findings | Main + Findings |",
        "|---|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        values = [row.main_conference, row.findings, row.main_plus_findings]
        if row.unit == "percent":
            displayed = [f"{value:.1f}%" for value in values]
        elif row.unit == "TFLOP/s":
            displayed = [
                f"{Decimal(str(value)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):,}"
                for value in values
            ]
        elif row.unit == "GPUs":
            displayed = [f"{value:,.0f}" for value in values]
        else:
            displayed = [f"{value:,.0f}" for value in values]
        lines.append(f"| {row.characteristic} | " + " | ".join(displayed) + " |")
    return "\n".join(lines)


def regression_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Outcome | Main Conference β (SE) [ΔR²] | Findings β (SE) [ΔR²] | Pooled β (SE) [ΔR²] | Findings − Main difference (p) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.outcome} | "
            f"{fmt_beta_se(row.main_beta, row.main_se, row.main_delta_r2)} | "
            f"{fmt_beta_se(row.findings_beta, row.findings_se, row.findings_delta_r2)} | "
            f"{fmt_beta_se(row.pooled_beta, row.pooled_se, row.pooled_delta_r2)} | "
            f"{row.difference_findings_minus_main:+.3f} ({row.difference_p:.3f}) |"
        )
    lines.append(
        f"| N | {int(table.main_n.iloc[0]):,} | {int(table.findings_n.iloc[0]):,} | "
        f"{int(table.pooled_n.iloc[0]):,} | {int(table.difference_test_n.iloc[0]):,} |"
    )
    return "\n".join(lines)


def high_capability_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Track | High-impact rate among high-capability papers | High-impact rate among other papers | Risk ratio | High-capability share among high-impact papers | High-capability papers not high-impact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.track_label} | {row.high_impact_rate_high_capability * 100:.1f}% | "
            f"{row.high_impact_rate_other * 100:.1f}% | {row.risk_ratio:.2f} | "
            f"{row.high_capability_share_high_impact * 100:.1f}% | "
            f"{row.high_capability_not_high_impact * 100:.1f}% |"
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    sample_table: pd.DataFrame,
    core_table: pd.DataFrame,
    high_table: pd.DataFrame,
) -> None:
    pooled_primary = core_table.loc[
        core_table["outcome_id"].eq("topic_year_percentile")
    ].iloc[0]
    findings_high = high_table.loc[high_table["track"].eq("findings")].iloc[0]
    report = f"""# Main Conference and Findings track-extension robustness

## Scope

This appendix analysis extends the ACL/EMNLP/NAACL main-conference corpus to
Findings papers. It preserves the paper's text-reported GPU measurement boundary:
reported capability is theoretical peak configuration capacity, not GPU-hours,
realized utilization, cost, energy, or causal treatment intensity.

## Sample comparison

{sample_markdown(sample_table)}

Median GPU count and capability are calculated among 2020-2025 strict papers
that report both a standardized GPU model and an explicit count.

## Core citation regressions

{regression_markdown(core_table)}

Cells report β (robust SE) [incremental R²]. Incremental R² is the full Spec-7
model R² minus the controls-only model R² on the identical sample. OLS and LPM
use HC3 robust standard errors; PPML uses HC0 robust standard errors and has no
ordinary R². The final column is the Findings-minus-main slope difference with
its two-sided Wald-test p-value. Holm-adjusted difference-test p-values are
available in the machine-readable table.

The pooled primary estimate is {pooled_primary.pooled_beta * 100:.2f} percentile
points per tenfold increase in reported GPU capability
(N={int(pooled_primary.pooled_n):,}, p={pooled_primary.pooled_p:.3g},
ΔR²={pooled_primary.pooled_delta_r2:.4f}). None of the five
Main-versus-Findings slope differences is significant at the 0.05 level.

## Does more reported capability ensure high impact?

{high_capability_markdown(high_table)}

In Findings, the high-impact rate is higher among annual top-20% capability
papers ({findings_high.high_impact_rate_high_capability * 100:.1f}% versus
{findings_high.high_impact_rate_other * 100:.1f}%; risk ratio
{findings_high.risk_ratio:.2f}), but {findings_high.high_capability_not_high_impact * 100:.1f}%
of high-capability papers are not venue-year-track top-10% cited. Reported
capability is therefore positively associated with citation impact but neither
sufficient nor necessary for high impact.

## Interpretation boundaries

- Results are conditional associations among papers with reportable and
  standardizable GPU evidence; they are not causal effects.
- Citation outcomes are incomplete proxies for scholarly value.
- The extension broadens publication tracks within ACL-family venues but does
  not establish generalizability to journals, workshops, arXiv, industrial
  reports, or other machine-learning venues.
- Findings lacks a directly comparable formal-award outcome, so award models are
  intentionally excluded.
"""
    path.write_text(report, encoding="utf-8")


def run_analysis(
    input_data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    quiet: bool = False,
) -> dict[str, object]:
    input_dir = (
        Path(input_data_dir)
        if input_data_dir is not None
        else ROOT / "data" / "analysis_ready" / "track_extension"
    )
    dirs = ensure_output_dirs(Path(output_dir) if output_dir is not None else None)
    papers = pd.read_csv(input_dir / PAPERS_FILE)
    membership = pd.read_csv(input_dir / MEMBERSHIP_FILE)

    if len(papers) != 12_724 or papers["paper_id"].nunique() != 12_724:
        raise ValueError("Track-extension paper input failed the frozen row/ID check")
    if len(membership) != 23_838 or membership["paper_id"].nunique() != 23_838:
        raise ValueError("Track-extension membership failed the frozen row/ID check")

    scopes = {scope: prepare_scope(papers, scope) for scope in TRACK_ORDER}
    core_long, core_wide = build_core_regression_tables(scopes)
    sample_table = build_sample_comparison(papers, membership, core_wide)
    high_table = build_high_capability_table(scopes)

    outputs = {
        "sample_comparison": dirs["tables"] / "track_sample_comparison.csv",
        "core_models_long": dirs["tables"] / "track_core_citation_models_long.csv",
        "core_regressions": dirs["tables"] / "track_core_citation_regressions.csv",
        "high_capability_impact": dirs["tables"] / "track_high_capability_impact.csv",
        "report": dirs["reports"] / "track_extension.md",
    }
    sample_table.to_csv(outputs["sample_comparison"], index=False)
    core_long.to_csv(outputs["core_models_long"], index=False)
    core_wide.to_csv(outputs["core_regressions"], index=False)
    high_table.to_csv(outputs["high_capability_impact"], index=False)
    write_report(outputs["report"], sample_table, core_wide, high_table)

    payload = {key: str(value) for key, value in outputs.items()}
    if not quiet:
        print(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_analysis(args.input_data_dir, args.output_dir, args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
