from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _quote(name: str) -> str:
    return f'Q("{name}")'


def assign_within_group_deciles(
    df: pd.DataFrame,
    value_col: str,
    group_cols: Sequence[str],
    output_col: str = "compute_decile",
) -> pd.DataFrame:
    result = df.copy()
    group_cols = list(group_cols)

    valid_mask = result[value_col].notna()
    for col in group_cols:
        valid_mask &= result[col].notna()

    result[output_col] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    if not valid_mask.any():
        return result

    if group_cols:
        rank_pct = (
            result.loc[valid_mask]
            .groupby(group_cols)[value_col]
            .rank(method="first", pct=True)
        )
    else:
        rank_pct = result.loc[valid_mask, value_col].rank(method="first", pct=True)

    deciles = np.ceil(rank_pct * 10).clip(1, 10).astype("Int64")
    result.loc[valid_mask, output_col] = deciles
    return result


def add_adjusted_citations(
    df: pd.DataFrame,
    outcome_col: str = "log1p_cites",
    year_col: str = "year_str",
    venue_col: str = "venue",
    topic_col: str = "primary_topic",
    team_size_col: str = "log1p_team_size",
    output_col: str = "adjusted_citations",
    strict_col: str | None = "is_strict",
):
    required_cols = [outcome_col, year_col, venue_col, topic_col, team_size_col]
    if strict_col and strict_col in df.columns:
        source = df.loc[pd.to_numeric(df[strict_col], errors="coerce").eq(1)].copy()
    else:
        source = df.copy()
    result = source.dropna(subset=required_cols).copy()
    result["year_venue"] = result[year_col].astype(str) + "_" + result[venue_col].astype(str)

    formula = " ~ ".join(
        [
            _quote(outcome_col),
            " + ".join(
                [
                    'C(Q("year_venue"))',
                    f"C({_quote(topic_col)})",
                    _quote(team_size_col),
                ]
            ),
        ]
    )
    model = smf.ols(formula, data=result).fit()
    result[output_col] = model.resid
    return result, model


def summarize_decile_curve(
    df: pd.DataFrame,
    value_col: str,
    decile_col: str = "compute_decile",
) -> pd.DataFrame:
    summary = (
        df.dropna(subset=[decile_col, value_col])
        .groupby(decile_col, dropna=False)[value_col]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "mean_value", "std": "std_value"})
        .sort_values(decile_col)
        .reset_index(drop=True)
    )

    summary["sem"] = summary["std_value"] / np.sqrt(summary["n"])
    ci_half_width = 1.96 * summary["sem"].fillna(0.0)
    summary["ci_low"] = summary["mean_value"] - ci_half_width
    summary["ci_high"] = summary["mean_value"] + ci_half_width
    return summary
