from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "paper_id",
    "raw_affiliation",
    "matched_display_name",
    "institution_id",
    "ror",
    "country_code",
    "institution_type",
]

DEFAULT_INPUT_PATH = Path(
    "data/processed/affiliations/emnlp2025/final/affiliation.xlsx"
)
DEFAULT_OUTPUT_FILENAMES = {
    "paper_level": "paper_level_variables.xlsx",
    "institution_level": "institution_level_variables.xlsx",
    "country_level": "country_level_variables.xlsx",
    "dictionary_xlsx": "variable_dictionary.xlsx",
    "dictionary_csv": "variable_dictionary.csv",
}


def normalize_column_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={column: normalize_column_name(column) for column in df.columns})
    missing = [column for column in REQUIRED_COLUMNS if column not in renamed.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_text}")

    normalized = renamed[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()
    return normalized


def load_affiliation_data(
    input_path: str | Path,
    sheet_name: str | None = None,
) -> tuple[pd.DataFrame, str]:
    workbook_path = Path(input_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {workbook_path}")

    with pd.ExcelFile(workbook_path) as excel_file:
        selected_sheet = sheet_name
        if selected_sheet is None:
            if "sciscinet_matches" in excel_file.sheet_names:
                selected_sheet = "sciscinet_matches"
            else:
                selected_sheet = excel_file.sheet_names[0]
        df = pd.read_excel(excel_file, sheet_name=selected_sheet)
    return _normalize_dataframe(df), selected_sheet


def _sorted_unique_values(series: pd.Series) -> list[str]:
    values = []
    seen = set()
    for value in series.fillna("").astype(str):
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        values.append(item)
    return sorted(values)


def _join_unique_values(series: pd.Series) -> str:
    return " | ".join(_sorted_unique_values(series))


def _unique_non_empty_count(series: pd.Series) -> int:
    return len(_sorted_unique_values(series))


def build_paper_level_variables(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("paper_id", dropna=False, sort=True)
    records: list[dict[str, object]] = []
    for paper_id, group in grouped:
        records.append(
            {
                "paper_id": paper_id,
                "affiliation_record_count": int(len(group)),
                "unique_raw_affiliation_count": _unique_non_empty_count(
                    group["raw_affiliation"]
                ),
                "unique_institution_count": _unique_non_empty_count(
                    group["matched_display_name"]
                ),
                "unique_institution_id_count": _unique_non_empty_count(
                    group["institution_id"]
                ),
                "unique_ror_count": _unique_non_empty_count(group["ror"]),
                "unique_country_count": _unique_non_empty_count(group["country_code"]),
                "match_success_flag": int(
                    _unique_non_empty_count(group["matched_display_name"]) > 0
                ),
                "raw_affiliation_list": _join_unique_values(group["raw_affiliation"]),
                "matched_display_name_list": _join_unique_values(
                    group["matched_display_name"]
                ),
                "institution_id_list": _join_unique_values(group["institution_id"]),
                "ror_list": _join_unique_values(group["ror"]),
                "country_code_list": _join_unique_values(group["country_code"]),
                "institution_type_list": _join_unique_values(group["institution_type"]),
            }
        )

    return pd.DataFrame(records).sort_values("paper_id").reset_index(drop=True)


def build_institution_level_variables(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.loc[df["matched_display_name"].ne("")].copy()
    grouped = filtered.groupby("matched_display_name", dropna=False, sort=True)
    records: list[dict[str, object]] = []
    for matched_display_name, group in grouped:
        records.append(
            {
                "matched_display_name": matched_display_name,
                "institution_affiliation_record_count": int(len(group)),
                "institution_paper_count_full": _unique_non_empty_count(group["paper_id"]),
                "paper_id_list": _join_unique_values(group["paper_id"]),
                "raw_affiliation_list": _join_unique_values(group["raw_affiliation"]),
                "institution_id_list": _join_unique_values(group["institution_id"]),
                "ror_list": _join_unique_values(group["ror"]),
                "country_code_list": _join_unique_values(group["country_code"]),
                "institution_type_list": _join_unique_values(group["institution_type"]),
                "unique_country_count": _unique_non_empty_count(group["country_code"]),
                "unique_institution_id_count": _unique_non_empty_count(
                    group["institution_id"]
                ),
                "unique_ror_count": _unique_non_empty_count(group["ror"]),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("matched_display_name")
        .reset_index(drop=True)
    )


def build_country_level_variables(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.loc[df["country_code"].ne("")].copy()
    grouped = filtered.groupby("country_code", dropna=False, sort=True)
    records: list[dict[str, object]] = []
    for country_code, group in grouped:
        records.append(
            {
                "country_code": country_code,
                "country_affiliation_record_count": int(len(group)),
                "country_paper_count_full": _unique_non_empty_count(group["paper_id"]),
                "country_institution_count": _unique_non_empty_count(
                    group["matched_display_name"]
                ),
                "paper_id_list": _join_unique_values(group["paper_id"]),
                "matched_display_name_list": _join_unique_values(
                    group["matched_display_name"]
                ),
                "institution_id_list": _join_unique_values(group["institution_id"]),
                "ror_list": _join_unique_values(group["ror"]),
                "institution_type_list": _join_unique_values(group["institution_type"]),
                "unique_institution_type_count": _unique_non_empty_count(
                    group["institution_type"]
                ),
            }
        )

    return pd.DataFrame(records).sort_values("country_code").reset_index(drop=True)


def _variable_dictionary_rows() -> list[dict[str, str]]:
    return [
        {
            "variable_name": "paper_id",
            "output_table": "input_affiliation",
            "level": "raw",
            "definition": "Paper identifier from the source affiliation table.",
            "construction_rule": "Directly read from affiliation.xlsx.",
            "source_field": "paper_id",
            "notes": "",
        },
        {
            "variable_name": "raw_affiliation",
            "output_table": "input_affiliation",
            "level": "raw",
            "definition": "Original affiliation string extracted from the paper.",
            "construction_rule": "Directly read from affiliation.xlsx.",
            "source_field": "raw_affiliation",
            "notes": "",
        },
        {
            "variable_name": "matched_display_name",
            "output_table": "input_affiliation",
            "level": "raw",
            "definition": "Matched institution display name used as the institution-level grouping key.",
            "construction_rule": "Directly read from affiliation.xlsx.",
            "source_field": "matched_display_name",
            "notes": "Blank values are excluded from institution-level aggregates.",
        },
        {
            "variable_name": "institution_id",
            "output_table": "input_affiliation",
            "level": "raw",
            "definition": "Institution identifier carried from the source match result.",
            "construction_rule": "Directly read from affiliation.xlsx.",
            "source_field": "institution_id",
            "notes": "Not used as the institution-level grouping key in this workflow.",
        },
        {
            "variable_name": "ror",
            "output_table": "input_affiliation",
            "level": "raw",
            "definition": "ROR identifier string from the source match result.",
            "construction_rule": "Directly read from affiliation.xlsx.",
            "source_field": "ror",
            "notes": "",
        },
        {
            "variable_name": "country_code",
            "output_table": "input_affiliation",
            "level": "raw",
            "definition": "Institution country code from the source match result.",
            "construction_rule": "Directly read from affiliation.xlsx.",
            "source_field": "country_code",
            "notes": "Blank values are excluded from country-level aggregates.",
        },
        {
            "variable_name": "institution_type",
            "output_table": "input_affiliation",
            "level": "raw",
            "definition": "Institution type label from the source match result.",
            "construction_rule": "Directly read from affiliation.xlsx.",
            "source_field": "institution_type",
            "notes": "",
        },
        {
            "variable_name": "affiliation_record_count",
            "output_table": "paper_level_variables",
            "level": "derived_paper_level",
            "definition": "Number of affiliation records linked to the paper.",
            "construction_rule": "Count rows per paper_id.",
            "source_field": "paper_id",
            "notes": "",
        },
        {
            "variable_name": "unique_raw_affiliation_count",
            "output_table": "paper_level_variables",
            "level": "derived_paper_level",
            "definition": "Number of distinct non-empty raw affiliation strings in the paper.",
            "construction_rule": "Count unique non-empty raw_affiliation values per paper_id.",
            "source_field": "raw_affiliation",
            "notes": "",
        },
        {
            "variable_name": "unique_institution_count",
            "output_table": "paper_level_variables",
            "level": "derived_paper_level",
            "definition": "Number of distinct non-empty matched institutions in the paper.",
            "construction_rule": "Count unique non-empty matched_display_name values per paper_id.",
            "source_field": "matched_display_name",
            "notes": "",
        },
        {
            "variable_name": "unique_institution_id_count",
            "output_table": "paper_level_variables",
            "level": "derived_paper_level",
            "definition": "Number of distinct non-empty institution IDs in the paper.",
            "construction_rule": "Count unique non-empty institution_id values per paper_id.",
            "source_field": "institution_id",
            "notes": "",
        },
        {
            "variable_name": "unique_ror_count",
            "output_table": "paper_level_variables",
            "level": "derived_paper_level",
            "definition": "Number of distinct non-empty ROR identifiers in the paper.",
            "construction_rule": "Count unique non-empty ror values per paper_id.",
            "source_field": "ror",
            "notes": "",
        },
        {
            "variable_name": "unique_country_count",
            "output_table": "paper_level_variables",
            "level": "derived_paper_level",
            "definition": "Number of distinct non-empty country codes in the paper.",
            "construction_rule": "Count unique non-empty country_code values per paper_id.",
            "source_field": "country_code",
            "notes": "",
        },
        {
            "variable_name": "match_success_flag",
            "output_table": "paper_level_variables",
            "level": "data_quality",
            "definition": "Flag indicating whether the paper has at least one non-empty matched institution name.",
            "construction_rule": "Set to 1 when any matched_display_name is non-empty within the paper, else 0.",
            "source_field": "matched_display_name",
            "notes": "",
        },
        {
            "variable_name": "raw_affiliation_list",
            "output_table": "paper_level_variables",
            "level": "derived_paper_level",
            "definition": "Pipe-separated list of distinct raw affiliations in the paper.",
            "construction_rule": "Sort unique non-empty raw_affiliation values and join with ' | '.",
            "source_field": "raw_affiliation",
            "notes": "",
        },
        {
            "variable_name": "matched_display_name_list",
            "output_table": "paper_level_variables",
            "level": "derived_paper_level",
            "definition": "Pipe-separated list of distinct matched institution names in the paper.",
            "construction_rule": "Sort unique non-empty matched_display_name values and join with ' | '.",
            "source_field": "matched_display_name",
            "notes": "",
        },
        {
            "variable_name": "institution_id_list",
            "output_table": "paper_level_variables",
            "level": "derived_paper_level",
            "definition": "Pipe-separated list of distinct institution IDs in the paper.",
            "construction_rule": "Sort unique non-empty institution_id values and join with ' | '.",
            "source_field": "institution_id",
            "notes": "",
        },
        {
            "variable_name": "ror_list",
            "output_table": "paper_level_variables",
            "level": "derived_paper_level",
            "definition": "Pipe-separated list of distinct ROR identifiers in the paper.",
            "construction_rule": "Sort unique non-empty ror values and join with ' | '.",
            "source_field": "ror",
            "notes": "",
        },
        {
            "variable_name": "country_code_list",
            "output_table": "paper_level_variables",
            "level": "derived_paper_level",
            "definition": "Pipe-separated list of distinct country codes in the paper.",
            "construction_rule": "Sort unique non-empty country_code values and join with ' | '.",
            "source_field": "country_code",
            "notes": "",
        },
        {
            "variable_name": "institution_type_list",
            "output_table": "paper_level_variables",
            "level": "derived_paper_level",
            "definition": "Pipe-separated list of distinct institution types in the paper.",
            "construction_rule": "Sort unique non-empty institution_type values and join with ' | '.",
            "source_field": "institution_type",
            "notes": "",
        },
        {
            "variable_name": "institution_affiliation_record_count",
            "output_table": "institution_level_variables",
            "level": "derived_institution_level",
            "definition": "Number of affiliation records assigned to the matched institution name.",
            "construction_rule": "Count rows per matched_display_name.",
            "source_field": "matched_display_name",
            "notes": "",
        },
        {
            "variable_name": "institution_paper_count_full",
            "output_table": "institution_level_variables",
            "level": "derived_institution_level",
            "definition": "Number of unique papers linked to the matched institution name.",
            "construction_rule": "Count unique paper_id values per matched_display_name.",
            "source_field": "paper_id, matched_display_name",
            "notes": "",
        },
        {
            "variable_name": "paper_id_list",
            "output_table": "institution_level_variables",
            "level": "derived_institution_level",
            "definition": "Pipe-separated list of unique paper IDs linked to the matched institution name.",
            "construction_rule": "Sort unique non-empty paper_id values and join with ' | '.",
            "source_field": "paper_id",
            "notes": "",
        },
        {
            "variable_name": "country_code_list",
            "output_table": "institution_level_variables",
            "level": "derived_institution_level",
            "definition": "Pipe-separated list of country codes observed for the matched institution name.",
            "construction_rule": "Sort unique non-empty country_code values and join with ' | '.",
            "source_field": "country_code",
            "notes": "",
        },
        {
            "variable_name": "unique_country_count",
            "output_table": "institution_level_variables",
            "level": "derived_institution_level",
            "definition": "Number of distinct countries observed for the matched institution name.",
            "construction_rule": "Count unique non-empty country_code values per matched_display_name.",
            "source_field": "country_code",
            "notes": "",
        },
        {
            "variable_name": "unique_institution_id_count",
            "output_table": "institution_level_variables",
            "level": "derived_institution_level",
            "definition": "Number of distinct institution IDs observed for the matched institution name.",
            "construction_rule": "Count unique non-empty institution_id values per matched_display_name.",
            "source_field": "institution_id",
            "notes": "Useful for spotting many-to-one name mappings.",
        },
        {
            "variable_name": "unique_ror_count",
            "output_table": "institution_level_variables",
            "level": "derived_institution_level",
            "definition": "Number of distinct ROR identifiers observed for the matched institution name.",
            "construction_rule": "Count unique non-empty ror values per matched_display_name.",
            "source_field": "ror",
            "notes": "Useful for spotting many-to-one name mappings.",
        },
        {
            "variable_name": "country_affiliation_record_count",
            "output_table": "country_level_variables",
            "level": "derived_country_level",
            "definition": "Number of affiliation records assigned to the country.",
            "construction_rule": "Count rows per country_code.",
            "source_field": "country_code",
            "notes": "",
        },
        {
            "variable_name": "country_paper_count_full",
            "output_table": "country_level_variables",
            "level": "derived_country_level",
            "definition": "Number of unique papers linked to the country.",
            "construction_rule": "Count unique paper_id values per country_code.",
            "source_field": "paper_id, country_code",
            "notes": "",
        },
        {
            "variable_name": "country_institution_count",
            "output_table": "country_level_variables",
            "level": "derived_country_level",
            "definition": "Number of distinct non-empty matched institution names observed in the country.",
            "construction_rule": "Count unique non-empty matched_display_name values per country_code.",
            "source_field": "matched_display_name, country_code",
            "notes": "",
        },
        {
            "variable_name": "matched_display_name_list",
            "output_table": "country_level_variables",
            "level": "derived_country_level",
            "definition": "Pipe-separated list of matched institution names observed in the country.",
            "construction_rule": "Sort unique non-empty matched_display_name values and join with ' | '.",
            "source_field": "matched_display_name",
            "notes": "",
        },
        {
            "variable_name": "institution_type_list",
            "output_table": "country_level_variables",
            "level": "derived_country_level",
            "definition": "Pipe-separated list of institution types observed in the country.",
            "construction_rule": "Sort unique non-empty institution_type values and join with ' | '.",
            "source_field": "institution_type",
            "notes": "",
        },
        {
            "variable_name": "unique_institution_type_count",
            "output_table": "country_level_variables",
            "level": "derived_country_level",
            "definition": "Number of distinct non-empty institution types observed in the country.",
            "construction_rule": "Count unique non-empty institution_type values per country_code.",
            "source_field": "institution_type",
            "notes": "",
        },
        {
            "variable_name": "author_count",
            "output_table": "not_directly_constructable",
            "level": "data_quality",
            "definition": "Number of authors on the paper.",
            "construction_rule": "Not constructed.",
            "source_field": "",
            "notes": "cannot_be_directly_constructed: author-level data is absent from affiliation.xlsx.",
        },
        {
            "variable_name": "author_country_count",
            "output_table": "not_directly_constructable",
            "level": "data_quality",
            "definition": "Number of author countries on the paper.",
            "construction_rule": "Not constructed.",
            "source_field": "",
            "notes": "cannot_be_directly_constructed: author-to-affiliation links are absent from affiliation.xlsx.",
        },
        {
            "variable_name": "first_author_country_code",
            "output_table": "not_directly_constructable",
            "level": "data_quality",
            "definition": "Country code associated with the first author.",
            "construction_rule": "Not constructed.",
            "source_field": "",
            "notes": "cannot_be_directly_constructed: author order is absent from affiliation.xlsx.",
        },
    ]


def generate_variable_dictionary(
    input_path: str | Path,
    output_xlsx: str | Path,
    output_csv: str | Path,
    sheet_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, selected_sheet = load_affiliation_data(input_path=input_path, sheet_name=sheet_name)

    dictionary_df = pd.DataFrame(_variable_dictionary_rows())
    metadata_df = pd.DataFrame(
        [
            {
                "item": "source_workbook",
                "value": str(Path(input_path)),
            },
            {
                "item": "source_sheet",
                "value": selected_sheet,
            },
            {
                "item": "identified_grain",
                "value": "one row per affiliation record with paper_id and raw_affiliation retained",
            },
            {
                "item": "row_count",
                "value": str(len(df)),
            },
            {
                "item": "required_columns",
                "value": ", ".join(REQUIRED_COLUMNS),
            },
        ]
    )

    output_xlsx_path = Path(output_xlsx)
    output_csv_path = Path(output_csv)
    output_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_xlsx_path) as writer:
        dictionary_df.to_excel(writer, sheet_name="variable_dictionary", index=False)
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)
    dictionary_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    return dictionary_df, metadata_df


def build_affiliation_outputs(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    sheet_name: str | None = None,
) -> dict[str, Path]:
    df, _ = load_affiliation_data(input_path=input_path, sheet_name=sheet_name)
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(input_path).parent
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    paper_df = build_paper_level_variables(df)
    institution_df = build_institution_level_variables(df)
    country_df = build_country_level_variables(df)

    paper_path = resolved_output_dir / DEFAULT_OUTPUT_FILENAMES["paper_level"]
    institution_path = resolved_output_dir / DEFAULT_OUTPUT_FILENAMES["institution_level"]
    country_path = resolved_output_dir / DEFAULT_OUTPUT_FILENAMES["country_level"]
    dictionary_xlsx_path = resolved_output_dir / DEFAULT_OUTPUT_FILENAMES["dictionary_xlsx"]
    dictionary_csv_path = resolved_output_dir / DEFAULT_OUTPUT_FILENAMES["dictionary_csv"]

    paper_df.to_excel(paper_path, index=False)
    institution_df.to_excel(institution_path, index=False)
    country_df.to_excel(country_path, index=False)
    generate_variable_dictionary(
        input_path=input_path,
        output_xlsx=dictionary_xlsx_path,
        output_csv=dictionary_csv_path,
        sheet_name=sheet_name,
    )

    return {
        "paper_level": paper_path,
        "institution_level": institution_path,
        "country_level": country_path,
        "dictionary_xlsx": dictionary_xlsx_path,
        "dictionary_csv": dictionary_csv_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build affiliation variable tables and a variable dictionary."
    )
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to affiliation.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated output files. Defaults to the input workbook directory.",
    )
    parser.add_argument(
        "--sheet-name",
        default=None,
        help="Optional worksheet name. Defaults to 'sciscinet_matches' when present, otherwise the first sheet.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    outputs = build_affiliation_outputs(
        input_path=args.input_path,
        output_dir=args.output_dir,
        sheet_name=args.sheet_name,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
