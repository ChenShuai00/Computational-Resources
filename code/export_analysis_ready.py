"""Export the frozen analysis inputs to deterministic, reviewable CSV files.

This is a maintainer utility, not part of the default reproduction command.  It
converts the legacy Excel/CSV inputs without changing their analytical content,
and creates two small audit datasets from local-only source artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ExportSpec:
    source: str
    target: str
    sheet: str | None = None
    columns: tuple[str, ...] | None = None


EXPORTS = (
    ExportSpec(
        "acl_arr_topics_all_acl_metadata_desirouter_complete_gpu_only.xlsx",
        "topics.csv",
        "topics",
    ),
    ExportSpec("acl_award_papers_2020_2025_gpu_only.xlsx", "awards.csv", "Awards"),
    ExportSpec(
        "acl_emnlp_naacl_2020_2025_gpu_normalized_gpu_only.xlsx",
        "gpu_rows.csv",
        "merged_gpu_normalized",
    ),
    ExportSpec(
        "acl_emnlp_naacl_2020_2025_gpu_normalized_gpu_only.xlsx",
        "gpu_coverage_summary.csv",
        "coverage_summary",
    ),
    ExportSpec("compute_paper_level_gpu_only.xlsx", "compute_papers.csv", "Sheet1"),
    ExportSpec(
        "compute_paper_level_with_contributions_gpu_only.xlsx",
        "compute_papers_with_contributions.csv",
        "Sheet1",
    ),
    ExportSpec("ml_hardware_gpu_only.xlsx", "hardware_catalog.csv", "ml_hardware"),
    ExportSpec(
        "ml_hardware_gpu_only.xlsx",
        "hardware_classification_map.csv",
        "_classification_map",
    ),
    ExportSpec(
        "openalex_paper_metadata_gpu_only.xlsx",
        "openalex_metadata.csv",
        "openalex_metadata",
    ),
    ExportSpec("organization_year_panel_gpu_only.csv", "organization_year_panel.csv"),
    ExportSpec("paper_compute_level_gpu_only.xlsx", "paper_compute_rows.csv", "Sheet1"),
    ExportSpec("paper_confounder_controls_gpu_only.csv", "paper_confounder_controls.csv"),
    ExportSpec("paper_level_org_variables_gpu_only.csv", "paper_organization_variables.csv"),
    ExportSpec(
        "paper_organization_long_gpu_only.csv",
        "paper_organizations.csv",
        columns=("paper_id", "org_id", "year", "org_name", "org_type", "org_country_code"),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source(path: Path, sheet: str | None) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")


def build_membership(corpus_gpu_root: Path, compute: pd.DataFrame) -> pd.DataFrame:
    corpus_ids = sorted(
        path.name[: -len("_gpu.json")]
        for path in corpus_gpu_root.glob("*/extract/*_gpu.json")
    )
    if len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError("Duplicate paper IDs found in the corpus extraction directories.")
    compute = compute.copy()
    compute["paper_id"] = compute["paper_id"].astype(str)
    model_ids = set(compute["paper_id"])
    strict_ids = set(compute.loc[pd.to_numeric(compute["is_strict"], errors="coerce").eq(1), "paper_id"])
    rows = pd.DataFrame({"paper_id": corpus_ids})
    rows["year"] = pd.to_numeric(rows["paper_id"].str.extract(r"^(\d{4})")[0], errors="raise").astype(int)
    rows["venue"] = rows["paper_id"].str.extract(r"^\d{4}\.([^-\.]+)-")[0].str.upper()
    rows["model_reported"] = rows["paper_id"].isin(model_ids).astype(int)
    rows["strict_reported"] = rows["paper_id"].isin(strict_ids).astype(int)
    expected = {"corpus": 13921, "model": 6900, "strict": 5360}
    observed = {
        "corpus": len(rows),
        "model": int(rows["model_reported"].sum()),
        "strict": int(rows["strict_reported"].sum()),
    }
    if observed != expected:
        raise ValueError(f"Unexpected sample counts: {observed}; expected {expected}")
    return rows


def build_consumption_audit(workbook: Path) -> pd.DataFrame:
    frame = pd.read_excel(workbook, sheet_name="annotation")
    columns = [
        "sample_id",
        "venue_year",
        "venue",
        "year",
        "paper_id",
        "consumption_visible_yes_no",
    ]
    out = frame[columns].rename(columns={"consumption_visible_yes_no": "consumption_signal_visible"})
    values = pd.to_numeric(out["consumption_signal_visible"], errors="raise").astype(int)
    out["consumption_signal_visible"] = values
    counts = values.value_counts().to_dict()
    if len(out) != 240 or counts != {0: 148, 1: 92}:
        raise ValueError(f"Unexpected consumption-audit counts: rows={len(out)}, counts={counts}")
    return out


def export(args: argparse.Namespace) -> None:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    schemas: dict[str, object] = {}

    frames: dict[str, pd.DataFrame] = {}
    for spec in EXPORTS:
        source = source_dir / spec.source
        frame = read_source(source, spec.sheet)
        if spec.columns is not None:
            frame = frame.loc[:, list(spec.columns)].copy()
        target = output_dir / spec.target
        write_csv(frame, target)
        reread = pd.read_csv(target)
        if reread.shape != frame.shape:
            raise ValueError(f"Round-trip shape mismatch for {target}")
        frames[spec.target] = frame
        manifest_rows.append(
            {
                "file": spec.target,
                "rows": len(frame),
                "columns": len(frame.columns),
                "sha256": sha256(target),
                "legacy_source": spec.source,
                "legacy_sheet": spec.sheet or "",
                "legacy_sha256": sha256(source),
            }
        )
        schemas[spec.target] = {
            "primary_key": "paper_id" if "paper_id" in frame.columns else None,
            "columns": [
                {
                    "name": str(column),
                    "dtype_at_export": str(frame[column].dtype),
                    "missing": int(frame[column].isna().sum()),
                }
                for column in frame.columns
            ],
        }

    membership = build_membership(args.corpus_gpu_root.resolve(), frames["compute_papers.csv"])
    membership_target = output_dir / "paper_sample_membership.csv"
    write_csv(membership, membership_target)
    manifest_rows.append(
        {
            "file": membership_target.name,
            "rows": len(membership),
            "columns": len(membership.columns),
            "sha256": sha256(membership_target),
            "legacy_source": "data/gpu/*/extract/*_gpu.json",
            "legacy_sheet": "",
            "legacy_sha256": "set_verified_by_exact_paper_id_membership",
        }
    )
    schemas[membership_target.name] = {
        "primary_key": "paper_id",
        "columns": [
            {"name": str(column), "dtype_at_export": str(membership[column].dtype), "missing": 0}
            for column in membership.columns
        ],
    }

    consumption = build_consumption_audit(args.consumption_workbook.resolve())
    consumption_target = output_dir / "consumption_audit_labels.csv"
    write_csv(consumption, consumption_target)
    manifest_rows.append(
        {
            "file": consumption_target.name,
            "rows": len(consumption),
            "columns": len(consumption.columns),
            "sha256": sha256(consumption_target),
            "legacy_source": str(args.consumption_workbook),
            "legacy_sheet": "annotation",
            "legacy_sha256": sha256(args.consumption_workbook.resolve()),
        }
    )
    schemas[consumption_target.name] = {
        "primary_key": "sample_id",
        "columns": [
            {"name": str(column), "dtype_at_export": str(consumption[column].dtype), "missing": 0}
            for column in consumption.columns
        ],
    }

    manifest = pd.DataFrame(manifest_rows).sort_values("file")
    write_csv(manifest, output_dir / "manifest.csv")
    (output_dir / "schema.json").write_text(
        json.dumps(schemas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--corpus-gpu-root", type=Path, required=True)
    parser.add_argument("--consumption-workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    export(build_parser().parse_args())
