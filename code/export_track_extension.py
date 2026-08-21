"""Export minimized Main/Findings inputs for the public track-extension analysis.

This is a maintainer utility, not part of the default offline reproduction run.
It reads the frozen local Main and Findings analysis bundles and writes only the
paper-level fields needed to reproduce the public appendix tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


COMPUTE_FILE = "compute_paper_level_with_contributions_gpu_only.xlsx"
METADATA_FILE = "openalex_paper_metadata_gpu_only.xlsx"
TOPIC_FILE = "acl_arr_topics_all_acl_metadata_desirouter_complete_gpu_only.xlsx"
ORG_FILE = "paper_level_org_variables_gpu_only.csv"

PAPER_COLUMNS = [
    "paper_id",
    "track",
    "year",
    "venue",
    "is_disclosed",
    "is_strict",
    "is_lb1_gfimp",
    "paper_gpu_num_total",
    "paper_max_row_compute_capability",
    "paper_max_row_compute_capability_gfimp_lb1",
    "team_size",
    "cited_by_count",
    "openalex_field_normalized_percentile",
    "primary_topic",
    "n_organizations",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )


def read_topic_table(path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    sheet = "topics" if "topics" in workbook.sheet_names else workbook.sheet_names[0]
    return pd.read_excel(workbook, sheet_name=sheet)


def build_track_papers(data_dir: Path, track: str) -> tuple[pd.DataFrame, list[Path]]:
    compute_path = data_dir / COMPUTE_FILE
    metadata_path = data_dir / METADATA_FILE
    topic_path = data_dir / TOPIC_FILE
    org_path = data_dir / ORG_FILE

    compute = pd.read_excel(compute_path)
    metadata = pd.read_excel(metadata_path)
    topics = read_topic_table(topic_path)
    organizations = pd.read_csv(org_path)

    if not compute["paper_id"].is_unique:
        raise ValueError(f"Duplicate compute paper IDs in {data_dir}")

    papers = (
        compute.merge(
            metadata[
                [
                    "source_acl_id",
                    "team_size",
                    "cited_by_count",
                    "citation_normalized_percentile.value",
                ]
            ].drop_duplicates("source_acl_id"),
            left_on="paper_id",
            right_on="source_acl_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            topics[["paper_id", "topic"]].drop_duplicates("paper_id"),
            on="paper_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            organizations[["paper_id", "n_organizations"]].drop_duplicates("paper_id"),
            on="paper_id",
            how="left",
            validate="one_to_one",
        )
    )

    papers["track"] = track
    papers["year"] = pd.to_numeric(papers["paper_year"], errors="raise").astype(int)
    papers["venue"] = papers["paper_venue"].astype(str).str.lower()
    papers["openalex_field_normalized_percentile"] = papers[
        "citation_normalized_percentile.value"
    ]
    papers["primary_topic"] = papers["topic"].fillna("Unknown").astype(str)
    papers = papers[PAPER_COLUMNS].copy()
    papers = papers.sort_values("paper_id").reset_index(drop=True)
    return papers, [compute_path, metadata_path, topic_path, org_path]


def build_membership(
    main_membership_path: Path,
    findings_manifest_path: Path,
    papers: pd.DataFrame,
) -> pd.DataFrame:
    main = pd.read_csv(main_membership_path)
    main["track"] = "main"
    main = main[["paper_id", "track", "year", "venue", "model_reported", "strict_reported"]]

    findings = pd.read_csv(findings_manifest_path)
    paper_ids = set(papers.loc[papers["track"].eq("findings"), "paper_id"])
    strict_ids = set(
        papers.loc[
            papers["track"].eq("findings")
            & pd.to_numeric(papers["is_strict"], errors="coerce").eq(1),
            "paper_id",
        ]
    )
    findings_membership = pd.DataFrame(
        {
            "paper_id": findings["anthology_id"].astype(str),
            "track": "findings",
            "year": pd.to_numeric(findings["year"], errors="raise").astype(int),
            "venue": findings["venue"].astype(str).str.upper(),
        }
    )
    findings_membership["model_reported"] = findings_membership["paper_id"].isin(paper_ids).astype(int)
    findings_membership["strict_reported"] = findings_membership["paper_id"].isin(strict_ids).astype(int)

    membership = pd.concat([main, findings_membership], ignore_index=True)
    membership = membership.sort_values(["track", "paper_id"]).reset_index(drop=True)
    if not membership["paper_id"].is_unique:
        raise ValueError("Track-extension membership contains duplicate paper IDs")

    observed = {
        "total": len(membership),
        "model": int(membership["model_reported"].sum()),
        "strict": int(membership["strict_reported"].sum()),
        "main_total": int(membership["track"].eq("main").sum()),
        "findings_total": int(membership["track"].eq("findings").sum()),
    }
    expected = {
        "total": 23_838,
        "model": 12_724,
        "strict": 9_546,
        "main_total": 13_921,
        "findings_total": 9_917,
    }
    if observed != expected:
        raise ValueError(f"Unexpected track-extension membership counts: {observed}; expected {expected}")
    return membership


def export(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    main, main_sources = build_track_papers(args.main_data_dir.resolve(), "main")
    findings, findings_sources = build_track_papers(args.findings_data_dir.resolve(), "findings")
    papers = pd.concat([main, findings], ignore_index=True)
    papers = papers.sort_values(["track", "paper_id"]).reset_index(drop=True)

    if len(papers) != 12_724 or papers["paper_id"].nunique() != 12_724:
        raise ValueError(f"Unexpected track-extension paper rows: {len(papers)}")
    if int(pd.to_numeric(papers["is_strict"], errors="coerce").eq(1).sum()) != 9_546:
        raise ValueError("Unexpected strict count in track-extension papers")

    membership = build_membership(
        args.main_membership.resolve(),
        args.findings_manifest.resolve(),
        papers,
    )

    papers_path = output_dir / "track_extension_papers.csv"
    membership_path = output_dir / "track_extension_membership.csv"
    write_csv(papers, papers_path)
    write_csv(membership, membership_path)

    source_paths = [
        *((f"main:{path.name}", path) for path in main_sources),
        *((f"findings:{path.name}", path) for path in findings_sources),
        ("main:paper_sample_membership.csv", args.main_membership.resolve()),
        ("findings:target_manifest.csv", args.findings_manifest.resolve()),
    ]
    provenance = {
        "release_scope": "minimal paper-level inputs for Main/Findings appendix robustness",
        "source_files": [
            {"role": role, "sha256": sha256(path)} for role, path in source_paths
        ],
        "counts": {
            "membership_rows": len(membership),
            "paper_rows": len(papers),
            "main_model_reported": int(papers["track"].eq("main").sum()),
            "findings_model_reported": int(papers["track"].eq("findings").sum()),
            "strict_rows": int(pd.to_numeric(papers["is_strict"], errors="coerce").eq(1).sum()),
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = pd.DataFrame(
        [
            {
                "file": papers_path.name,
                "rows": len(papers),
                "columns": len(papers.columns),
                "sha256": sha256(papers_path),
            },
            {
                "file": membership_path.name,
                "rows": len(membership),
                "columns": len(membership.columns),
                "sha256": sha256(membership_path),
            },
        ]
    ).sort_values("file")
    write_csv(manifest, output_dir / "manifest.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-data-dir", type=Path, required=True)
    parser.add_argument("--findings-data-dir", type=Path, required=True)
    parser.add_argument("--main-membership", type=Path, required=True)
    parser.add_argument("--findings-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    export(build_parser().parse_args())
