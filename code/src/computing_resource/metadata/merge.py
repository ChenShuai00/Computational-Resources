import argparse
from pathlib import Path
from typing import Callable, Optional

from computing_resource.common.io import build_logger, iter_json_files, load_json, write_json
from computing_resource.config import load_config


def iter_acl_files(input_root: Path, conference: Optional[str] = None):
    yield from iter_json_files(input_root, conference=conference)


def failed_manifest_path(output_root: Path, conference: Optional[str] = None) -> Path:
    if conference:
        return output_root / conference / "failed_files.json"
    return output_root / "failed_files.json"


def write_failed_manifest(
    output_root: Path,
    failures: list[dict],
    conference: Optional[str] = None,
) -> Optional[Path]:
    if not failures:
        return None
    return write_json(
        {
            "step": "merge",
            "conference": conference,
            "failed_count": len(failures),
            "failures": failures,
        },
        failed_manifest_path(output_root, conference),
    )


def merge_records(
    acl_record: dict,
    openalex_record: Optional[dict],
    semantic_scholar_record: Optional[dict],
    relative_path: Path,
) -> dict:
    return {
        "acl": acl_record,
        "openalex": openalex_record,
        "semantic_scholar": semantic_scholar_record,
        "merge_meta": {
            "relative_path": relative_path.as_posix(),
            "acl_id": acl_record.get("anthology_id"),
            "openalex_exists": openalex_record is not None,
            "openalex_matched": (openalex_record or {}).get("matched"),
            "semantic_scholar_exists": semantic_scholar_record is not None,
            "semantic_scholar_matched": (semantic_scholar_record or {}).get("matched"),
        },
    }


def run_merge_file(
    acl_path: Path,
    openalex_path: Path,
    semantic_scholar_path: Path,
    output_path: Path,
    relative_path: Optional[Path] = None,
    logger: Optional[Callable[[str], None]] = None,
    overwrite: bool = False,
) -> Path:
    if output_path.exists() and not overwrite:
        if logger is not None:
            logger(f"[INFO] Skipping existing file: {output_path}")
        return output_path

    acl_record = load_json(acl_path)
    openalex_record = load_json(openalex_path) if openalex_path.exists() else None
    semantic_scholar_record = load_json(semantic_scholar_path) if semantic_scholar_path.exists() else None
    merged = merge_records(
        acl_record,
        openalex_record,
        semantic_scholar_record,
        relative_path=relative_path or output_path.name,
    )
    saved_path = write_json(merged, output_path)
    if logger is not None:
        logger(f"[INFO] Saved merged metadata: {saved_path}")
    return saved_path


def build_parser() -> argparse.ArgumentParser:
    config = load_config()
    parser = argparse.ArgumentParser(description="Merge ACL and OpenAlex metadata")
    parser.add_argument("--acl-root", default=str(config["paths"]["acl_metadata_root"]))
    parser.add_argument("--openalex-root", default=str(config["paths"]["openalex_root"]))
    parser.add_argument("--semantic-scholar-root", default=str(config["paths"]["semantic_scholar_root"]))
    parser.add_argument("--output-root", default=str(config["paths"]["merged_metadata_root"]))
    parser.add_argument("--conference", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = build_logger(Path(args.log_file) if args.log_file else None)
    acl_root = Path(args.acl_root)
    openalex_root = Path(args.openalex_root)
    semantic_scholar_root = Path(args.semantic_scholar_root)
    output_root = Path(args.output_root)
    acl_files = list(iter_acl_files(acl_root, conference=args.conference))

    logger(
        "[INFO] Starting merge: "
        f"acl_root={acl_root} openalex_root={openalex_root} "
        f"semantic_scholar_root={semantic_scholar_root} files={len(acl_files)}"
    )
    failures = []
    for index, acl_path in enumerate(acl_files, start=1):
        rel_path = acl_path.relative_to(acl_root)
        output_path = output_root / rel_path
        openalex_path = openalex_root / rel_path
        semantic_scholar_path = semantic_scholar_root / rel_path
        logger(f"[INFO] Merging {index}/{len(acl_files)}: {rel_path}")
        try:
            run_merge_file(
                acl_path,
                openalex_path,
                semantic_scholar_path,
                output_path,
                relative_path=rel_path,
                logger=logger,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            logger(f"[WARN] Merge failed for {acl_path}, continuing: {exc}")
            failures.append(
                {
                    "input_path": acl_path.as_posix(),
                    "relative_path": rel_path.as_posix(),
                    "error": str(exc),
                }
            )

    manifest_path = write_failed_manifest(output_root, failures, conference=args.conference)
    if manifest_path is not None:
        logger(f"[WARN] Merge failures recorded at: {manifest_path}")


if __name__ == "__main__":
    main()
