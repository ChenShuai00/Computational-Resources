import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

import requests

from computing_resource.common.io import build_logger, derive_output_path, iter_json_files, load_json, maybe_sleep, write_json
from computing_resource.config import load_config


OPENALEX_API_BASE = "https://api.openalex.org"


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def openalex_get_json(
    url: str,
    params: Optional[dict],
    api_key: str,
    email: Optional[str],
    timeout: float,
) -> dict:
    params = dict(params or {})
    params["api_key"] = api_key
    if email:
        params["mailto"] = email
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def extract_openalex_id(entity_id: Optional[str], expected_prefix: str) -> Optional[str]:
    if not entity_id:
        return None
    entity_id = entity_id.strip().rstrip("/")
    if entity_id.startswith("https://openalex.org/"):
        entity_id = entity_id.removeprefix("https://openalex.org/")
    if entity_id.startswith(expected_prefix):
        return entity_id
    return None


def first_acl_author(record: dict) -> Optional[str]:
    authors = record.get("authors") or []
    return authors[0] if authors else None


def first_openalex_author(work: dict) -> Optional[str]:
    authorships = work.get("authorships") or []
    if not authorships:
        return None
    return ((authorships[0].get("author") or {}).get("display_name"))


def is_confident_search_match(acl_record: dict, work: dict) -> bool:
    if normalize_text(acl_record.get("title")) != normalize_text(work.get("display_name")):
        return False

    acl_year = str(acl_record.get("year") or "").strip()
    work_year = str(work.get("publication_year") or "").strip()
    if acl_year and work_year and acl_year != work_year:
        return False

    acl_author = normalize_text(first_acl_author(acl_record))
    work_author = normalize_text(first_openalex_author(work))
    if acl_author and work_author and acl_author != work_author:
        return False

    return True


def build_query_info(acl_record: dict) -> dict:
    return {
        "doi": acl_record.get("doi"),
        "title": acl_record.get("title"),
        "year": acl_record.get("year"),
        "first_author": first_acl_author(acl_record),
    }


def fetch_openalex_author(
    author_id: Optional[str],
    api_key: str,
    email: Optional[str] = None,
    timeout: float = 30.0,
    get_json: Callable[[str, Optional[dict], str, Optional[str], float], dict] = openalex_get_json,
) -> dict:
    author_key = extract_openalex_id(author_id, "A")
    if not author_key:
        raise ValueError(f"Invalid OpenAlex author id: {author_id}")
    return get_json(f"{OPENALEX_API_BASE}/authors/{author_key}", None, api_key, email, timeout)


def enrich_openalex_authors(
    work: Optional[dict],
    api_key: str,
    email: Optional[str] = None,
    timeout: float = 30.0,
    get_json: Callable[[str, Optional[dict], str, Optional[str], float], dict] = openalex_get_json,
    logger: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    if not work:
        return []

    authors_enriched = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        author_id = author.get("id")
        enriched = {
            "author_id": author_id,
            "display_name": author.get("display_name"),
            "author_position": authorship.get("author_position"),
            "is_corresponding": authorship.get("is_corresponding", False),
            "author_raw": None,
        }
        try:
            enriched["author_raw"] = fetch_openalex_author(
                author_id,
                api_key=api_key,
                email=email,
                timeout=timeout,
                get_json=get_json,
            )
        except Exception as exc:
            enriched["fetch_error"] = str(exc)
            if logger is not None:
                logger(f"[WARN] OpenAlex author lookup failed for {author_id}: {exc}")
        authors_enriched.append(enriched)
    return authors_enriched


def fetch_openalex_record(
    acl_record: dict,
    api_key: str,
    email: Optional[str] = None,
    timeout: float = 30.0,
    get_json: Callable[[str, Optional[dict], str, Optional[str], float], dict] = openalex_get_json,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    query = build_query_info(acl_record)
    doi = acl_record.get("doi")

    if doi:
        doi_url = f"{OPENALEX_API_BASE}/works/doi:{quote(doi, safe='')}"
        if logger is not None:
            logger(f"[INFO] OpenAlex DOI lookup: {doi}")
        try:
            work = get_json(doi_url, None, api_key, email, timeout)
            return {
                "source_acl_id": acl_record["anthology_id"],
                "matched": True,
                "match_strategy": "doi",
                "query": query,
                "openalex_raw": work,
                "authors_enriched": enrich_openalex_authors(
                    work,
                    api_key=api_key,
                    email=email,
                    timeout=timeout,
                    get_json=get_json,
                    logger=logger,
                ),
            }
        except Exception as exc:
            if logger is not None:
                logger(f"[WARN] DOI lookup failed for {acl_record['anthology_id']}: {exc}")

    params = {"search": acl_record.get("title") or "", "per-page": 5}
    year = str(acl_record.get("year") or "").strip()
    if year:
        params["filter"] = f"publication_year:{year}"
    if logger is not None:
        logger(f"[INFO] OpenAlex fallback search: {acl_record.get('title')}")
    result = get_json(f"{OPENALEX_API_BASE}/works", params, api_key, email, timeout)
    for work in result.get("results", []):
        if is_confident_search_match(acl_record, work):
            return {
                "source_acl_id": acl_record["anthology_id"],
                "matched": True,
                "match_strategy": "title_year_author",
                "query": query,
                "openalex_raw": work,
                "authors_enriched": enrich_openalex_authors(
                    work,
                    api_key=api_key,
                    email=email,
                    timeout=timeout,
                    get_json=get_json,
                    logger=logger,
                ),
            }

    return {
        "source_acl_id": acl_record["anthology_id"],
        "matched": False,
        "match_strategy": None,
        "query": query,
        "openalex_raw": None,
    }


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
            "step": "openalex",
            "conference": conference,
            "failed_count": len(failures),
            "failures": failures,
        },
        failed_manifest_path(output_root, conference),
    )


def run_openalex_file(
    input_path: Path,
    output_path: Path,
    api_key: str,
    email: Optional[str] = None,
    timeout: float = 30.0,
    sleep_seconds: float = 1.0,
    sleep_fn=time.sleep,
    logger: Optional[Callable[[str], None]] = None,
    get_json: Callable[[str, Optional[dict], str, Optional[str], float], dict] = openalex_get_json,
) -> Path:
    if output_path.exists():
        try:
            existing_data = load_json(output_path)
        except json.JSONDecodeError as exc:
            existing_data = None
            if logger is not None:
                logger(f"[WARN] Existing OpenAlex metadata is unreadable, rebuilding {output_path}: {exc}")

        if existing_data is not None:
            if existing_data.get("authors_enriched") is not None:
                if logger is not None:
                    logger(f"[INFO] Skipping existing enriched file: {output_path}")
                return output_path
            if existing_data.get("matched") and existing_data.get("openalex_raw"):
                if logger is not None:
                    logger(f"[INFO] Enriching existing OpenAlex metadata: {output_path}")
                existing_data["authors_enriched"] = enrich_openalex_authors(
                    existing_data.get("openalex_raw"),
                    api_key=api_key,
                    email=email,
                    timeout=timeout,
                    get_json=get_json,
                    logger=logger,
                )
                saved_path = write_json(existing_data, output_path)
                if logger is not None:
                    logger(f"[INFO] Saved OpenAlex metadata: {saved_path}")
                return saved_path
            if logger is not None:
                logger(f"[WARN] Existing OpenAlex metadata is incomplete, rebuilding {output_path}")

    if logger is not None:
        logger(f"[INFO] Reading ACL metadata: {input_path}")
        if sleep_seconds > 0:
            logger(f"[INFO] Sleeping {sleep_seconds:.1f}s before OpenAlex request")
    maybe_sleep(sleep_seconds, sleep_fn)
    acl_record = load_json(input_path)
    data = fetch_openalex_record(
        acl_record,
        api_key=api_key,
        email=email,
        timeout=timeout,
        get_json=get_json,
        logger=logger,
    )
    saved_path = write_json(data, output_path)
    if logger is not None:
        logger(f"[INFO] Saved OpenAlex metadata: {saved_path}")
    return saved_path


def build_parser() -> argparse.ArgumentParser:
    config = load_config()
    parser = argparse.ArgumentParser(description="Fetch OpenAlex work metadata from ACL metadata")
    parser.add_argument("--input-root", default=str(config["paths"]["acl_metadata_root"]))
    parser.add_argument("--output-root", default=str(config["paths"]["openalex_root"]))
    parser.add_argument("--conference", default=None, help="Optional conference/volume subdirectory, such as 2025.emnlp-main")
    parser.add_argument("--api-key", default=None, help="OpenAlex API key; defaults to OPENALEX_API_KEY")
    parser.add_argument("--email", default=None, help="OpenAlex polite-pool email; defaults to OPENALEX_EMAIL")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--log-file", default=None)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def resolve_api_credentials(args) -> tuple[str, Optional[str]]:
    api_key = args.api_key or os.getenv("OPENALEX_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAlex API key. Pass --api-key or set OPENALEX_API_KEY.")
    email = args.email or os.getenv("OPENALEX_EMAIL")
    return api_key, email


def main(argv=None):
    args = parse_args(argv)
    logger = build_logger(Path(args.log_file) if args.log_file else None)
    api_key, email = resolve_api_credentials(args)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    acl_files = list(iter_acl_files(input_root, conference=args.conference))

    logger(
        f"[INFO] Starting OpenAlex fetch: input_root={input_root} output_root={output_root} files={len(acl_files)}"
    )
    failures = []
    for index, input_path in enumerate(acl_files, start=1):
        output_path = derive_output_path(input_path, input_root, output_root)
        logger(f"[INFO] Processing {index}/{len(acl_files)}: {input_path}")
        try:
            run_openalex_file(
                input_path,
                output_path,
                api_key=api_key,
                email=email,
                timeout=args.timeout,
                sleep_seconds=args.sleep_seconds,
                logger=logger,
            )
        except Exception as exc:
            logger(f"[WARN] OpenAlex failed for {input_path}, continuing: {exc}")
            failures.append(
                {
                    "input_path": input_path.as_posix(),
                    "relative_path": input_path.relative_to(input_root).as_posix(),
                    "error": str(exc),
                }
            )

    manifest_path = write_failed_manifest(output_root, failures, conference=args.conference)
    if manifest_path is not None:
        logger(f"[WARN] OpenAlex failures recorded at: {manifest_path}")


if __name__ == "__main__":
    main()
