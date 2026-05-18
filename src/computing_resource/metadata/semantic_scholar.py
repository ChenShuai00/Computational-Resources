import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Optional

import requests

from computing_resource.common.io import build_logger, derive_output_path, iter_json_files, load_json, maybe_sleep, write_json
from computing_resource.config import load_config


SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_PAPER_SEARCH_ENDPOINT = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search/bulk"
SEMANTIC_SCHOLAR_PAPER_FIELDS = "paperId,title,year,authors,externalIds,url"
SEMANTIC_SCHOLAR_PAPER_DETAIL_FIELDS = (
    "paperId,title,year,"
    "authors.authorId,authors.name,authors.externalIds,authors.affiliations,"
    "externalIds,url,abstract,venue,publicationDate,publicationTypes,"
    "citationCount,influentialCitationCount,referenceCount,isOpenAccess,"
    "openAccessPdf,fieldsOfStudy,references"
)
SEMANTIC_SCHOLAR_DETAIL_REQUIRED_FIELDS = {
    "abstract",
    "venue",
    "publicationDate",
    "publicationTypes",
    "citationCount",
    "influentialCitationCount",
    "referenceCount",
    "isOpenAccess",
    "openAccessPdf",
    "fieldsOfStudy",
    "references",
}
SEMANTIC_SCHOLAR_SEARCH_MAX_PAGES = 3
SEMANTIC_SCHOLAR_MAX_RETRIES = 3
SEMANTIC_SCHOLAR_RETRY_BACKOFF_SECONDS = 5.0


def is_semantic_scholar_rate_limit_error(exc: Exception) -> bool:
    if not isinstance(exc, requests.HTTPError):
        return False
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429


def semantic_scholar_retry_delay(response: Optional[requests.Response], attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    now = datetime.now(retry_at.tzinfo)
                    return max(0.0, (retry_at - now).total_seconds())
                except (TypeError, ValueError):
                    pass
    return SEMANTIC_SCHOLAR_RETRY_BACKOFF_SECONDS * (2 ** attempt)


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def build_title_query(title: Optional[str]) -> str:
    normalized_title = " ".join((title or "").split())
    if not normalized_title:
        return ""
    escaped_title = normalized_title.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped_title}"'


def semantic_scholar_get_json(
    url: str,
    params: Optional[dict],
    api_key: Optional[str],
    timeout: float,
    max_retries: int = SEMANTIC_SCHOLAR_MAX_RETRIES,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(max_retries + 1):
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if is_semantic_scholar_rate_limit_error(exc) and attempt < max_retries:
                sleep_fn(semantic_scholar_retry_delay(getattr(exc, "response", None), attempt))
                continue
            raise
        return response.json()

    raise RuntimeError("Semantic Scholar request failed after retries")


def first_acl_author(record: dict) -> Optional[str]:
    authors = record.get("authors") or []
    return authors[0] if authors else None


def first_semantic_scholar_author(paper: dict) -> Optional[str]:
    authors = paper.get("authors") or []
    if not authors:
        return None
    return (authors[0] or {}).get("name")


def build_query_info(acl_record: dict) -> dict:
    return {
        "doi": acl_record.get("doi"),
        "title": acl_record.get("title"),
        "year": acl_record.get("year"),
        "first_author": first_acl_author(acl_record),
    }


def summarize_affiliations(affiliations) -> list[str]:
    if not affiliations:
        return []
    if isinstance(affiliations, str):
        return [affiliations]
    summary = []
    for affiliation in affiliations:
        if isinstance(affiliation, str):
            cleaned = affiliation.strip()
        elif isinstance(affiliation, dict):
            cleaned = (
                affiliation.get("name")
                or affiliation.get("displayName")
                or affiliation.get("institution")
                or affiliation.get("label")
            )
        else:
            cleaned = str(affiliation).strip()
        if cleaned:
            summary.append(cleaned)
    return summary


def semantic_scholar_raw_has_required_fields(raw: Optional[dict]) -> bool:
    if not isinstance(raw, dict):
        return False
    if not all(field in raw for field in SEMANTIC_SCHOLAR_DETAIL_REQUIRED_FIELDS):
        return False
    authors = raw.get("authors")
    if not isinstance(authors, list) or not authors:
        return False
    for author in authors:
        if not isinstance(author, dict) or "affiliations" not in author:
            return False
    return True


def semantic_scholar_authors_summary_is_complete(summary: Optional[list[dict]]) -> bool:
    if not isinstance(summary, list) or not summary:
        return False
    for author in summary:
        if not isinstance(author, dict) or "affiliations" not in author:
            return False
    return True


def fetch_semantic_scholar_paper_details(
    paper_id: str,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    get_json: Callable[..., dict] = semantic_scholar_get_json,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    if logger is not None:
        logger(f"[INFO] Semantic Scholar paper details: {paper_id}")
    return get_json(
        f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}",
        {"fields": SEMANTIC_SCHOLAR_PAPER_DETAIL_FIELDS},
        api_key,
        timeout,
    )


def iter_semantic_scholar_bulk_search_results(
    title: Optional[str],
    year: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    get_json: Callable[..., dict] = semantic_scholar_get_json,
    logger: Optional[Callable[[str], None]] = None,
    sleep_seconds: float = 0.0,
    sleep_fn=time.sleep,
    max_pages: int = SEMANTIC_SCHOLAR_SEARCH_MAX_PAGES,
):
    query = build_title_query(title)
    if not query:
        return

    if logger is not None:
        logger(f"[INFO] Semantic Scholar title search: {title}")

    base_params = {"query": query, "fields": SEMANTIC_SCHOLAR_PAPER_FIELDS}
    if year:
        base_params["year"] = year
    token = None
    for page in range(max_pages):
        maybe_sleep(sleep_seconds, sleep_fn)
        params = dict(base_params)
        if token:
            params["token"] = token
            if logger is not None:
                logger(f"[INFO] Semantic Scholar title search continuation page {page + 1}")
        result = get_json(SEMANTIC_SCHOLAR_PAPER_SEARCH_ENDPOINT, params, api_key, timeout)
        yield result

        token = result.get("token")
        if not token:
            break


def summarize_authors(paper: Optional[dict]) -> list[dict]:
    if not paper:
        return []

    summary = []
    for author in paper.get("authors") or []:
        summary.append(
            {
                "author_id": author.get("authorId"),
                "name": author.get("name"),
                "external_ids": author.get("externalIds") or {},
                "affiliations": summarize_affiliations(author.get("affiliations")),
            }
        )
    return summary


def is_confident_search_match(acl_record: dict, paper: dict) -> bool:
    if normalize_text(acl_record.get("title")) != normalize_text(paper.get("title")):
        return False

    acl_year = str(acl_record.get("year") or "").strip()
    paper_year = str(paper.get("year") or "").strip()
    if acl_year and paper_year and acl_year != paper_year:
        return False

    acl_author = normalize_text(first_acl_author(acl_record))
    paper_author = normalize_text(first_semantic_scholar_author(paper))
    if acl_author and paper_author and acl_author != paper_author:
        return False

    return True


def fetch_semantic_scholar_record(
    acl_record: dict,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    get_json: Callable[..., dict] = semantic_scholar_get_json,
    logger: Optional[Callable[[str], None]] = None,
    sleep_seconds: float = 0.0,
    sleep_fn=time.sleep,
) -> dict:
    query = build_query_info(acl_record)
    title = acl_record.get("title")
    year = str(acl_record.get("year") or "").strip() or None
    for result in iter_semantic_scholar_bulk_search_results(
        title,
        year=year,
        api_key=api_key,
        timeout=timeout,
        get_json=get_json,
        logger=logger,
        sleep_seconds=sleep_seconds,
        sleep_fn=sleep_fn,
    ):
        for paper in result.get("data", []):
            if is_confident_search_match(acl_record, paper):
                paper_id = paper.get("paperId")
                if not paper_id:
                    raise ValueError("Semantic Scholar search result is missing paperId")
                paper_details = fetch_semantic_scholar_paper_details(
                    paper_id,
                    api_key=api_key,
                    timeout=timeout,
                    get_json=get_json,
                    logger=logger,
                )
                return {
                    "source_acl_id": acl_record["anthology_id"],
                    "matched": True,
                    "match_strategy": "title_year_author",
                    "query": query,
                    "semantic_scholar_raw": paper_details,
                    "authors_summary": summarize_authors(paper_details),
                }

    return {
        "source_acl_id": acl_record["anthology_id"],
        "matched": False,
        "match_strategy": None,
        "query": query,
        "semantic_scholar_raw": None,
        "authors_summary": [],
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
            "step": "semantic-scholar",
            "conference": conference,
            "failed_count": len(failures),
            "failures": failures,
        },
        failed_manifest_path(output_root, conference),
    )


def run_semantic_scholar_file(
    input_path: Path,
    output_path: Path,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    sleep_seconds: float = 1.0,
    sleep_fn=time.sleep,
    logger: Optional[Callable[[str], None]] = None,
    get_json: Callable[..., dict] = semantic_scholar_get_json,
) -> Optional[Path]:
    if output_path.exists():
        try:
            existing_data = load_json(output_path)
        except json.JSONDecodeError as exc:
            existing_data = None
            if logger is not None:
                logger(f"[WARN] Existing Semantic Scholar metadata is unreadable, rebuilding {output_path}: {exc}")

        if existing_data is not None:
            matched = bool(existing_data.get("matched"))
            raw = existing_data.get("semantic_scholar_raw")
            has_complete_raw = semantic_scholar_raw_has_required_fields(raw)
            has_complete_authors_summary = semantic_scholar_authors_summary_is_complete(
                existing_data.get("authors_summary")
            )

            if matched:
                if has_complete_raw and has_complete_authors_summary:
                    if logger is not None:
                        logger(f"[INFO] Skipping existing enriched file: {output_path}")
                    return output_path
                if has_complete_raw:
                    if logger is not None:
                        logger(f"[INFO] Enriching existing Semantic Scholar metadata: {output_path}")
                    existing_data["authors_summary"] = summarize_authors(raw)
                    saved_path = write_json(existing_data, output_path)
                    if logger is not None:
                        logger(f"[INFO] Saved Semantic Scholar metadata: {saved_path}")
                    return saved_path
            elif existing_data.get("authors_summary") is not None:
                if logger is not None:
                    logger(f"[INFO] Skipping existing unmatched file: {output_path}")
                return output_path
            if logger is not None:
                logger(f"[WARN] Existing Semantic Scholar metadata is incomplete, rebuilding {output_path}")

    if logger is not None:
        logger(f"[INFO] Reading ACL metadata: {input_path}")
    acl_record = load_json(input_path)
    try:
        data = fetch_semantic_scholar_record(
            acl_record,
            api_key=api_key,
            timeout=timeout,
            get_json=get_json,
            logger=logger,
            sleep_seconds=sleep_seconds,
            sleep_fn=sleep_fn,
        )
    except requests.RequestException as exc:
        if logger is not None:
            logger(f"[WARN] Semantic Scholar request failed for {input_path}, skipping for now: {exc}")
        return None
    saved_path = write_json(data, output_path)
    if logger is not None:
        logger(f"[INFO] Saved Semantic Scholar metadata: {saved_path}")
    return saved_path


def build_parser() -> argparse.ArgumentParser:
    config = load_config()
    parser = argparse.ArgumentParser(description="Fetch Semantic Scholar paper metadata from ACL metadata (title search / bulk)")
    parser.add_argument("--input-root", default=str(config["paths"]["acl_metadata_root"]))
    parser.add_argument("--output-root", default=str(config["paths"]["semantic_scholar_root"]))
    parser.add_argument("--conference", default=None, help="Optional conference/volume subdirectory, such as 2025.emnlp-main")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Semantic Scholar API key; defaults to SEMANTIC_SCHOLAR_API_KEY",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Seconds to wait before each Semantic Scholar request; pass 0 to disable")
    parser.add_argument("--log-file", default=None)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def resolve_api_key(args) -> Optional[str]:
    return args.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")


def main(argv=None):
    args = parse_args(argv)
    logger = build_logger(Path(args.log_file) if args.log_file else None)
    api_key = resolve_api_key(args)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    acl_files = list(iter_acl_files(input_root, conference=args.conference))

    logger(
        "[INFO] Starting Semantic Scholar fetch: "
        f"input_root={input_root} output_root={output_root} files={len(acl_files)}"
    )
    failures = []
    for index, input_path in enumerate(acl_files, start=1):
        output_path = derive_output_path(input_path, input_root, output_root)
        logger(f"[INFO] Processing {index}/{len(acl_files)}: {input_path}")
        try:
            run_semantic_scholar_file(
                input_path,
                output_path,
                api_key=api_key,
                timeout=args.timeout,
                sleep_seconds=args.sleep_seconds,
                logger=logger,
            )
        except Exception as exc:
            logger(f"[WARN] Semantic Scholar failed for {input_path}, continuing: {exc}")
            failures.append(
                {
                    "input_path": input_path.as_posix(),
                    "relative_path": input_path.relative_to(input_root).as_posix(),
                    "error": str(exc),
                }
            )

    manifest_path = write_failed_manifest(output_root, failures, conference=args.conference)
    if manifest_path is not None:
        logger(f"[WARN] Semantic Scholar failures recorded at: {manifest_path}")


if __name__ == "__main__":
    main()
