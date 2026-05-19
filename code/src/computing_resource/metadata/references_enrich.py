import argparse
from collections.abc import Iterable
import json
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Optional

import requests

from computing_resource.common.io import build_logger, iter_json_files, load_json, maybe_sleep, write_json
from computing_resource.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_METADATA_ROOT = PROJECT_ROOT / "data" / "interim" / "reference_metadata"
SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_PAPER_BATCH_ENDPOINT = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/batch"
SEMANTIC_SCHOLAR_PAPER_SEARCH_ENDPOINT = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search/bulk"
REFERENCE_DETAIL_FIELDS = (
    "paperId,corpusId,title,year,authors.authorId,authors.name,authors.externalIds,authors.affiliations,"
    "externalIds,url,abstract,venue,publicationVenue,publicationDate,publicationTypes,"
    "citationCount,influentialCitationCount,referenceCount,isOpenAccess,"
    "openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,journal,citationStyles,references"
)
REFERENCE_SEARCH_FIELDS = "paperId,title,year,authors,externalIds,url"
TITLE_FALLBACK_PROGRESS_INTERVAL = 100
TITLE_FALLBACK_STATE_FILENAME = "title_fallback_state.json"
SEMANTIC_SCHOLAR_MAX_RETRIES = 3
SEMANTIC_SCHOLAR_RETRY_BACKOFF_SECONDS = 5.0


def default_output_root() -> Path:
    config = load_config()
    return config["paths"].get("reference_metadata_root", DEFAULT_REFERENCE_METADATA_ROOT)


def build_parser() -> argparse.ArgumentParser:
    config = load_config()
    parser = argparse.ArgumentParser(description="Enrich standalone reference metadata from Semantic Scholar reference data")
    parser.add_argument("--input-root", default=str(config["paths"]["semantic_scholar_root"]))
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--conference", required=True, help="conference/volume subdirectory, such as 2025.emnlp-main")
    parser.add_argument("--api-key", default=None, help="Semantic Scholar API key; defaults to SEMANTIC_SCHOLAR_API_KEY")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--throttle-seconds", type=float, default=1.0, help="Seconds to wait between detail-fetch batches")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--enable-title-fallback",
        action="store_true",
        help="Enable title-based fallback lookup for references missing paperId",
    )
    parser.add_argument("--log-file", default=None)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def iter_semantic_scholar_files(input_root: Path, conference: Optional[str] = None):
    yield from iter_json_files(input_root, conference=conference)


def extract_reference_targets(record: dict, source_path: Path) -> dict:
    references = ((record or {}).get("semantic_scholar_raw") or {}).get("references") or []
    reference_ids = []
    unresolved_references = []
    source_paper_id = record.get("source_acl_id")
    source_path_str = source_path.as_posix()

    for reference in references:
        paper_id = reference.get("paperId")
        if paper_id:
            reference_ids.append(paper_id)
            continue
        unresolved_references.append(
            {
                "source_paper_id": source_paper_id,
                "source_path": source_path_str,
                "title": reference.get("title"),
                "year": reference.get("year"),
                "external_ids": reference.get("externalIds") or {},
            }
        )

    return {
        "reference_ids": reference_ids,
        "unresolved_references": unresolved_references,
    }


def collect_reference_targets(input_paths: Iterable[Path]) -> dict:
    reference_ids = []
    unresolved_references = []
    seen_reference_ids = set()

    for input_path in sorted(Path(path) for path in input_paths):
        record = load_json(input_path)
        extracted = extract_reference_targets(record, input_path)
        for reference_id in extracted["reference_ids"]:
            if reference_id in seen_reference_ids:
                continue
            seen_reference_ids.add(reference_id)
            reference_ids.append(reference_id)
        unresolved_references.extend(extracted["unresolved_references"])

    return {
        "reference_ids": reference_ids,
        "unresolved_references": unresolved_references,
    }


def normalize_reference_title(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def build_reference_title_query(title: Optional[str]) -> str:
    normalized_title = " ".join((title or "").split())
    if not normalized_title:
        return ""
    escaped_title = normalized_title.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped_title}"'


def build_unresolved_reference_key(unresolved_reference: dict) -> str:
    return json.dumps(unresolved_reference, ensure_ascii=False, sort_keys=True)


def match_reference_candidate(unresolved_reference: dict, candidate: dict) -> bool:
    if normalize_reference_title(unresolved_reference.get("title")) != normalize_reference_title(candidate.get("title")):
        return False

    unresolved_year = str(unresolved_reference.get("year") or "").strip()
    if unresolved_year:
        candidate_year = str(candidate.get("year") or "").strip()
        if candidate_year != unresolved_year:
            return False

    return True


def chunked(items: Iterable[str], size: int) -> Iterable[list[str]]:
    chunk = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def is_retryable_semantic_scholar_http_error(exc: Exception) -> bool:
    if not isinstance(exc, requests.HTTPError):
        return False
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600)


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


def is_retryable_semantic_scholar_post_error(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ),
    )


def semantic_scholar_post_json(
    url: str,
    payload: dict,
    api_key: Optional[str],
    timeout: float,
    max_retries: int = SEMANTIC_SCHOLAR_MAX_RETRIES,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[dict]:
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    body = dict(payload)
    params = {}
    fields = body.pop("fields", None)
    if fields:
        params["fields"] = fields

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=body, params=params or None, headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.HTTPError as exc:
            if is_retryable_semantic_scholar_http_error(exc) and attempt < max_retries:
                maybe_sleep(semantic_scholar_retry_delay(getattr(exc, "response", None), attempt), sleep_fn=sleep_fn)
                continue
            raise
        except requests.RequestException as exc:
            if is_retryable_semantic_scholar_post_error(exc) and attempt < max_retries:
                maybe_sleep(SEMANTIC_SCHOLAR_RETRY_BACKOFF_SECONDS * (2 ** attempt), sleep_fn=sleep_fn)
                continue
            raise

        data = response.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data") or []
        raise TypeError("Semantic Scholar batch response must be a list or dict")

    raise RuntimeError("Semantic Scholar batch request failed after retries")


def semantic_scholar_get_json(
    url: str,
    params: Optional[dict],
    api_key: Optional[str],
    timeout: float,
):
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_reference_details_batch(
    paper_ids: Iterable[str],
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    batch_size: int = 100,
    post_json: Callable[..., list[dict]] = semantic_scholar_post_json,
    logger: Optional[Callable[[str], None]] = None,
    throttle_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, dict]:
    fetched = {}
    paper_id_chunks = list(chunked(paper_ids, batch_size))
    for index, paper_id_chunk in enumerate(paper_id_chunks, start=1):
        if not paper_id_chunk:
            continue
        if logger is not None:
            logger(
                f"[INFO] Fetching reference detail batch {index}/{len(paper_id_chunks)}: size={len(paper_id_chunk)}"
            )
        response = post_json(
            SEMANTIC_SCHOLAR_PAPER_BATCH_ENDPOINT,
            {"ids": paper_id_chunk, "fields": REFERENCE_DETAIL_FIELDS},
            api_key,
            timeout,
        )
        for paper in response or []:
            if not isinstance(paper, dict):
                continue
            paper_id = paper.get("paperId")
            if paper_id:
                fetched[paper_id] = paper
        if index < len(paper_id_chunks):
            maybe_sleep(throttle_seconds, sleep_fn=sleep_fn)
    return fetched


def search_reference_by_title(
    unresolved_reference: dict,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    get_json: Callable[..., dict] = semantic_scholar_get_json,
) -> Optional[dict]:
    query = build_reference_title_query(unresolved_reference.get("title"))
    if not query:
        return None

    params = {"query": query, "fields": REFERENCE_SEARCH_FIELDS}
    year = unresolved_reference.get("year")
    if year not in (None, ""):
        params["year"] = year

    response = get_json(SEMANTIC_SCHOLAR_PAPER_SEARCH_ENDPOINT, params, api_key, timeout)
    candidates = response.get("data") if isinstance(response, dict) else response
    for candidate in candidates or []:
        if match_reference_candidate(unresolved_reference, candidate):
            return candidate
    return None


def resolve_unresolved_references(
    unresolved_references: list[dict],
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    get_json: Callable[..., dict] = semantic_scholar_get_json,
    logger: Optional[Callable[[str], None]] = None,
    state_path: Optional[Path] = None,
    conference: Optional[str] = None,
) -> dict:
    resolved_reference_ids = []
    resolved_references = []
    remaining_unresolved_references = []
    seen_reference_ids = set()
    unresolved_reference_map = {
        build_unresolved_reference_key(unresolved_reference): unresolved_reference
        for unresolved_reference in unresolved_references
    }
    processed_entries = {}

    if state_path is not None and state_path.exists():
        state = load_json(state_path)
        state_entries = state.get("processed_entries") if isinstance(state, dict) else {}
        if isinstance(state_entries, dict):
            processed_entries = {
                key: value
                for key, value in state_entries.items()
                if key in unresolved_reference_map and isinstance(value, dict)
            }
        if logger is not None and processed_entries:
            logger(
                "[INFO] Resuming title fallback from state: "
                f"processed={len(processed_entries)} "
                f"remaining={len(unresolved_references) - len(processed_entries)}"
            )

    if logger is not None:
        logger(f"[INFO] Resolving unresolved references by title: count={len(unresolved_references)}")

    def persist_state(completed: bool) -> None:
        if state_path is None:
            return
        resolved_ids_for_state = []
        seen_ids_for_state = set()
        for key in unresolved_reference_map:
            entry = processed_entries.get(key)
            if not entry:
                continue
            paper_id = entry.get("paper_id")
            if paper_id and paper_id not in seen_ids_for_state:
                seen_ids_for_state.add(paper_id)
                resolved_ids_for_state.append(paper_id)
        write_json(
            {
                "conference": conference,
                "completed": completed,
                "processed_count": len(processed_entries),
                "total_count": len(unresolved_references),
                "resolved_reference_ids": resolved_ids_for_state,
                "processed_entries": processed_entries,
            },
            state_path,
        )

    for index, unresolved_reference in enumerate(unresolved_references, start=1):
        unresolved_key = build_unresolved_reference_key(unresolved_reference)
        cached_entry = processed_entries.get(unresolved_key)
        if cached_entry is not None:
            cached_reference = cached_entry.get("unresolved_reference") or unresolved_reference
            cached_paper_id = cached_entry.get("paper_id")
            if cached_entry.get("status") == "resolved" and cached_paper_id and cached_paper_id not in seen_reference_ids:
                seen_reference_ids.add(cached_paper_id)
                resolved_reference_ids.append(cached_paper_id)
                resolved_references.append({**cached_reference, **(cached_entry.get("candidate") or {})})
            elif cached_entry.get("status") != "resolved":
                remaining_unresolved_references.append(cached_reference)
            if logger is not None and (
                index % TITLE_FALLBACK_PROGRESS_INTERVAL == 0 or index == len(unresolved_references)
            ):
                logger(f"[INFO] Title fallback progress: processed={index}/{len(unresolved_references)}")
            continue
        try:
            candidate = search_reference_by_title(
                unresolved_reference,
                api_key=api_key,
                timeout=timeout,
                get_json=get_json,
            )
        except requests.RequestException as exc:
            if logger is not None:
                logger(
                    "[WARN] Title fallback failed: "
                    f"title={unresolved_reference.get('title')!r} "
                    f"year={unresolved_reference.get('year')!r} "
                    f"error={exc}"
                )
            remaining_unresolved_references.append(unresolved_reference)
            continue
        candidate_id = (candidate or {}).get("paperId")
        if candidate_id and candidate_id not in seen_reference_ids:
            seen_reference_ids.add(candidate_id)
            resolved_reference_ids.append(candidate_id)
            resolved_references.append({**unresolved_reference, **candidate})
            processed_entries[unresolved_key] = {
                "status": "resolved",
                "paper_id": candidate_id,
                "candidate": candidate,
                "unresolved_reference": unresolved_reference,
            }
            persist_state(completed=False)
            continue
        if candidate_id:
            processed_entries[unresolved_key] = {
                "status": "resolved",
                "paper_id": candidate_id,
                "candidate": candidate,
                "unresolved_reference": unresolved_reference,
            }
        else:
            processed_entries[unresolved_key] = {
                "status": "unresolved",
                "unresolved_reference": unresolved_reference,
            }
            remaining_unresolved_references.append(unresolved_reference)
        persist_state(completed=False)
        if logger is not None and (
            index % TITLE_FALLBACK_PROGRESS_INTERVAL == 0 or index == len(unresolved_references)
        ):
            logger(f"[INFO] Title fallback progress: processed={index}/{len(unresolved_references)}")

    persist_state(completed=True)

    if logger is not None:
        logger(
            "[INFO] Title fallback summary: "
            f"resolved={len(resolved_reference_ids)} "
            f"remaining_unresolved={len(remaining_unresolved_references)}"
        )

    return {
        "resolved_reference_ids": resolved_reference_ids,
        "resolved_references": resolved_references,
        "remaining_unresolved_references": remaining_unresolved_references,
    }


def reference_output_dir(output_root: Path, conference: str) -> Path:
    return Path(output_root) / conference


def title_fallback_state_path(output_root: Path, conference: str) -> Path:
    return reference_output_dir(output_root, conference) / TITLE_FALLBACK_STATE_FILENAME


def reference_output_path(output_root: Path, conference: str, paper_id: str) -> Path:
    return reference_output_dir(output_root, conference) / f"{paper_id}.json"


def write_reference_detail(output_root: Path, conference: str, paper_id: str, detail: dict, overwrite: bool) -> tuple[Path, bool]:
    output_path = reference_output_path(output_root, conference, paper_id)
    if output_path.exists() and not overwrite:
        return output_path, False
    return write_json(detail, output_path), True


def write_unresolved_references(output_root: Path, conference: str, unresolved_references: list[dict]) -> Path:
    return write_json(unresolved_references, reference_output_dir(output_root, conference) / "unresolved_references.json")


def write_reference_index(output_root: Path, conference: str, index: dict) -> Path:
    return write_json(index, reference_output_dir(output_root, conference) / "index.json")


def run_references_enrich(
    input_root: Path,
    output_root: Path,
    conference: str,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    batch_size: int = 100,
    throttle_seconds: float = 1.0,
    overwrite: bool = False,
    enable_title_fallback: bool = False,
    fetch_reference_details_batch: Callable[..., dict[str, dict]] = fetch_reference_details_batch,
    get_json: Callable[..., dict] = semantic_scholar_get_json,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    input_files = list(iter_semantic_scholar_files(Path(input_root), conference=conference))
    collected = collect_reference_targets(input_files)
    direct_reference_ids = collected["reference_ids"]
    unresolved_references = collected["unresolved_references"]
    if logger is not None:
        logger(
            "[INFO] Collected reference targets: "
            f"files={len(input_files)} "
            f"direct_reference_ids={len(direct_reference_ids)} "
            f"unresolved_references={len(unresolved_references)}"
        )
    resolved_reference_ids = []
    remaining_unresolved_references = unresolved_references
    if enable_title_fallback:
        resolved = resolve_unresolved_references(
            unresolved_references,
            api_key=api_key,
            timeout=timeout,
            get_json=get_json,
            logger=logger,
            state_path=title_fallback_state_path(output_root, conference),
            conference=conference,
        )
        resolved_reference_ids = resolved["resolved_reference_ids"]
        remaining_unresolved_references = resolved["remaining_unresolved_references"]
    elif logger is not None and unresolved_references:
        logger(
            "[INFO] Title fallback disabled: "
            f"keeping_unresolved={len(unresolved_references)}"
        )

    reference_ids = []
    seen_reference_ids = set()
    for paper_id in direct_reference_ids + resolved_reference_ids:
        if paper_id in seen_reference_ids:
            continue
        seen_reference_ids.add(paper_id)
        reference_ids.append(paper_id)

    existing_reference_ids = {
        paper_id
        for paper_id in reference_ids
        if reference_output_path(output_root, conference, paper_id).exists()
    }
    reference_ids_to_fetch = [paper_id for paper_id in reference_ids if overwrite or paper_id not in existing_reference_ids]
    reference_id_batches_to_fetch = list(chunked(reference_ids_to_fetch, batch_size))
    fetched_count = 0
    for index, paper_id_batch in enumerate(reference_id_batches_to_fetch, start=1):
        fetch_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
            "batch_size": batch_size,
            "throttle_seconds": throttle_seconds,
        }
        if logger is not None:
            logger(
                f"[INFO] Fetching reference detail batch {index}/{len(reference_id_batches_to_fetch)}: "
                f"size={len(paper_id_batch)}"
            )
        fetched_details = fetch_reference_details_batch(
            paper_id_batch,
            **fetch_kwargs,
        )
        for paper_id in paper_id_batch:
            detail = fetched_details.get(paper_id)
            if detail is None:
                continue
            _, written = write_reference_detail(output_root, conference, paper_id, detail, overwrite=overwrite)
            if written:
                fetched_count += 1

    skipped_existing = len(existing_reference_ids) if not overwrite else 0
    write_unresolved_references(output_root, conference, remaining_unresolved_references)
    write_reference_index(
        output_root,
        conference,
        {
            "conference": conference,
            "semantic_scholar_files": len(input_files),
            "reference_ids_total": len(reference_ids),
            "reference_ids_fetched": fetched_count,
            "reference_ids_skipped_existing": skipped_existing,
            "reference_ids_resolved_by_title": len(resolved_reference_ids),
            "unresolved_total": len(remaining_unresolved_references),
        },
    )
    if logger is not None:
        logger(
            "[INFO] Reference enrichment write summary: "
            f"fetched={fetched_count} "
            f"skipped_existing={skipped_existing} "
            f"resolved_by_title={len(resolved_reference_ids)} "
            f"unresolved={len(remaining_unresolved_references)}"
        )
    return {
        "conference": conference,
        "semantic_scholar_files": len(input_files),
        "reference_ids": reference_ids,
        "unresolved_references": remaining_unresolved_references,
        "reference_ids_fetched": fetched_count,
        "reference_ids_skipped_existing": skipped_existing,
        "reference_ids_resolved_by_title": len(resolved_reference_ids),
    }


def resolve_api_key(args) -> Optional[str]:
    return args.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")


def main(argv=None):
    args = parse_args(argv)
    logger = build_logger(Path(args.log_file) if args.log_file else None)
    api_key = resolve_api_key(args)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    logger(
        "[INFO] Starting reference enrichment: "
        f"input_root={input_root} output_root={output_root} conference={args.conference}"
    )
    result = run_references_enrich(
        input_root=input_root,
        output_root=output_root,
        conference=args.conference,
        api_key=api_key,
        timeout=args.timeout,
        batch_size=args.batch_size,
        throttle_seconds=args.throttle_seconds,
        overwrite=args.overwrite,
        enable_title_fallback=args.enable_title_fallback,
        logger=logger,
    )
    logger(
        "[INFO] Finished reference enrichment: "
        f"files={result['semantic_scholar_files']} "
        f"reference_ids={len(result['reference_ids'])} "
        f"fetched={result['reference_ids_fetched']} "
        f"resolved_by_title={result.get('reference_ids_resolved_by_title', 0)} "
        f"skipped_existing={result['reference_ids_skipped_existing']} "
        f"unresolved={len(result['unresolved_references'])}"
    )
    return result
