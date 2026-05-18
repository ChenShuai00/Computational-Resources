import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import time
from typing import Any
import zipfile

import requests

from computing_resource.common.io import build_logger
from computing_resource.common.io import write_json
from computing_resource.config import load_config


DEFAULT_API_BASE_URL = "https://mineru.net/api/v4"
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_POLL_INTERVAL_SECONDS = 5
DEFAULT_BATCH_SIZE = 200
DEFAULT_MODEL_VERSION = "vlm"
DEFAULT_TOKEN_ENV = "MINERU_API_TOKEN"
DEFAULT_DOWNLOAD_WORKERS = 4
DEFAULT_MAX_DOWNLOAD_RETRIES = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch parse PDFs with the hosted MinerU API and download zip results.")
    parser.add_argument("--input-dir")
    parser.add_argument("--conference")
    parser.add_argument("--output-dir")
    parser.add_argument("--token")
    parser.add_argument("--token-env", default=DEFAULT_TOKEN_ENV)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--pattern", default="*.pdf")
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--is-ocr", dest="is_ocr", action="store_true")
    parser.add_argument("--no-is-ocr", dest="is_ocr", action="store_false")
    parser.set_defaults(is_ocr=None)
    parser.add_argument("--language")
    parser.add_argument("--download-results", dest="download_results", action="store_true", default=True)
    parser.add_argument("--no-download-results", dest="download_results", action="store_false")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--extract-results-json")
    parser.add_argument("--download-workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS)
    parser.add_argument("--max-download-retries", type=int, default=DEFAULT_MAX_DOWNLOAD_RETRIES)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        if not args.conference:
            raise SystemExit("--conference is required when --output-dir is omitted.")
        config = load_config()
        args.output_dir = str(config["paths"]["parses_root"] / args.conference)
    if args.download_only and args.retry_failed:
        raise SystemExit("--download-only and --retry-failed cannot be used together.")
    if not args.download_only and not args.input_dir:
        raise SystemExit("--input-dir is required unless --download-only is used.")
    if args.download_only:
        args.token = args.token or os.getenv(args.token_env)
    else:
        args.token = resolve_token(args.token, args.token_env)
    return args


def resolve_token(token: str | None, token_env: str) -> str:
    if token:
        return token
    env_value = os.getenv(token_env)
    if env_value:
        return env_value
    raise SystemExit(f"MinerU token is required via --token or environment variable {token_env}.")


def collect_pdf_paths_from_dir(input_dir: str | Path, pattern: str = "*.pdf") -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")
    pdf_paths = sorted(path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() == ".pdf")
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in directory: {root}")
    return pdf_paths


def chunked(items: list[Path], size: int) -> list[list[Path]]:
    batch_size = max(int(size or 1), 1)
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _auth_headers(token: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def request_upload_batch(pdf_paths: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    files = []
    for pdf_path in pdf_paths:
        file_info: dict[str, Any] = {
            "name": pdf_path.name,
            "data_id": pdf_path.stem,
        }
        if args.is_ocr is not None:
            file_info["is_ocr"] = args.is_ocr
        files.append(file_info)

    payload: dict[str, Any] = {
        "files": files,
        "model_version": args.model_version,
    }
    if args.language is not None:
        payload["language"] = args.language

    response = requests.post(
        f"{args.api_base_url.rstrip('/')}/file-urls/batch",
        headers=_auth_headers(args.token),
        json=payload,
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Upload-link request failed: status={response.status_code}, text={response.text}")
    result = response.json()
    if result.get("code") != 0:
        raise RuntimeError(f"Upload-link request failed: code={result.get('code')}, msg={result.get('msg')}")
    return result


def upload_file(upload_url: str, pdf_path: Path, timeout: int = 300) -> None:
    with pdf_path.open("rb") as handle:
        response = requests.put(upload_url, data=handle, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Upload failed for {pdf_path}: status={response.status_code}")


def get_batch_result(batch_id: str, token: str, api_base_url: str) -> dict[str, Any]:
    response = requests.get(
        f"{api_base_url.rstrip('/')}/extract-results/batch/{batch_id}",
        headers=_auth_headers(token),
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Batch result request failed: status={response.status_code}, text={response.text}")
    result = response.json()
    if result.get("code") != 0:
        raise RuntimeError(f"Batch result request failed: code={result.get('code')}, msg={result.get('msg')}")
    return result


def wait_batch_result(
    batch_id: str,
    token: str,
    api_base_url: str,
    poll_interval: int,
    timeout: int,
) -> dict[str, Any]:
    start_time = time.time()
    while True:
        result = get_batch_result(batch_id, token, api_base_url)
        extract_results = result.get("data", {}).get("extract_result") or []
        running_count = sum(1 for item in extract_results if item.get("state") not in {"done", "failed"})
        if running_count == 0:
            return result
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timed out waiting for batch {batch_id}")
        time.sleep(poll_interval)


def save_extract_results_csv(extract_results: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "file_name",
        "state",
        "full_zip_url",
        "err_msg",
        "data_id",
        "extracted_pages",
        "total_pages",
        "start_time",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in extract_results:
            progress = item.get("extract_progress", {}) or {}
            writer.writerow(
                {
                    "file_name": item.get("file_name", ""),
                    "state": item.get("state", ""),
                    "full_zip_url": item.get("full_zip_url", ""),
                    "err_msg": item.get("err_msg", ""),
                    "data_id": item.get("data_id", ""),
                    "extracted_pages": progress.get("extracted_pages", ""),
                    "total_pages": progress.get("total_pages", ""),
                    "start_time": progress.get("start_time", ""),
                }
            )


def save_summary_csv(
    upload_results: list[dict[str, Any]],
    extract_results: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    fieldnames = [
        "batch_index",
        "batch_id",
        "file_name",
        "state",
        "full_zip_url",
        "err_msg",
        "data_id",
        "extracted_pages",
        "total_pages",
        "start_time",
    ]
    batch_index_map = {item["batch_id"]: item["batch_index"] for item in upload_results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for batch_id, items in extract_results.items():
            batch_index = batch_index_map.get(batch_id, "")
            for item in items:
                progress = item.get("extract_progress", {}) or {}
                writer.writerow(
                    {
                        "batch_index": batch_index,
                        "batch_id": batch_id,
                        "file_name": item.get("file_name", ""),
                        "state": item.get("state", ""),
                        "full_zip_url": item.get("full_zip_url", ""),
                        "err_msg": item.get("err_msg", ""),
                        "data_id": item.get("data_id", ""),
                        "extracted_pages": progress.get("extracted_pages", ""),
                        "total_pages": progress.get("total_pages", ""),
                        "start_time": progress.get("start_time", ""),
                    }
                )


def download_and_extract_zip(download_url: str, output_dir: Path) -> None:
    response = requests.get(download_url, timeout=300, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Download failed: status={response.status_code}, url={download_url}")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "result.zip"
    archive_path.write_bytes(response.content)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(output_dir)
    archive_path.unlink()


def _artifact_dir(output_root: Path, paper_id: str) -> Path:
    return output_root / paper_id / "auto"


def _has_existing_extraction(output_root: Path, paper_id: str) -> bool:
    output_dir = _artifact_dir(output_root, paper_id)
    return (output_dir / f"{paper_id}.md").exists() or (output_dir / "full.md").exists()


def _paper_id_from_item(item: dict[str, Any]) -> str:
    return item.get("data_id") or Path(item.get("file_name", "")).stem


def _load_extract_results(path: Path) -> dict[str, list[dict[str, Any]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_done_paper_ids(extract_results_by_batch: dict[str, list[dict[str, Any]]]) -> set[str]:
    done_ids: set[str] = set()
    for item in _flatten_extract_results(extract_results_by_batch):
        if item.get("state") == "done":
            done_ids.add(_paper_id_from_item(item))
    return done_ids


def _load_existing_extract_results_if_present(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    return _load_extract_results(path)


def _select_pending_pdf_paths(
    pdf_paths: list[Path],
    output_root: Path,
    existing_extract_results: dict[str, list[dict[str, Any]]],
) -> tuple[list[Path], int]:
    existing_done_ids = _existing_done_paper_ids(existing_extract_results)
    pending_paths: list[Path] = []
    skipped_count = 0
    for path in pdf_paths:
        paper_id = path.stem
        if _has_existing_extraction(output_root, paper_id) or paper_id in existing_done_ids:
            skipped_count += 1
            continue
        pending_paths.append(path)
    return pending_paths, skipped_count


def _download_one_result(item: dict[str, Any], output_root: Path, max_retries: int) -> dict[str, Any]:
    paper_id = _paper_id_from_item(item)
    if item.get("state") != "done":
        return {
            "status": "failed",
            "paper_id": paper_id,
            "error": item.get("err_msg") or f"Unexpected state: {item.get('state')}",
        }
    download_url = item.get("full_zip_url")
    if not download_url:
        return {
            "status": "failed",
            "paper_id": paper_id,
            "error": "Missing full_zip_url for done result",
        }
    output_dir = _artifact_dir(output_root, paper_id)
    if _has_existing_extraction(output_root, paper_id):
        return {"status": "skipped", "paper_id": paper_id}
    attempt = 0
    while True:
        try:
            download_and_extract_zip(download_url, output_dir)
            return {"status": "downloaded", "paper_id": paper_id}
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                return {"status": "failed", "paper_id": paper_id, "error": str(exc)}


def _download_successful_results(
    extract_results: list[dict[str, Any]],
    output_root: Path,
    download_workers: int,
    max_download_retries: int,
    logger=None,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    downloaded_count = 0
    skipped_count = 0
    failure_count = 0
    failures: list[dict[str, Any]] = []
    total = len(extract_results)
    processed = 0

    with ThreadPoolExecutor(max_workers=max(int(download_workers or 1), 1)) as executor:
        futures = [
            executor.submit(_download_one_result, item, output_root, max(int(max_download_retries or 0), 0))
            for item in extract_results
        ]
        for future in as_completed(futures):
            processed += 1
            result = future.result()
            status = result["status"]
            if status == "downloaded":
                downloaded_count += 1
            elif status == "skipped":
                skipped_count += 1
            else:
                failure_count += 1
                failures.append({"paper_id": result["paper_id"], "error": result["error"]})
                if logger is not None:
                    logger(f"[mineru-hosted-api] download failed for {result['paper_id']}: {result['error']}")
            if logger is not None and (processed == 1 or processed == total or processed % 25 == 0):
                logger(
                    f"[mineru-hosted-api] download progress {processed}/{total}: "
                    f"downloaded={downloaded_count} skipped={skipped_count} failed={failure_count}"
                )
    return downloaded_count, skipped_count, failure_count, failures


def _flatten_extract_results(extract_results_by_batch: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [item for items in extract_results_by_batch.values() for item in items]


def _select_retry_failed_pdf_paths(
    input_dir: str | Path,
    pattern: str,
    extract_results_by_batch: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    pdf_paths = collect_pdf_paths_from_dir(input_dir, pattern)
    by_stem = {path.stem: path for path in pdf_paths}
    failed_ids = []
    for item in _flatten_extract_results(extract_results_by_batch):
        if item.get("state") == "failed":
            failed_ids.append(_paper_id_from_item(item))
    selected = []
    missing = []
    for paper_id in failed_ids:
        path = by_stem.get(paper_id)
        if path is None:
            missing.append(paper_id)
            continue
        selected.append(path)
    if missing:
        raise FileNotFoundError(f"Missing source PDFs for failed items: {', '.join(missing[:10])}")
    return selected


def run_mineru_hosted_api_batch(args: argparse.Namespace, logger=None) -> dict[str, Any]:
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_dir = output_root / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    extract_results_path = Path(args.extract_results_json) if args.extract_results_json else output_root / "all_extract_results.json"

    upload_results: list[dict[str, Any]] = []
    all_extract_results: dict[str, list[dict[str, Any]]] = _load_existing_extract_results_if_present(extract_results_path)
    failed_batches: list[dict[str, Any]] = []
    success_count = 0
    skipped_count = 0
    failure_count = 0

    if args.download_only:
        if logger is not None:
            logger(f"[mineru-hosted-api] download-only mode using {extract_results_path}")
        all_extract_results = _load_extract_results(extract_results_path)
        flat_results = _flatten_extract_results(all_extract_results)
        downloaded_count, skipped_count, download_failure_count, download_failures = _download_successful_results(
            flat_results,
            output_root,
            args.download_workers,
            args.max_download_retries,
            logger=logger,
        )
        success_count += downloaded_count
        failure_count += download_failure_count
        failed_batches.extend(download_failures)
        if logger is not None:
            logger(
                f"[mineru-hosted-api] downloads complete: downloaded={downloaded_count} "
                f"skipped={skipped_count} failed={download_failure_count}"
            )
        total_pdfs = len(flat_results)
        save_summary_csv(upload_results, all_extract_results, output_root / "all_extract_results_summary.csv")
        summary = {
            "input_dir": str(Path(args.input_dir)) if args.input_dir else None,
            "output_dir": str(output_root),
            "total_pdfs": total_pdfs,
            "success_count": success_count,
            "skipped_count": skipped_count,
            "failure_count": failure_count,
            "failed_batches": failed_batches,
        }
        write_json(summary, output_root / "parse_summary.json")
        return summary

    if args.retry_failed:
        if logger is not None:
            logger(f"[mineru-hosted-api] retry-failed mode using {extract_results_path}")
        existing_extract_results = _load_extract_results(extract_results_path)
        pdf_paths = _select_retry_failed_pdf_paths(args.input_dir, args.pattern, existing_extract_results)
        if logger is not None:
            logger(f"[mineru-hosted-api] retry-failed selected {len(pdf_paths)} pdfs")
    else:
        all_pdf_paths = collect_pdf_paths_from_dir(args.input_dir, args.pattern)
        pdf_paths, existing_skipped_count = _select_pending_pdf_paths(all_pdf_paths, output_root, all_extract_results)
        skipped_count += existing_skipped_count
        if logger is not None:
            logger(
                f"[mineru-hosted-api] resume skip existing completed pdfs={existing_skipped_count} "
                f"pending={len(pdf_paths)}"
            )
    batches = chunked(pdf_paths, args.batch_size)
    for batch_index, pdf_batch in enumerate(batches, start=1):
        if logger is not None:
            logger(f"[mineru-hosted-api] batch {batch_index}/{len(batches)}: pdfs={len(pdf_batch)}")
        try:
            upload_payload = request_upload_batch(pdf_batch, args)
            batch_id = upload_payload["data"]["batch_id"]
            upload_urls = upload_payload["data"]["file_urls"]
            if len(upload_urls) != len(pdf_batch):
                raise RuntimeError("Hosted MinerU returned mismatched upload URLs")
            for pdf_path, upload_url in zip(pdf_batch, upload_urls):
                upload_file(upload_url, pdf_path)
            upload_record = {
                "batch_index": batch_index,
                "batch_id": batch_id,
                "uploaded_files": [str(path) for path in pdf_batch],
                "raw_response": upload_payload,
            }
            write_json(upload_record, batch_dir / f"upload_record_batch_{batch_index}.json")
            upload_results.append(
                {
                    "batch_index": batch_index,
                    "batch_id": batch_id,
                    "uploaded_files": [str(path) for path in pdf_batch],
                }
            )

            batch_result = wait_batch_result(
                batch_id=batch_id,
                token=args.token,
                api_base_url=args.api_base_url,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
            )
            extract_results = batch_result.get("data", {}).get("extract_result") or []
            all_extract_results[batch_id] = extract_results
            write_json(batch_result, batch_dir / f"{batch_id}_result.json")
            save_extract_results_csv(extract_results, batch_dir / f"{batch_id}_result.csv")

            if args.download_results:
                batch_success_count, batch_skipped_count, batch_failure_count, batch_failures = _download_successful_results(
                    extract_results,
                    output_root,
                    args.download_workers,
                    args.max_download_retries,
                    logger=logger,
                )
                success_count += batch_success_count
                skipped_count += batch_skipped_count
                failure_count += batch_failure_count
                failed_batches.extend(
                    {
                        "batch_index": batch_index,
                        "batch_id": batch_id,
                        "paper_id": failure["paper_id"],
                        "error": failure["error"],
                    }
                    for failure in batch_failures
                )
        except Exception as exc:
            failure_count += len(pdf_batch)
            failed_batches.append(
                {
                    "batch_index": batch_index,
                    "pdf_paths": [str(path) for path in pdf_batch],
                    "error": str(exc),
                }
            )

    write_json(upload_results, output_root / "all_upload_batches.json")
    write_json(all_extract_results, output_root / "all_extract_results.json")
    save_summary_csv(upload_results, all_extract_results, output_root / "all_extract_results_summary.csv")

    summary = {
        "input_dir": str(Path(args.input_dir)),
        "output_dir": str(output_root),
        "total_pdfs": len(pdf_paths) + skipped_count if not args.retry_failed else len(pdf_paths),
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failure_count": failure_count,
        "failed_batches": failed_batches,
    }
    write_json(summary, output_root / "parse_summary.json")
    return summary


def main(argv=None):
    args = parse_args(argv)
    summary = run_mineru_hosted_api_batch(args, logger=build_logger())
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
