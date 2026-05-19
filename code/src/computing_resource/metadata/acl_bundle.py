import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import requests

from computing_resource.common.io import build_logger, write_json
from computing_resource.config import load_config
from computing_resource.metadata import acl


EVENT_PROGRESS_LOG_INTERVAL = 25


def conference_dir_name_from_event_url(url: str) -> str:
    venue, year = acl.extract_event_info(url)
    return f"{venue}{year}"


def conference_dir_name_from_paper_url(url: str) -> str:
    anthology_id = acl.extract_anthology_id(url)
    year, venue_track, *_ = anthology_id.split(".")
    venue = venue_track.split("-")[0]
    return f"{venue}{year}"


def default_pdf_output_path(url: str) -> Path:
    anthology_id = acl.extract_anthology_id(url)
    return load_config()["paths"]["papers_root"] / conference_dir_name_from_paper_url(url) / f"{anthology_id}.pdf"


def default_event_pdf_output_dir(url: str) -> Path:
    return load_config()["paths"]["papers_root"] / conference_dir_name_from_event_url(url)


def output_pdf_path_for_paper_url(url: str, output_dir: Path) -> Path:
    return output_dir / f"{acl.extract_anthology_id(url)}.pdf"


def download_pdf_file(
    url: str,
    output_path: Path,
    timeout: float = 30.0,
    logger: Optional[Callable[[str], None]] = None,
    max_attempts: int = 3,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    last_error = None

    for attempt in range(1, max_attempts + 1):
        response = None
        try:
            response = requests.get(
                url,
                timeout=timeout,
                stream=True,
                headers={"User-Agent": "computing_resource/acl-bundle"},
            )
            response.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
            tmp_path.replace(output_path)
            if logger is not None:
                logger(f"[INFO] Saved PDF to {output_path}")
            return output_path
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if logger is not None and attempt < max_attempts:
                logger(f"[WARN] PDF download interrupted, retrying {attempt}/{max_attempts}: {url}")
        finally:
            if response is not None:
                response.close()

    assert last_error is not None
    raise last_error


def run_single_paper_bundle(
    url: str,
    metadata_output_path: Path,
    pdf_output_path: Path,
    timeout: float = 30.0,
    sleep_seconds: float = 1.0,
    download_pdf: bool = False,
    sleep_fn=None,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    pdf_part_path = pdf_output_path.with_suffix(pdf_output_path.suffix + ".part")

    if metadata_output_path.exists() and (not download_pdf or pdf_output_path.exists()):
        if logger is not None:
            logger(f"[INFO] Skipping existing file: {metadata_output_path}")
            if download_pdf:
                logger(f"[INFO] Skipping existing file: {pdf_output_path}")
        return {
            "metadata_path": metadata_output_path,
            "pdf_path": pdf_output_path,
            "data": None,
        }

    if download_pdf and pdf_part_path.exists():
        pdf_part_path.unlink()
        if logger is not None:
            logger(f"[INFO] Removing stale partial PDF: {pdf_part_path}")

    data = acl.fetch_paper_metadata(
        url,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
        sleep_fn=sleep_fn,
        logger=logger,
    )
    if metadata_output_path.exists():
        if logger is not None:
            logger(f"[INFO] Skipping existing file: {metadata_output_path}")
    else:
        write_json(data, metadata_output_path)
        if logger is not None:
            logger(f"[INFO] Saved metadata to {metadata_output_path}")

    if download_pdf:
        if pdf_output_path.exists():
            if logger is not None:
                logger(f"[INFO] Skipping existing file: {pdf_output_path}")
        else:
            download_pdf_file(data["pdf_url"], pdf_output_path, timeout=timeout, logger=logger)

    return {
        "metadata_path": metadata_output_path,
        "pdf_path": pdf_output_path,
        "data": data,
    }


def run_event_bundle(
    url: str,
    metadata_output_dir: Path,
    pdf_output_dir: Path,
    timeout: float = 30.0,
    sleep_seconds: float = 1.0,
    download_pdf: bool = False,
    workers: int = 4,
    sleep_fn=None,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    if logger is not None:
        logger(f"[INFO] Fetching event page: {url}")
    paper_urls, discovery_backend, discovery_warning = acl.discover_event_paper_urls(
        url,
        fetch_html=lambda: acl.fetch_page_html(url, timeout=timeout),
        logger=logger,
    )
    total_papers = len(paper_urls)
    if logger is not None:
        logger(f"[INFO] Event paper count: total={total_papers}")
    papers = []
    failures = []
    completed = 0
    max_workers = max(1, workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_paper_bundle,
                paper_url,
                metadata_output_path=acl.output_path_for_paper_url(paper_url, metadata_output_dir),
                pdf_output_path=output_pdf_path_for_paper_url(paper_url, pdf_output_dir),
                timeout=timeout,
                sleep_seconds=sleep_seconds,
                download_pdf=download_pdf,
                sleep_fn=sleep_fn,
                logger=logger,
            ): paper_url
            for paper_url in paper_urls
        }

        for future in as_completed(futures):
            paper_url = futures[future]
            try:
                papers.append(future.result())
            except requests.RequestException as exc:
                failures.append({"url": paper_url, "error": str(exc)})
                if logger is not None:
                    logger(f"[ERROR] Failed paper: {paper_url} ({exc})")
            completed += 1
            if logger is not None and completed % EVENT_PROGRESS_LOG_INTERVAL == 0:
                logger(
                    "[INFO] Event progress: "
                    f"completed={completed} succeeded={len(papers)} failed={len(failures)} total={total_papers}"
                )

    if logger is not None and completed and completed % EVENT_PROGRESS_LOG_INTERVAL != 0:
        logger(
            "[INFO] Event progress: "
            f"completed={completed} succeeded={len(papers)} failed={len(failures)} total={total_papers}"
        )

    index = acl.build_event_index(url, acl.main_track_volume_ids(url), paper_urls)
    index["discovery_backend"] = discovery_backend
    if discovery_warning:
        index["discovery_warning"] = discovery_warning
    index_path = metadata_output_dir / "index.json"
    write_json(index, index_path)
    if logger is not None:
        logger(f"[INFO] Saved event index: {index_path}")
        logger(
            f"[INFO] Event summary: total={len(paper_urls)} succeeded={len(papers)} failed={len(failures)}"
        )
    return {"papers": papers, "failures": failures, "index_path": index_path}


def iter_legacy_paper_urls(
    conference_name: str,
    conference_year: str,
    paper_type: str,
    total_num: int,
):
    for index in range(1, total_num):
        yield f"https://aclanthology.org/{conference_year}.{conference_name}-{paper_type}.{index}/"


def build_parser() -> argparse.ArgumentParser:
    config = load_config()
    parser = argparse.ArgumentParser(description="Unified ACL metadata fetcher and PDF downloader")
    parser.add_argument("--url", default=None, help="ACL paper page or event page URL")
    parser.add_argument("--metadata-output", default=None, help="Output JSON path in single-paper mode")
    parser.add_argument("--pdf-output", default=None, help="Output PDF path in single-paper mode")
    parser.add_argument("--metadata-output-dir", default=None, help="Metadata output directory in event or legacy mode")
    parser.add_argument("--pdf-output-dir", default=None, help="PDF output directory in event or legacy mode")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--download-pdf", action="store_true", help="Download PDFs explicitly; by default only metadata is fetched")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent fetch workers in event mode")
    parser.add_argument("--log-file", default=None)

    parser.add_argument("--conference-name", default=config["conference"]["name"])
    parser.add_argument("--conference-year", default=str(config["conference"]["year"]))
    parser.add_argument("--paper-type", default=config["conference"]["paper_type"])
    parser.add_argument("--total-num", type=int, default=int(config["conference"]["total_num"]))
    parser.add_argument("--paper-folder", default=None)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = build_logger(Path(args.log_file) if args.log_file else None)

    if args.url:
        if acl.is_event_url(args.url):
            metadata_output_dir = (
                Path(args.metadata_output_dir) if args.metadata_output_dir else acl.default_event_output_dir(args.url)
            )
            pdf_output_dir = (
                Path(args.pdf_output_dir) if args.pdf_output_dir else default_event_pdf_output_dir(args.url)
            )
            run_event_bundle(
                args.url,
                metadata_output_dir=metadata_output_dir,
                pdf_output_dir=pdf_output_dir,
                timeout=args.timeout,
                sleep_seconds=args.sleep_seconds,
                download_pdf=args.download_pdf,
                workers=args.workers,
                logger=logger,
            )
            return

        metadata_output_path = Path(args.metadata_output) if args.metadata_output else acl.default_output_path(args.url)
        pdf_output_path = Path(args.pdf_output) if args.pdf_output else default_pdf_output_path(args.url)
        run_single_paper_bundle(
            args.url,
            metadata_output_path=metadata_output_path,
            pdf_output_path=pdf_output_path,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
            download_pdf=args.download_pdf,
            logger=logger,
        )
        return

    volume_id = f"{args.conference_year}.{args.conference_name}-{args.paper_type}"
    metadata_output_dir = (
        Path(args.metadata_output_dir)
        if args.metadata_output_dir
        else load_config()["paths"]["acl_metadata_root"] / volume_id
    )
    pdf_output_dir = (
        Path(args.pdf_output_dir) if args.pdf_output_dir else Path(args.paper_folder or default_event_pdf_output_dir(f"https://aclanthology.org/events/{args.conference_name}-{args.conference_year}/"))
    )

    for paper_url in iter_legacy_paper_urls(
        conference_name=args.conference_name,
        conference_year=args.conference_year,
        paper_type=args.paper_type,
        total_num=args.total_num,
    ):
        try:
            run_single_paper_bundle(
                paper_url,
                metadata_output_path=acl.output_path_for_paper_url(paper_url, metadata_output_dir),
                pdf_output_path=output_pdf_path_for_paper_url(paper_url, pdf_output_dir),
                timeout=args.timeout,
                sleep_seconds=args.sleep_seconds,
                download_pdf=args.download_pdf,
                logger=logger,
            )
        except requests.HTTPError as exc:
            logger(f"[ERROR] Failed to fetch {paper_url}: {exc}")


if __name__ == "__main__":
    main()
