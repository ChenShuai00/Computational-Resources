import argparse
import importlib
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Optional

import requests

from computing_resource.common.io import build_logger, maybe_sleep, write_json
from computing_resource.config import load_config


ACL_BASE_URL = "https://aclanthology.org"
ACL_URL_RE = re.compile(r"^https?://aclanthology\.org/(?P<anthology_id>[^/?#]+)/?$")
ACL_EVENT_URL_RE = re.compile(r"^https?://aclanthology\.org/events/(?P<venue>[a-z0-9-]+)-(?P<year>\d{4})/?$")


def clean_text(value: str) -> str:
    return " ".join(value.split())


def extract_event_info(url: str) -> tuple[str, str]:
    match = ACL_EVENT_URL_RE.match(url.strip())
    if not match:
        raise ValueError(
            "URL must be an ACL Anthology event page like "
            "'https://aclanthology.org/events/emnlp-2025/'"
        )
    return match.group("venue"), match.group("year")


def is_event_url(url: str) -> bool:
    return ACL_EVENT_URL_RE.match(url.strip()) is not None


def extract_anthology_id(url: str) -> str:
    match = ACL_URL_RE.match(url.strip())
    if not match:
        raise ValueError(
            "URL must be an ACL Anthology paper page like "
            "'https://aclanthology.org/2025.emnlp-main.1/'"
        )
    anthology_id = match.group("anthology_id")
    if anthology_id.endswith(".pdf"):
        raise ValueError("URL must point to the paper page, not the PDF file")
    return anthology_id


def normalize_url(url: str) -> str:
    anthology_id = extract_anthology_id(url)
    return f"{ACL_BASE_URL}/{anthology_id}/"


def normalize_event_url(url: str) -> str:
    venue, year = extract_event_info(url)
    return f"{ACL_BASE_URL}/events/{venue}-{year}/"


def main_track_volume_ids(url: str) -> list[str]:
    venue, year = extract_event_info(url)

    if venue == "acl":
        return [f"{year}.acl-long", f"{year}.acl-main"]

    if venue == "naacl":
        return [f"{year}.naacl-main"]

    if venue == "emnlp":
        return [f"{year}.emnlp-main"]

    if venue == "eacl":
        return [f"{year}.eacl-long"]

    raise ValueError(f"Unsupported venue for main track extraction: {venue}")


def extract_event_paper_urls(html: str, source_url: str) -> list[str]:
    volume_ids = [re.escape(id) for id in main_track_volume_ids(source_url)]
    volume_id_pattern = "|".join(volume_ids)
    pattern = re.compile(
        rf'href=(?:"|\')?/(?P<volume_id>(?:{volume_id_pattern}))\.(?P<number>\d+)/(?:"|\')?'
    )
    urls_by_number: dict[int, str] = {}

    for match in pattern.finditer(html):
        number = int(match.group("number"))
        if number == 0:
            continue
        anthology_id = f'{match.group("volume_id")}.{number}'
        urls_by_number[number] = f"{ACL_BASE_URL}/{anthology_id}/"

    if not urls_by_number:
        raise ValueError(f"Failed to find main-track paper URLs for {main_track_volume_ids(source_url)}")

    return [urls_by_number[number] for number in sorted(urls_by_number)]


def _load_acl_anthology():
    module = importlib.import_module("acl_anthology")
    anthology_cls = getattr(module, "Anthology")
    return anthology_cls.from_repo()


def _paper_url_from_object(paper) -> Optional[str]:
    web_url = getattr(paper, "web_url", None)
    if web_url:
        return normalize_url(web_url)

    anthology_id = getattr(paper, "full_id", None) or getattr(paper, "id", None)
    if anthology_id:
        return f"{ACL_BASE_URL}/{anthology_id}/"

    return None


def _event_volume_full_id(volume) -> Optional[str]:
    full_id = getattr(volume, "full_id", None)
    if full_id:
        return full_id

    collection_id = getattr(volume, "collection_id", None)
    volume_id = getattr(volume, "id", None)
    if collection_id and volume_id:
        return f"{collection_id}-{volume_id}"

    if volume_id:
        return volume_id

    return None


def _extract_event_paper_urls_from_anthology(anthology, source_url: str) -> list[str]:
    event_id = f"{extract_event_info(source_url)[0]}-{extract_event_info(source_url)[1]}"
    event = anthology.get_event(event_id)
    if event is None:
        raise ValueError(f"Failed to find event {event_id} in acl-anthology")

    main_track_ids = set(main_track_volume_ids(source_url))
    paper_urls: list[str] = []

    for volume in event.volumes():
        volume_full_id = _event_volume_full_id(volume)
        if volume_full_id not in main_track_ids:
            continue
        for paper in volume.papers():
            paper_url = _paper_url_from_object(paper)
            if paper_url is None:
                continue
            anthology_id = extract_anthology_id(paper_url)
            if anthology_id.endswith(".0"):
                continue
            paper_urls.append(normalize_url(paper_url))

    if not paper_urls:
        raise ValueError(f"Failed to find main-track paper URLs for {main_track_volume_ids(source_url)}")

    return paper_urls


def discover_event_paper_urls(
    source_url: str,
    event_html: Optional[str] = None,
    fetch_html: Optional[Callable[[], str]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> tuple[list[str], str, Optional[str]]:
    discovery_warning: Optional[str] = None
    try:
        anthology = _load_acl_anthology()
        paper_urls = _extract_event_paper_urls_from_anthology(anthology, source_url)
        if logger is not None:
            logger(f"[INFO] Discovered {len(paper_urls)} main-track paper URLs via acl-anthology")
        return paper_urls, "acl_anthology", None
    except Exception as exc:
        discovery_warning = str(exc)
        if logger is not None:
            logger(f"[WARN] acl-anthology discovery failed for {source_url}, falling back to HTML parsing: {exc}")

    if event_html is None:
        if fetch_html is None:
            raise ValueError("HTML fallback requires either event_html or fetch_html")
        event_html = fetch_html()

    paper_urls = extract_event_paper_urls(event_html, source_url)
    if logger is not None:
        logger(f"[INFO] Discovered {len(paper_urls)} main-track paper URLs")
    return paper_urls, "html_fallback", discovery_warning


def default_output_path(url: str) -> Path:
    anthology_id = extract_anthology_id(url)
    return load_config()["paths"]["acl_metadata_root"] / f"{anthology_id}.json"


def default_event_output_dir(url: str) -> Path:
    return load_config()["paths"]["acl_metadata_root"] / main_track_volume_ids(url)[0]


class ACLPageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tag_stack: list[str] = []
        self.current_h2_depth: Optional[int] = None
        self.current_title_parts: list[str] = []
        self.current_title_href: Optional[str] = None
        self.after_title = False
        self.in_author_paragraph = False
        self.in_author_link = False
        self.current_author_parts: list[str] = []
        self.authors: list[str] = []
        self.capture_h5 = False
        self.current_h5_parts: list[str] = []
        self.pending_abstract = False
        self.abstract_capture_depth: Optional[int] = None
        self.abstract_parts: list[str] = []
        self.in_dl = False
        self.capture_dt = False
        self.current_dt_parts: list[str] = []
        self.current_dt: Optional[str] = None
        self.capture_dd = False
        self.current_dd_parts: list[str] = []
        self.current_dd_hrefs: list[str] = []
        self.metadata: dict[str, dict[str, list[str] | str]] = {}
        self.capture_bibtex = False
        self.bibtex_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.tag_stack.append(tag)

        if tag == "h2":
            self.current_h2_depth = len(self.tag_stack)
            self.current_title_parts = []
            self.current_title_href = None
        elif tag == "a" and self.current_h2_depth is not None and self.tag_stack[-2] == "h2":
            self.current_title_href = attrs_dict.get("href")

        if self.after_title and tag == "p":
            self.in_author_paragraph = True
            self.after_title = False
        elif self.in_author_paragraph and tag == "a":
            self.in_author_link = True
            self.current_author_parts = []

        if tag == "h5":
            self.capture_h5 = True
            self.current_h5_parts = []
        elif self.pending_abstract and tag not in {"script", "style"}:
            self.pending_abstract = False
            self.abstract_capture_depth = len(self.tag_stack)
            self.abstract_parts = []

        if tag == "dl":
            self.in_dl = True
        elif self.in_dl and tag == "dt":
            self.capture_dt = True
            self.current_dt_parts = []
        elif self.in_dl and tag == "dd":
            self.capture_dd = True
            self.current_dd_parts = []
            self.current_dd_hrefs = []
        elif self.capture_dd and tag == "a":
            href = attrs_dict.get("href")
            if href:
                self.current_dd_hrefs.append(href)

        if tag == "pre" and attrs_dict.get("id") == "citeBibtexContent":
            self.capture_bibtex = True
            self.bibtex_parts = []

    def handle_endtag(self, tag):
        if tag == "h2" and self.current_h2_depth is not None and len(self.tag_stack) == self.current_h2_depth:
            self.after_title = True
            self.current_h2_depth = None
        elif self.in_author_paragraph and tag == "p":
            self.in_author_paragraph = False
        elif self.in_author_link and tag == "a":
            author = clean_text("".join(self.current_author_parts))
            if author:
                self.authors.append(author)
            self.in_author_link = False
            self.current_author_parts = []

        if self.capture_h5 and tag == "h5":
            heading = clean_text("".join(self.current_h5_parts))
            self.capture_h5 = False
            self.current_h5_parts = []
            if heading == "Abstract":
                self.pending_abstract = True

        if self.abstract_capture_depth is not None and len(self.tag_stack) == self.abstract_capture_depth and tag == self.tag_stack[-1]:
            self.abstract_capture_depth = None

        if self.capture_dt and tag == "dt":
            self.capture_dt = False
            self.current_dt = clean_text("".join(self.current_dt_parts)).rstrip(":")
            self.current_dt_parts = []
        elif self.capture_dd and tag == "dd":
            text = clean_text("".join(self.current_dd_parts))
            if self.current_dt:
                self.metadata[self.current_dt] = {
                    "text": text,
                    "hrefs": list(self.current_dd_hrefs),
                }
            self.capture_dd = False
            self.current_dd_parts = []
            self.current_dd_hrefs = []
            self.current_dt = None
        elif self.in_dl and tag == "dl":
            self.in_dl = False

        if self.capture_bibtex and tag == "pre":
            self.capture_bibtex = False

        if self.tag_stack:
            self.tag_stack.pop()

    def handle_data(self, data):
        if self.current_h2_depth is not None:
            self.current_title_parts.append(data)
        if self.in_author_link:
            self.current_author_parts.append(data)
        if self.capture_h5:
            self.current_h5_parts.append(data)
        if self.abstract_capture_depth is not None:
            self.abstract_parts.append(data)
        if self.capture_dt:
            self.current_dt_parts.append(data)
        if self.capture_dd:
            self.current_dd_parts.append(data)
        if self.capture_bibtex:
            self.bibtex_parts.append(data)


def parse_acl_html(html: str, source_url: str) -> dict:
    parser = ACLPageParser()
    parser.feed(html)
    parser.close()

    title = clean_text("".join(parser.current_title_parts))
    metadata = parser.metadata
    anthology_id = metadata.get("Anthology ID", {}).get("text") or extract_anthology_id(source_url)
    bibtex = "".join(parser.bibtex_parts).strip()

    if not title:
        raise ValueError("Failed to extract paper title from ACL Anthology page")
    if not anthology_id:
        raise ValueError("Failed to extract anthology ID from ACL Anthology page")

    def meta_text(key: str) -> Optional[str]:
        value = metadata.get(key, {}).get("text")
        return value if value else None

    def meta_href(key: str) -> Optional[str]:
        hrefs = metadata.get(key, {}).get("hrefs", [])
        return hrefs[0] if hrefs else None

    return {
        "anthology_id": anthology_id,
        "title": title,
        "authors": parser.authors,
        "abstract": clean_text("".join(parser.abstract_parts)),
        "pdf_url": meta_href("PDF") or parser.current_title_href,
        "checklist_url": meta_href("Checklist"),
        "venue": meta_text("Venue"),
        "booktitle": meta_text("Volume"),
        "month": meta_text("Month"),
        "year": meta_text("Year"),
        "address": meta_text("Address"),
        "pages": meta_text("Pages"),
        "publisher": meta_text("Publisher"),
        "doi": meta_text("DOI"),
        "bibkey": meta_text("Bibkey"),
        "bibtex": bibtex,
        "source_url": normalize_url(source_url),
    }


def fetch_page_html(url: str, timeout: float = 30.0) -> str:
    normalized_url = normalize_event_url(url) if is_event_url(url) else normalize_url(url)
    response = requests.get(
        normalized_url,
        timeout=timeout,
        headers={"User-Agent": "computing_resource/acl-page-fetcher"},
    )
    response.raise_for_status()
    return response.text


def build_event_index(source_url: str, volume_id: str | list[str], paper_urls: list[str]) -> dict:
    volume_ids = [volume_id] if isinstance(volume_id, str) else list(volume_id)

    index = {
        "source_url": normalize_event_url(source_url),
        "volume_id": volume_ids[0],
        "paper_count": len(paper_urls),
        "paper_urls": paper_urls,
    }
    if len(volume_ids) > 1:
        index["volume_ids"] = volume_ids
    return index


def output_path_for_paper_url(url: str, output_dir: Path) -> Path:
    return output_dir / f"{extract_anthology_id(url)}.json"


def fetch_paper_metadata(
    url: str,
    timeout: float = 30.0,
    sleep_seconds: float = 1.0,
    sleep_fn=time.sleep,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    if logger is not None:
        logger(f"[INFO] Fetching paper page: {url}")
        if sleep_seconds > 0:
            logger(f"[INFO] Sleeping {sleep_seconds:.1f}s before request")
    maybe_sleep(sleep_seconds, sleep_fn)
    html = fetch_page_html(url, timeout=timeout)
    data = parse_acl_html(html, url)
    if logger is not None:
        logger(f'[INFO] Parsed paper metadata: {data["anthology_id"]}')
    return data


def run_single_paper(
    url: str,
    output_path: Path,
    timeout: float = 30.0,
    sleep_seconds: float = 1.0,
    sleep_fn=time.sleep,
    logger: Optional[Callable[[str], None]] = None,
) -> Path:
    if output_path.exists():
        if logger is not None:
            logger(f"[INFO] Skipping existing file: {output_path}")
        return output_path

    data = fetch_paper_metadata(
        url,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
        sleep_fn=sleep_fn,
        logger=logger,
    )
    saved_path = write_json(data, output_path)
    if logger is not None:
        logger(f"[INFO] Saved metadata to {saved_path}")
    return saved_path


def fetch_event_papers(
    url: str,
    output_dir: Path,
    timeout: float = 30.0,
    sleep_seconds: float = 1.0,
    sleep_fn=time.sleep,
    logger: Optional[Callable[[str], None]] = None,
) -> list[Path]:
    if logger is not None:
        logger(f"[INFO] Fetching event page: {url}")
        if sleep_seconds > 0:
            logger(f"[INFO] Sleeping {sleep_seconds:.1f}s before request")
    maybe_sleep(sleep_seconds, sleep_fn)
    event_html: Optional[str] = None

    def load_event_html() -> str:
        nonlocal event_html
        if event_html is None:
            event_html = fetch_page_html(url, timeout=timeout)
        return event_html

    paper_urls, discovery_backend, discovery_warning = discover_event_paper_urls(
        url,
        fetch_html=load_event_html,
        logger=logger,
    )
    written_paths: list[Path] = []

    for index, paper_url in enumerate(paper_urls, start=1):
        paper_path = output_path_for_paper_url(paper_url, output_dir)
        if paper_path.exists():
            written_paths.append(paper_path)
            if logger is not None:
                logger(f"[INFO] Skipping existing file: {paper_path}")
            continue
        if logger is not None:
            logger(f"[INFO] Fetching paper {index}/{len(paper_urls)}: {paper_url}")
        data = fetch_paper_metadata(
            paper_url,
            timeout=timeout,
            sleep_seconds=sleep_seconds,
            sleep_fn=sleep_fn,
            logger=logger,
        )
        written_paths.append(write_json(data, paper_path))
        if logger is not None:
            logger(f"[INFO] Saved paper metadata: {paper_path}")

    index_path = output_dir / "index.json"
    index = build_event_index(url, main_track_volume_ids(url), paper_urls)
    index["discovery_backend"] = discovery_backend
    if discovery_warning:
        index["discovery_warning"] = discovery_warning
    written_paths.append(write_json(index, index_path))
    if logger is not None:
        logger(f"[INFO] Saved event index: {index_path}")
    return written_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch an ACL Anthology paper page and save structured JSON")
    parser.add_argument("--url", required=True, help="ACL Anthology paper page URL")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path; defaults to data/external/metadata/acl/<anthology_id>.json",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Seconds to wait before each network request; pass 0 to disable")
    parser.add_argument("--log-file", default=None, help="Optional log file path; defaults to console output only")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = build_logger(Path(args.log_file) if args.log_file else None)

    logger(
        f"[INFO] Starting fetch: url={args.url} timeout={args.timeout} sleep_seconds={args.sleep_seconds}"
    )

    try:
        if is_event_url(args.url):
            output_dir = Path(args.output) if args.output else default_event_output_dir(args.url)
            logger(f"[INFO] Mode: event output_dir={output_dir}")
            written_paths = fetch_event_papers(
                args.url,
                output_dir,
                timeout=args.timeout,
                sleep_seconds=args.sleep_seconds,
                logger=logger,
            )
            logger(f"[INFO] Saved {len(written_paths) - 1} paper metadata files and index to {output_dir}")
            return

        output_path = Path(args.output) if args.output else default_output_path(args.url)
        logger(f"[INFO] Mode: paper output_path={output_path}")
        run_single_paper(
            args.url,
            output_path,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
            logger=logger,
        )
    except Exception as exc:
        logger(f"[ERROR] Failed to fetch {args.url}: {exc}")
        raise


if __name__ == "__main__":
    main()
