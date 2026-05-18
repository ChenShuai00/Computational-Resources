from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from xml.sax.saxutils import escape
import zipfile

import requests


ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from computing_resource.classification.emnlp_topic import (
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    classify_title_abstract,
    load_openalex_title_abstract,
    load_openalex_title,
    load_title_abstract_with_mineru_fallback,
    resolve_api_key,
)
from computing_resource.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config


def render_progress(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[{}] 0/0".format("-" * width)
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def print_progress(current: int, total: int) -> None:
    end = "\n" if total > 0 and current >= total else "\r"
    print(render_progress(current=current, total=total), file=sys.stderr, end=end, flush=True)


def derive_xlsx_output_path(output_jsonl: Path) -> Path:
    return output_jsonl.with_suffix(".xlsx")


def read_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _column_letter(index: int) -> str:
    result = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xlsx_inline_cell(cell_ref: str, value: object) -> str:
    text = "" if value is None else str(value)
    return f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'


def write_topic_xlsx(records: list[dict[str, object]], output_path: Path) -> None:
    columns = ["paper_id", "title", "topic", "confidence", "needs_review", "reason"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_xml: list[str] = []
    all_rows = [{column: column for column in columns}] + [
        {column: record.get(column, "") for column in columns} for record in records
    ]
    for row_index, row in enumerate(all_rows, start=1):
        cells = []
        for column_index, column in enumerate(columns, start=1):
            cell_ref = f"{_column_letter(column_index)}{row_index}"
            cells.append(_xlsx_inline_cell(cell_ref, row.get(column, "")))
        rows_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(rows_xml)}</sheetData>'
        "</worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="topics" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        "</Relationships>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/styles.xml", styles_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def load_existing_paper_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    existing_ids: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        paper_id = str(record.get("paper_id", "")).strip()
        if paper_id:
            existing_ids.add(paper_id)
    return existing_ids


def build_record(paper_id: str, title: str, result: dict[str, object]) -> dict[str, object]:
    record = {"paper_id": paper_id, "title": title}
    record.update(result)
    return record


def build_empty_record(paper_id: str, title: str) -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "title": title,
        "topic": "",
        "confidence": "",
        "needs_review": "",
        "reason": "",
    }


def classify_openalex_file(
    file_path: Path,
    *,
    api_key: str,
    model: str,
    api_base: str,
    timeout: float,
    mineru_root: Path | None,
) -> dict[str, object]:
    paper_id = file_path.stem
    title, abstract = load_title_abstract_with_mineru_fallback(file_path, mineru_root=mineru_root)
    result = classify_title_abstract(
        title=title,
        abstract=abstract,
        api_key=api_key,
        model=model,
        api_base=api_base,
        timeout=timeout,
    )
    return build_record(paper_id=paper_id, title=title, result=result)


def run_batch(
    *,
    input_dir: Path,
    output_jsonl: Path,
    api_key: str,
    model: str,
    api_base: str,
    timeout: float,
    limit: int,
    skip_existing: bool,
    workers: int,
    mineru_root: Path | None,
    recursive: bool,
) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = load_existing_paper_ids(output_jsonl) if skip_existing else set()
    iterator = input_dir.rglob("*.json") if recursive else input_dir.glob("*.json")
    files = sorted(file_path for file_path in iterator if file_path.name.lower() != "index.json")
    if limit > 0:
        files = files[:limit]
    pending_files = [file_path for file_path in files if file_path.stem not in existing_ids]
    total = len(pending_files)

    with output_jsonl.open("a", encoding="utf-8") as handle:
        if total == 0:
            print_progress(current=0, total=0)
            return 0

        max_workers = max(1, int(workers))
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    classify_openalex_file,
                    file_path,
                    api_key=api_key,
                    model=model,
                    api_base=api_base,
                    timeout=timeout,
                    mineru_root=mineru_root,
                ): file_path
                for file_path in pending_files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                completed += 1
                try:
                    record = future.result()
                except ValueError as exc:
                    print(f"warning: {exc}", file=sys.stderr)
                    record = build_empty_record(
                        paper_id=file_path.stem,
                        title=load_openalex_title(file_path),
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print_progress(current=completed, total=total)
                    continue
                except requests.exceptions.RequestException as exc:
                    print(f"warning: request failed for {file_path.stem}: {exc}", file=sys.stderr)
                    print_progress(current=completed, total=total)
                    continue

                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print_progress(current=completed, total=total)
    write_topic_xlsx(read_jsonl_records(output_jsonl), derive_xlsx_output_path(output_jsonl))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify EMNLP 2025 topics from titles and abstracts")
    parser.add_argument("--title", default="", help="Paper title")
    parser.add_argument("--abstract", default="", help="Paper abstract")
    parser.add_argument("--paper-id", default="", help="Paper ID matching the OpenAlex filename, such as 2025.emnlp-main.1")
    parser.add_argument("--input-dir", default="", help="Directory of per-paper OpenAlex JSON files in batch mode")
    parser.add_argument("--output-jsonl", default="", help="Output JSONL path in batch mode")
    parser.add_argument("--limit", type=int, default=0, help="Maximum papers to process in batch mode; 0 means unlimited")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent request count in batch mode")
    parser.add_argument("--skip-existing", action="store_true", help="Skip paper_id values already present in the output JSONL in batch mode")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan JSON files under --input-dir in batch mode")
    parser.add_argument(
        "--mineru-root",
        default=str(PROJECT_ROOT / "data" / "mineru_results"),
        help="MinerU result root used when OpenAlex title/abstract is missing",
    )
    parser.add_argument(
        "--openalex-dir",
        default=str(PROJECT_ROOT / "data" / "external" / "metadata" / "openalex" / "2025.emnlp-main"),
        help="Directory of per-paper OpenAlex JSON files",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--api-key", default="", help="Optional direct API key")
    parser.add_argument("--model", default="", help="Optional model override")
    parser.add_argument("--api-base", default="", help="Optional API endpoint override")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    llm_cfg = config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {}
    model = args.model or str(llm_cfg.get("model", DEFAULT_MODEL))
    api_base = args.api_base or str(llm_cfg.get("api_base", "")).rstrip("/")
    if api_base.endswith("/chat/completions"):
        request_url = api_base
    elif api_base.endswith("/v1"):
        request_url = f"{api_base}/chat/completions"
    elif api_base:
        request_url = f"{api_base}/v1/chat/completions"
    else:
        request_url = DEFAULT_API_BASE
    api_key = resolve_api_key(config=config, cli_api_key=args.api_key)

    if args.input_dir or args.output_jsonl:
        if not args.input_dir or not args.output_jsonl:
            raise SystemExit("Batch mode requires both --input-dir and --output-jsonl.")
        return run_batch(
            input_dir=Path(args.input_dir),
            output_jsonl=Path(args.output_jsonl),
            api_key=api_key,
            model=model,
            api_base=request_url,
            timeout=args.timeout,
            limit=args.limit,
            skip_existing=args.skip_existing,
            workers=args.workers,
            mineru_root=Path(args.mineru_root) if args.mineru_root else None,
            recursive=args.recursive,
        )

    if args.paper_id:
        paper_path = Path(args.openalex_dir) / f"{args.paper_id}.json"
        title, abstract = load_openalex_title_abstract(paper_path)
    else:
        title = args.title.strip()
        abstract = args.abstract.strip()
        if not title or not abstract:
            raise SystemExit("Either provide --paper-id or both --title and --abstract.")

    result = classify_title_abstract(
        title=title,
        abstract=abstract,
        api_key=api_key,
        model=model,
        api_base=request_url,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
