import argparse
import asyncio
import base64
import csv
import inspect
import json
import mimetypes
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

from computing_resource.config import load_config


INSTITUTION_KEYWORDS = (
    "university",
    "institute",
    "research",
    "lab",
    "center",
    "centre",
    "school",
    "college",
    "academy",
    "department",
)

COMPANY_KEYWORDS = (
    "microsoft",
    "google",
    "meta",
    "amazon",
    "apple",
    "nvidia",
    "openai",
    "huawei",
    "alibaba",
    "bytedance",
    "corp",
    "corporation",
    "inc",
    "llc",
)

LOWERCASE_TOKENS = {
    "and",
    "of",
    "for",
    "at",
    "the",
    "in",
    "on",
}

AFFILIATION_SPLIT_MARKER = "<<<AFF_SPLIT>>>"
EXCEL_ILLEGAL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_GPT_AFFILIATION_MODEL = "openai/gpt-4o-mini-2024-07-18"
AFFILIATIONS_ROOT = load_config()["paths"]["affiliations_root"]
CHECKPOINT_SYNC_EVERY = 25


def split_before_abstract(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"(?im)^\s{0,3}(?:#{1,6}\s*)?abstract\s*$", text)
    if not match:
        return text.strip()
    return text[: match.start()].strip()


def insert_missing_spaces(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("Mell on", "Mellon")
    text = text.replace("Harb in", "Harbin")
    text = text.replace("Jil in", "Jilin")
    text = text.replace("Dubl in", "Dublin")
    text = text.replace("Informati on", "Information")
    text = text.replace("Transformati on", "Transformation")
    text = text.replace("Innovati on", "Innovation")
    text = text.replace("Communicati on", "Communication")
    text = text.replace("Amaz on", "Amazon")
    text = text.replace("Ben Guri on", "Ben Gurion")
    text = text.replace("Megag on", "Megagon")
    text = text.replace("Deep Auto.ai", "DeepAuto.ai")
    text = re.sub(r"(?<=[a-zà-öø-ÿ])(?=[A-ZÀ-ÖØ-Þ])", " ", text)
    text = re.sub(r"(?<=[A-ZÀ-ÖØ-Þ])(?=[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ])", " ", text)
    text = re.sub(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-zÀ-ÖØ-öø-ÿ])", " ", text)
    for token in sorted(LOWERCASE_TOKENS, key=len, reverse=True):
        text = re.sub(
            rf"(?<=[A-Za-zÀ-ÖØ-öø-ÿ]){token}(?=[A-ZÀ-ÖØ-Þ])",
            f" {token} ",
            text,
        )
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def normalize_case_tokens(text: str) -> str:
    parts = text.split()
    normalized: list[str] = []
    for index, part in enumerate(parts):
        if index > 0 and part.lower() in LOWERCASE_TOKENS:
            normalized.append(part.lower())
        else:
            normalized.append(part)
    return " ".join(normalized)


def clean_affiliation_segment(segment: str) -> str:
    segment = segment.strip(" ,;:")
    segment = re.sub(r"https?://\S+|www\.\S+", "", segment, flags=re.IGNORECASE)
    segment = re.sub(r"\S+@\S+", "", segment)
    segment = re.sub(r"\{[^}]*$", "", segment)
    segment = re.sub(r"<sup>\s*.*?\s*</sup>", f" {AFFILIATION_SPLIT_MARKER} ", segment, flags=re.IGNORECASE)
    segment = re.sub(r"\s*[†‡♣♢§¶♥♠♣♦]+\s*", f" {AFFILIATION_SPLIT_MARKER} ", segment)
    segment = segment.replace("|", f" {AFFILIATION_SPLIT_MARKER} ")
    segment = insert_missing_spaces(segment)
    segment = normalize_case_tokens(segment)
    return segment.strip(" ,;:")


def dedupe_affiliations(affiliations: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for affiliation in affiliations:
        cleaned = clean_affiliation_segment(affiliation)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def merge_affiliation_records(
    base_records: list[dict[str, object]],
    override_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    override_by_paper_id = {
        str(record.get("paper_id") or ""): record
        for record in override_records
        if str(record.get("paper_id") or "")
    }
    merged_records: list[dict[str, object]] = []
    for base_record in base_records:
        paper_id = str(base_record.get("paper_id") or "")
        merged_record = dict(override_by_paper_id.get(paper_id, base_record))
        merged_record["affiliations"] = dedupe_affiliations(list(merged_record.get("affiliations") or []))
        merged_records.append(merged_record)
    seen_paper_ids = {
        str(record.get("paper_id") or "")
        for record in merged_records
        if str(record.get("paper_id") or "")
    }
    for override_record in override_records:
        paper_id = str(override_record.get("paper_id") or "")
        if not paper_id or paper_id in seen_paper_ids:
            continue
        appended_record = dict(override_record)
        appended_record["affiliations"] = dedupe_affiliations(list(appended_record.get("affiliations") or []))
        merged_records.append(appended_record)
    return merged_records


def resolve_local_mineru_markdown_path(paper_dir: str | Path) -> Path | None:
    root = Path(paper_dir)
    candidate_paths = [
        root / "full.md",
        root / "auto" / "full.md",
        root / "auto" / f"{root.name}.md",
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    auto_markdown_paths = sorted(path for path in (root / "auto").glob("*.md") if path.is_file())
    if auto_markdown_paths:
        return auto_markdown_paths[0]
    return None


def load_local_mineru_full_md_front_matter(paper_dir: str | Path) -> str:
    markdown_path = resolve_local_mineru_markdown_path(paper_dir)
    if markdown_path is not None:
        return split_before_abstract(markdown_path.read_text(encoding="utf-8"))
    return ""


def normalize_openai_affiliation_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def normalize_openai_affiliation_details(raw_affiliations: object) -> list[dict[str, object]]:
    if not isinstance(raw_affiliations, list):
        return []
    normalized_details: list[dict[str, object]] = []
    seen_institutions: set[str] = set()
    for item in raw_affiliations:
        raw_affiliation_text = ""
        institution = ""
        country: str | None = None
        if isinstance(item, str):
            raw_affiliation_text = normalize_openai_affiliation_text(item)
            institution = clean_affiliation_segment(item)
        elif isinstance(item, dict):
            raw_affiliation_text = normalize_openai_affiliation_text(item.get("raw_affiliation_text"))
            institution = clean_affiliation_segment(str(item.get("institution") or ""))
            if not institution and raw_affiliation_text:
                institution = clean_affiliation_segment(raw_affiliation_text)
            raw_country = item.get("country")
            if raw_country is not None:
                country = normalize_openai_affiliation_text(raw_country) or None
        else:
            continue
        if not institution:
            continue
        if not raw_affiliation_text:
            raw_affiliation_text = institution
        if institution in seen_institutions:
            continue
        seen_institutions.add(institution)
        normalized_details.append(
            {
                "raw_affiliation_text": raw_affiliation_text,
                "institution": institution,
                "country": country,
            }
        )
    return normalized_details


def parse_openai_affiliation_payload(content: object) -> dict[str, object]:
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        content = "\n".join(text_parts).strip()
    if not isinstance(content, str) or not content.strip():
        return {"affiliations": [], "affiliation_details": []}

    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        return {"affiliations": [], "affiliation_details": []}
    normalized_details = normalize_openai_affiliation_details(parsed.get("affiliations", []))
    return {
        "affiliations": [str(item["institution"]) for item in normalized_details],
        "affiliation_details": normalized_details,
        "transcribed_text": str(parsed.get("transcribed_text") or "").strip(),
    }


def build_openai_affiliation_schema(include_transcribed_text: bool = False) -> dict[str, object]:
    properties = {
        "affiliations": {
            "type": "array",
            "description": "Distinct author affiliations extracted from the front matter.",
            "items": {
                "type": "object",
                "properties": {
                    "raw_affiliation_text": {"type": "string"},
                    "institution": {"type": "string"},
                    "country": {"type": ["string", "null"]},
                },
                "required": ["raw_affiliation_text", "institution", "country"],
                "additionalProperties": False,
            },
        }
    }
    required = ["affiliations"]
    if include_transcribed_text:
        properties["transcribed_text"] = {"type": "string"}
        required.append("transcribed_text")
    return {
        "name": "affiliation_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def build_openai_affiliation_request_with_examples(front_matter_text: str, model: str) -> dict[str, object]:
    schema = build_openai_affiliation_schema(include_transcribed_text=False)
    system_prompt = """
You extract structured author affiliations from the first-page front matter of academic papers.
The input may contain OCR noise, line breaks, superscripts, footnote markers, author names, emails, URLs, publisher text, and copyright text.

Your task is to extract only the author-affiliated institutions.

Rules:
1. Return only distinct author affiliations.
2. Extract institution-level affiliations only.
3. Remove or ignore author names, email addresses, URLs, postal addresses, zip codes, phone numbers, footnote markers, superscripts, and explanatory notes.
4. Do not return departments, schools, faculties, laboratories, centers, or institutes inside a university unless they are clearly functioning as the main institution named in the paper.
5. If multiple subunits belong to the same parent institution, normalize them to one institution entry.
6. Keep genuinely different institutions separate.
7. If a country is explicitly stated in the affiliation text, extract it; otherwise set country to null.
8. Do not infer the country from institution name, city name, email domain, or background knowledge.
9. Ignore non-author organizations such as publishers, conference organizers, copyright holders, indexing services, and funding agencies unless they are clearly listed as an author affiliation.
10. Preserve the institution name as written in the paper as much as possible, but remove obvious noise.
11. If no author affiliation is present, return an empty affiliations array.
12. raw_affiliation_text should be the relevant affiliation phrase from the source, cleaned only minimally.

Examples:

Example 1
Input:
John Smith1,2, Alice Lee2
1 Department of Computer Science, Stanford University, USA
2 Google Research, Mountain View, CA, USA
john@stanford.edu

Output:
{
  "affiliations": [
    {
      "raw_affiliation_text": "Department of Computer Science, Stanford University, USA",
      "institution": "Stanford University",
      "country": "USA"
    },
    {
      "raw_affiliation_text": "Google Research, Mountain View, CA, USA",
      "institution": "Google Research",
      "country": "USA"
    }
  ]
}
""".strip()
    user_prompt = f"""
Extract all distinct author affiliations from the following academic paper front matter.

For each affiliation, return:
- raw_affiliation_text: the original affiliation phrase or line
- institution: institution name only
- country: only if explicitly stated in the text, otherwise null

Merge duplicate mentions of the same institution.

Front matter text:
{front_matter_text.strip()}
""".strip()
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_schema", "json_schema": schema},
    }


def build_openai_affiliation_request(front_matter_text: str, model: str) -> dict[str, object]:
    return build_openai_affiliation_request_with_examples(front_matter_text=front_matter_text, model=model)


def parse_openai_affiliation_response(response_payload: dict[str, object]) -> dict[str, object]:
    choices = response_payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return {"affiliations": [], "affiliation_details": []}
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return {"affiliations": [], "affiliation_details": []}
    parsed = parse_openai_affiliation_payload(message.get("content", ""))
    return {
        "affiliations": parsed["affiliations"],
        "affiliation_details": parsed["affiliation_details"],
    }


def request_openai_affiliations(
    front_matter_text: str,
    api_key: str,
    model: str,
    api_base: str = OPENROUTER_API_BASE,
    timeout: float = 60.0,
) -> dict[str, object]:
    payload = build_openai_affiliation_request(front_matter_text=front_matter_text, model=model)
    client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(**payload)
    return parse_openai_affiliation_response(response.model_dump())


def parse_openai_affiliation_image_response(response_payload: dict[str, object]) -> dict[str, object]:
    choices = response_payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return {"affiliations": [], "affiliation_details": [], "transcribed_text": ""}
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return {"affiliations": [], "affiliation_details": [], "transcribed_text": ""}
    parsed = parse_openai_affiliation_payload(message.get("content", ""))
    return {
        "affiliations": parsed["affiliations"],
        "affiliation_details": parsed["affiliation_details"],
        "transcribed_text": str(parsed.get("transcribed_text") or "").strip(),
    }


def build_data_uri_for_image(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def request_openai_affiliations_from_images(
    front_matter_text: str,
    image_paths: list[Path],
    api_key: str,
    model: str,
    api_base: str = OPENROUTER_API_BASE,
    timeout: float = 60.0,
) -> dict[str, object]:
    schema = build_openai_affiliation_schema(include_transcribed_text=True)
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "These images come from the author-and-affiliation front matter of an academic paper. "
                "Transcribe only the visible author, affiliation, and email text. "
                "Do not infer missing text. Then extract structured affiliations with raw_affiliation_text, institution, and country.\n\n"
                f"Front matter markdown context:\n{front_matter_text.strip()}"
            ),
        }
    ]
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": build_data_uri_for_image(image_path)}})

    client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You read author-affiliation blocks from paper front matter images. Return JSON only."},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )
    return parse_openai_affiliation_image_response(response.model_dump())


def derive_paper_id_from_local_dir_name(dir_name: str) -> str:
    if dir_name.endswith("_first_page"):
        return dir_name[: -len("_first_page")]
    return dir_name


def _extract_title_text_from_content_block(block: dict[str, object]) -> str:
    content = block.get("content")
    if not isinstance(content, dict):
        return ""
    title_content = content.get("title_content")
    if not isinstance(title_content, list):
        return ""
    parts: list[str] = []
    for item in title_content:
        if isinstance(item, dict) and isinstance(item.get("content"), str):
            parts.append(item["content"])
    return " ".join(parts).strip()


def load_front_matter_image_paths(paper_dir: str | Path) -> list[Path]:
    root = Path(paper_dir)
    content_list_path = root / "auto" / "content_list_v2.json"
    image_paths: list[Path] = []
    if content_list_path.exists():
        try:
            payload = json.loads(content_list_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, list):
            reached_abstract = False
            for page in payload:
                if reached_abstract or not isinstance(page, list):
                    continue
                for block in page:
                    if not isinstance(block, dict):
                        continue
                    if str(block.get("type") or "").strip().lower() == "title":
                        if _extract_title_text_from_content_block(block).strip().lower() == "abstract":
                            reached_abstract = True
                            break
                    if str(block.get("type") or "").strip().lower() != "image":
                        continue
                    content = block.get("content")
                    if not isinstance(content, dict):
                        continue
                    image_source = content.get("image_source")
                    if not isinstance(image_source, dict):
                        continue
                    relative_path = image_source.get("path")
                    if isinstance(relative_path, str) and relative_path.strip():
                        candidate_path = root / "auto" / Path(relative_path)
                        if candidate_path.exists():
                            image_paths.append(candidate_path)
    if image_paths:
        deduped: list[Path] = []
        seen: set[Path] = set()
        for image_path in image_paths:
            if image_path not in seen:
                seen.add(image_path)
                deduped.append(image_path)
        return deduped

    front_matter_text = load_local_mineru_full_md_front_matter(root)
    matches = re.findall(r"!\[\]\((images/[^)]+)\)", front_matter_text)
    deduped_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for relative_path in matches:
        candidate_path = root / "auto" / Path(relative_path)
        if candidate_path.exists() and candidate_path not in seen_paths:
            seen_paths.add(candidate_path)
            deduped_paths.append(candidate_path)
    return deduped_paths


def should_use_front_matter_image_fallback(front_matter_text: str, affiliations: list[str]) -> bool:
    if affiliations:
        return False
    stripped = front_matter_text.strip()
    if not stripped or "images/" not in stripped:
        return False
    lowered = stripped.lower()
    if any(keyword in lowered for keyword in INSTITUTION_KEYWORDS + COMPANY_KEYWORDS):
        return False
    non_image_lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip() and not line.strip().startswith("![](")
    ]
    return len(non_image_lines) <= 3


def extract_affiliations_from_local_mineru_dir_with_gpt(
    paper_dir: str | Path,
    api_key: str,
    model: str,
    api_base: str = OPENROUTER_API_BASE,
    timeout: float = 60.0,
) -> dict[str, object]:
    root = Path(paper_dir)
    paper_id = derive_paper_id_from_local_dir_name(root.name)
    try:
        raw_text = load_local_mineru_full_md_front_matter(root)
        ocr_fallback_used = False
        ocr_transcribed_text = ""
        ocr_image_paths: list[str] = []
        text_result = request_openai_affiliations(
            front_matter_text=raw_text,
            api_key=api_key,
            model=model,
            api_base=api_base,
            timeout=timeout,
        ) if raw_text else {"affiliations": [], "affiliation_details": []}
        affiliations = list(text_result.get("affiliations") or [])
        affiliation_details = list(text_result.get("affiliation_details") or [])
        if raw_text and should_use_front_matter_image_fallback(raw_text, affiliations):
            image_paths = load_front_matter_image_paths(root)
            if image_paths:
                fallback_result = request_openai_affiliations_from_images(
                    front_matter_text=raw_text,
                    image_paths=image_paths,
                    api_key=api_key,
                    model=model,
                    api_base=api_base,
                    timeout=timeout,
                )
                fallback_affiliations = list(fallback_result.get("affiliations") or [])
                if fallback_affiliations or str(fallback_result.get("transcribed_text") or "").strip():
                    ocr_fallback_used = True
                    ocr_transcribed_text = str(fallback_result.get("transcribed_text") or "").strip()
                    ocr_image_paths = [str(path.relative_to(root)).replace("\\", "/") for path in image_paths]
                    if fallback_affiliations:
                        affiliations = fallback_affiliations
                        affiliation_details = list(fallback_result.get("affiliation_details") or [])
        return {
            "paper_id": paper_id,
            "source_pdf": "",
            "uploaded_pdf": "",
            "batch_id": "",
            "data_id": root.name,
            "status": "done" if raw_text else "missing",
            "raw_first_page_text": raw_text,
            "affiliation_lines": [line.strip() for line in raw_text.splitlines() if line.strip()],
            "affiliations": dedupe_affiliations(affiliations),
            "affiliation_details": normalize_openai_affiliation_details(affiliation_details),
            "ocr_fallback_used": ocr_fallback_used,
            "ocr_image_paths": ocr_image_paths,
            "ocr_transcribed_text": ocr_transcribed_text,
        }
    except Exception as exc:
        return {
            "paper_id": paper_id,
            "source_pdf": "",
            "uploaded_pdf": "",
            "batch_id": "",
            "data_id": root.name,
            "status": "failed",
            "raw_first_page_text": "",
            "affiliation_lines": [],
            "affiliations": [],
            "affiliation_details": [],
            "ocr_fallback_used": False,
            "ocr_image_paths": [],
            "ocr_transcribed_text": "",
            "error": str(exc),
        }


async def extract_affiliations_from_local_mineru_dir_with_gpt_async(
    paper_dir: str | Path,
    api_key: str,
    model: str,
    api_base: str = OPENROUTER_API_BASE,
    timeout: float = 60.0,
    retries: int = 0,
) -> dict[str, object]:
    attempts = 0
    while True:
        record = await asyncio.to_thread(
            extract_affiliations_from_local_mineru_dir_with_gpt,
            paper_dir,
            api_key,
            model,
            api_base,
            timeout,
        )
        if str(record.get("status") or "") != "failed" or attempts >= retries:
            return record
        attempts += 1


async def run_gpt_md_tasks_async(
    candidate_dirs: list[Path],
    api_key: str,
    model: str,
    api_base: str,
    timeout: float,
    concurrency: int,
    retries: int,
    on_record_completed=None,
) -> list[dict[str, object]]:
    semaphore = asyncio.Semaphore(max(concurrency, 1))
    total = len(candidate_dirs)

    async def run_one(index: int, paper_dir: Path) -> tuple[int, dict[str, object]]:
        async with semaphore:
            record = await extract_affiliations_from_local_mineru_dir_with_gpt_async(
                paper_dir,
                api_key=api_key,
                model=model,
                api_base=api_base,
                timeout=timeout,
                retries=retries,
            )
            return index, record

    print(f"[gpt-md] start: total={total} concurrency={max(concurrency, 1)}", file=sys.stderr, flush=True)
    tasks = [asyncio.create_task(run_one(index, paper_dir)) for index, paper_dir in enumerate(candidate_dirs)]
    ordered_results: list[dict[str, object] | None] = [None] * total
    completed = 0
    for task in asyncio.as_completed(tasks):
        index, record = await task
        ordered_results[index] = record
        completed += 1
        if on_record_completed is not None:
            callback_result = on_record_completed(record, completed, total)
            if inspect.isawaitable(callback_result):
                await callback_result
        affiliation_count = len(record.get("affiliations", [])) if isinstance(record.get("affiliations"), list) else 0
        print(
            f"[gpt-md] progress: {completed}/{total} paper_id={record.get('paper_id', '')} "
            f"status={record.get('status', '')} affiliations={affiliation_count}",
            file=sys.stderr,
            flush=True,
        )
    return [record for record in ordered_results if record is not None]


def sanitize_excel_cell_value(value: object) -> object:
    if isinstance(value, str):
        return EXCEL_ILLEGAL_CHARACTERS_RE.sub("", value)
    return value


def write_affiliation_xlsx(records: list[dict[str, object]], output_path: str | Path) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "affiliations"
    headers = [
        "paper_id",
        "source_pdf",
        "uploaded_pdf",
        "batch_id",
        "data_id",
        "status",
        "affiliations",
        "raw_first_page_text",
        "ocr_fallback_used",
        "ocr_image_paths",
        "ocr_transcribed_text",
        "error",
        "affiliation_details_json",
    ]
    sheet.append([sanitize_excel_cell_value(header) for header in headers])
    for record in records:
        sheet.append(
            [
                sanitize_excel_cell_value(record.get("paper_id", "")),
                sanitize_excel_cell_value(record.get("source_pdf", "")),
                sanitize_excel_cell_value(record.get("uploaded_pdf", "")),
                sanitize_excel_cell_value(record.get("batch_id", "")),
                sanitize_excel_cell_value(record.get("data_id", "")),
                sanitize_excel_cell_value(record.get("status", "")),
                sanitize_excel_cell_value(" | ".join(record.get("affiliations", []))),
                sanitize_excel_cell_value(record.get("raw_first_page_text", "")),
                sanitize_excel_cell_value(str(bool(record.get("ocr_fallback_used"))).lower()),
                sanitize_excel_cell_value(" | ".join(record.get("ocr_image_paths", []))),
                sanitize_excel_cell_value(record.get("ocr_transcribed_text", "")),
                sanitize_excel_cell_value(record.get("error", "")),
                sanitize_excel_cell_value(json.dumps(record.get("affiliation_details", []), ensure_ascii=False)),
            ]
        )
    workbook.save(output_path)


def write_mineru_affiliation_outputs(records: list[dict[str, object]], output_dir: str | Path) -> None:
    output_root = Path(output_dir)
    per_paper_dir = output_root / "per_paper"
    per_paper_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        paper_id = str(record["paper_id"])
        (per_paper_dir / f"{paper_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (output_root / "affiliations.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )

    csv_path = output_root / "affiliations.csv"
    fieldnames = [
        "paper_id",
        "source_pdf",
        "uploaded_pdf",
        "batch_id",
        "data_id",
        "status",
        "affiliations",
        "raw_first_page_text",
        "ocr_fallback_used",
        "ocr_image_paths",
        "ocr_transcribed_text",
        "error",
        "affiliation_details_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "paper_id": record.get("paper_id", ""),
                    "source_pdf": record.get("source_pdf", ""),
                    "uploaded_pdf": record.get("uploaded_pdf", ""),
                    "batch_id": record.get("batch_id", ""),
                    "data_id": record.get("data_id", ""),
                    "status": record.get("status", ""),
                    "affiliations": " | ".join(record.get("affiliations", [])),
                    "raw_first_page_text": record.get("raw_first_page_text", ""),
                    "ocr_fallback_used": str(bool(record.get("ocr_fallback_used"))).lower(),
                    "ocr_image_paths": " | ".join(record.get("ocr_image_paths", [])),
                    "ocr_transcribed_text": record.get("ocr_transcribed_text", ""),
                    "error": record.get("error", ""),
                    "affiliation_details_json": json.dumps(record.get("affiliation_details", []), ensure_ascii=False),
                }
            )
    write_affiliation_xlsx(records, output_root / "affiliations.xlsx")


def checkpoint_records_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "checkpoints" / "records.jsonl"


def append_affiliation_checkpoint_record(record: dict[str, object], output_dir: str | Path) -> None:
    checkpoint_path = checkpoint_records_path(output_dir)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_affiliation_rows(input_csv: str | Path) -> list[dict[str, str]]:
    with Path(input_csv).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_affiliation_jsonl_records(input_path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    path = Path(input_path)
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def explode_affiliation_field(affiliation_field: str) -> list[str]:
    if not affiliation_field:
        return []
    return [part.strip() for part in affiliation_field.split("|") if part.strip()]


def select_local_mineru_dirs_for_gpt(
    input_root: str | Path,
    existing_csv: str | Path | None = None,
    existing_records: list[dict[str, object]] | None = None,
    only_empty: bool = False,
    status_filter: set[str] | None = None,
    resume_incomplete: bool = False,
) -> list[Path]:
    root = Path(input_root)
    candidate_dirs = sorted(candidate for candidate in root.iterdir() if candidate.is_dir() and candidate.name != "batches")
    record_rows = existing_records
    if record_rows is None and existing_csv:
        record_rows = load_affiliation_records_from_csv(existing_csv)
    if not record_rows:
        return candidate_dirs
    if status_filter:
        wanted_status = {status.strip().lower() for status in status_filter if status.strip()}
        matching_paper_ids = {
            str(row.get("paper_id") or "").strip()
            for row in record_rows
            if str(row.get("status") or "").strip().lower() in wanted_status
        }
        return [paper_dir for paper_dir in candidate_dirs if derive_paper_id_from_local_dir_name(paper_dir.name) in matching_paper_ids]
    if not only_empty:
        if not resume_incomplete:
            return candidate_dirs
        completed_paper_ids = {
            str(row.get("paper_id") or "").strip()
            for row in record_rows
            if str(row.get("status") or "").strip().lower() == "done" and str(row.get("affiliations") or "").strip()
        }
        return [paper_dir for paper_dir in candidate_dirs if derive_paper_id_from_local_dir_name(paper_dir.name) not in completed_paper_ids]

    empty_paper_ids = {
        str(row.get("paper_id") or "").strip()
        for row in record_rows
        if not str(row.get("affiliations") or "").strip()
    }
    return [paper_dir for paper_dir in candidate_dirs if derive_paper_id_from_local_dir_name(paper_dir.name) in empty_paper_ids]


def load_affiliation_records_from_csv(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in iter_affiliation_rows(path):
        raw_first_page_text = str(row.get("raw_first_page_text") or "")
        raw_affiliation_details = str(row.get("affiliation_details_json") or "").strip()
        try:
            parsed_affiliation_details = json.loads(raw_affiliation_details) if raw_affiliation_details else []
        except Exception:
            parsed_affiliation_details = []
        records.append(
            {
                "paper_id": str(row.get("paper_id") or ""),
                "source_pdf": str(row.get("source_pdf") or ""),
                "uploaded_pdf": str(row.get("uploaded_pdf") or ""),
                "batch_id": str(row.get("batch_id") or ""),
                "data_id": str(row.get("data_id") or ""),
                "status": str(row.get("status") or ""),
                "raw_first_page_text": raw_first_page_text,
                "affiliation_lines": [line.strip() for line in raw_first_page_text.splitlines() if line.strip()],
                "affiliations": explode_affiliation_field(str(row.get("affiliations") or "")),
                "affiliation_details": normalize_openai_affiliation_details(parsed_affiliation_details),
                "ocr_fallback_used": str(row.get("ocr_fallback_used") or "").strip().lower() == "true",
                "ocr_image_paths": explode_affiliation_field(str(row.get("ocr_image_paths") or "")),
                "ocr_transcribed_text": str(row.get("ocr_transcribed_text") or ""),
                "error": str(row.get("error") or ""),
            }
        )
    return records


def load_existing_affiliation_records(existing_csv_path: str | Path | None, output_dir: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if existing_csv_path is not None and Path(existing_csv_path).exists():
        records = merge_affiliation_records(records, load_affiliation_records_from_csv(existing_csv_path))
    checkpoint_path = checkpoint_records_path(output_dir)
    if checkpoint_path.exists():
        records = merge_affiliation_records(records, iter_affiliation_jsonl_records(checkpoint_path))
    return records


def run_gpt_md_affiliation_extraction(args: argparse.Namespace) -> list[dict[str, object]]:
    config = load_config()
    output_root = Path(args.output_dir) if args.output_dir else config["paths"]["affiliations_root"] / args.conference
    output_root.mkdir(parents=True, exist_ok=True)
    auto_existing_csv = output_root / "affiliations.csv"
    existing_csv_path = Path(args.existing_csv) if args.existing_csv else (auto_existing_csv if auto_existing_csv.exists() else None)
    existing_records = load_existing_affiliation_records(existing_csv_path, output_root)
    resume_incomplete = bool(existing_records and not args.existing_csv and not args.only_empty and not args.status_filter)

    api_key = os.environ.get(args.api_key_env or "OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(f"Missing API key in environment variable: {args.api_key_env}")

    input_root = Path(args.input_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    candidate_dirs = select_local_mineru_dirs_for_gpt(
        input_root=input_root,
        existing_csv=existing_csv_path,
        existing_records=existing_records,
        only_empty=args.only_empty,
        status_filter={args.status_filter} if args.status_filter else None,
        resume_incomplete=resume_incomplete,
    )
    if args.limit:
        candidate_dirs = candidate_dirs[: args.limit]
    if resume_incomplete and existing_csv_path is not None:
        print(f"[gpt-md] resume: existing_csv={existing_csv_path} remaining={len(candidate_dirs)}", file=sys.stderr, flush=True)
    merged_records = list(existing_records)

    def on_record_completed(record: dict[str, object], completed_count: int, total_count: int) -> None:
        nonlocal merged_records
        merged_records = merge_affiliation_records(merged_records, [record])
        append_affiliation_checkpoint_record(record, output_root)
        if completed_count % CHECKPOINT_SYNC_EVERY == 0 or completed_count == total_count:
            write_mineru_affiliation_outputs(merged_records, output_root)

    records = asyncio.run(
        run_gpt_md_tasks_async(
            candidate_dirs=candidate_dirs,
            api_key=api_key,
            model=args.model,
            api_base=args.api_base,
            timeout=args.timeout_seconds,
            concurrency=args.concurrency,
            retries=args.retries,
            on_record_completed=on_record_completed,
        )
    )
    merged_records = merge_affiliation_records(merged_records, records)
    write_mineru_affiliation_outputs(merged_records, output_root)
    return merged_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract affiliations from MinerU full.md front matter with GPT.")
    parser.add_argument("command", choices=["gpt-md"])
    parser.add_argument("--input-dir")
    parser.add_argument("--conference")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--api-base", default=OPENROUTER_API_BASE)
    parser.add_argument("--model", default=DEFAULT_GPT_AFFILIATION_MODEL)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--output-dir")
    parser.add_argument("--existing-csv")
    parser.add_argument("--only-empty", action="store_true")
    parser.add_argument("--status-filter")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.input_dir:
        parser.error("gpt-md requires --input-dir")
    if not args.conference:
        parser.error("gpt-md requires --conference")
    records = run_gpt_md_affiliation_extraction(args)
    config = load_config()
    output_dir = Path(args.output_dir) if args.output_dir else config["paths"]["affiliations_root"] / args.conference
    print(json.dumps({"records": len(records), "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
