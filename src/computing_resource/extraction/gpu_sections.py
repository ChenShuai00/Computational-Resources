from __future__ import annotations

import hashlib
import json
from pathlib import Path

from computing_resource.pipeline.section import (
    extract_sections,
    extract_start_with_letter_dot_number,
    extract_title_level,
    merge_article_section,
)

SECTION_DOC_CACHE_SCHEMA_VERSION = 1


def _find_parse_markdown(parse_dir: Path) -> Path:
    auto_dir = parse_dir / "auto"
    full_md = auto_dir / "full.md"
    if full_md.exists():
        return full_md

    paper_md = auto_dir / f"{parse_dir.name}.md"
    if paper_md.exists():
        return paper_md

    raise FileNotFoundError(f"No markdown parse artifact found under {auto_dir}")


def _cached_section_path(parse_dir: Path) -> Path:
    return parse_dir / f"{parse_dir.name}_sectioned.json"


def load_parse_markdown_metadata(parse_dir: str | Path) -> dict:
    parse_path = Path(parse_dir)
    md_path = _find_parse_markdown(parse_path)
    full_text = md_path.read_text(encoding="utf-8")
    return {
        "source_md_path": md_path,
        "full_text": full_text,
        "source_md_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
    }


def _build_section_doc_meta(md_path: Path, full_text: str) -> dict:
    return {
        "schema_version": SECTION_DOC_CACHE_SCHEMA_VERSION,
        "source_md_path": str(md_path),
        "source_md_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
    }


def _serialize_section_doc_for_cache(section_doc: dict) -> dict:
    return {
        key: value
        for key, value in section_doc.items()
        if key not in {"source_md_path", "source_section_path", "full_text"}
    }


def _write_section_doc_cache(cache_path: Path, section_doc: dict) -> None:
    cache_path.write_text(
        json.dumps(_serialize_section_doc_for_cache(section_doc), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _is_cached_section_doc_fresh(section_doc: dict, source_meta: dict) -> bool:
    meta = section_doc.get("_section_doc_meta")
    if not isinstance(meta, dict):
        return False
    if meta.get("schema_version") != SECTION_DOC_CACHE_SCHEMA_VERSION:
        return False
    return meta.get("source_md_sha256") == source_meta["source_md_sha256"]


def _appendix_sections(raw_appendix: list[tuple[str, str]]) -> list[dict]:
    appendix_section = []
    appendix_total_content = ""

    for appendix_title, appendix_content in raw_appendix:
        appendix_total_content += appendix_content + "\n\n"
        appendix_dict: dict[str, str] = {}
        numbered_prefix = extract_start_with_letter_dot_number(appendix_title)
        first_letter = appendix_title[0]
        second_letter = appendix_title[1] if len(appendix_title) > 1 else "A"

        if numbered_prefix:
            appendix_dict["section_number"] = numbered_prefix
            appendix_dict["section_title"] = appendix_title[len(numbered_prefix) :].lstrip().lower()
            appendix_dict["raw_section_title"] = appendix_title
            appendix_dict["content"] = appendix_content
            appendix_section.append(appendix_dict)
        elif appendix_section and (not first_letter.isalpha() or second_letter.islower()):
            appendix_section[-1]["content"] += f"{appendix_title} {appendix_content}"
        else:
            appendix_dict["section_number"] = first_letter
            appendix_dict["section_title"] = appendix_title[1:].lstrip().lower()
            appendix_dict["raw_section_title"] = appendix_title
            appendix_dict["content"] = appendix_content
            appendix_section.append(appendix_dict)

    if appendix_total_content:
        appendix_section.append(
            {
                "section_number": "-1",
                "section_title": "appendix",
                "raw_section_title": "appendix",
                "content": appendix_total_content.strip(),
            }
        )

    return merge_article_section(appendix_section)


def _build_section_doc(parse_dir: Path, md_path: Path, full_text: str | None = None) -> dict:
    if full_text is None:
        full_text = md_path.read_text(encoding="utf-8")
    raw_sections = extract_sections(full_text)
    if len(raw_sections) < 2:
        raise ValueError(f"Unexpected markdown structure in {md_path}")

    paper_title, paper_author_institution = raw_sections[0]
    _, abstract_content = raw_sections[1]
    section_doc = {
        "paper_id": parse_dir.name,
        "source_md_path": md_path,
        "full_text": full_text,
        "title": paper_title,
        "authors_institution": paper_author_institution,
        "abstract": abstract_content,
        "sections": [],
        "appendix": [],
        "_section_doc_meta": _build_section_doc_meta(md_path, full_text),
    }

    appendix_raw: list[tuple[str, str]] = []
    for index in range(2, len(raw_sections)):
        raw_section_title, section_content = raw_sections[index]
        level, section_num, section_title = extract_title_level(raw_section_title)
        section_doc["sections"].append(
            {
                "level": level,
                "section_number": section_num,
                "raw_section_title": raw_section_title,
                "section_title": section_title,
                "content": section_content,
            }
        )
        if section_title == "references":
            appendix_raw = raw_sections[index + 1 :]
            break

    section_doc["sections"] = merge_article_section(section_doc["sections"])
    section_doc["appendix"] = _appendix_sections(appendix_raw)
    return section_doc


def load_section_doc_from_parse_dir(parse_dir: str | Path) -> dict:
    parse_path = Path(parse_dir)
    cache_path = _cached_section_path(parse_path)
    source_meta: dict | None = None
    try:
        source_meta = load_parse_markdown_metadata(parse_path)
    except FileNotFoundError:
        source_meta = None

    if cache_path.exists():
        section_doc = json.loads(cache_path.read_text(encoding="utf-8"))
        if source_meta is None:
            section_doc["source_section_path"] = cache_path
            return section_doc
        if _is_cached_section_doc_fresh(section_doc, source_meta):
            section_doc["source_section_path"] = cache_path
            section_doc["source_md_path"] = source_meta["source_md_path"]
            section_doc["full_text"] = source_meta["full_text"]
            return section_doc

    if source_meta is None:
        raise FileNotFoundError(f"No markdown parse artifact found under {parse_path / 'auto'}")

    section_doc = _build_section_doc(
        parse_path,
        source_meta["source_md_path"],
        full_text=source_meta["full_text"],
    )
    _write_section_doc_cache(cache_path, section_doc)
    section_doc["source_section_path"] = cache_path
    section_doc["source_md_path"] = source_meta["source_md_path"]
    section_doc["full_text"] = source_meta["full_text"]
    return section_doc
