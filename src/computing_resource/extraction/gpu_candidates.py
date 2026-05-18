from __future__ import annotations

import re
from typing import Any

from computing_resource.extraction.gpu_catalog import HardwareCatalog
from computing_resource.extraction.gpu_name_rules import standardize_hardware_text


def _normalize_title(title: str | None) -> str:
    return " ".join((title or "").lower().split())


def _is_appendix_section(section_number: str | None) -> bool:
    if not section_number:
        return False
    return section_number[0].isalpha()


def _matches_appendix_pattern(normalized_title: str, patterns: list[str]) -> bool:
    return any(pattern in normalized_title for pattern in patterns)


def _contains_term(normalized_text: str, term: str) -> bool:
    normalized_term = standardize_hardware_text(term)
    if not normalized_term:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])"
    return re.search(pattern, normalized_text, flags=re.IGNORECASE) is not None


def _requires_hardware_context(term: str) -> bool:
    normalized_term = standardize_hardware_text(term)
    return bool(re.fullmatch(r"[a-z]\d{2}", normalized_term) or re.fullmatch(r"\d{4}", normalized_term))


def _has_hardware_context_around_term(normalized_text: str, term: str) -> bool:
    normalized_term = standardize_hardware_text(term)
    if not normalized_term:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])"
    context_terms = (
        "gpu",
        "gpus",
        "tpu",
        "tpus",
        "nvidia",
        "amd",
        "rtx",
        "tesla",
        "quadro",
        "geforce",
        "accelerator",
        "accelerators",
        "device",
        "devices",
    )
    for match in re.finditer(pattern, normalized_text, flags=re.IGNORECASE):
        start = max(0, match.start() - 32)
        end = min(len(normalized_text), match.end() + 32)
        context = normalized_text[start:end]
        if any(_contains_term(context, context_term) for context_term in context_terms):
            return True
    return False


def _has_keyword_hit(text: str, keywords: list[str]) -> bool:
    normalized_text = standardize_hardware_text(text)
    for keyword in keywords:
        keyword_text = str(keyword)
        if not _contains_term(normalized_text, keyword_text):
            continue
        if _requires_hardware_context(keyword_text) and not _has_hardware_context_around_term(normalized_text, keyword_text):
            continue
        return True
    return False


def _has_alias_hit(text: str, catalog: HardwareCatalog) -> bool:
    normalized_text = standardize_hardware_text(text)
    for alias in catalog.alias_to_names:
        if not _contains_term(normalized_text, alias):
            continue
        if _requires_hardware_context(alias) and not _has_hardware_context_around_term(normalized_text, alias):
            continue
        return True
    return False


def _has_count_pattern_hit(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _iter_section_windows(section_doc: dict[str, Any]):
    for group_name in ("sections", "appendix"):
        for section in section_doc.get(group_name, []):
            yield group_name, section
            for sub_section in section.get("sub_section", []):
                yield group_name, sub_section


def has_hardware_signal(text: str, rules: dict[str, Any], catalog: HardwareCatalog) -> bool:
    hardware_keywords = list(rules.get("hardware_keywords", []))
    count_patterns = list(rules.get("count_patterns", []))
    return (
        _has_alias_hit(text, catalog)
        or _has_keyword_hit(text, hardware_keywords)
        or _has_count_pattern_hit(text, count_patterns)
    )


def build_candidate_windows(section_doc: dict[str, Any], rules: dict[str, Any], catalog: HardwareCatalog) -> list[dict]:
    strong_keep_titles = set(rules.get("strong_keep_titles", []))
    soft_skip_titles = set(rules.get("soft_skip_titles", []))
    hard_skip_titles = set(rules.get("hard_skip_titles", []))
    appendix_keep_patterns = list(rules.get("appendix_keep_patterns", []))
    hardware_keywords = list(rules.get("hardware_keywords", []))
    count_patterns = list(rules.get("count_patterns", []))

    windows = []
    seen_keys = set()
    for group_name, section in _iter_section_windows(section_doc):
        section_title = section.get("section_title", "")
        normalized_title = _normalize_title(section_title)
        section_number = section.get("section_number", "")
        if group_name == "appendix" and section_number == "-1":
            continue
        dedupe_key = (
            group_name,
            section_number,
            normalized_title,
            standardize_hardware_text(section.get("content", "") or ""),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        text = section.get("content", "") or ""
        alias_hit = _has_alias_hit(text, catalog)
        keyword_hit = _has_keyword_hit(text, hardware_keywords)
        count_pattern_hit = _has_count_pattern_hit(text, count_patterns)
        appendix_pattern_hit = _is_appendix_section(section_number) and _matches_appendix_pattern(
            normalized_title, appendix_keep_patterns
        )

        include = False
        rule_tier = ""
        matched_by_rule = False

        if normalized_title in hard_skip_titles:
            if alias_hit or keyword_hit or count_pattern_hit:
                include = True
                rule_tier = "hard_skip_override"
        elif normalized_title in strong_keep_titles or appendix_pattern_hit:
            include = True
            rule_tier = "strong_keep"
            matched_by_rule = True
        elif normalized_title in soft_skip_titles:
            include = alias_hit or keyword_hit or count_pattern_hit
            if include:
                rule_tier = "soft_skip"
        else:
            include = alias_hit or keyword_hit or count_pattern_hit
            if include:
                rule_tier = "conditional_keep"

        if not include:
            continue

        windows.append(
            {
                "window_id": f"{section_doc['paper_id']}::{group_name}::{section_number or normalized_title}",
                "paper_id": section_doc["paper_id"],
                "section_number": section_number,
                "section_title": section_title,
                "normalized_section_title": normalized_title,
                "window_text": text,
                "rule_tier": rule_tier,
                "matched_by_rule": matched_by_rule,
                "matched_by_alias": alias_hit,
                "matched_by_keyword": keyword_hit,
                "matched_by_count_pattern": count_pattern_hit,
            }
        )

    return windows
