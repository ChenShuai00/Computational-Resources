from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from computing_resource.extraction.gpu_name_rules import (
    GENERIC_UNRESOLVED_TOKENS,
    SPECIAL_CANONICAL_RULES,
    extract_input_memory,
    generate_catalog_aliases,
    should_include_catalog_row,
    standardize_hardware_text,
)
from computing_resource.extraction.gpu_family_rules import (
    resolve_post_unresolved_family_rules,
    resolve_pre_unresolved_family_rules,
)


REQUIRED_CATALOG_COLUMNS = {
    "Hardware name",
    "Manufacturer",
    "Type",
    "Memory (bytes)",
}
MODEL_OR_FRAMEWORK_NAME_REASON = "input is a model or framework name, not a GPU device"
MODEL_NAME_TOKENS = {
    "alexnet",
    "bart",
    "bert",
    "chatgpt",
    "claude",
    "codellama",
    "cursor",
    "deepseek",
    "electra",
    "gemma",
    "gpt",
    "gru",
    "llama",
    "lstm",
    "magicoder",
    "mistral",
    "modernbert",
    "mt5",
    "parlai",
    "qwen",
    "resnet",
    "roberta",
    "transformer",
    "vgg",
    "xlnet",
}
MODEL_NAME_HARDWARE_EXCLUSION_TOKENS = {
    "amd",
    "apple",
    "ascend",
    "cuda",
    "dgx",
    "geforce",
    "google",
    "gpu",
    "gtx",
    "huawei",
    "instinct",
    "intel",
    "mi",
    "npu",
    "nvidia",
    "quadro",
    "radeon",
    "rtx",
    "tesla",
    "tpu",
}


@dataclass(frozen=True)
class HardwareMatch:
    raw_hardware_name: str
    cleaned_hardware_name: str
    benchmark_hardware_name: str | None
    match_status: str
    candidate_names: tuple[str, ...]


@dataclass(frozen=True)
class HardwareResolution:
    raw_hardware_name: str
    cleaned_hardware_name: str
    normalized_hardware_name: str
    benchmark_hardware_name: str | None
    match_status: str
    match_method: str
    normalization_reason: str
    candidate_names: tuple[str, ...]
    extracted_memory: str


@dataclass(frozen=True)
class HardwareCatalog:
    source_path: Path
    rows: tuple[dict, ...]
    normalized_to_name: dict[str, str]
    alias_to_names: dict[str, tuple[str, ...]]
    row_map: dict[str, dict]

    @property
    def alias_to_name(self) -> dict[str, str]:
        return {
            alias: names[0]
            for alias, names in self.alias_to_names.items()
            if len(names) == 1
        }


def _load_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_xlsx_shared_strings(archive: ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []

    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings = []
    for shared_string in root.findall("a:si", namespace):
        parts = [node.text or "" for node in shared_string.findall(".//a:t", namespace)]
        strings.append("".join(parts))
    return strings


def _load_xlsx_date_style_ids(archive: ZipFile) -> set[int]:
    try:
        root = ElementTree.fromstring(archive.read("xl/styles.xml"))
    except KeyError:
        return set()

    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    custom_numfmts = {
        int(node.attrib["numFmtId"]): node.attrib.get("formatCode", "").lower()
        for node in root.findall("a:numFmts/a:numFmt", namespace)
    }
    builtin_date_numfmt_ids = {14, 15, 16, 17, 18, 19, 20, 21, 22, 45, 46, 47}
    date_style_ids = set()
    cell_formats = root.find("a:cellXfs", namespace)
    if cell_formats is None:
        return date_style_ids

    for style_id, xf in enumerate(cell_formats.findall("a:xf", namespace)):
        numfmt_id = int(xf.attrib.get("numFmtId", "0"))
        if numfmt_id in builtin_date_numfmt_ids:
            date_style_ids.add(style_id)
            continue
        format_code = custom_numfmts.get(numfmt_id, "")
        if format_code and all(token in format_code for token in ("y", "m", "d")):
            date_style_ids.add(style_id)
    return date_style_ids


def _resolve_xlsx_sheet_path(archive: ZipFile) -> str:
    namespace = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    first_sheet = workbook.find("a:sheets/a:sheet", namespace)
    if first_sheet is None:
        raise ValueError("xlsx benchmark must contain at least one worksheet")

    relationship_id = first_sheet.attrib.get(
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    )
    relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relationship in relationships.findall("rel:Relationship", namespace):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib["Target"].lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError("xlsx benchmark worksheet relationship is missing")


def _column_index_from_ref(cell_ref: str) -> int:
    letters = "".join(character for character in cell_ref if character.isalpha())
    index = 0
    for character in letters:
        index = index * 26 + (ord(character.upper()) - ord("A") + 1)
    return index - 1


def _convert_excel_serial_to_date(value: str) -> str:
    date_value = datetime(1899, 12, 30) + timedelta(days=float(value))
    return date_value.strftime("%Y-%m-%d")


def _extract_xlsx_cell_value(cell, shared_strings: list[str], date_style_ids: set[int], namespace: dict) -> object:
    cell_type = cell.attrib.get("t")
    style_id = int(cell.attrib.get("s", "0"))
    value_node = cell.find("a:v", namespace)

    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//a:t", namespace))
    if value_node is None:
        return ""

    value = value_node.text or ""
    if cell_type == "s":
        return shared_strings[int(value)]
    if style_id in date_style_ids and value:
        return _convert_excel_serial_to_date(value)
    return value


def _load_xlsx_rows(path: Path) -> list[dict]:
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared_strings = _load_xlsx_shared_strings(archive)
        date_style_ids = _load_xlsx_date_style_ids(archive)
        worksheet_path = _resolve_xlsx_sheet_path(archive)
        worksheet = ElementTree.fromstring(archive.read(worksheet_path))

    sheet_data = worksheet.find("a:sheetData", namespace)
    if sheet_data is None:
        return []

    rows = []
    header = []
    for row_node in sheet_data.findall("a:row", namespace):
        row_values = {}
        for cell in row_node.findall("a:c", namespace):
            cell_ref = cell.attrib.get("r", "")
            row_values[_column_index_from_ref(cell_ref)] = _extract_xlsx_cell_value(
                cell,
                shared_strings,
                date_style_ids,
                namespace,
            )

        if not row_values:
            continue

        width = max(row_values) + 1
        ordered_values = [row_values.get(index, "") for index in range(width)]
        if not header:
            header = ordered_values
            continue

        if len(ordered_values) < len(header):
            ordered_values.extend([""] * (len(header) - len(ordered_values)))

        rows.append({header[index]: ordered_values[index] for index in range(len(header))})
    return rows


def _load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".xlsx":
        return _load_xlsx_rows(path)
    return _load_csv_rows(path)


def _resolve_catalog_path(path: Path) -> Path:
    if path.exists():
        return path

    # Support the legacy workbook layout that lived under paper_section_gpu/.
    legacy_segment = "paper_section_gpu"
    if legacy_segment in path.parts:
        legacy_index = path.parts.index(legacy_segment)
        candidate = Path(*path.parts[:legacy_index]) / "ml_hardware" / path.name
        if candidate.exists():
            return candidate
        path = candidate

    path_parts = tuple(part.lower() for part in path.parts)
    config_candidate = Path(__file__).resolve().parents[3] / "config" / "ml_hardware" / path.name
    if (
        path.name.lower() == "ml_hardware.xlsx"
        and config_candidate.exists()
        and len(path_parts) >= 5
        and path_parts[-5:] == ("data", "processed", "gpu", "ml_hardware", "ml_hardware.xlsx")
    ):
        return config_candidate

    return path


def load_hardware_catalog(catalog_path: str | Path) -> HardwareCatalog:
    path = Path(catalog_path)
    path = _resolve_catalog_path(path)
    rows = _load_rows(path)
    if not rows:
        raise ValueError(f"No catalog rows found in {path}")

    missing_columns = REQUIRED_CATALOG_COLUMNS.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Catalog is missing required columns: {sorted(missing_columns)}")

    normalized_to_name: dict[str, str] = {}
    row_map: dict[str, dict] = {}
    alias_lookup: defaultdict[str, set[str]] = defaultdict(set)

    for row in rows:
        if not should_include_catalog_row(row):
            continue

        hardware_name = (row.get("Hardware name") or "").strip()
        if not hardware_name:
            continue

        normalized_name = standardize_hardware_text(hardware_name)
        normalized_to_name[normalized_name] = hardware_name
        row_map[hardware_name] = row

    exact_names = set(normalized_to_name)
    for hardware_name, row in row_map.items():
        for alias in generate_catalog_aliases(hardware_name, row, exact_names):
            alias_lookup[alias].add(hardware_name)

    return HardwareCatalog(
        source_path=path,
        rows=tuple(row_map[name] for name in row_map),
        normalized_to_name=normalized_to_name,
        alias_to_names={alias: tuple(sorted(names)) for alias, names in alias_lookup.items()},
        row_map=row_map,
    )


def match_hardware_name(raw_hardware_name: str, catalog: HardwareCatalog) -> HardwareMatch:
    cleaned_name = standardize_hardware_text(raw_hardware_name)
    if not cleaned_name:
        return HardwareMatch(raw_hardware_name, cleaned_name, None, "unmatched", ())

    exact_name = catalog.normalized_to_name.get(cleaned_name)
    if exact_name:
        return HardwareMatch(raw_hardware_name, cleaned_name, exact_name, "exact_match", (exact_name,))

    alias_names = catalog.alias_to_names.get(cleaned_name, ())
    if len(alias_names) == 1:
        return HardwareMatch(raw_hardware_name, cleaned_name, alias_names[0], "alias_match", alias_names)
    if alias_names:
        return HardwareMatch(raw_hardware_name, cleaned_name, None, "ambiguous", alias_names)
    return HardwareMatch(raw_hardware_name, cleaned_name, None, "unmatched", ())


def _match_cleaned_hardware_name(
    raw_hardware_name: str,
    cleaned_name: str,
    catalog: HardwareCatalog,
) -> HardwareMatch:
    if not cleaned_name:
        return HardwareMatch(raw_hardware_name, cleaned_name, None, "unmatched", ())

    exact_name = catalog.normalized_to_name.get(cleaned_name)
    if exact_name:
        return HardwareMatch(raw_hardware_name, cleaned_name, exact_name, "exact_match", (exact_name,))

    alias_names = catalog.alias_to_names.get(cleaned_name, ())
    if len(alias_names) == 1:
        return HardwareMatch(raw_hardware_name, cleaned_name, alias_names[0], "alias_match", alias_names)
    if alias_names:
        return HardwareMatch(raw_hardware_name, cleaned_name, None, "ambiguous", alias_names)
    return HardwareMatch(raw_hardware_name, cleaned_name, None, "unmatched", ())


def _canonicalize_match_text(cleaned_name: str) -> str:
    tokens = cleaned_name.split()
    if len(tokens) < 2:
        return cleaned_name

    trailing_noise_tokens = {
        "gpu",
        "gpus",
        "accelerator",
        "accelerators",
        "hardware",
        "device",
        "devices",
        "nvidia",
        "amd",
        "google",
        "huawei",
        "apple",
    }
    end_index = len(tokens)
    while end_index > 0 and tokens[end_index - 1] in trailing_noise_tokens:
        end_index -= 1

    if end_index == len(tokens):
        return cleaned_name

    candidate = " ".join(tokens[:end_index]).strip()
    if not candidate:
        return cleaned_name
    if not re.search(r"\d", candidate):
        return cleaned_name
    if not re.search(r"[a-z]", candidate):
        return cleaned_name
    return candidate


def _lookup_benchmark_name(canonical_name: str, catalog: HardwareCatalog) -> str | None:
    normalized_name = standardize_hardware_text(canonical_name)
    exact_name = catalog.normalized_to_name.get(normalized_name)
    if exact_name:
        return exact_name
    alias_names = catalog.alias_to_names.get(normalized_name, ())
    return alias_names[0] if len(alias_names) == 1 else None


def _default_explicit_benchmark_name(
    normalized_hardware_name: str,
    cleaned_hardware_name: str,
    extracted_memory: str,
    catalog: HardwareCatalog,
) -> str | None:
    normalized_name = standardize_hardware_text(normalized_hardware_name)
    if normalized_name in {"nvidia a40"}:
        return _lookup_benchmark_name("NVIDIA A40 PCIe", catalog)
    if normalized_name in {"nvidia rtx a100"}:
        benchmark_name = "NVIDIA A100 PCIe 80GB" if extracted_memory == "80GB" or "80gb" in cleaned_hardware_name else "NVIDIA A100 PCIe 40GB"
        return _lookup_benchmark_name(benchmark_name, catalog)
    if normalized_name in {"nvidia h800"}:
        if "sxm" in cleaned_hardware_name:
            return _lookup_benchmark_name("NVIDIA H800 SXM5", catalog)
        if "nvl" in cleaned_hardware_name:
            return _lookup_benchmark_name("NVIDIA H800 NVL", catalog)
        return _lookup_benchmark_name("NVIDIA H800 PCIe", catalog)
    if normalized_name in {"nvidia h200"}:
        if "sxm" in cleaned_hardware_name:
            return _lookup_benchmark_name("NVIDIA H200 SXM5", catalog)
        return _lookup_benchmark_name("NVIDIA H200 PCIe", catalog)
    if normalized_name in {"nvidia gh200"}:
        return _lookup_benchmark_name("NVIDIA GH200", catalog)
    if normalized_name in {"nvidia a100"}:
        benchmark_name = "NVIDIA A100 80GB" if extracted_memory == "80GB" or "80gb" in cleaned_hardware_name else "NVIDIA A100 40GB"
        return _lookup_benchmark_name(benchmark_name, catalog)
    return None


def _build_rule_resolution(
    raw_hardware_name: str,
    normalized_text: str,
    special_name: str,
    catalog: HardwareCatalog,
    normalization_reason: str,
    extracted_memory: str,
) -> HardwareResolution:
    benchmark_name = _lookup_benchmark_name(special_name, catalog) or _default_explicit_benchmark_name(
        normalized_hardware_name=special_name,
        cleaned_hardware_name=normalized_text,
        extracted_memory=extracted_memory,
        catalog=catalog,
    )
    normalized_hardware_name = benchmark_name or special_name
    return HardwareResolution(
        raw_hardware_name=raw_hardware_name,
        cleaned_hardware_name=normalized_text,
        normalized_hardware_name=normalized_hardware_name,
        benchmark_hardware_name=benchmark_name,
        match_status="rule_match",
        match_method="rule",
        normalization_reason=normalization_reason,
        candidate_names=(benchmark_name,) if benchmark_name else (special_name,),
        extracted_memory=extracted_memory,
    )


def _classify_unresolved_reason(reason_text: str, tokens: set[str]) -> str | None:
    if not reason_text:
        return "empty_hardware_name"
    if reason_text == "tpu":
        return "input is a generic TPU reference"
    if tokens.intersection(MODEL_NAME_TOKENS) and not tokens.intersection(MODEL_NAME_HARDWARE_EXCLUSION_TOKENS):
        return MODEL_OR_FRAMEWORK_NAME_REASON
    if "unknown" in tokens or tokens.issubset(GENERIC_UNRESOLVED_TOKENS):
        return "input is generic or unknown"
    if "instance" in tokens or re.search(r"\b[a-z]\d+\.\d*xlarge\b", reason_text):
        return "input is an instance type, not a GPU model"
    if any(token in tokens for token in {"xeon", "cpu"}) or (
        any(token in tokens for token in {"core", "cores"})
        and "tensor" not in tokens
        and not any(token in tokens for token in {"v100", "p100", "a100", "h100", "k80", "k40", "t4"})
    ):
        return "input is not a GPU device"
    if re.fullmatch(r"\d{1,3}gb gpu", reason_text) or re.fullmatch(r"gpu \d{1,3}gb", reason_text):
        return "input describes capacity without a specific GPU model"
    if re.fullmatch(r"(nvidia|amd|apple|huawei|google)\s+gpu", reason_text):
        return "input names a vendor but not a specific GPU model"
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?\d{1,3}gb", reason_text) or re.fullmatch(
        r"\d{1,3}gb\s+nvidia", reason_text
    ):
        return "input describes vendor and capacity without a specific GPU model"
    if reason_text in {"rtx gpu", "gtx gpu", "nvidia rtx", "nvidia gtx"}:
        return "input names a GPU family without a specific model"
    return None


def _resolve_family_variant(
    *,
    normalized_text: str,
    extracted_memory: str,
    family_patterns: tuple[str, ...],
    explicit_memory_to_benchmark: dict[str, str],
    default_benchmark_name: str | None,
    default_normalized_name: str,
) -> str | None:
    if not any(re.fullmatch(pattern, normalized_text) for pattern in family_patterns):
        return None

    for memory_value, benchmark_name in explicit_memory_to_benchmark.items():
        if extracted_memory == memory_value or memory_value.lower() in normalized_text:
            return benchmark_name

    return default_benchmark_name or default_normalized_name


def _refine_family_match(
    raw_hardware_name: str,
    cleaned_name: str,
    normalized_name: str,
    benchmark_name: str | None,
    extracted_memory: str,
    match_status: str,
    match_method: str,
    normalization_reason: str,
    candidate_names: tuple[str, ...],
) -> HardwareResolution | None:
    normalized_family = standardize_hardware_text(normalized_name)
    refined_name = _resolve_family_variant(
        normalized_text=normalized_family,
        extracted_memory=extracted_memory,
        family_patterns=(r"nvidia tesla v100",),
        explicit_memory_to_benchmark={
            "32GB": "NVIDIA Tesla V100 PCIe 32 GB",
            "16GB": "NVIDIA Tesla V100 PCIe 16 GB",
        },
        default_benchmark_name="NVIDIA Tesla V100 PCIe 16 GB",
        default_normalized_name=benchmark_name or normalized_name,
    )
    if refined_name is None:
        refined_name = _resolve_family_variant(
            normalized_text=normalized_family,
            extracted_memory=extracted_memory,
            family_patterns=(r"nvidia tesla p100", r"nvidia p100"),
            explicit_memory_to_benchmark={
                "12GB": "NVIDIA Tesla P100 PCIe 12GB",
                "16GB": "NVIDIA Tesla P100 PCIe 16GB",
            },
            default_benchmark_name="NVIDIA Tesla P100 PCIe 16GB",
            default_normalized_name=benchmark_name or normalized_name,
        )

    if refined_name is None:
        return None

    return HardwareResolution(
        raw_hardware_name=raw_hardware_name,
        cleaned_hardware_name=cleaned_name,
        normalized_hardware_name=refined_name,
        benchmark_hardware_name=refined_name,
        match_status=match_status,
        match_method=match_method,
        normalization_reason="normalized NVIDIA special-case rule",
        candidate_names=(refined_name,),
        extracted_memory=extracted_memory,
    )


def resolve_hardware_name(raw_hardware_name: str, catalog: HardwareCatalog) -> HardwareResolution:
    cleaned_name = standardize_hardware_text(raw_hardware_name)
    match_text = _canonicalize_match_text(cleaned_name)
    extracted_memory = extract_input_memory(raw_hardware_name)

    if re.fullmatch(r"(?:nvidia\s+)?dgxa\s*100(?:\s+station)?(?:\s+\d{1,3}gb)?", cleaned_name) or re.fullmatch(
        r"(?:nvidia\s+)?dgx\s*a\s*100(?:\s+station)?(?:\s+\d{1,3}gb)?", cleaned_name
    ):
        benchmark_name = _lookup_benchmark_name("NVIDIA DGX A100", catalog)
        return HardwareResolution(
            raw_hardware_name=raw_hardware_name,
            cleaned_hardware_name=cleaned_name,
            normalized_hardware_name="NVIDIA DGX A100",
            benchmark_hardware_name=benchmark_name,
            match_status="rule_match",
            match_method="rule",
            normalization_reason="normalized NVIDIA special-case rule",
            candidate_names=(benchmark_name,) if benchmark_name else ("NVIDIA DGX A100",),
            extracted_memory=extracted_memory,
        )

    if cleaned_name == "titan x pascal" and "nvidia" not in raw_hardware_name.lower():
        return HardwareResolution(
            raw_hardware_name=raw_hardware_name,
            cleaned_hardware_name=cleaned_name,
            normalized_hardware_name="NVIDIA GeForce GTX TITAN X",
            benchmark_hardware_name=_lookup_benchmark_name("NVIDIA GeForce GTX TITAN X", catalog),
            match_status="rule_match",
            match_method="rule",
            normalization_reason="normalized NVIDIA special-case rule",
            candidate_names=("NVIDIA GeForce GTX TITAN X",),
            extracted_memory=extracted_memory,
        )

    match = _match_cleaned_hardware_name(raw_hardware_name, match_text, catalog)

    if match.match_status in {"exact_match", "alias_match"}:
        benchmark_name = match.benchmark_hardware_name
        if benchmark_name == "NVIDIA H20" and "sxm" not in cleaned_name and "pcie" not in cleaned_name and not extracted_memory:
            return HardwareResolution(
                raw_hardware_name=raw_hardware_name,
                cleaned_hardware_name=cleaned_name,
                normalized_hardware_name="",
                benchmark_hardware_name=None,
                match_status="unmatched",
                match_method="unmatched",
                normalization_reason="input names a GPU family without a specific model",
                candidate_names=match.candidate_names,
                extracted_memory=extracted_memory,
            )
        refined = _refine_family_match(
            raw_hardware_name=raw_hardware_name,
            cleaned_name=cleaned_name,
            normalized_name=benchmark_name or raw_hardware_name,
            benchmark_name=benchmark_name,
            extracted_memory=extracted_memory,
            match_status=match.match_status,
            match_method="exact" if match.match_status == "exact_match" else "alias",
            normalization_reason=match.match_status,
            candidate_names=match.candidate_names,
        )
        if refined:
            return refined
        return HardwareResolution(
            raw_hardware_name=raw_hardware_name,
            cleaned_hardware_name=cleaned_name,
            normalized_hardware_name=benchmark_name or raw_hardware_name,
            benchmark_hardware_name=benchmark_name,
            match_status=match.match_status,
            match_method="exact" if match.match_status == "exact_match" else "alias",
            normalization_reason=match.match_status,
            candidate_names=match.candidate_names,
            extracted_memory=extracted_memory,
        )

    normalized_text = match_text
    special_name = SPECIAL_CANONICAL_RULES.get(normalized_text)
    if special_name:
        benchmark_name = _lookup_benchmark_name(special_name, catalog) or _default_explicit_benchmark_name(
            normalized_hardware_name=special_name,
            cleaned_hardware_name=cleaned_name,
            extracted_memory=extracted_memory,
            catalog=catalog,
        )
        normalized_hardware_name = benchmark_name or special_name
        return HardwareResolution(
            raw_hardware_name=raw_hardware_name,
            cleaned_hardware_name=cleaned_name,
            normalized_hardware_name=normalized_hardware_name,
            benchmark_hardware_name=benchmark_name,
            match_status="rule_match",
            match_method="rule",
            normalization_reason="special_canonical_rule",
            candidate_names=(benchmark_name,) if benchmark_name else (special_name,),
            extracted_memory=extracted_memory,
        )

    def _build_named_resolution(normalized_hardware_name: str, normalization_reason: str) -> HardwareResolution:
        benchmark_name = _lookup_benchmark_name(normalized_hardware_name, catalog) or _default_explicit_benchmark_name(
            normalized_hardware_name=normalized_hardware_name,
            cleaned_hardware_name=normalized_text,
            extracted_memory=extracted_memory,
            catalog=catalog,
        )
        resolved_name = benchmark_name or normalized_hardware_name
        return HardwareResolution(
            raw_hardware_name=raw_hardware_name,
            cleaned_hardware_name=normalized_text,
            normalized_hardware_name=resolved_name,
            benchmark_hardware_name=benchmark_name,
            match_status="rule_match",
            match_method="rule",
            normalization_reason=normalization_reason,
            candidate_names=(benchmark_name,) if benchmark_name else (normalized_hardware_name,),
            extracted_memory=extracted_memory,
        )

    family_resolution = resolve_pre_unresolved_family_rules(
        raw_hardware_name=raw_hardware_name,
        normalized_text=normalized_text,
        catalog=catalog,
        extracted_memory=extracted_memory,
        build_rule_resolution=_build_rule_resolution,
        resolve_family_variant=_resolve_family_variant,
        build_named_resolution=_build_named_resolution,
    )
    if family_resolution is not None:
        return family_resolution
    reason_text = cleaned_name
    tokens = set(reason_text.split())
    reason = _classify_unresolved_reason(reason_text, tokens)
    if reason is not None:
        return HardwareResolution(
            raw_hardware_name=raw_hardware_name,
            cleaned_hardware_name=normalized_text,
            normalized_hardware_name="",
            benchmark_hardware_name=None,
            match_status="unmatched",
            match_method="unmatched",
            normalization_reason=reason,
            candidate_names=match.candidate_names,
            extracted_memory=extracted_memory,
        )

    def _build_unmatched_resolution(normalization_reason: str) -> HardwareResolution:
        return HardwareResolution(
            raw_hardware_name=raw_hardware_name,
            cleaned_hardware_name=normalized_text,
            normalized_hardware_name="",
            benchmark_hardware_name=None,
            match_status="unmatched",
            match_method="unmatched",
            normalization_reason=normalization_reason,
            candidate_names=match.candidate_names,
            extracted_memory=extracted_memory,
        )

    family_resolution = resolve_post_unresolved_family_rules(
        raw_hardware_name=raw_hardware_name,
        normalized_text=normalized_text,
        catalog=catalog,
        extracted_memory=extracted_memory,
        build_rule_resolution=_build_rule_resolution,
        build_unmatched_resolution=_build_unmatched_resolution,
    )
    if family_resolution is not None:
        return family_resolution
    return _build_unmatched_resolution("no normalization rule matched")

