from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from openpyxl import Workbook

from computing_resource.extraction.gpu_catalog import (
    HardwareCatalog,
    load_hardware_catalog,
    resolve_hardware_name,
)


EXPORT_HEADERS = [
    "source_file",
    "gpu_name",
    "gpu_num",
    "benchmark_gpu_name",
    "gpu_vendor",
    "benchmark_generation",
    "benchmark_family",
    "benchmark_release_date",
    "benchmark_release_price_usd",
    "benchmark_tdp_w",
    "benchmark_memory_bytes",
    "benchmark_memory_bandwidth_bytes_per_s",
    "benchmark_fp32_flops",
    "benchmark_tf32_flops",
    "benchmark_tensor_fp16_bf16_flops",
    "benchmark_fp8_flops",
    "benchmark_max_performance",
    "benchmark_energy_efficiency",
    "benchmark_process_size_nm",
    "normalize_status",
    "normalize_reason",
]


@dataclass(frozen=True)
class BenchmarkColumns:
    name: str | None
    vendor: Any
    generation: Any
    family: Any
    release_date: Any
    release_price_usd: Any
    tdp_w: Any
    memory_bytes: Any
    memory_bandwidth_bytes_per_s: Any
    fp32_flops: Any
    tf32_flops: Any
    tensor_fp16_bf16_flops: Any
    fp8_flops: Any
    max_performance: Any
    energy_efficiency: Any
    process_size_nm: Any


def _clean_benchmark_value(value: Any) -> Any:
    if value in ("", None):
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.lower() == "none":
            return None
        try:
            return datetime.strptime(normalized, "%Y-%m-%d")
        except ValueError:
            pass
        if normalized.isdigit() or (
            normalized.startswith("-") and normalized[1:].isdigit()
        ):
            try:
                return int(normalized)
            except ValueError:
                return normalized
        try:
            numeric = float(normalized)
        except ValueError:
            return normalized
        return int(numeric) if numeric.is_integer() else numeric
    return value


def _normalize_status(match_status: str | None) -> str:
    if match_status in {"unmatched", "ambiguous"}:
        return "unresolved"
    return "normalized"


def _benchmark_columns_for_row(row: dict[str, Any] | None) -> BenchmarkColumns:
    row = row or {}
    return BenchmarkColumns(
        name=row.get("Hardware name"),
        vendor=_clean_benchmark_value(row.get("Manufacturer")),
        generation=_clean_benchmark_value(row.get("Generation")),
        family=_clean_benchmark_value(row.get("Family") or row.get("product category")),
        release_date=_clean_benchmark_value(row.get("Release date")),
        release_price_usd=_clean_benchmark_value(row.get("Release price (USD)")),
        tdp_w=_clean_benchmark_value(row.get("TDP (W)")),
        memory_bytes=_clean_benchmark_value(row.get("Memory (bytes)")),
        memory_bandwidth_bytes_per_s=_clean_benchmark_value(row.get("Memory bandwidth (byte/s)")),
        fp32_flops=_clean_benchmark_value(row.get("FP32 (single precision) performance (FLOP/s)")),
        tf32_flops=_clean_benchmark_value(row.get("TF32 (TensorFloat-32) performance (FLOP/s)")),
        tensor_fp16_bf16_flops=_clean_benchmark_value(
            row.get("Tensor-FP16/BF16 performance (FLOP/s)")
        ),
        fp8_flops=_clean_benchmark_value(row.get("FP8 performance (FLOP/s)")),
        max_performance=_clean_benchmark_value(row.get("Max performance")),
        energy_efficiency=_clean_benchmark_value(row.get("Energy efficiency")),
        process_size_nm=_clean_benchmark_value(row.get("Process size (nm)")),
    )


def _legacy_normalized_rows(doc: dict[str, Any], catalog: HardwareCatalog) -> list[dict[str, Any]]:
    pred_resources = doc.get("pred_resources")
    if isinstance(pred_resources, dict):
        gpu_rows = pred_resources.get("gpu")
    else:
        gpu_rows = None
    if not isinstance(gpu_rows, list):
        pred_result = doc.get("pred_result")
        gpu_rows = pred_result.get("pred") if isinstance(pred_result, dict) else []
    if not isinstance(gpu_rows, list):
        return []

    normalized_rows: list[dict[str, Any]] = []
    for item in gpu_rows:
        if not isinstance(item, dict):
            continue
        raw_name = item.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        resolution = resolve_hardware_name(raw_name, catalog)
        normalized_rows.append(
            {
                "raw_hardware_name": raw_name,
                "count": item.get("num"),
                "benchmark_hardware_name": resolution.benchmark_hardware_name,
                "match_status": resolution.match_status,
                "normalization_reason": resolution.normalization_reason,
            }
        )
    return normalized_rows


def _iter_normalized_rows(input_dir: Path, catalog: HardwareCatalog) -> Iterable[tuple[str, dict[str, Any]]]:
    for path in sorted(input_dir.glob("*_gpu.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("normalized_extractions")
        if not isinstance(rows, list):
            rows = _legacy_normalized_rows(data, catalog)
        source_file = (
            data.get("pred_resources", {}).get("source_file")
            if isinstance(data.get("pred_resources"), dict)
            else None
        )
        source_file = source_file if isinstance(source_file, str) and source_file.strip() else path.name
        for row in rows:
            yield source_file, row


def build_export_rows(input_dir: str | Path, catalog: HardwareCatalog) -> list[dict[str, Any]]:
    source_dir = Path(input_dir)
    export_rows: list[dict[str, Any]] = []

    for source_file, row in _iter_normalized_rows(source_dir, catalog):
        benchmark_name = row.get("benchmark_hardware_name")
        benchmark_row = catalog.row_map.get(benchmark_name) if benchmark_name else None
        benchmark = _benchmark_columns_for_row(benchmark_row)

        export_rows.append(
            {
                "source_file": source_file,
                "gpu_name": row.get("raw_hardware_name"),
                "gpu_num": row.get("count"),
                "benchmark_gpu_name": benchmark.name,
                "gpu_vendor": benchmark.vendor,
                "benchmark_generation": benchmark.generation,
                "benchmark_family": benchmark.family,
                "benchmark_release_date": benchmark.release_date,
                "benchmark_release_price_usd": benchmark.release_price_usd,
                "benchmark_tdp_w": benchmark.tdp_w,
                "benchmark_memory_bytes": benchmark.memory_bytes,
                "benchmark_memory_bandwidth_bytes_per_s": benchmark.memory_bandwidth_bytes_per_s,
                "benchmark_fp32_flops": benchmark.fp32_flops,
                "benchmark_tf32_flops": benchmark.tf32_flops,
                "benchmark_tensor_fp16_bf16_flops": benchmark.tensor_fp16_bf16_flops,
                "benchmark_fp8_flops": benchmark.fp8_flops,
                "benchmark_max_performance": benchmark.max_performance,
                "benchmark_energy_efficiency": benchmark.energy_efficiency,
                "benchmark_process_size_nm": benchmark.process_size_nm,
                "normalize_status": _normalize_status(row.get("match_status")),
                "normalize_reason": row.get("normalization_reason"),
            }
        )

    return export_rows


def export_normalized_gpu_excel(
    input_dir: str | Path,
    output_path: str | Path,
    catalog_path: str | Path,
) -> Path:
    catalog = load_hardware_catalog(catalog_path)
    rows = build_export_rows(input_dir, catalog)

    workbook = Workbook()
    worksheet = workbook.active
    output = Path(output_path)
    worksheet.title = output.stem[:31]
    worksheet.append(EXPORT_HEADERS)
    for row in rows:
        worksheet.append([row.get(header) for header in EXPORT_HEADERS])

    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    return output
