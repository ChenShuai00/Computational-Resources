from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

from computing_resource.extraction.gpu_name_rules import (
    TPU_V3_DEFAULT_PATTERNS,
    TPU_VERSION_TOKEN_PATTERN,
    TPU_VERSIONED_PATTERNS,
)


if TYPE_CHECKING:
    from computing_resource.extraction.gpu_catalog import HardwareCatalog, HardwareResolution


class RuleResolutionBuilder(Protocol):
    def __call__(
        self,
        raw_hardware_name: str,
        normalized_text: str,
        normalized_hardware_name: str,
        catalog: HardwareCatalog,
        normalization_reason: str,
        extracted_memory: str,
    ) -> HardwareResolution: ...


class FamilyVariantResolver(Protocol):
    def __call__(
        self,
        *,
        normalized_text: str,
        extracted_memory: str,
        family_patterns: tuple[str, ...],
        explicit_memory_to_benchmark: dict[str, str],
        default_benchmark_name: str | None,
        default_normalized_name: str,
    ) -> str | None: ...


class UnmatchedResolutionBuilder(Protocol):
    def __call__(self, normalization_reason: str) -> HardwareResolution: ...


class NamedResolutionBuilder(Protocol):
    def __call__(self, normalized_hardware_name: str, normalization_reason: str) -> HardwareResolution: ...


def _matches_any_pattern(normalized_text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.fullmatch(pattern, normalized_text) for pattern in patterns)


def _resolve_h_accelerator_series(
    *,
    raw_hardware_name: str,
    normalized_text: str,
    catalog: HardwareCatalog,
    extracted_memory: str,
    build_rule_resolution: RuleResolutionBuilder,
) -> HardwareResolution | None:
    tokens = set(normalized_text.split())
    family = None
    if "gh200" in tokens:
        family = "gh200"
    elif "h200" in tokens:
        family = "h200"
    elif "h20" in tokens:
        family = "h20"
    elif "h800" in tokens:
        family = "h800"
    elif "h100" in tokens or normalized_text == "hopper" or (
        "hopper" in tokens and ("nvidia" in tokens or "tesla" in tokens or "gpu" in tokens)
    ):
        family = "h100"

    if family is None:
        return None

    recognized_tokens = {
        "nvidia",
        "tesla",
        "gpu",
        "gpus",
        "rtx",
        "geforce",
        "hopper",
        "pcie",
        "sxm",
        "sxm5",
        "nvl",
        "nvlink",
        "gh200",
        "h200",
        "h20",
        "h100",
        "h800",
    }
    for token in tokens:
        if token in recognized_tokens:
            continue
        if re.fullmatch(r"\d{1,3}gb", token):
            continue
        return None

    if family == "gh200":
        benchmark = "NVIDIA GH200"
    elif family == "h200":
        benchmark = "NVIDIA H200 SXM5" if "sxm" in tokens or "sxm5" in tokens else "NVIDIA H200 PCIe"
    elif family == "h20":
        if extracted_memory == "141GB":
            benchmark = "NVIDIA H20 SXM5 141GB"
        elif "sxm" in tokens or "sxm5" in tokens or extracted_memory == "96GB":
            benchmark = "NVIDIA H20 SXM5 96GB"
        else:
            return None
    elif family == "h800":
        if "nvl" in tokens or "nvlink" in tokens or extracted_memory in {"94GB", "95GB", "96GB"}:
            benchmark = "NVIDIA H800 NVL"
        elif "sxm" in tokens or "sxm5" in tokens:
            benchmark = "NVIDIA H800 SXM5"
        else:
            benchmark = "NVIDIA H800 PCIe"
    else:
        if "nvl" in tokens or "nvlink" in tokens or extracted_memory in {"94GB", "95GB", "96GB"}:
            benchmark = "NVIDIA H100 NVL"
        elif "sxm" in tokens or "sxm5" in tokens:
            benchmark = "NVIDIA H100 SXM5 80GB"
        else:
            benchmark = "NVIDIA H100 PCIe"

    return build_rule_resolution(
        raw_hardware_name,
        normalized_text,
        benchmark,
        catalog,
        "normalized NVIDIA accelerator rule",
        extracted_memory,
    )


def _build_versioned_tpu_resolution(
    *,
    raw_hardware_name: str,
    normalized_text: str,
    catalog: HardwareCatalog,
    extracted_memory: str,
    build_rule_resolution: RuleResolutionBuilder,
    patterns: tuple[str, ...],
) -> HardwareResolution | None:
    if not _matches_any_pattern(normalized_text, patterns):
        return None

    version_match = re.search(TPU_VERSION_TOKEN_PATTERN, normalized_text)
    if version_match is None:
        return None

    return build_rule_resolution(
        raw_hardware_name,
        normalized_text,
        f"Google TPU {version_match.group(0)}",
        catalog,
        "normalized TPU special-case rule",
        extracted_memory,
    )


def _resolve_nvidia_accelerator_pre_unresolved(
    *,
    raw_hardware_name: str,
    normalized_text: str,
    catalog: HardwareCatalog,
    extracted_memory: str,
    build_rule_resolution: RuleResolutionBuilder,
    resolve_family_variant: FamilyVariantResolver,
    build_named_resolution: NamedResolutionBuilder,
) -> HardwareResolution | None:
    if re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?rtx\s+a100(?:\s+\d{1,3}gb)?",
        normalized_text,
    ):
        benchmark = "NVIDIA A100 PCIe 80GB" if "80gb" in normalized_text else "NVIDIA A100 PCIe 40GB"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )

    if re.fullmatch(
        r"(?:google\s+)?(?:nvidia\s+)?(?:gpu\s+)?(?:tesla\s+)?a100(?:\s+(?:pcie|sxm4?|sxm))?(?:\s+\d{1,3}gb)?(?:\s+(?:pcie|sxm4?|sxm))?(?:\s+(?:nvidia|gpu|google))?",
        normalized_text,
    ) or re.fullmatch(
        r"\d{1,3}gb\s+(?:google\s+)?(?:nvidia\s+)?(?:gpu\s+)?(?:tesla\s+)?a100(?:\s+(?:pcie|sxm4?|sxm))?",
        normalized_text,
    ):
        if "pcie" in normalized_text:
            benchmark = "NVIDIA A100 PCIe 80GB" if "80gb" in normalized_text else "NVIDIA A100 PCIe 40GB"
        elif "sxm" in normalized_text:
            benchmark = "NVIDIA A100 SXM4 80 GB" if "80gb" in normalized_text else "NVIDIA A100 SXM4 40 GB"
        elif "80gb" in normalized_text:
            benchmark = "NVIDIA A100 80GB"
        else:
            benchmark = "NVIDIA A100 40GB"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )

    if re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:tesla\s+)?a800(?:\s+(?:pcie|sxm4?|sxm))?(?:\s+\d{1,3}gb)?(?:\s+(?:pcie|sxm4?|sxm))?(?:\s+(?:nvidia|gpu))?",
        normalized_text,
    ) or re.fullmatch(
        r"\d{1,3}gb\s+(?:nvidia\s+)?(?:gpu\s+)?(?:tesla\s+)?a800(?:\s+(?:pcie|sxm4?|sxm))?",
        normalized_text,
    ):
        if "sxm" in normalized_text:
            benchmark = "NVIDIA A800 SXM"
        elif "80gb" in normalized_text:
            benchmark = "NVIDIA A800 PCIe 80 GB"
        else:
            benchmark = "NVIDIA A800 PCIe 40 GB"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )

    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:tesla\s+)?a30(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA A30",
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )

    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:tesla\s+)?a10g(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA A10G",
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )

    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:tesla\s+)?l40s?(?:\s+\d{1,3}gb)?", normalized_text):
        benchmark = "NVIDIA L40S" if "l40s" in normalized_text else "NVIDIA L40"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )

    if re.fullmatch(r"(?:nvidia\s+)?(?:tesla\s+)?l4(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA L4",
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )

    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:tesla\s+)?l20(?:\s+(?:pcie|pcle))?(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA L20",
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )

    if re.fullmatch(r"(?:nvidia\s+)?geforce\s+a100(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA A100 40GB",
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )

    if re.fullmatch(r"nvidia\s+ampere(?:\s+gpu)?(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA A100 40GB",
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )

    if re.fullmatch(r"(?:nvidia\s+)?a100\s+gpu\s+\d{1,3}gb", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA A100 40GB",
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )

    if re.fullmatch(r"(?:nvidia\s+)?a100\s+sxm(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA A100 40GB",
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )

    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:tesla\s+)?a40(?:\s+gpu)?(?:\s+pcie)?(?:\s+\d{1,3}gb)?(?:\s+gpu)?", normalized_text) or re.fullmatch(
        r"\d{1,3}gb\s+(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:tesla\s+)?a40(?:\s+gpu)?(?:\s+pcie)?(?:\s+gpu)?",
        normalized_text,
    ):
        benchmark = "NVIDIA A40 PCIe"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )

    family_resolution = resolve_family_variant(
        normalized_text=normalized_text,
        extracted_memory=extracted_memory,
        family_patterns=(
            r"(?:nvidia\s+)?dgx[- ]?\d+\s+(?:tesla\s+)?v100(?:\s+\d{1,3}gb)?",
            r"(?:nvidia\s+)?(?:tesla\s+)?v100\s+dgx[- ]?\d+(?:\s+\d{1,3}gb)?",
            r"(?:nvidia\s+)?(?:tesla\s+)?v100\s+dgx\s*\d+(?:\s+\d{1,3}gb)?",
        ),
        explicit_memory_to_benchmark={
            "32GB": "NVIDIA Tesla V100 DGXS 32 GB",
            "16GB": "NVIDIA Tesla V100 DGXS 16 GB",
        },
        default_benchmark_name="NVIDIA Tesla V100 DGXS 16 GB",
        default_normalized_name="NVIDIA Tesla V100 DGXS 16 GB",
    )
    if family_resolution is not None:
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            family_resolution,
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )

    family_resolution = resolve_family_variant(
        normalized_text=normalized_text,
        extracted_memory=extracted_memory,
        family_patterns=(
            r"(?:nvidia\s+)?(?:gpu\s+)?(?:tesla\s+)?v100(?:\s+gpu)?(?:\s+\d{1,3}gb)?",
            r"(?:nvidia\s+)?(?:rtx\s+)?v100(?:\s+\d{1,3}gb)?",
            r"(?:nvidia\s+)?v100\s+nvlink(?:\s+\d{1,3}gb)?",
            r"\d{1,3}gb\s+(?:nvidia\s+)?(?:gpu\s+)?(?:tesla\s+)?v100(?:\s+gpu)?",
            r"v100\s+nvidia(?:\s+gpu)?",
            r"nvidia\s+gv100\s+tesla\s+v100(?:\s+gpu)?",
            r"(?:nvidia\s+)?volta\s+v100(?:\s+gpu)?(?:\s+\d{1,3}gb)?",
            r"\d{1,3}gb\s+(?:nvidia\s+)?volta\s+v100(?:\s+gpu)?",
            r"v100\s+volta(?:\s+gpu)?(?:\s+\d{1,3}gb)?",
            r"titan\s+v100(?:\s+gpu)?",
            r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:gtx\s*)?titan\s+v100(?:\s+\d{1,3}gb)?",
            r"\d{1,3}gb\s+(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:gtx\s*)?titan\s+v100",
        ),
        explicit_memory_to_benchmark={
            "32GB": "NVIDIA Tesla V100 PCIe 32 GB",
            "16GB": "NVIDIA Tesla V100 PCIe 16 GB",
        },
        default_benchmark_name="NVIDIA Tesla V100 PCIe 16 GB",
        default_normalized_name="NVIDIA Tesla V100 PCIe 16 GB",
    )
    if family_resolution is not None:
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            family_resolution,
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )

    if re.fullmatch(r"(?:nvidia\s+)?(?:rtx\s+)?v100(?:\s+\d{1,3}gb)?", normalized_text):
        benchmark = "NVIDIA Tesla V100 PCIe 32 GB" if "32gb" in normalized_text else "NVIDIA Tesla V100 PCIe 16 GB"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )

    if re.fullmatch(r"(?:nvidia\s+)?volta(?:\s+\d{1,3}gb)?", normalized_text):
        benchmark = "NVIDIA Tesla V100 PCIe 32 GB" if "32gb" in normalized_text else "NVIDIA Tesla V100 PCIe 16 GB"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )

    family_resolution = resolve_family_variant(
        normalized_text=normalized_text,
        extracted_memory=extracted_memory,
        family_patterns=(
            r"(?:nvidia\s+)?(?:tesla\s+)?p100(?:\s+gpu)?(?:\s+\d{1,3}gb)?",
            r"\d{1,3}gb\s+(?:nvidia\s+)?(?:tesla\s+)?p100(?:\s+gpu)?",
        ),
        explicit_memory_to_benchmark={
            "12GB": "NVIDIA Tesla P100 PCIe 12GB",
            "16GB": "NVIDIA Tesla P100 PCIe 16GB",
        },
        default_benchmark_name="NVIDIA Tesla P100 PCIe 16GB",
        default_normalized_name="NVIDIA Tesla P100 PCIe 16GB",
    )
    if family_resolution is not None:
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            family_resolution,
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )

    if re.fullmatch(r"(?:nvidia\s+)?(?:quadro\s+)?gp100(?:\s+\d{2,3}gb)?", normalized_text):
        return build_named_resolution("NVIDIA Quadro GP100", "normalized NVIDIA special-case rule")

    if re.fullmatch(r"(?:nvidia\s+)?dgx\s*1(?:\s+station)?", normalized_text):
        return build_named_resolution("NVIDIA DGX-1 Station", "normalized NVIDIA special-case rule")

    if re.fullmatch(r"(?:nvidia\s+)?dgx\s*2(?:\s+station)?", normalized_text):
        return build_named_resolution("NVIDIA DGX-2 Station", "normalized NVIDIA special-case rule")

    return None


def _resolve_nvidia_accelerator_post_unresolved(
    *,
    raw_hardware_name: str,
    normalized_text: str,
    catalog: HardwareCatalog,
    extracted_memory: str,
    build_rule_resolution: RuleResolutionBuilder,
) -> HardwareResolution | None:
    h_series_resolution = _resolve_h_accelerator_series(
        raw_hardware_name=raw_hardware_name,
        normalized_text=normalized_text,
        catalog=catalog,
        extracted_memory=extracted_memory,
        build_rule_resolution=build_rule_resolution,
    )
    if h_series_resolution is not None:
        return h_series_resolution

    if normalized_text == "nvidia gh200" or re.fullmatch(r"(?:nvidia\s+)?gh200(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GH200",
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?hopper(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA H100 PCIe",
            catalog,
            "hopper_default_rule",
            extracted_memory,
        )
    if re.fullmatch(
        r"(?:nvidia\s+)?h200(?:\s+(?:sxm5?|pcie))?(?:\s+\d{1,3}gb)?(?:\s+(?:sxm5?|pcie))?",
        normalized_text,
    ) or re.fullmatch(
        r"\d{1,3}gb\s+(?:nvidia\s+)?h200(?:\s+(?:sxm5?|pcie))?(?:\s+(?:sxm5?|pcie))?",
        normalized_text,
    ):
        benchmark = "NVIDIA H200 SXM5" if "sxm" in normalized_text else "NVIDIA H200 PCIe"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )
    if re.fullmatch(
        r"(?:nvidia\s+)?(?:tesla\s+)?(?:rtx\s+)?h800(?:\s+(?:sxm5?|nvl|pcie))?(?:\s+\d{1,3}gb)?(?:\s+(?:sxm5?|nvl|pcie))?",
        normalized_text,
    ) or re.fullmatch(
        r"\d{1,3}gb\s+(?:nvidia\s+)?(?:tesla\s+)?(?:rtx\s+)?h800(?:\s+(?:sxm5?|nvl|pcie))?(?:\s+(?:sxm5?|nvl|pcie))?",
        normalized_text,
    ):
        if "sxm" in normalized_text:
            benchmark = "NVIDIA H800 SXM5"
        elif "nvl" in normalized_text:
            benchmark = "NVIDIA H800 NVL"
        else:
            benchmark = "NVIDIA H800 PCIe"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?h100(?:\s+(?:sxm5?|nvl|pcie))?(?:\s+\d{1,3}gb)?(?:\s+(?:sxm5?|nvl|pcie))?", normalized_text) or re.fullmatch(
        r"\d{1,3}gb\s+(?:nvidia\s+)?h100(?:\s+(?:sxm5?|nvl|pcie))?(?:\s+(?:sxm5?|nvl|pcie))?",
        normalized_text,
    ):
        if "sxm" in normalized_text:
            benchmark = "NVIDIA H100 SXM5 80GB"
        elif "nvl" in normalized_text:
            benchmark = "NVIDIA H100 NVL"
        else:
            benchmark = "NVIDIA H100 PCIe"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?a10(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA A10",
            catalog,
            "nvidia_accelerator_family_rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?h100(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA H100 PCIe",
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:tesla\s+)?t4(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Tesla T4",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:tesla\s+)?v100\s+pcie\s+32gb", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Tesla V100 PCIe 32 GB",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?v100\s+sxm2\s+16gb", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Tesla V100 SXM2 16 GB",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:tesla\s+)?m40(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA M40",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    return None


def _resolve_nvidia_geforce_family(
    *,
    raw_hardware_name: str,
    normalized_text: str,
    catalog: HardwareCatalog,
    extracted_memory: str,
    build_rule_resolution: RuleResolutionBuilder,
) -> HardwareResolution | None:
    if re.fullmatch(r"(?:\d{2,3}gb\s+)?4090\s+(?:rtx|gpu)(?:\s+\d{2,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?4090(?:\s+\d{1,3}gb)?",
        normalized_text,
    ) or re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?4090(?:\s+\d{1,3}gb)?",
        normalized_text,
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce RTX 4090",
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?4080(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?4080(?:\s+\d{1,3}gb)?",
        normalized_text,
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce RTX 4080",
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?4070(?:\s*ti)?(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?4070(?:\s*ti)?(?:\s+\d{1,3}gb)?",
        normalized_text,
    ):
        benchmark = "NVIDIA GeForce RTX 4070 Ti" if "ti" in normalized_text else "NVIDIA GeForce RTX 4070"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?5000(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Quadro RTX 5000",
            catalog,
            "normalized RTX family rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:tesla\s+)?(?:rtx\s+)?8000(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Quadro RTX 8000",
            catalog,
            "normalized RTX family rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?6000\s+ada(?:\s+generation)?(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA RTX 6000 Ada Generation",
            catalog,
            "normalized RTX family rule",
            extracted_memory,
        )
    if re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?a6000\s+ada(?:\s+generation)?(?:\s+\d{1,3}gb)?",
        normalized_text,
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA RTX 6000 Ada Generation",
            catalog,
            "normalized RTX family rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?3090(?:\s*ti)?(?:\s+\d{2,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?3090(?:\s*ti)?(?:\s+\d{2,3}gb)?",
        normalized_text,
    ) or re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?ti\s+3090(?:\s+\d{2,3}gb)?", normalized_text):
        benchmark = "NVIDIA GeForce RTX 3090 Ti" if "ti" in normalized_text else "NVIDIA GeForce RTX 3090"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?3080(?:\s*ti)?(?:\s+\d{1,3}gb)?(?:\s+nvidia)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:gtx\s+)?3080(?:\s*ti)?(?:\s+\d{1,3}gb)?(?:\s+nvidia)?",
        normalized_text,
    ) or re.fullmatch(r"nvidia\s+gtx\s*3080ti", normalized_text):
        benchmark = "NVIDIA GeForce RTX 3080 Ti" if "ti" in normalized_text else "NVIDIA GeForce RTX 3080"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?3070(?:\s*ti)?(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?3070(?:\s*ti)?(?:\s+\d{1,3}gb)?",
        normalized_text,
    ):
        benchmark = "NVIDIA GeForce RTX 3070 Ti" if "ti" in normalized_text else "NVIDIA GeForce RTX 3070"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?2080(?:\s*ti|\s*s)?(?:\s+\d{2,3}gb)?",
        normalized_text,
    ) or re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?2080(?:\s*ti|\s*s)?(?:\s+\d{2,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?2080\s+rtx(?:\s+\d{2,3}gb)?",
        normalized_text,
    ) or re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?2080rtx(?:\s+\d{2,3}gb)?", normalized_text):
        if "ti" in normalized_text:
            benchmark = "NVIDIA GeForce RTX 2080 Ti"
        elif " s" in normalized_text or normalized_text.endswith("2080s"):
            benchmark = "NVIDIA GeForce RTX 2080 Super"
        else:
            benchmark = "NVIDIA GeForce RTX 2080"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:\d{2,3}gb\s+)?3090\s+(?:rtx|gpu)(?:\s+\d{2,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:geforce\s+)?(?:rtx\s+)?3090(?:\s*ti)?(?:\s+\d{2,3}gb)?", normalized_text
    ) or re.fullmatch(
        r"(?:nvidia\s+)?(?:geforce\s+)?rtx3090(?:ti)?(?:\s+\d{2,3}gb)?", normalized_text
    ) or re.fullmatch(r"\d{2,3}gb\s+(?:nvidia\s+)?(?:geforce\s+)?(?:rtx\s+)?3090(?:\s*ti)?", normalized_text):
        benchmark = "NVIDIA GeForce RTX 3090 Ti" if "ti" in normalized_text else "NVIDIA GeForce RTX 3090"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?gtx\s*3090(?:\s+\d{2,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?gtx3090(?:\s+\d{2,3}gb)?", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce RTX 3090",
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?gtx\s*2080(?:\s+\d{2,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?gtx2080(?:\s+\d{2,3}gb)?", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce RTX 2080",
            catalog,
            "normalized GeForce GTX-to-RTX correction rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?2070(?:\s+super)?(?:\s+\d{1,3}(?:gb|g))?", normalized_text):
        benchmark = "NVIDIA GeForce RTX 2070 SUPER" if "super" in normalized_text else "NVIDIA GeForce RTX 2070"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?2800(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce RTX 2080",
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?gtx\s*1080(?:\s*ti)?(?:\s+\d{2,3}gb)?", normalized_text) or re.fullmatch(
        r"\d{2,3}gb\s+(?:nvidia\s+)?(?:geforce\s+)?gtx\s*1080(?:\s*ti)?", normalized_text
    ):
        benchmark = "NVIDIA GeForce GTX 1080 Ti" if "ti" in normalized_text else "NVIDIA GeForce GTX 1080"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized GeForce GTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s*)?1080(?:\s*ti)?(?:\s+\d{2,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s*)?1080ti(?:\s+\d{2,3}gb)?", normalized_text
    ):
        benchmark = "NVIDIA GeForce GTX 1080 Ti" if "ti" in normalized_text else "NVIDIA GeForce GTX 1080"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized GeForce GTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:geforce\s+)?gtx\s*980(?:\s*ti)?(?:\s+\d{2,3}gb)?", normalized_text) or re.fullmatch(
        r"\d{2,3}gb\s+(?:nvidia\s+)?(?:geforce\s+)?gtx\s*980(?:\s*ti)?", normalized_text
    ):
        benchmark = "NVIDIA GeForce GTX 980 Ti" if "ti" in normalized_text else "NVIDIA GeForce GTX 980"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized GeForce GTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s*)?a6000(?:\s+\d{2,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA RTX A6000",
            catalog,
            "normalized RTX family rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s*)?ax6000(?:\s+\d{2,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA RTX A6000",
            catalog,
            "normalized RTX family rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:quadro\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s*)?a5000(?:\s+gpu)?(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:quadro\s+)?(?:gpu\s+)?(?:geforce\s+)?a5000(?:\s+gpu)?(?:\s+\d{1,3}gb)?",
        normalized_text,
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA RTX A5000",
            catalog,
            "normalized RTX family rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s*)?a5500(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA RTX A5500",
            catalog,
            "normalized RTX family rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s*)?a4500(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA RTX A4500",
            catalog,
            "normalized RTX family rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?3080(?:\s*ti)?(?:\s+gpu)?(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:gtx\s+)?3080(?:\s*ti)?(?:\s+gpu)?(?:\s+\d{1,3}gb)?",
        normalized_text,
    ) or re.fullmatch(r"nvidia\s+gtx\s*3080ti", normalized_text):
        benchmark = "NVIDIA GeForce RTX 3080 Ti" if "ti" in normalized_text else "NVIDIA GeForce RTX 3080"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:rtx\s*)?3060(?:\s+\d{1,3}(?:gb|g))?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce RTX 3060",
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s+)?1650(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce GTX 1650",
            catalog,
            "normalized GeForce GTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:gtx\s+)?k80(?:\s+\d{2,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Tesla K80",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?p40(?:\s+\d{2,3}gb)?", normalized_text) or re.fullmatch(
        r"p40\s+nvidia", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Tesla P40",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:quadro\s+)?gp100(?:\s+\d{2,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Quadro GP100",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:tesla\s+)?v100s(?:\s+\d{2,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Tesla V100S PCIe 32 GB",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:tesla\s+)?v100(?:\s+sxm2|\s+pcie|\s+sxm|[- ]sxm2|[- ]pcie|[- ]sxm)?(?:\s+\d{2,3}gb)?(?:\s+ls)?", normalized_text):
        if "sxm2" in normalized_text:
            if "32gb" in normalized_text:
                benchmark = "NVIDIA Tesla V100 SXM2 32 GB"
            else:
                benchmark = "NVIDIA Tesla V100 SXM2 16 GB"
        elif "pcie" in normalized_text:
            benchmark = "NVIDIA Tesla V100 PCIe 32 GB" if "32gb" in normalized_text else "NVIDIA Tesla V100 PCIe 16 GB"
        elif "sxm" in normalized_text:
            benchmark = "NVIDIA Tesla V100 SXM2 32 GB" if "32gb" in normalized_text else "NVIDIA Tesla V100 SXM2 16 GB"
        else:
            benchmark = "NVIDIA Tesla V100 PCIe 32 GB" if "32gb" in normalized_text else "NVIDIA Tesla V100 PCIe 16 GB"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            benchmark,
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:tesla\s+)?p100(?:\s+sxm2|\s+pcie|[- ]sxm2|[- ]pcie)?(?:\s+\d{2,3}gb)?(?:\s+ls)?", normalized_text):
        if "sxm2" in normalized_text:
            special_name = "NVIDIA Tesla P100 SXM2"
        elif "pcie" in normalized_text:
            special_name = "NVIDIA Tesla P100 PCIe 12GB" if "12gb" in normalized_text else "NVIDIA Tesla P100 PCIe 16GB"
        else:
            special_name = "NVIDIA Tesla P100 PCIe 16GB"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            special_name,
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:quadro\s+)?p1000(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Quadro P1000",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    return None


def _resolve_nvidia_titan_family(
    *,
    raw_hardware_name: str,
    normalized_text: str,
    catalog: HardwareCatalog,
    extracted_memory: str,
    build_rule_resolution: RuleResolutionBuilder,
) -> HardwareResolution | None:
    if normalized_text in {"nvidia titan", "nvidia geforce titan", "titan"}:
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce GTX TITAN",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?titan\s+rtx(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA TITAN RTX",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?rtx\s+titan(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"titan\s+\d{1,3}gb\s+rtx", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA TITAN RTX",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?titan\s+v(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Titan V",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if (
        "nvidia" in raw_hardware_name.lower()
        and re.fullmatch(r"(?:nvidia\s+)?(?:geforce\s+)?(?:gtx\s+)?titan\s+x\s+pascal(?:\s+\d{1,3}gb)?", normalized_text)
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA TITAN X Pascal",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:geforce\s+)?(?:gtx\s+)?titan\s+x\s+maxwell(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce GTX TITAN X",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?titan\s+xp(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?titan\s+xp(?:\s+\d{1,3}\s+\d{3}m)?", normalized_text
    ) or re.fullmatch(
        r"nvidia gp102 \[titan xp\]", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA TITAN Xp",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:geforce\s+)?(?:gtx\s+)?titan\s+x(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:geforce\s+)?titan\s+x\s+pascal", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce GTX TITAN X",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s*)?titan(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:rtx\s*)?titan\s+rtx(?:\s+\d{1,3}gb)?", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA TITAN RTX",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    return None


def _resolve_tpu_family(
    *,
    raw_hardware_name: str,
    normalized_text: str,
    catalog: HardwareCatalog,
    extracted_memory: str,
    build_rule_resolution: RuleResolutionBuilder,
) -> HardwareResolution | None:
    family_resolution = _build_versioned_tpu_resolution(
        raw_hardware_name=raw_hardware_name,
        normalized_text=normalized_text,
        catalog=catalog,
        extracted_memory=extracted_memory,
        build_rule_resolution=build_rule_resolution,
        patterns=TPU_VERSIONED_PATTERNS,
    )
    if family_resolution is not None:
        return family_resolution

    if _matches_any_pattern(normalized_text, TPU_V3_DEFAULT_PATTERNS):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "Google TPU v3",
            catalog,
            "tpu_v3_default_rule",
            extracted_memory,
        )
    return None


def _resolve_other_vendor_family(
    *,
    raw_hardware_name: str,
    normalized_text: str,
    catalog: HardwareCatalog,
    extracted_memory: str,
    build_rule_resolution: RuleResolutionBuilder,
) -> HardwareResolution | None:
    if re.fullmatch(r"(?:amd(?:\s+radeon\s+instinct|\s+instinct)?[\s-]+)?mi250x(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "AMD Radeon Instinct MI250X",
            catalog,
            "normalized AMD instinct shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:amd[\s-]+)?mi210(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "AMD Instinct MI210",
            catalog,
            "normalized AMD special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:amd[\s-]+)?mi200(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "AMD Instinct MI200",
            catalog,
            "normalized AMD instinct shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:ascend\s+)?910b(?:\s+npu)?", normalized_text) or re.fullmatch(
        r"ascend\s+910b(?:\s+\d{1,3}gb)?(?:\s+npu)?", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "Huawei Ascend 910B",
            catalog,
            "normalized Huawei special-case rule",
            extracted_memory,
        )
    return None


def _resolve_post_unresolved_tail_rules(
    *,
    raw_hardware_name: str,
    normalized_text: str,
    catalog: HardwareCatalog,
    extracted_memory: str,
    build_rule_resolution: RuleResolutionBuilder,
    build_unmatched_resolution: UnmatchedResolutionBuilder,
) -> HardwareResolution | None:
    if re.fullmatch(r"(?:nvidia\s+)?hopper(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA H100 PCIe",
            catalog,
            "hopper_default_rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?a10(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA A10",
            catalog,
            "nvidia_accelerator_family_rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?h100(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA H100 PCIe",
            catalog,
            "normalized NVIDIA accelerator rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:tesla\s+)?t4(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Tesla T4",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:tesla\s+)?v100\s+pcie\s+32gb", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Tesla V100 PCIe 32 GB",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?v100\s+sxm2\s+16gb", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Tesla V100 SXM2 16 GB",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:tesla\s+)?m40(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA M40",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?rtx\s*2080(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?2080(?:\s+\d{1,3}gb)?", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce RTX 2080",
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:geforce\s+)?(?:rtx\s*)?2080\s*ti(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce RTX 2080 Ti",
            catalog,
            "normalized GeForce RTX shorthand rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?gtx\s*2080\s*ti(?:\s+gpu)?(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce RTX 2080 Ti",
            catalog,
            "normalized GTX-to-RTX correction rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:geforce\s+)?gtx\s*2080\s*ti(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce RTX 2080 Ti",
            catalog,
            "normalized GTX-to-RTX correction rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?gtx\s*2080(?:\s+gpu)?(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce RTX 2080",
            catalog,
            "normalized GTX-to-RTX correction rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:geforce\s+)?(?:gtx\s*)?1080(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce GTX 1080",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:quadro\s+)?gv100(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Quadro GV100",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:geforce\s+)?gtx\s*1080\s*ti(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:geforce\s+)?1080ti(?:\s+\d{1,3}gb)?(?:\s+gddr5x)?", normalized_text
    ) or re.fullmatch(
        r"(?:nvidia\s+)?gt1080ti(?:\s+\d{1,3}gb)?", normalized_text
    ) or re.fullmatch(
        r"(?:nvidia\s+)?1080ti\s+gpu(?:\s+\d{1,3}gb)?", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce GTX 1080 Ti",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:geforce\s+)?(?:quadro\s+)?(?:rtx\s+)?8000(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:geforce\s+)?(?:quadro\s+)?(?:rtx\s+)?8000(?:\s+\d{1,3}gb)?\s+gpu",
        normalized_text,
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Quadro RTX 8000",
            catalog,
            "normalized RTX professional special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:quadro\s+)?rtx\s*6000(?:\s*ti)?(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Quadro RTX 6000",
            catalog,
            "normalized RTX professional special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?quadro\s+r8000(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?quadro\s+8000(?:\s+\d{1,3}gb)?", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Quadro RTX 8000",
            catalog,
            "normalized RTX professional special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?quadro\s+6000(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Quadro RTX 6000",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?rtx\s*a6000(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA RTX A6000",
            catalog,
            "normalized RTX family rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?(?:tesla\s+)?k80(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Tesla K80",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:tesla\s+)?k40c?(?:\s+\d{1,3}gb)?", normalized_text):
        special_name = "NVIDIA Tesla K40c" if "k40c" in normalized_text else "NVIDIA Tesla K40s"
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            special_name,
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:tesla\s+)?p40(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Tesla P40",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"(?:nvidia\s+)?(?:quadro\s+)?rtx\s*6000(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Quadro RTX 6000",
            catalog,
            "normalized RTX professional special-case rule",
            extracted_memory,
        )
    if re.fullmatch(r"npu(?:\s+\d{1,3}gb)?", normalized_text):
        reason = "input describes an NPU without a specific model"
    elif re.fullmatch(r"(?:nvidia\s+)?rtx\s*1080(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?1080\s+gtx(?:\s+\d{1,3}gb)?", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce GTX 1080",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    elif re.fullmatch(r"(?:nvidia\s+)?(?:geforce\s+)?gtx\s*1070(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?1070(?:\s+\d{1,3}gb)?", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce GTX 1070",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    elif re.fullmatch(r"(?:nvidia\s+)?(?:geforce\s+)?(?:gtx\s*)?960(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce GTX 960",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    elif re.fullmatch(r"(?:nvidia\s+)?(?:geforce\s+)?gtx\s*980\s*ti(?:\s+\d{1,3}gb)?", normalized_text) or re.fullmatch(
        r"(?:nvidia\s+)?(?:geforce\s+)?980ti(?:\s+\d{1,3}gb)?", normalized_text
    ):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce GTX 980 Ti",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    elif re.fullmatch(r"(?:nvidia\s+)?(?:gpu\s+)?p100(?:\s+\d{1,3}gb)?", normalized_text):
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA Tesla P100 PCIe 16GB",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    elif normalized_text == "titan x equivalent gpu":
        reason = "input is a comparative description, not a specific GPU model"
    elif normalized_text == "nvidia ti tan":
        return build_rule_resolution(
            raw_hardware_name,
            normalized_text,
            "NVIDIA GeForce GTX TITAN",
            catalog,
            "normalized NVIDIA special-case rule",
            extracted_memory,
        )
    elif re.search(r"\bmx\d{3}\b", normalized_text):
        reason = "input names a mobile GPU outside the benchmark set"
    else:
        reason = "no normalization rule matched"

    return build_unmatched_resolution(reason)


def resolve_pre_unresolved_family_rules(
    *,
    raw_hardware_name: str,
    normalized_text: str,
    catalog: HardwareCatalog,
    extracted_memory: str,
    build_rule_resolution: RuleResolutionBuilder,
    resolve_family_variant: FamilyVariantResolver,
    build_named_resolution: NamedResolutionBuilder,
) -> HardwareResolution | None:
    return _resolve_nvidia_accelerator_pre_unresolved(
        raw_hardware_name=raw_hardware_name,
        normalized_text=normalized_text,
        catalog=catalog,
        extracted_memory=extracted_memory,
        build_rule_resolution=build_rule_resolution,
        resolve_family_variant=resolve_family_variant,
        build_named_resolution=build_named_resolution,
    )


def resolve_post_unresolved_family_rules(
    *,
    raw_hardware_name: str,
    normalized_text: str,
    catalog: HardwareCatalog,
    extracted_memory: str,
    build_rule_resolution: RuleResolutionBuilder,
    build_unmatched_resolution: UnmatchedResolutionBuilder,
) -> HardwareResolution | None:
    family_resolution = _resolve_nvidia_accelerator_post_unresolved(
        raw_hardware_name=raw_hardware_name,
        normalized_text=normalized_text,
        catalog=catalog,
        extracted_memory=extracted_memory,
        build_rule_resolution=build_rule_resolution,
    )
    if family_resolution is not None:
        return family_resolution

    family_resolution = _resolve_nvidia_geforce_family(
        raw_hardware_name=raw_hardware_name,
        normalized_text=normalized_text,
        catalog=catalog,
        extracted_memory=extracted_memory,
        build_rule_resolution=build_rule_resolution,
    )
    if family_resolution is not None:
        return family_resolution

    family_resolution = _resolve_nvidia_titan_family(
        raw_hardware_name=raw_hardware_name,
        normalized_text=normalized_text,
        catalog=catalog,
        extracted_memory=extracted_memory,
        build_rule_resolution=build_rule_resolution,
    )
    if family_resolution is not None:
        return family_resolution

    family_resolution = _resolve_tpu_family(
        raw_hardware_name=raw_hardware_name,
        normalized_text=normalized_text,
        catalog=catalog,
        extracted_memory=extracted_memory,
        build_rule_resolution=build_rule_resolution,
    )
    if family_resolution is not None:
        return family_resolution

    family_resolution = _resolve_other_vendor_family(
        raw_hardware_name=raw_hardware_name,
        normalized_text=normalized_text,
        catalog=catalog,
        extracted_memory=extracted_memory,
        build_rule_resolution=build_rule_resolution,
    )
    if family_resolution is not None:
        return family_resolution

    return _resolve_post_unresolved_tail_rules(
        raw_hardware_name=raw_hardware_name,
        normalized_text=normalized_text,
        catalog=catalog,
        extracted_memory=extracted_memory,
        build_rule_resolution=build_rule_resolution,
        build_unmatched_resolution=build_unmatched_resolution,
    )
