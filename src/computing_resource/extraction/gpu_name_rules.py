from __future__ import annotations

import re


GENERIC_UNRESOLVED_TOKENS = {"accelerator", "gpu", "processor", "hardware", "device", "unknown"}
SPECIAL_CANONICAL_RULES = {
    "cloud tpu v3": "Google TPU v3",
    "google cloud tpu v3": "Google TPU v3",
    "google tpu v3": "Google TPU v3",
    "h80o": "NVIDIA H800",
    "a100l": "NVIDIA A100",
}
TPU_VERSION_TOKEN_PATTERN = r"\bv(\d+(?:i)?)\b"
TPU_VERSIONED_PATTERNS = (
    r"(?:google\s+)?cloud\s+tpu\s+v(\d+(?:i)?)",
    r"tpu\s+v(\d+(?:i)?)\s+\d+",
    r"cloud\s+v(\d+(?:i)?)\s+tpu(?:\s+\d{1,3}gb)?",
    r"v(\d+(?:i)?)[\s-]+\d+\s+tpu",
)
TPU_V3_DEFAULT_PATTERNS = (
    r"v(\d+(?:i)?)\s+\d+\s+(?:google\s+)?cloud\s+tpus?",
    r"v(\d+(?:i)?)\s+(?:google\s+)?cloud\s+tpus?",
    r"(?:google\s+)?cloud\s+tpus?\s+v(\d+(?:i)?)(?:\s+\d{1,3}gb)?",
    r"(?:google\s+)?v(\d+(?:i)?)\s+tpu(?:\s+\d{1,3}gb)?",
    r"(?:google\s+)?cloud\s+tpu\s+pods",
    r"(?:google\s+)?cloud\s+tpu",
)


def standardize_hardware_text(text: str | None) -> str:
    if text is None:
        return ""

    value = str(text).strip().lower()
    if not value:
        return ""

    value = value.replace("_", " ").replace("/", " ")
    value = re.sub(r"[\-]+", " ", value)
    value = re.sub(r"[(),]+", " ", value)
    value = re.sub(r"\brunpod\b", " ", value)
    value = re.sub(r"\bhgx\b", " ", value)
    value = re.sub(r"\bti\s+tan\b", "titan", value)
    value = re.sub(r"\bti\-tan\b", "titan", value)
    value = re.sub(r"\btelsa\b", "tesla", value)
    value = re.sub(r"\btitian\b", "titan", value)
    value = re.sub(r"\brtc\b", "rtx", value)
    value = re.sub(r"\bturbo\b", " ", value)
    value = re.sub(r"\bcoll?ab\b", " ", value)
    value = re.sub(r"\b(gigabyte|msi|asus|zotac|evga|palit|pny|colorful|inno3d|gainward)\b", " ", value)
    value = re.sub(r"\btensor\s+cores?\b", " ", value)
    value = re.sub(r"\b\d+\s*cores?\b", " ", value)
    value = re.sub(r"\bpci\s*e\b", "pcie", value)
    value = re.sub(r"\bpcl[e3]\b", "pcie", value)
    value = re.sub(r"\btpuv(\d+(?:i)?)\b", r"tpu v\1", value)
    value = re.sub(r"\bcloudtpu\s*v(\d+(?:i)?)\b", r"cloud tpu v\1", value)
    value = re.sub(r"\btpu\s+v(\d+(?:i)?)-(\d+)\b", r"tpu v\1 \2", value)
    value = re.sub(r"\btitanx\b", "titan x", value)
    value = re.sub(r"\btitanxp\b", "titan xp", value)
    value = re.sub(r"\btitanrtx\b", "titan rtx", value)
    value = re.sub(r"\b1080i\b", "1080 ti", value)
    value = re.sub(r"\bgtx\s*(\d{3,4})\b", r"gtx \1", value)
    value = re.sub(r"\brtx\s*(\d{3,4})\b", r"rtx \1", value)
    value = re.sub(r"\bgtx\s+([2-9]\d{3})(\s*ti)?\b", r"rtx \1\2", value)
    value = re.sub(r"\b(\d{4})rtx\b", r"\1 rtx", value)
    value = re.sub(r"\bnvidia\s+(\d{4})\s+rtx\b", r"nvidia rtx \1", value)
    value = re.sub(r"\b(\d{4})\s+rtx\b", r"rtx \1", value)
    value = re.sub(r"\brtx\s+nvidia\s+(a\d{4})\b", r"nvidia rtx \1", value)
    value = re.sub(r"\bgddr\d+x?\b", " ", value)
    value = re.sub(r"\bquadro\s+(r?8000)\b", r"quadro \1", value)
    value = re.sub(r"\bquadro\s+q6000\b", "quadro 6000", value)
    value = re.sub(r"\btesla\s+v100s\b", "tesla v100s", value)
    value = re.sub(r"\bv100s\b", "v100s", value)
    value = re.sub(r"\bgeforce\s+nvidia\b", "nvidia geforce", value)
    value = re.sub(r"\bnvidia\s+geforce\b", "nvidia geforce", value)
    value = re.sub(r"\bh[1l]oo\b", "h100", value)
    value = re.sub(r"\bh1o0\b", "h100", value)
    value = re.sub(r"\bh10o\b", "h100", value)
    value = re.sub(r"\ba10o\b", "a100", value)
    value = re.sub(r"\bai00\b", "a100", value)
    value = re.sub(r"\ba60o0\b", "a6000", value)
    value = re.sub(r"\brtx(\d{3,4})ada\b", r"rtx \1 ada", value)
    value = re.sub(r"\ba(\d{4})ada\b", r"a\1 ada", value)
    value = re.sub(r"\b(\d{3,4})ada\b", r"\1 ada", value)
    value = re.sub(r"\bnvidia(?=\d{3,4}\b)", "nvidia ", value)
    value = re.sub(r"\b([ahlvkptm])\s*(\d{2,4})\b", r"\1\2", value)
    value = re.sub(r"\b([ahlvkptm]\d{2,4}s?)\s+nvidia\b", r"nvidia \1", value)
    value = re.sub(r"\brtx\s+nvidia\s+(a\d{4})\b", r"nvidia rtx \1", value)
    value = re.sub(r"\b(\d{4})\s*ti\b", r"\1ti", value)
    value = re.sub(r"\b(\d{4})ti\b", r"\1 ti", value)
    value = re.sub(r"\b(\d{2,3})\s*g\b", r"\1gb", value)
    value = re.sub(r"\b(\d{1,3})\s*gib\b", r"\1gb", value)
    value = re.sub(r"\b(\d{1,3})\.\d+\s*gb\b", r"\1gb", value)
    value = re.sub(r"\b(\d{2,3})\s*gb\b", r"\1gb", value)
    value = re.sub(r"\b(v100|p100|a100|h100|h200|h800|h20)\s+(\d{2,3}gb)\s+(sxm\d?|pcie|nvl)\b", r"\1 \3 \2", value)
    value = re.sub(r"\b(\d{3,4})ada\b", r"\1 ada", value)
    value = re.sub(r"\bsxm\s*(\d)\b", r"sxm\1", value)
    value = re.sub(r"\bsmx\b", "sxm", value)
    value = re.sub(r"\bsxm(\d{1,3}gb)\b", r"sxm \1", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_input_memory(text: str | None) -> str:
    normalized_text = standardize_hardware_text(text)
    capacity_match = re.search(r"\b(\d{1,3})gb\b", normalized_text)
    if capacity_match:
        return f"{capacity_match.group(1)}GB"
    return ""


def should_include_catalog_row(row: dict) -> bool:
    row_type = (row.get("Type") or "").strip().lower()
    if row_type in {"gpu", "tpu", "station", "npu"}:
        return True
    if row_type:
        return False

    hardware_name = standardize_hardware_text(row.get("Hardware name") or "")
    if not hardware_name:
        return False

    return bool(
        re.search(
            r"\b("
            r"tpu|rtx|gtx|quadro|tesla|titan|geforce|"
            r"rtx\d{4}|gtx\d{3,4}|"
            r"[ahlv]\d{2,4}|gh\d{3}|mi\d{3,4}|"
            r"ascend|instinct|radeon|apple m\d"
            r")\b",
            hardware_name,
        )
    )


def _add_alias(aliases: set[str], value: str | None) -> None:
    normalized = standardize_hardware_text(value)
    if normalized:
        aliases.add(normalized)


def _catalog_memory_aliases(normalized_name: str, row: dict) -> tuple[str, ...]:
    values: set[str] = set()
    for match in re.findall(r"\b(\d{1,3})gb\b", normalized_name):
        values.add(f"{int(match)}gb")

    raw_bytes = (row.get("Memory (bytes)") or "").strip()
    if raw_bytes:
        try:
            memory_bytes = float(raw_bytes)
        except ValueError:
            memory_bytes = 0.0
        if memory_bytes > 0:
            values.add(f"{int(round(memory_bytes / 1_000_000_000))}gb")

    return tuple(sorted(values))


def generate_catalog_aliases(hardware_name: str, row: dict, exact_names: set[str]) -> set[str]:
    normalized_name = standardize_hardware_text(hardware_name)
    aliases: set[str] = set()
    memory_aliases = _catalog_memory_aliases(normalized_name, row)

    manufacturer = (row.get("Manufacturer") or "").strip()
    model_only = normalized_name
    if manufacturer:
        manufacturer_token = standardize_hardware_text(manufacturer)
        if model_only.startswith(f"{manufacturer_token} "):
            model_only = model_only[len(manufacturer_token) + 1 :]
            _add_alias(aliases, model_only)

    if normalized_name.startswith("google tpu"):
        version_match = re.search(TPU_VERSION_TOKEN_PATTERN, normalized_name)
        if version_match:
            version = version_match.group(1)
            _add_alias(aliases, f"tpu v{version}")
            _add_alias(aliases, f"cloud tpu v{version}")
            _add_alias(aliases, f"google cloud tpu v{version}")

    if normalized_name.startswith("nvidia ") and re.search(r"\b[ahlv]\d{2,4}s?\b", normalized_name):
        accel_match = re.search(r"\b([ahlv]\d{2,4}s?)\b", normalized_name)
        if accel_match:
            model = accel_match.group(1)
            _add_alias(aliases, model)
            _add_alias(aliases, f"nvidia {model}")
            for memory_alias in memory_aliases:
                _add_alias(aliases, f"{model} {memory_alias}")
                _add_alias(aliases, f"nvidia {model} {memory_alias}")
            if "sxm" in normalized_name:
                _add_alias(aliases, f"{model} sxm")
                _add_alias(aliases, f"nvidia {model} sxm")
            if "pcie" in normalized_name:
                _add_alias(aliases, f"{model} pcie")
                _add_alias(aliases, f"nvidia {model} pcie")
            if "nvl" in normalized_name:
                _add_alias(aliases, f"{model} nvl")
                _add_alias(aliases, f"nvidia {model} nvl")

    geforce_rtx_match = re.search(r"\bnvidia\s+geforce\s+rtx\s+(\d{4})(?:\s+(ti))?\b", normalized_name)
    if geforce_rtx_match:
        model = geforce_rtx_match.group(1)
        ti_suffix = f" {geforce_rtx_match.group(2)}" if geforce_rtx_match.group(2) else ""
        shorthand = f"{model}{ti_suffix}"
        compact = shorthand.replace(" ", "")
        _add_alias(aliases, shorthand)
        _add_alias(aliases, f"rtx {shorthand}")
        _add_alias(aliases, f"nvidia rtx {shorthand}")
        _add_alias(aliases, f"{model} rtx{ti_suffix}")
        _add_alias(aliases, f"nvidia {model}")
        _add_alias(aliases, f"nvidia {model} rtx{ti_suffix}")
        _add_alias(aliases, compact)
        _add_alias(aliases, f"rtx{compact}")
        if int(model) >= 2000:
            _add_alias(aliases, f"gtx {shorthand}")
            _add_alias(aliases, f"nvidia gtx {shorthand}")
            _add_alias(aliases, f"nvidia geforce gtx {shorthand}")
        for memory_alias in memory_aliases:
            _add_alias(aliases, f"{shorthand} {memory_alias}")
            _add_alias(aliases, f"rtx {shorthand} {memory_alias}")
            _add_alias(aliases, f"nvidia {model} {memory_alias}")
            _add_alias(aliases, f"{model} rtx {memory_alias}")
            if int(model) >= 2000:
                _add_alias(aliases, f"gtx {shorthand} {memory_alias}")
                _add_alias(aliases, f"nvidia geforce gtx {shorthand} {memory_alias}")

    geforce_gtx_match = re.search(r"\bnvidia\s+geforce\s+gtx\s+(\d{3,4})(?:\s+(ti))?\b", normalized_name)
    if geforce_gtx_match:
        model = geforce_gtx_match.group(1)
        ti_suffix = f" {geforce_gtx_match.group(2)}" if geforce_gtx_match.group(2) else ""
        shorthand = f"{model}{ti_suffix}"
        compact = shorthand.replace(" ", "")
        _add_alias(aliases, f"gtx {shorthand}")
        _add_alias(aliases, f"nvidia gtx {shorthand}")
        _add_alias(aliases, compact)
        for memory_alias in memory_aliases:
            _add_alias(aliases, f"gtx {shorthand} {memory_alias}")
            _add_alias(aliases, f"nvidia gtx {shorthand} {memory_alias}")

    rtx_ada_match = re.search(r"\bnvidia\s+rtx\s+(\d{4})\s+ada\s+generation\b", normalized_name)
    if rtx_ada_match:
        model = rtx_ada_match.group(1)
        _add_alias(aliases, f"rtx {model} ada")
        _add_alias(aliases, f"nvidia rtx {model} ada")
        _add_alias(aliases, f"{model} ada")
        for memory_alias in memory_aliases:
            _add_alias(aliases, f"rtx {model} ada {memory_alias}")
            _add_alias(aliases, f"nvidia rtx {model} ada {memory_alias}")

    if normalized_name.startswith("nvidia ") and re.search(r"\brtx\s*3090\b", normalized_name):
        _add_alias(aliases, "rtx 3090")
        _add_alias(aliases, "geforce rtx 3090")
        _add_alias(aliases, "nvidia geforce rtx 3090")
        _add_alias(aliases, "rtx3090")
    if normalized_name.startswith("nvidia ") and re.search(r"\brtx\s*2080\s*ti\b", normalized_name):
        _add_alias(aliases, "rtx 2080 ti")
        _add_alias(aliases, "geforce rtx 2080 ti")
        _add_alias(aliases, "nvidia geforce rtx 2080 ti")
    if normalized_name.startswith("nvidia ") and re.search(r"\brtx\s*2080\b", normalized_name):
        _add_alias(aliases, "rtx 2080")
        _add_alias(aliases, "geforce rtx 2080")
        _add_alias(aliases, "nvidia geforce rtx 2080")
    if normalized_name.startswith("nvidia ") and re.search(r"\brtx\s*1080\s*ti\b", normalized_name):
        _add_alias(aliases, "gtx 1080 ti")
        _add_alias(aliases, "geforce gtx 1080 ti")
        _add_alias(aliases, "nvidia geforce gtx 1080 ti")
    if normalized_name.startswith("nvidia ") and re.search(r"\brtx\s*1080\b", normalized_name):
        _add_alias(aliases, "gtx 1080")
        _add_alias(aliases, "geforce gtx 1080")
        _add_alias(aliases, "nvidia geforce gtx 1080")
    if normalized_name.startswith("nvidia ") and re.search(r"\bgtx\s*980\s*ti\b", normalized_name):
        _add_alias(aliases, "gtx 980 ti")
        _add_alias(aliases, "geforce gtx 980 ti")
    if normalized_name.startswith("nvidia ") and re.search(r"\bgtx\s*980\b", normalized_name):
        _add_alias(aliases, "gtx 980")
        _add_alias(aliases, "geforce gtx 980")
    if normalized_name.startswith("nvidia ") and re.search(r"\bgtx\s*1070\b", normalized_name):
        _add_alias(aliases, "gtx 1070")
        _add_alias(aliases, "geforce gtx 1070")
    if normalized_name.startswith("nvidia ") and re.search(r"\brtx\s*3090\s*ti\b", normalized_name):
        _add_alias(aliases, "rtx 3090 ti")
        _add_alias(aliases, "geforce rtx 3090 ti")
    if normalized_name.startswith("nvidia ") and re.search(r"\bquadro\s+r?8000\b", normalized_name):
        _add_alias(aliases, "quadro 8000")
        _add_alias(aliases, "quadro r8000")
        _add_alias(aliases, "nvidia quadro r8000")
    if normalized_name.startswith("nvidia ") and re.search(r"\bquadro\s+6000\b", normalized_name):
        _add_alias(aliases, "quadro 6000")
        _add_alias(aliases, "nvidia quadro 6000")
    if normalized_name.startswith("nvidia ") and re.search(r"\btitan\s*xp\b", normalized_name):
        _add_alias(aliases, "titan xp")
        _add_alias(aliases, "geforce titan xp")
    if normalized_name.startswith("nvidia ") and re.search(r"\btitan\s*x\b", normalized_name):
        _add_alias(aliases, "titan x")
        _add_alias(aliases, "geforce titan x")
    if normalized_name.startswith("nvidia ") and re.search(r"\btitan\s*rtx\b", normalized_name):
        _add_alias(aliases, "titan rtx")
        _add_alias(aliases, "geforce titan rtx")
    if normalized_name.startswith("nvidia ") and re.search(r"\bv100s\b", normalized_name):
        _add_alias(aliases, "tesla v100s")
        _add_alias(aliases, "nvidia tesla v100s")
    if normalized_name.startswith("nvidia ") and re.search(r"\bq6000\b", normalized_name):
        _add_alias(aliases, "quadro q6000")
    if normalized_name.startswith("nvidia ") and re.search(r"\bgp100\b", normalized_name):
        _add_alias(aliases, "quadro gp100")
        _add_alias(aliases, "nvidia quadro gp100")

    if "rtx a" in normalized_name:
        rtx_a = normalized_name.replace("rtx a", "rtxa")
        _add_alias(aliases, rtx_a)

    return {alias for alias in aliases if alias not in exact_names}
