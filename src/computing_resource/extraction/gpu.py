import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import yaml

from computing_resource.config import load_config
from computing_resource.extraction.gpu_candidates import build_candidate_windows, has_hardware_signal
from computing_resource.extraction.gpu_catalog import (
    HardwareCatalog,
    load_hardware_catalog,
    resolve_hardware_name,
)
from computing_resource.extraction.gpu_name_rules import standardize_hardware_text
from computing_resource.extraction.gpu_sections import load_parse_markdown_metadata, load_section_doc_from_parse_dir

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised via runtime error in build_tagger
    OpenAI = None


LLM_INPUT_MAX_CHARS = 12000
LLM_INPUT_CHUNK_OVERLAP = 300
GPU_EXTRACTION_SCHEMA_VERSION = 1
DEFAULT_GPU_LLM_MODEL = "gpt-4o-mini"
DEFAULT_GPU_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_GPU_API_KEY_ENV = "OPENROUTER_API_KEY"
DEEPSEEK_GPU_API_BASE = "https://api.deepseek.com"
DEEPSEEK_GPU_API_KEY_ENV = "DEEPSEEK_API_KEY"


class QuotaExhaustedError(RuntimeError):
    """Raised when the upstream model API reports depleted credits or quota."""


class FatalModelRequestError(RuntimeError):
    """Raised when the upstream model API rejects requests in a non-recoverable way."""


def is_quota_exhausted_error(exc: Exception) -> bool:
    message = str(exc).lower()
    quota_signals = (
        "insufficient balance",
        "insufficient_balance",
        "insufficient credits",
        "insufficient credit",
        "quota exceeded",
        "quota_exceeded",
        "credit exhausted",
        "out of credits",
        "billing hard limit",
        "exceeded your current quota",
    )
    return any(signal in message for signal in quota_signals)


def is_fatal_model_request_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 403:
        return True

    message = str(exc).lower()
    fatal_signals = (
        "permissiondeniederror",
        "permission denied",
        "error code: 403",
        "status code: 403",
        "403 forbidden",
        "request is prohibited",
        "terms of service",
        "terms of service violation",
        "violation of provider terms of service",
    )
    return any(signal in message for signal in fatal_signals)


def is_retryable_model_request_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    message = str(exc).lower()
    retryable_signals = (
        "rate limit",
        "rate_limit",
        "too many requests",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
        "gateway",
        "server error",
    )
    return any(signal in message for signal in retryable_signals)


def get_unpred_content(unpred_dict: Dict[str, Any]) -> List[Dict[str, str]]:
    unpred_gpu_content_list = unpred_dict.get("gpu_content", [])
    unpred_content_dict_list = []
    seen_section_numbers = set()

    for item in unpred_gpu_content_list:
        section_num = item.get("section_number")
        content = item.get("content")

        if section_num == "-1" or section_num in seen_section_numbers:
            continue

        if content:
            unpred_content_dict_list.append({"sec_num": section_num, "sec_content": content})
            seen_section_numbers.add(section_num)

        for sub in item.get("sub_section", []):
            sub_num = sub.get("section_number")
            sub_content = sub.get("content")
            if sub_num in seen_section_numbers or sub_num == "-1":
                continue
            if sub_content:
                unpred_content_dict_list.append({"sec_num": sub_num, "sec_content": sub_content})
                seen_section_numbers.add(sub_num)

    return unpred_content_dict_list


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def normalize_entity_name(entity_list: List[Dict[str, Any]]) -> None:
    for entity in entity_list:
        if "name" in entity and isinstance(entity["name"], str):
            entity["name"] = entity["name"].replace("O", "0")


def entities_match(pred_entity: Dict[str, Any], gold_entity: Dict[str, Any], match_type: str = "exact") -> bool:
    pred_type = normalize_text(str(pred_entity.get("name", "")))
    pred_num = normalize_text(str(pred_entity.get("num")))
    gold_type = normalize_text(str(gold_entity.get("name", "")))
    gold_num = normalize_text(str(gold_entity.get("num")))

    if match_type == "name_only":
        return pred_type == gold_type
    if match_type == "num_only":
        return pred_num == gold_num
    return (pred_type == gold_type) and (pred_num == gold_num)


class NERMetrics:
    def __init__(self, match_type: str = "name_only"):
        self.match_type = match_type
        self.reset()

    def reset(self) -> None:
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0

    def update(self, pred_list: List[Dict[str, Any]], gold_list: List[Dict[str, Any]]) -> None:
        matched_gold = [False] * len(gold_list)

        for pred_entity in pred_list:
            matched = False
            for i, gold_entity in enumerate(gold_list):
                if not matched_gold[i] and entities_match(pred_entity, gold_entity, self.match_type):
                    self.total_tp += 1
                    matched_gold[i] = True
                    matched = True
                    break
            if not matched:
                self.total_fp += 1

        for i in range(len(gold_list)):
            if not matched_gold[i]:
                self.total_fn += 1

    def compute(self) -> Dict[str, float]:
        tp = self.total_tp
        fp = self.total_fp
        fn = self.total_fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn,
        }


INSTRUCTION_SHORT = textwrap.dedent(
    r"""
    Extract only the hardware actually used by the authors in this paper's own experiments.
    Return JSON only: [{"name": "...", "num": ...}]

    Rules:
    - Extract GPUs, TPUs, NPUs, IPUs, and other relevant accelerators.
    - If there is no qualifying hardware, return exactly [].
    - "num" is an integer if explicitly stated, 1 if a single device is clearly indicated, otherwise null.
    - Include memory size in the name when it specifies the hardware config.
    - Ignore CPUs, memory, runtime, and storage as standalone resources.
    - Do not extract model names, LLM names, frameworks, APIs, cloud/services, software libraries, datasets, or algorithms as hardware.
    - Do not extract hardware mentioned only in prior work, related work, citations, baselines from other papers, comparisons, historical background, hypothetical setups, or future work.
    - If both prior work and current paper hardware appear in the same sentence, extract only the current paper's hardware.
    - Do not treat footnotes, citation markers, or superscripts as quantities unless clearly semantic.
    - Output JSON only, with no explanation.
    """
).strip()

DEMOS_TEXT = textwrap.dedent(
    r"""
    input: The models are trained using the AdamW optimizer (Kingma and Ba, 2014) on 4 Nvidia V100-32G GPUs for Qwen2-0.5B models and 16 Nvidia V100-32G GPUs for Mistral-7B.
    output: [{"name": "Nvidia V100 32G", "num": 4}, {"name": "Nvidia V100 32G", "num": 16}]

    input: We use Group Relative Policy Optimization (GRPO) (Shao et al., 2024) for model optimization.
    output: []

    input: Full Mode1</td><td>36h on NVIDIA A100</td></tr><tr><td>Router-Tuning</td><td>Block / MLP / Attn</td><td>Token / Sequence</td><td>Finetuning</td><td>Router</td><td>15m on NVIDIA A6000</td></tr></table>
    output: [{"name": "NVIDIA A6000", "num": null}, {"name": "NVIDIA A100", "num": null}]

    input: We run all evaluations on 4 NVIDIA A100 GPUs, each with 80 GB of memory.
    output: [{"name": "NVIDIA A100 80GB", "num": 4}]

    input: the calibration process is completed within 10 minutes using a single RTX 4090 GPU. Regarding efficiency, we evaluate the encoder latency on NVIDIA RTX 4090, NVIDIA RTX A6000, and Apple M1 Pro.
    output: [{"name": "RTX 4090", "num": 1}, {"name": "NVIDIA RTX A6000", "num": null}, {"name": "Apple M1 Pro", "num": null}]

    input: All experiments were performed using 512 CPU cores, 8 Nvidia RTX A6000 (48GB) GPUs, and 1024 GB of memory.
    output: [{"name": "Nvidia RTX A6000 48GB", "num": 8}]

    input: All experiments were conducted on NVIDIA A100 GPUs with 40GB or 80GB memory configurations. Running the full set of main experiments, including all primary tables, required approximately 21 days using 8 GPUs in parallel.
    output: [{"name": "NVIDIA A100 40GB", "num": 8}, {"name": "NVIDIA A100 80GB", "num": 8}]

    input: We conduct all experiments on an NVIDIA A40 GPU.
    output: [{"name": "NVIDIA A40", "num": 1}]

    input: We train the mT5-large model on 18 Cloud TPU V3 chips.
    output: [{"name": "Cloud TPU V3", "num": 18}]

    input: Our experiments use either 8 NVIDIA A100 80GB GPUs or 8 NVIDIA H100 GPUs depending on model size.
    output: [{"name": "NVIDIA A100 80GB", "num": 8}, {"name": "NVIDIA H100", "num": 8}]

    input: Previous work trained the model on 64 NVIDIA V100 GPUs, but we do not use that setup here.
    output: []

    input: Brown et al. (2020) used 1024 TPU v3 chips for pretraining.
    output: []

    input: We compare against a baseline model originally trained on 16 V100 GPUs.
    output: []

    input: Unlike Smith et al. (2023), who used 64 NVIDIA V100 GPUs, we train our model on 8 NVIDIA A100 GPUs.
    output: [{"name": "NVIDIA A100", "num": 8}]

    input: Prior work used TPUv3-128, while all our experiments are run on 4 A100 GPUs.
    output: [{"name": "A100", "num": 4}]

    input: We reproduce the baseline on 4 NVIDIA A100 GPUs and train our method on 8 NVIDIA A100 GPUs.
    output: [{"name": "NVIDIA A100", "num": 4}, {"name": "NVIDIA A100", "num": 8}]

    input: The baseline reported results on 8 V100 GPUs, but in this paper we only evaluate on a single NVIDIA A6000 GPU.
    output: [{"name": "NVIDIA A6000", "num": 1}]

    input: We use GPT-4o, Claude, Gemini, Qwen, LLaMA, Mistral, and vLLM for evaluation and inference.
    output: []

    input: We evaluate Qwen2.5-7B, Qwen2.5-14B, Llama3.1-8B, and Meta-Llama3-8B-Instruct on all datasets.
    output: []

    input: We call the Azure Batch REST API service to run GPT-4o requests.
    output: []

    input: We compare LLaVA-Video-7B and CogVideoX-5B on video generation benchmarks.
    output: []

    input:
    output: []
    """
).strip()

PROMPT_MODE_FEW_SHOT = "few_shot"
PROMPT_MODE_ZERO_SHOT = "zero_shot"
PROMPT_MODES = {PROMPT_MODE_FEW_SHOT, PROMPT_MODE_ZERO_SHOT}


def build_prompt(
    text: str,
    use_system_hint: bool = False,
    prompt_mode: str = PROMPT_MODE_FEW_SHOT,
) -> str:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

    parts = [INSTRUCTION_SHORT]
    if prompt_mode == PROMPT_MODE_FEW_SHOT:
        parts.append(DEMOS_TEXT)
    parts.append(f"input: {text}")
    parts.append("output:")

    return "\n\n".join(parts)


def resolve_gpu_llm_settings(
    model: str | None = None,
    api_base: str | None = None,
    api_key_env: str | None = None,
) -> Dict[str, str]:
    config = load_config()
    llm_config = config.get("llm", {}) if isinstance(config, dict) else {}
    resolved_model = str(model or llm_config.get("model") or DEFAULT_GPU_LLM_MODEL)
    if api_base or api_key_env:
        return {
            "model": resolved_model,
            "api_base": str(api_base or llm_config.get("api_base") or DEFAULT_GPU_API_BASE),
            "api_key_env": str(api_key_env or llm_config.get("api_key_env") or DEFAULT_GPU_API_KEY_ENV),
        }
    if resolved_model.startswith("deepseek"):
        return {
            "model": resolved_model,
            "api_base": DEEPSEEK_GPU_API_BASE,
            "api_key_env": DEEPSEEK_GPU_API_KEY_ENV,
        }
    return {
        "model": resolved_model,
        "api_base": str(llm_config.get("api_base") or DEFAULT_GPU_API_BASE),
        "api_key_env": str(llm_config.get("api_key_env") or DEFAULT_GPU_API_KEY_ENV),
    }


def build_tagger(
    model: str | None = None,
    prompt_mode: str = PROMPT_MODE_FEW_SHOT,
    api_base: str | None = None,
    api_key_env: str | None = None,
):
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

    settings = resolve_gpu_llm_settings(model=model, api_base=api_base, api_key_env=api_key_env)
    api_key = os.getenv(settings["api_key_env"])
    if not api_key:
        raise RuntimeError(f'Missing environment variable: {settings["api_key_env"]}')

    client_state = threading.local()
    system_prompt_parts = [INSTRUCTION_SHORT]
    if prompt_mode == PROMPT_MODE_FEW_SHOT:
        system_prompt_parts.append(DEMOS_TEXT)
    system_prompt = "\n\n".join(system_prompt_parts)
    max_retries = 5

    def get_client():
        client = getattr(client_state, "client", None)
        if client is None:
            if OpenAI is None:
                raise RuntimeError("Missing dependency: openai. Install with `pip install openai`.")
            client = OpenAI(api_key=api_key, base_url=settings["api_base"])
            client_state.client = client
        return client

    def tagger(input_text: str) -> str:
        if not (input_text or "").strip():
            return "[]"
        response = None
        for attempt in range(max_retries):
            try:
                response = get_client().chat.completions.create(
                    model=settings["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_text},
                    ],
                    stream=False,
                    temperature=0,
                    max_tokens=150,
                )
                break
            except Exception as exc:
                if is_quota_exhausted_error(exc):
                    raise QuotaExhaustedError(str(exc)) from exc
                if is_fatal_model_request_error(exc):
                    if attempt == max_retries - 1:
                        raise FatalModelRequestError(str(exc)) from exc
                    time.sleep(2**attempt)
                    continue
                if is_retryable_model_request_error(exc):
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2**attempt)
                    continue
                if "connection" not in str(exc).lower() or attempt == max_retries - 1:
                    raise
                time.sleep(2**attempt)
        if not response.choices:
            return ""
        message = response.choices[0].message
        content = message.content or ""
        print(f"[gpu][llm-output] {content}")
        return content

    return tagger


def _build_runtime_tagger(
    model: str | None = None,
):
    if model is None:
        return build_tagger()
    return build_tagger(model=model)


def _build_runtime_tagger_factory(
    model: str | None = None,
):
    if model is None:
        return build_tagger

    def factory():
        return build_tagger(model=model)

    return factory


def load_gpu_rules(rules_path: str | Path | None = None) -> Dict[str, Any]:
    config = load_config()
    path = Path(rules_path) if rules_path is not None else config["paths"]["gpu_rules_path"]
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_default_catalog(catalog_path: str | Path | None = None) -> HardwareCatalog:
    if catalog_path is not None:
        return load_hardware_catalog(catalog_path)
    default_path = load_config()["paths"]["gpu_root"] / "ml_hardware" / "ml_hardware.xlsx"
    return load_hardware_catalog(default_path)


def _sha256_jsonable(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _catalog_signature(catalog: HardwareCatalog | Any) -> str:
    source_path = getattr(catalog, "source_path", None)
    if source_path:
        candidate = Path(source_path)
        if candidate.exists() and candidate.is_file():
            return hashlib.sha256(candidate.read_bytes()).hexdigest()
    payload = {
        "normalized_to_name": getattr(catalog, "normalized_to_name", {}),
        "alias_to_names": getattr(catalog, "alias_to_names", {}),
        "row_map": getattr(catalog, "row_map", {}),
    }
    return _sha256_jsonable(payload)


def _build_extraction_meta(parse_dir: Path, rules: Dict[str, Any], catalog: HardwareCatalog | Any) -> Dict[str, Any]:
    source_meta = load_parse_markdown_metadata(parse_dir)
    return {
        "schema_version": GPU_EXTRACTION_SCHEMA_VERSION,
        "source_md_path": str(source_meta["source_md_path"]),
        "source_md_sha256": source_meta["source_md_sha256"],
        "rules_sha256": _sha256_jsonable(rules),
        "catalog_sha256": _catalog_signature(catalog),
    }


EXTRACTION_OUTPUT_FIELDS = {
    "candidate_windows",
    "raw_extractions",
    "normalized_extractions",
    "review_flags",
    "fulltext_fallback_triggered",
    "fulltext_fallback_windows",
    "fulltext_fallback_extractions",
    "pred_result",
    "extraction_meta",
}


def _build_input_doc_extraction_meta(
    file_path: Path,
    doc: Dict[str, Any],
    rules: Dict[str, Any],
    catalog: HardwareCatalog | Any,
) -> Dict[str, Any]:
    source_doc = {
        key: value
        for key, value in doc.items()
        if key not in EXTRACTION_OUTPUT_FIELDS
    }
    return {
        "schema_version": GPU_EXTRACTION_SCHEMA_VERSION,
        "source_json_path": str(file_path),
        "source_json_sha256": _sha256_jsonable(source_doc),
        "rules_sha256": _sha256_jsonable(rules),
        "catalog_sha256": _catalog_signature(catalog),
    }


def _classify_output_freshness(output_path: Path, current_meta: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"fresh": False, "reason": "invalid_output_json"}

    if not isinstance(payload, dict):
        return {"fresh": False, "reason": "invalid_output_payload"}

    meta = payload.get("extraction_meta")
    if not isinstance(meta, dict):
        return {"fresh": False, "reason": "missing_extraction_meta"}
    if meta.get("schema_version") != GPU_EXTRACTION_SCHEMA_VERSION:
        return {"fresh": False, "reason": "schema_version_changed"}
    if meta.get("source_md_sha256") != current_meta["source_md_sha256"]:
        return {"fresh": False, "reason": "source_markdown_changed"}
    if meta.get("rules_sha256") != current_meta["rules_sha256"]:
        return {"fresh": False, "reason": "gpu_rules_changed"}
    if meta.get("catalog_sha256") != current_meta["catalog_sha256"]:
        return {"fresh": False, "reason": "hardware_catalog_changed"}
    return {"fresh": True, "reason": "fresh"}


def _classify_input_doc_freshness(doc: Dict[str, Any], current_meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = doc.get("extraction_meta")
    if not isinstance(meta, dict):
        return {"fresh": False, "reason": "missing_extraction_meta"}
    if meta.get("schema_version") != GPU_EXTRACTION_SCHEMA_VERSION:
        return {"fresh": False, "reason": "schema_version_changed"}
    if meta.get("source_json_sha256") != current_meta["source_json_sha256"]:
        return {"fresh": False, "reason": "source_json_changed"}
    if meta.get("rules_sha256") != current_meta["rules_sha256"]:
        return {"fresh": False, "reason": "gpu_rules_changed"}
    if meta.get("catalog_sha256") != current_meta["catalog_sha256"]:
        return {"fresh": False, "reason": "hardware_catalog_changed"}
    return {"fresh": True, "reason": "fresh"}


def _candidate_prediction_payloads(raw_text: str) -> List[str]:
    candidates = [raw_text]
    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
    candidates.extend(match.group(1).strip() for match in fence_pattern.finditer(raw_text))

    first_array = raw_text.find("[")
    last_array = raw_text.rfind("]")
    if first_array != -1 and last_array != -1 and last_array > first_array:
        candidates.append(raw_text[first_array : last_array + 1].strip())

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def parse_prediction_result(raw_text: str) -> tuple[List[Dict[str, Any]], str | None]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return [], "empty_model_output"

    result = None
    for candidate in _candidate_prediction_payloads(raw_text):
        try:
            result = json.loads(candidate)
            break
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(candidate)
                break
            except (ValueError, SyntaxError):
                continue
    if result is None:
        return [], "model_output_parse_failed"
    if not isinstance(result, list):
        return [], "model_output_not_list"
    if any(not isinstance(item, dict) for item in result):
        return [], "model_output_not_object_list"
    return result, None


def parse_prediction(raw_text: str) -> List[Dict[str, Any]]:
    return parse_prediction_result(raw_text)[0]


def _chunk_text_for_llm(text: str, chunk_size: int = LLM_INPUT_MAX_CHARS, overlap: int = LLM_INPUT_CHUNK_OVERLAP) -> List[str]:
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return []
    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(cleaned_text):
        end = min(len(cleaned_text), start + chunk_size)
        chunks.append(cleaned_text[start:end].strip())
        if end >= len(cleaned_text):
            break
        start += step
    return [chunk for chunk in chunks if chunk]


def _select_relevant_llm_snippets(text: str, rules: Dict[str, Any], catalog: HardwareCatalog) -> List[str]:
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return []
    if len(cleaned_text) <= LLM_INPUT_MAX_CHARS:
        return [cleaned_text]

    sentence_splits = re.split(r"(?<=[.!?。！？])\s+|\n+", cleaned_text)
    relevant_sentences = [
        sentence.strip()
        for sentence in sentence_splits
        if sentence.strip() and has_hardware_signal(sentence, rules, catalog)
    ]

    if not relevant_sentences:
        return _chunk_text_for_llm(cleaned_text)

    snippets = []
    current = ""
    for sentence in relevant_sentences:
        if len(sentence) > LLM_INPUT_MAX_CHARS:
            snippets.extend(_chunk_text_for_llm(sentence))
            current = ""
            continue
        if not current:
            current = sentence
            continue
        candidate = f"{current} {sentence}"
        if len(candidate) > LLM_INPUT_MAX_CHARS:
            snippets.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        snippets.append(current)

    if not snippets:
        return _chunk_text_for_llm(cleaned_text)
    return snippets


def predict_for_doc(
    doc: Dict[str, Any],
    tagger,
    rules: Dict[str, Any] | None = None,
    catalog: HardwareCatalog | None = None,
) -> Dict[str, Any]:
    if "sections" in doc and "appendix" in doc:
        return predict_for_section_doc(doc, tagger, rules=rules, catalog=catalog)

    unpred_content_dict_list = get_unpred_content(doc)
    pred_gpu_type_num_list = []
    pred_sec = []

    for unpred_content_dict in unpred_content_dict_list:
        unpred_sec_num = unpred_content_dict["sec_num"]
        unpred_content = unpred_content_dict["sec_content"]
        pred_result, _ = parse_prediction_result(tagger(unpred_content))
        if pred_result:
            pred_gpu_type_num_list.extend(pred_result)
            pred_sec.append(unpred_sec_num)

    normalize_entity_name(pred_gpu_type_num_list)
    return {"sec": pred_sec, "pred": pred_gpu_type_num_list}


def infer_hardware_type(raw_name: str) -> str:
    normalized_name = standardize_hardware_text(raw_name)
    return "TPU" if "tpu" in normalized_name else "GPU"


def normalize_extraction(raw_extraction: Dict[str, Any], catalog: HardwareCatalog) -> Dict[str, Any]:
    resolution = resolve_hardware_name(raw_extraction.get("raw_hardware_name", ""), catalog)
    normalized_name = resolution.normalized_hardware_name or raw_extraction.get("raw_hardware_name", "")
    review_required = resolution.match_status in {"ambiguous", "unmatched"} or raw_extraction.get("count") is None
    review_reason = ""
    if resolution.match_status == "unmatched":
        review_reason = "catalog_unmatched"
    elif resolution.match_status == "ambiguous":
        review_reason = "ambiguous_variant"
    elif raw_extraction.get("count") is None:
        review_reason = "count_missing"

    return {
        "window_id": raw_extraction["window_id"],
        "hardware_type": raw_extraction["hardware_type"],
        "raw_hardware_name": raw_extraction["raw_hardware_name"],
        "cleaned_hardware_name": resolution.cleaned_hardware_name,
        "normalized_hardware_name": normalized_name,
        "benchmark_hardware_name": resolution.benchmark_hardware_name,
        "count": raw_extraction.get("count"),
        "count_text": raw_extraction.get("count_text"),
        "section_number": raw_extraction["section_number"],
        "section_title": raw_extraction["section_title"],
        "normalized_section_title": raw_extraction["normalized_section_title"],
        "evidence": raw_extraction["evidence"],
        "match_status": resolution.match_status,
        "match_method": resolution.match_method,
        "match_confidence": "high" if resolution.match_status in {"exact_match", "alias_match", "rule_match"} else "low",
        "normalization_reason": resolution.normalization_reason,
        "catalog_candidate_count": len(resolution.candidate_names),
        "review_required": review_required,
        "review_reason": review_reason,
        "extraction_source": raw_extraction.get("extraction_source", "section_primary"),
    }


def should_use_llm_fallback(match_status: str, candidate_count: int) -> bool:
    return match_status in {"ambiguous", "unmatched"} and candidate_count != 1


def accept_embedding_candidate(
    top1_score: float,
    top2_score: float,
    score_threshold: float,
    margin_threshold: float,
    type_compatible: bool,
    vendor_compatible: bool,
    memory_conflict: bool,
    variant_conflict: bool,
) -> bool:
    if top1_score < score_threshold:
        return False
    if (top1_score - top2_score) < margin_threshold:
        return False
    if not type_compatible or not vendor_compatible:
        return False
    if memory_conflict or variant_conflict:
        return False
    return True


def audit_doc_extractions(
    candidate_windows: List[Dict[str, Any]],
    normalized_extractions: List[Dict[str, Any]],
    catalog: HardwareCatalog,
    parse_failures: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    _ = catalog
    matched_window_ids = {row["window_id"] for row in normalized_extractions}
    review_flags = list(parse_failures or [])
    parse_failure_window_ids = {flag["source_window_id"] for flag in review_flags if "source_window_id" in flag}

    for window in candidate_windows:
        if window["window_id"] in matched_window_ids:
            continue
        if window["window_id"] in parse_failure_window_ids:
            continue
        if window.get("matched_by_keyword") or window.get("matched_by_alias"):
            review_flags.append(
                {
                    "source_window_id": window["window_id"],
                    "audit_reason": "keyword_hit_without_extraction",
                    "audit_method": "rule_audit",
                    "section_number": window["section_number"],
                    "section_title": window["section_title"],
                    "evidence": window["window_text"],
                }
            )
        elif window.get("matched_by_rule"):
            review_flags.append(
                {
                    "source_window_id": window["window_id"],
                    "audit_reason": "rule_window_without_extraction",
                    "audit_method": "rule_audit",
                    "section_number": window["section_number"],
                    "section_title": window["section_title"],
                    "evidence": window["window_text"],
                }
            )

    return {"review_flags": review_flags}


def _build_raw_extractions(
    candidate_windows: List[Dict[str, Any]],
    tagger,
    rules: Dict[str, Any],
    catalog: HardwareCatalog,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_extractions = []
    parse_failures = []
    for window in candidate_windows:
        for chunk_index, chunk_text in enumerate(
            _select_relevant_llm_snippets(window["window_text"], rules=rules, catalog=catalog),
            start=1,
        ):
            try:
                model_output = tagger(chunk_text)
            except QuotaExhaustedError:
                raise
            except FatalModelRequestError:
                raise
            except Exception as exc:
                parse_failures.append(
                    {
                        "source_window_id": window["window_id"],
                        "audit_reason": "model_request_failed",
                        "audit_method": "model_request_audit",
                        "section_number": window["section_number"],
                        "section_title": window["section_title"],
                        "evidence": chunk_text,
                        "request_error": f"{type(exc).__name__}: {exc}",
                        "chunk_index": chunk_index,
                    }
                )
                continue

            pred_result, parse_error = parse_prediction_result(model_output)
            if parse_error:
                parse_failures.append(
                    {
                        "source_window_id": window["window_id"],
                        "audit_reason": "model_output_parse_failed",
                        "audit_method": "model_output_audit",
                        "section_number": window["section_number"],
                        "section_title": window["section_title"],
                        "evidence": chunk_text,
                        "parse_error": parse_error,
                        "chunk_index": chunk_index,
                    }
                )
                continue
            for item in pred_result:
                for raw_name, resolved_count in split_predicted_item(item):
                    raw_extractions.append(
                        {
                            "window_id": window["window_id"],
                            "hardware_type": infer_hardware_type(raw_name),
                            "raw_hardware_name": raw_name,
                            "count": resolved_count,
                            "count_text": str(resolved_count) if resolved_count is not None else "",
                            "evidence": chunk_text,
                            "section_number": window["section_number"],
                            "section_title": window["section_title"],
                            "normalized_section_title": window["normalized_section_title"],
                        }
                    )
    return raw_extractions, parse_failures


def split_composite_hardware_name(raw_name: str) -> List[str]:
    text = (raw_name or "").strip()
    if not text:
        return []

    variant_match = re.fullmatch(
        r"(?i)\s*([A-Za-z]+\d{2,4})[- ](\d{2,3})GB/(\d{2,3})GB\s*",
        text,
    )
    if variant_match:
        model = variant_match.group(1).upper()
        return [f"{model} {variant_match.group(2)}GB", f"{model} {variant_match.group(3)}GB"]

    slash_split = [part.strip() for part in text.split("/") if part.strip()]
    if len(slash_split) > 1:
        return slash_split

    or_split = [part.strip() for part in re.split(r"\s+or\s+", text, flags=re.IGNORECASE) if part.strip()]
    if len(or_split) > 1:
        return or_split

    return [text]


def split_predicted_item(item: Dict[str, Any]) -> List[tuple[str, Any]]:
    raw_name = (item.get("name") or "").strip()
    inherited_count = item.get("num")
    if not raw_name:
        return []

    variant_with_shared_count = re.fullmatch(
        r"(?i)(\d+)\s*(?:x\s*)?([A-Za-z]+\d{2,4})[- ](\d{2,3})GB/(\d{2,3})GB",
        raw_name,
    )
    if variant_with_shared_count:
        count = int(variant_with_shared_count.group(1))
        model = variant_with_shared_count.group(2).upper()
        return [
            (f"{model} {variant_with_shared_count.group(3)}GB", count),
            (f"{model} {variant_with_shared_count.group(4)}GB", count),
        ]

    counted_segments = []
    for segment in re.split(r"(?i)\s+and\s+|,", raw_name):
        segment = segment.strip()
        if not segment:
            continue
        match = re.fullmatch(
            r"(?i)(\d+)\s*(?:x\s*)?([A-Za-z]+(?:[-/ ]?[A-Za-z0-9]+)*(?:\s+\d{2,3}GB)?)",
            segment,
        )
        if not match:
            continue
        candidate_name = match.group(2).strip()
        if re.search(r"[A-Za-z]\d{2,4}", candidate_name):
            counted_segments.append((candidate_name, int(match.group(1))))

    if len(counted_segments) >= 2:
        result = []
        for candidate_name, candidate_count in counted_segments:
            for split_name in split_composite_hardware_name(candidate_name):
                result.append((split_name, candidate_count))
        return result

    return [(name, inherited_count) for name in split_composite_hardware_name(raw_name)]


def _build_pred_result(normalized_extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
    sec = []
    pred = []
    for row in normalized_extractions:
        if row["section_number"] not in sec:
            sec.append(row["section_number"])
        pred.append({"name": row["normalized_hardware_name"], "num": row["count"]})
    return {"sec": sec, "pred": pred}


def _load_full_text(section_doc: Dict[str, Any]) -> str:
    if section_doc.get("full_text"):
        return str(section_doc["full_text"])
    source_md_path = section_doc.get("source_md_path")
    if source_md_path:
        return Path(source_md_path).read_text(encoding="utf-8")
    return ""


def _chunk_full_text(full_text: str, chunk_size: int = 3000, overlap: int = 300) -> List[str]:
    text = (full_text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def _dedupe_normalized_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped = []
    seen = set()
    for row in rows:
        key = (
            row.get("normalized_hardware_name"),
            row.get("count"),
            standardize_hardware_text(row.get("evidence", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _dedupe_fulltext_fallback_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped = []
    seen = set()
    for row in rows:
        key = (
            row.get("normalized_hardware_name"),
            row.get("benchmark_hardware_name"),
            row.get("count"),
            row.get("raw_hardware_name"),
            row.get("extraction_source"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _natural_sort_key(text: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text))


def _should_run_fulltext_fallback(
    normalized_extractions: List[Dict[str, Any]],
    review_flags: List[Dict[str, Any]],
) -> bool:
    _ = review_flags
    return not normalized_extractions


def _build_fulltext_fallback_text(
    section_doc: Dict[str, Any],
    candidate_windows: List[Dict[str, Any]],
    raw_extractions: List[Dict[str, Any]],
) -> str:
    full_text = _load_full_text(section_doc)
    if not full_text:
        return ""

    consumed_window_ids = {row.get("window_id") for row in raw_extractions if row.get("window_id")}
    if not consumed_window_ids:
        return full_text

    remaining_text = full_text
    for window in candidate_windows:
        if window["window_id"] not in consumed_window_ids:
            continue
        window_text = (window.get("window_text") or "").strip()
        if not window_text:
            continue
        remaining_text = remaining_text.replace(window_text, " ", 1)
    return remaining_text


def _run_fulltext_fallback(
    section_doc: Dict[str, Any],
    tagger,
    rules: Dict[str, Any],
    catalog: HardwareCatalog,
    candidate_windows: List[Dict[str, Any]],
    raw_extractions: List[Dict[str, Any]],
    normalized_extractions: List[Dict[str, Any]],
    review_flags: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not _should_run_fulltext_fallback(normalized_extractions, review_flags):
        return {
            "fulltext_fallback_triggered": False,
            "fulltext_fallback_windows": [],
            "fulltext_fallback_extractions": [],
            "fulltext_fallback_review_flags": [],
        }

    fallback_text = _build_fulltext_fallback_text(section_doc, candidate_windows, raw_extractions)
    if not fallback_text or not fallback_text.strip():
        return {
            "fulltext_fallback_triggered": False,
            "fulltext_fallback_windows": [],
            "fulltext_fallback_extractions": [],
            "fulltext_fallback_review_flags": [],
        }
    if normalized_extractions and not has_hardware_signal(fallback_text, rules, catalog):
        return {
            "fulltext_fallback_triggered": False,
            "fulltext_fallback_windows": [],
            "fulltext_fallback_extractions": [],
            "fulltext_fallback_review_flags": [],
        }

    windows = []
    extractions = []
    review_flags = []
    for index, chunk in enumerate(_chunk_full_text(fallback_text), start=1):
        window_id = f"{section_doc.get('paper_id', 'paper')}::fulltext::{index}"
        windows.append({"window_id": window_id, "window_text": chunk})
        for chunk_index, chunk_text in enumerate(
            _select_relevant_llm_snippets(chunk, rules=rules, catalog=catalog),
            start=1,
        ):
            try:
                model_output = tagger(chunk_text)
            except QuotaExhaustedError:
                raise
            except FatalModelRequestError:
                raise
            except Exception as exc:
                review_flags.append(
                    {
                        "source_window_id": window_id,
                        "audit_reason": "model_request_failed",
                        "audit_method": "fulltext_fallback_audit",
                        "section_number": "",
                        "section_title": "fulltext fallback",
                        "evidence": chunk_text,
                        "request_error": f"{type(exc).__name__}: {exc}",
                        "chunk_index": chunk_index,
                    }
                )
                continue

            pred_result, parse_error = parse_prediction_result(model_output)
            if parse_error:
                review_flags.append(
                    {
                        "source_window_id": window_id,
                        "audit_reason": "model_output_parse_failed",
                        "audit_method": "fulltext_fallback_audit",
                        "section_number": "",
                        "section_title": "fulltext fallback",
                        "evidence": chunk_text,
                        "parse_error": parse_error,
                        "chunk_index": chunk_index,
                    }
                )
                continue
            for item in pred_result:
                for raw_name, resolved_count in split_predicted_item(item):
                    raw_row = {
                        "window_id": window_id,
                        "hardware_type": infer_hardware_type(raw_name),
                        "raw_hardware_name": raw_name,
                        "count": resolved_count,
                        "count_text": str(resolved_count) if resolved_count is not None else "",
                        "evidence": chunk_text,
                        "section_number": "",
                        "section_title": "fulltext fallback",
                        "normalized_section_title": "fulltext fallback",
                        "extraction_source": "fulltext_fallback",
                    }
                    extractions.append(normalize_extraction(raw_row, catalog))

    return {
        "fulltext_fallback_triggered": True,
        "fulltext_fallback_windows": windows,
        "fulltext_fallback_extractions": _dedupe_fulltext_fallback_rows(extractions),
        "fulltext_fallback_review_flags": review_flags,
    }


def predict_for_section_doc(
    section_doc: Dict[str, Any],
    tagger,
    rules: Dict[str, Any] | None = None,
    catalog: HardwareCatalog | None = None,
) -> Dict[str, Any]:
    if rules is None or catalog is None:
        raise ValueError("rules and catalog are required for section_doc prediction")

    candidate_windows = build_candidate_windows(section_doc, rules, catalog)
    raw_extractions, parse_failures = _build_raw_extractions(candidate_windows, tagger, rules, catalog)
    normalized_extractions = [normalize_extraction(row, catalog) for row in raw_extractions]
    audit = audit_doc_extractions(candidate_windows, normalized_extractions, catalog, parse_failures=parse_failures)
    fallback = _run_fulltext_fallback(
        section_doc,
        tagger,
        rules,
        catalog,
        candidate_windows,
        raw_extractions,
        normalized_extractions,
        audit["review_flags"],
    )
    combined_extractions = _dedupe_normalized_rows(
        normalized_extractions + fallback["fulltext_fallback_extractions"]
    )
    pred_result = _build_pred_result(combined_extractions)
    return {
        "paper_id": section_doc.get("paper_id"),
        "candidate_windows": candidate_windows,
        "raw_extractions": raw_extractions,
        "normalized_extractions": combined_extractions,
        "review_flags": audit["review_flags"] + fallback["fulltext_fallback_review_flags"],
        "fulltext_fallback_triggered": fallback["fulltext_fallback_triggered"],
        "fulltext_fallback_windows": fallback["fulltext_fallback_windows"],
        "fulltext_fallback_extractions": fallback["fulltext_fallback_extractions"],
        "pred_result": pred_result,
    }


def _iter_parse_dirs(input_dir: Path) -> List[Path]:
    parse_dirs = []
    for path in sorted(input_dir.iterdir(), key=lambda item: _natural_sort_key(item.name)):
        if not path.is_dir():
            continue
        if path.name.startswith(".") or path.name == "batches":
            continue
        if not (path / "auto").is_dir():
            continue
        parse_dirs.append(path)
    return parse_dirs


def _predict_single_parse_dir(
    parse_dir: Path,
    target_dir: Path,
    overwrite: bool,
    rules: Dict[str, Any],
    catalog: HardwareCatalog,
    tagger=None,
    tagger_factory=None,
    stop_event: threading.Event | None = None,
) -> Dict[str, Any]:
    if stop_event is not None and stop_event.is_set():
        raise QuotaExhaustedError("GPU extraction stopped after quota exhaustion.")
    output_path = target_dir / f"{parse_dir.name}_gpu.json"
    if output_path.exists() and not overwrite:
        return {"paper_id": parse_dir.name, "status": "skipped", "output_path": output_path}

    if tagger is None:
        if tagger_factory is None:
            raise ValueError("tagger or tagger_factory is required")
        tagger = tagger_factory()

    section_doc = load_section_doc_from_parse_dir(parse_dir)
    result = predict_for_doc(section_doc, tagger, rules=rules, catalog=catalog)
    result["extraction_meta"] = _build_extraction_meta(parse_dir, rules, catalog)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"paper_id": parse_dir.name, "status": "written", "output_path": output_path}


def scan_stale_extraction_outputs(input_dir: str, output_dir: str = "") -> Dict[str, Any]:
    input_path = Path(input_dir)
    rules = load_gpu_rules()
    catalog = load_default_catalog()
    target_dir = Path(output_dir) if output_dir else load_config()["paths"]["gpu_root"] / input_path.name

    parse_dirs = _iter_parse_dirs(input_path)
    stale_files = []
    missing_files = []
    fresh_count = 0

    for parse_dir in parse_dirs:
        output_path = target_dir / f"{parse_dir.name}_gpu.json"
        if not output_path.exists():
            missing_files.append({"paper_id": parse_dir.name, "output_path": str(output_path)})
            continue
        current_meta = _build_extraction_meta(parse_dir, rules, catalog)
        freshness = _classify_output_freshness(output_path, current_meta)
        if freshness["fresh"]:
            fresh_count += 1
            continue
        stale_files.append(
            {
                "paper_id": parse_dir.name,
                "output_path": str(output_path),
                "reason": freshness["reason"],
            }
        )

    return {
        "total": len(parse_dirs),
        "fresh_count": fresh_count,
        "stale_count": len(stale_files),
        "missing_count": len(missing_files),
        "stale_files": stale_files,
        "missing_files": missing_files,
    }


def run_predict(
    input_dir: str,
    overwrite: bool,
    output_dir: str = "",
    workers: int = 1,
    model: str | None = None,
) -> Dict[str, Any]:
    input_path = Path(input_dir)
    rules = load_gpu_rules()
    catalog = load_default_catalog()

    if any(path.is_dir() for path in input_path.iterdir()):
        target_dir = Path(output_dir) if output_dir else load_config()["paths"]["gpu_root"] / input_path.name
        target_dir.mkdir(parents=True, exist_ok=True)
        failures = []
        summary = {
            "processed": 0,
            "written": 0,
            "skipped": 0,
            "succeeded": 0,
            "failed": 0,
            "failures": failures,
        }
        parse_dirs = _iter_parse_dirs(input_path)
        total = len(parse_dirs)
        if workers <= 1:
            tagger = _build_runtime_tagger(model=model)
            for index, parse_dir in enumerate(parse_dirs, start=1):
                summary["processed"] += 1
                print(f"[gpu] {index}/{total} {parse_dir.name}")
                try:
                    task_result = _predict_single_parse_dir(
                        parse_dir,
                        target_dir,
                        overwrite,
                        rules=rules,
                        catalog=catalog,
                        tagger=tagger,
                    )
                except QuotaExhaustedError:
                    print(f"[gpu] stopping after quota exhaustion at {parse_dir.name}")
                    raise
                except FatalModelRequestError as exc:
                    print(f"[gpu] stopping after fatal model request error at {parse_dir.name}: {exc}")
                    raise
                except Exception as exc:
                    summary["failed"] += 1
                    failures.append(
                        {
                            "paper_id": parse_dir.name,
                            "parse_dir": str(parse_dir),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(f"[gpu] failed for {parse_dir.name}: {exc}")
                    continue
                if task_result["status"] == "skipped":
                    summary["skipped"] += 1
                    print(f"[gpu] skipped {parse_dir.name}")
                else:
                    summary["written"] += 1
                    print(f"[gpu] wrote {task_result['output_path'].name}")
                summary["succeeded"] = summary["written"]
        else:
            future_to_meta = {}
            completed = 0
            stop_event = threading.Event()
            tagger_factory = _build_runtime_tagger_factory(model=model)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                for index, parse_dir in enumerate(parse_dirs, start=1):
                    summary["processed"] += 1
                    print(f"[gpu] queued {index}/{total} {parse_dir.name}")
                    future = executor.submit(
                        _predict_single_parse_dir,
                        parse_dir,
                        target_dir,
                        overwrite,
                        rules,
                        catalog,
                        None,
                        tagger_factory,
                        stop_event,
                    )
                    future_to_meta[future] = parse_dir

                for future in concurrent.futures.as_completed(future_to_meta):
                    parse_dir = future_to_meta[future]
                    completed += 1
                    try:
                        task_result = future.result()
                    except QuotaExhaustedError:
                        stop_event.set()
                        for pending_future in future_to_meta:
                            if pending_future is not future:
                                pending_future.cancel()
                        print(f"[gpu] stopping after quota exhaustion at {parse_dir.name}")
                        raise
                    except FatalModelRequestError as exc:
                        stop_event.set()
                        for pending_future in future_to_meta:
                            if pending_future is not future:
                                pending_future.cancel()
                        print(f"[gpu] stopping after fatal model request error at {parse_dir.name}: {exc}")
                        raise
                    except Exception as exc:
                        summary["failed"] += 1
                        failures.append(
                            {
                                "paper_id": parse_dir.name,
                                "parse_dir": str(parse_dir),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        print(f"[gpu] done {completed}/{total} failed for {parse_dir.name}: {exc}")
                        continue
                    if task_result["status"] == "skipped":
                        summary["skipped"] += 1
                        print(f"[gpu] done {completed}/{total} skipped {parse_dir.name}")
                    else:
                        summary["written"] += 1
                        print(f"[gpu] done {completed}/{total} wrote {task_result['output_path'].name}")
                    summary["succeeded"] = summary["written"]
        if failures:
            failure_report = {
                "processed": summary["processed"],
                "written": summary["written"],
                "skipped": summary["skipped"],
                "succeeded": summary["succeeded"],
                "failed": summary["failed"],
                "failures": failures,
            }
            (target_dir / "_gpu_failures.json").write_text(
                json.dumps(failure_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(
            f"[gpu] completed: processed={summary['processed']} "
            f"written={summary['written']} skipped={summary['skipped']} "
            f"failed={summary['failed']}"
        )
        return summary

    tagger = _build_runtime_tagger(model=model)
    summary = {"processed": 0, "written": 0, "skipped": 0, "succeeded": 0, "failed": 0, "failures": []}
    for file_path in sorted(input_path.iterdir()):
        if not file_path.is_file():
            continue
        print(file_path)
        summary["processed"] += 1
        try:
            with file_path.open("r", encoding="utf-8") as f:
                doc = json.load(f)
            current_meta = _build_input_doc_extraction_meta(file_path, doc, rules, catalog)
            if not overwrite:
                freshness = _classify_input_doc_freshness(doc, current_meta)
                if freshness["fresh"]:
                    summary["skipped"] += 1
                    print("skip")
                    continue
            prediction = predict_for_doc(doc, tagger, rules=rules, catalog=catalog)
            if isinstance(prediction, dict):
                if "pred_result" in prediction:
                    doc.update(prediction)
                else:
                    doc["pred_result"] = prediction
            else:
                doc["pred_result"] = prediction
            doc["extraction_meta"] = current_meta
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=4)
            summary["written"] += 1
            summary["succeeded"] += 1
        except QuotaExhaustedError:
            print(f"[gpu] stopping after quota exhaustion at {file_path.name}")
            raise
        except FatalModelRequestError as exc:
            print(f"[gpu] stopping after fatal model request error at {file_path.name}: {exc}")
            raise
        except Exception as exc:
            summary["failed"] += 1
            summary["failures"].append(
                {
                    "source": str(file_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[gpu] failed for {file_path.name}: {exc}")
    return summary


def run_evaluate(
    eval_dir: str,
    match_type: str,
    save_predictions: str,
    model: str | None = None,
) -> None:
    tagger = _build_runtime_tagger(model=model)
    metrics = NERMetrics(match_type=match_type)
    predictions = []

    for file_name in sorted(os.listdir(eval_dir)):
        file_path = os.path.join(eval_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        pred_result = predict_for_doc(doc, tagger)
        predicted = pred_result["pred"]
        gold = doc.get("gpu", [])
        normalize_entity_name(gold)
        metrics.update(predicted, gold)
        predictions.append(
            {
                "source": file_path,
                "sec": pred_result["sec"],
                "predicted": predicted,
                "gold": gold,
            }
        )
        print(file_path)

    result = metrics.compute()
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if save_predictions:
        with open(save_predictions, "w", encoding="utf-8") as f:
            json.dump({"metrics": result, "predictions": predictions}, f, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    config = load_config()
    conference = config["conference"]
    conference_id = f'{conference["name"]}{conference["year"]}'
    parser = argparse.ArgumentParser(description="Extract GPU/TPU information from section content")
    parser.add_argument("--mode", choices=["predict", "evaluate"], default="predict")
    parser.add_argument(
        "--input-dir",
        default=str(config["paths"]["parses_root"] / conference_id),
    )
    parser.add_argument("--output-dir", default="", help="Directory for new results in predict mode")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing pred_result in predict mode")
    parser.add_argument("--workers", type=int, default=1, help="Number of paper-level workers in predict mode")
    parser.add_argument("--model", default=None, help="Override the default LLM model name")
    parser.add_argument("--eval-dir", default="emnlp2025_papers_gpu_train_test")
    parser.add_argument("--match-type", choices=["name_only", "num_only", "exact"], default="name_only")
    parser.add_argument("--save-predictions", default="", help="Save prediction details to a JSON file in evaluate mode")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.mode == "predict":
        run_predict(
            input_dir=args.input_dir,
            overwrite=args.overwrite,
            output_dir=args.output_dir,
            workers=args.workers,
            model=args.model,
        )
    else:
        run_evaluate(
            eval_dir=args.eval_dir,
            match_type=args.match_type,
            save_predictions=args.save_predictions,
            model=args.model,
        )


if __name__ == "__main__":
    main()
