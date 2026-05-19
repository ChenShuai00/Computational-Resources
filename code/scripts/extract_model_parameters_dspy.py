import argparse
import ast
import json
import os
import re
from typing import Any, Dict, List, Tuple

try:
    import yaml
except ImportError:
    yaml = None


INSTRUCTION = """
### Task
Extract only concrete model names and corresponding parameter scales from paper text.

### Output Format
Return a JSON array:
[{"model": "<model name>", "parameters": "<parameter scale or context length or empty string>"}]

### Inclusion Rules
- Include named models/checkpoints/APIs, e.g., GPT-4, Claude-3.5-Sonnet, Llama-3.1-8B-Instruct, Qwen2.5-32B.
- `parameters` only keeps scales/context length such as 124M, 7B, 70B, 1.1B, 32K, 128K.
- If parameter scale is not explicitly given, use empty string.

### Exclusion Rules
- Exclude training methods/algorithms: DPO, LoRA, KTO, GRPO, RLHF.
- Exclude benchmarks/datasets/metrics: MMBench, BLEU, ROUGE, F1, CLIPScore.
- Exclude generic nouns without concrete identity, e.g., "language model", "encoder", "baseline model".
- Exclude hardware names.

### Constraints
- Return JSON only, no explanation.
- If nothing valid is found, return [].
"""


DEMOS_TEXT = """
input: We evaluate Llama2-7B, Llama2-13B and Qwen2.5-32B on downstream tasks.
output: [{"model": "Llama2-7B", "parameters": "7B"}, {"model": "Llama2-13B", "parameters": "13B"}, {"model": "Qwen2.5-32B", "parameters": "32B"}]

input: We compare GPT-4 and Claude-3.5 in our human study.
output: [{"model": "GPT-4", "parameters": ""}, {"model": "Claude-3.5", "parameters": ""}]

input: We train with LoRA and DPO, and evaluate on MMBench with BLEU and ROUGE.
output: []

input: We use Claude-3.5-Sonnet and Doubao-Pro-128K APIs for comparison.
output: [{"model": "Claude-3.5-Sonnet", "parameters": ""}, {"model": "Doubao-Pro-128K", "parameters": "128K"}]

input: The experiments use AdamW with lr=1e-4 and batch size 64.
output: []
"""

# Canonical aliases over compact model tokens (alnum-only, lowercase).
MODEL_ALIASES: Dict[str, str] = {
    "gpt4o": "gpt4o",
    "gpt4o20240806": "gpt4o",
    "gpt4o20240513": "gpt4o",
    "gpt4omini": "gpt4omini",
    "gpt40mini": "gpt4omini",
    "gpt4omini20240718": "gpt4omini",
    "chatgpt": "chatgpt",
    "qwenl5": "qwen15",
    "qwenl514b": "qwen1514b",
    "qwenl572b": "qwen1572b",
}
ACTIVE_MODEL_ALIASES: Dict[str, str] = dict(MODEL_ALIASES)
FAMILY_VARIANT_SUFFIXES = ("base", "large", "xl", "xxl")


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def normalize_model_token(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"(?<=\.)\s+", "", text)
    token = re.sub(r"[^a-z0-9]+", "", text)

    # Common OCR / formatting noise.
    token = token.replace("7ob", "70b")
    token = token.replace("gpt40", "gpt4o")

    # Common formatting variant: "...instruct3b" vs "...instruct".
    token = re.sub(r"(instruction|instruct)\d+(?:\.\d+)?[bkmg]$", r"\1", token)

    # Remove optional suffixes frequently added in prose.
    token = re.sub(r"(instruction|instruct|chat|it)$", "", token)
    return token


def configure_model_aliases(config: Dict[str, Any]) -> None:
    global ACTIVE_MODEL_ALIASES
    aliases = dict(MODEL_ALIASES)

    llm_cfg = config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {}
    raw_aliases = llm_cfg.get("model_aliases", {})
    if isinstance(raw_aliases, dict):
        for src, dst in raw_aliases.items():
            src_token = normalize_model_token(str(src))
            dst_token = normalize_model_token(str(dst))
            if src_token and dst_token:
                aliases[src_token] = dst_token

    ACTIVE_MODEL_ALIASES = aliases


def normalize_param(text: str) -> str:
    text = normalize_text(text)
    text = text.replace(" ", "")
    text = text.replace(",", "")
    text = text.replace("_", "")
    text = text.replace("-", "")

    # Normalize null-like values first (before OCR fixes such as o -> 0).
    if text in {"none", "null", "na", "n/a"}:
        return ""

    # Treat API markers as missing parameter scales.
    if text == "api" or text.startswith("api("):
        return ""

    # Non-scale size adjectives are usually model variant names, not parameter scales.
    if text in {"base", "large", "small", "medium", "mini"}:
        return ""

    # OCR noise normalization for alphanumeric scales (e.g., 7OB -> 70B).
    if any(ch.isdigit() for ch in text):
        text = text.replace("o", "0")

    # Normalize magnitude words: 380 million -> 380m, 1.1 billion -> 1.1b
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(million|mn)\b", r"\1m", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(billion|bn)\b", r"\1b", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(thousand|k)\b", r"\1k", text)
    return text


def strip_param_suffix_from_model_token(token: str, param_norm: str) -> str:
    if not token or not param_norm:
        return token
    compact_param = re.sub(r"[^a-z0-9]+", "", param_norm)
    if compact_param and token.endswith(compact_param):
        return token[: -len(compact_param)]
    if token.endswith(param_norm):
        return token[: -len(param_norm)]
    return token


def normalize_model(text: str, parameters: str = "") -> str:
    token = normalize_model_token(text)
    token = strip_param_suffix_from_model_token(token, normalize_param(parameters))
    # Canonical alias mapping (supports config extension/override).
    return ACTIVE_MODEL_ALIASES.get(token, token)


def normalize_model_family(text: str, parameters: str = "") -> str:
    token = normalize_model(text, parameters)
    for suffix in FAMILY_VARIANT_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix):
            return token[: -len(suffix)]
    return token


def normalize_model_surface(text: str) -> str:
    model = str(text).strip()
    # Normalize spacing after dots: "MR. Judge-3B" -> "MR.Judge-3B"
    model = re.sub(r"(?<=\.)\s+", "", model)
    model = re.sub(r"\s{2,}", " ", model)
    return model


def parse_prediction(raw_text: str) -> List[Dict[str, Any]]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return []
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            result = ast.literal_eval(raw_text)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(result, list):
        return []

    parsed = []
    for item in result:
        if not isinstance(item, dict):
            continue
        model = normalize_model_surface(item.get("model", ""))
        parameters = str(item.get("parameters", "")).strip()
        if not model:
            continue
        parsed.append({"model": model, "parameters": parameters})
    return parsed


def get_content_blocks_with_sections(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    block_items: List[Dict[str, str]] = []
    seen_sections = set()

    for item in doc.get("gpu_content", []):
        sec_num = str(item.get("section_number", "")).strip()
        content = item.get("content")
        if sec_num and sec_num != "-1" and isinstance(content, str) and content.strip() and sec_num not in seen_sections:
            block_items.append({"sec_num": sec_num, "sec_content": content})
            seen_sections.add(sec_num)

        for sub in item.get("sub_section", []):
            sub_num = str(sub.get("section_number", "")).strip()
            sub_content = sub.get("content")
            if (
                sub_num
                and sub_num != "-1"
                and isinstance(sub_content, str)
                and sub_content.strip()
                and sub_num not in seen_sections
            ):
                block_items.append({"sec_num": sub_num, "sec_content": sub_content})
                seen_sections.add(sub_num)

    return block_items


def extract_blocks_from_sectioned_doc(sectioned_doc: Dict[str, Any], target_sections: List[str]) -> List[str]:
    sections = sectioned_doc.get("sections", [])
    appendix = sectioned_doc.get("appendix", [])
    if not isinstance(sections, list):
        sections = []
    if not isinstance(appendix, list):
        appendix = []
    all_sections = sections + appendix

    normalized_targets = []
    for raw in target_sections:
        tok = normalize_text(str(raw))
        if tok and tok != "nan":
            normalized_targets.append(tok)

    if not normalized_targets:
        return []

    if "appendices" in normalized_targets:
        normalized_targets.append("appendix")

    selected = []
    for section in all_sections:
        if not isinstance(section, dict):
            continue
        sec_no = normalize_text(section.get("section_number", ""))
        sec_title = normalize_text(section.get("section_title", ""))
        raw_title = normalize_text(section.get("raw_section_title", ""))
        matched = any(t == sec_no or t == sec_title or t in raw_title for t in normalized_targets)

        if not matched:
            for t in normalized_targets:
                if not t.startswith("appendix"):
                    continue
                suffix = t.replace("appendix", "", 1).strip()
                if not suffix:
                    if "appendix" in sec_title or "appendix" in raw_title:
                        matched = True
                        break
                if suffix and (suffix == sec_no or raw_title.startswith(f"{suffix} ") or sec_title.startswith(suffix)):
                    matched = True
                    break

        if matched:
            selected.append(section)

    blocks = []
    for section in selected:
        content = section.get("content")
        if isinstance(content, str) and content.strip():
            blocks.append(content)
        for sub in section.get("sub_section", []):
            if not isinstance(sub, dict):
                continue
            sub_content = sub.get("content")
            if isinstance(sub_content, str) and sub_content.strip():
                blocks.append(sub_content)
    return blocks


def extract_blocks_with_sections_from_sectioned_doc(
    sectioned_doc: Dict[str, Any], target_sections: List[str]
) -> List[Dict[str, str]]:
    sections = sectioned_doc.get("sections", [])
    appendix = sectioned_doc.get("appendix", [])
    if not isinstance(sections, list):
        sections = []
    if not isinstance(appendix, list):
        appendix = []
    all_sections = sections + appendix

    normalized_targets = []
    for raw in target_sections:
        tok = normalize_text(str(raw))
        if tok and tok != "nan":
            normalized_targets.append(tok)

    if not normalized_targets:
        return []

    if "appendices" in normalized_targets:
        normalized_targets.append("appendix")

    selected = []
    for section in all_sections:
        if not isinstance(section, dict):
            continue
        sec_no = normalize_text(section.get("section_number", ""))
        sec_title = normalize_text(section.get("section_title", ""))
        raw_title = normalize_text(section.get("raw_section_title", ""))
        matched = any(t == sec_no or t == sec_title or t in raw_title for t in normalized_targets)

        if not matched:
            for t in normalized_targets:
                if not t.startswith("appendix"):
                    continue
                suffix = t.replace("appendix", "", 1).strip()
                if not suffix:
                    if "appendix" in sec_title or "appendix" in raw_title:
                        matched = True
                        break
                if suffix and (suffix == sec_no or raw_title.startswith(f"{suffix} ") or sec_title.startswith(suffix)):
                    matched = True
                    break

        if matched:
            selected.append(section)

    block_items = []
    seen_sections = set()
    for section in selected:
        sec_num = str(section.get("section_number", "")).strip()
        content = section.get("content")
        if sec_num and sec_num != "-1" and isinstance(content, str) and content.strip() and sec_num not in seen_sections:
            block_items.append({"sec_num": sec_num, "sec_content": content})
            seen_sections.add(sec_num)
        for sub in section.get("sub_section", []):
            if not isinstance(sub, dict):
                continue
            sub_num = str(sub.get("section_number", "")).strip()
            sub_content = sub.get("content")
            if (
                sub_num
                and sub_num != "-1"
                and isinstance(sub_content, str)
                and sub_content.strip()
                and sub_num not in seen_sections
            ):
                block_items.append({"sec_num": sub_num, "sec_content": sub_content})
                seen_sections.add(sub_num)
    return block_items


def get_fallback_content_blocks_with_sections(
    doc: Dict[str, Any], file_path: str, input_dir: str, config: Dict[str, Any]
) -> List[Dict[str, str]]:
    paths_cfg = config.get("paths", {}) if isinstance(config.get("paths", {}), dict) else {}
    paper_section_root = str(paths_cfg.get("paper_section_root", "")).strip() or "paper_section"
    subset_name = os.path.basename(os.path.normpath(input_dir))

    file_name = os.path.basename(file_path)
    if not file_name.endswith("_gpu.json"):
        return []
    paper_id = file_name[: -len("_gpu.json")]
    sectioned_name = f"{paper_id}_sectioned.json"
    sectioned_path = os.path.join(paper_section_root, subset_name, sectioned_name)

    if not os.path.exists(sectioned_path):
        return []

    raw_sections = doc.get("raw_gpu_sections", [])
    if not isinstance(raw_sections, list):
        return []

    with open(sectioned_path, "r", encoding="utf-8") as f:
        sectioned_doc = json.load(f)
    return extract_blocks_with_sections_from_sectioned_doc(sectioned_doc=sectioned_doc, target_sections=raw_sections)


def deduplicate_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[Tuple[str, str]] = set()
    deduped: List[Dict[str, str]] = []
    for item in entities:
        model = normalize_model_surface(item.get("model", ""))
        parameters = str(item.get("parameters", "")).strip()
        key = (normalize_model(model, parameters), normalize_param(parameters))
        if not model or key in seen:
            continue
        seen.add(key)
        deduped.append({"model": model, "parameters": parameters})
    return deduped


GENERIC_MODEL_TOKENS = {
    "deepseek",
    "gemini",
    "claude",
    "mistral",
    "llama",
    "qwen",
    "gpt",
    "whisper",
    "glm4air",
}


def prune_generic_specific_conflicts(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not entities:
        return entities
    normalized = [
        normalize_model(item.get("model", ""), item.get("parameters", "")) for item in entities
    ]
    drop_idx = set()
    for i, tok_i in enumerate(normalized):
        if tok_i not in GENERIC_MODEL_TOKENS:
            continue
        for j, tok_j in enumerate(normalized):
            if i == j:
                continue
            if tok_j.startswith(tok_i) and len(tok_j) > len(tok_i):
                drop_idx.add(i)
                break
    return [item for idx, item in enumerate(entities) if idx not in drop_idx]


def canonical_key(entity: Dict[str, Any], match_type: str) -> Tuple[str, str]:
    param = normalize_param(str(entity.get("parameters", "")))
    model = normalize_model(str(entity.get("model", "")), str(entity.get("parameters", "")))
    if match_type == "model_only":
        return model, ""
    if match_type == "parameter_only":
        return "", param
    return model, param


def clean_entities_for_eval(items: Any, match_type: str, dedup: bool, drop_noise: bool) -> List[Dict[str, str]]:
    if not isinstance(items, list):
        return []
    cleaned: List[Dict[str, str]] = []
    seen = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        model = str(it.get("model", "")).strip()
        if not model:
            model = str(it.get("model_name", "")).strip()
        model = normalize_model_surface(model)
        param = str(it.get("parameters", "")).strip()
        if not model and not param:
            continue
        if drop_noise and is_likely_noise_entity(model, param):
            continue
        entity = {"model": model, "parameters": param}
        if dedup:
            key = canonical_key(entity, match_type)
            if key in seen:
                continue
            seen.add(key)
        cleaned.append(entity)
    return cleaned


def is_likely_noise_entity(model: str, parameters: str) -> bool:
    model_norm = normalize_text(model)
    param_norm = normalize_param(parameters)
    model_token = normalize_model(model)
    compact = normalize_model_token(model)

    # Generic placeholders that are usually not concrete model names.
    if model_norm in {"model", "full model", "base model", "our model"}:
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?[bkmg]\s*model", model_norm):
        return True
    if model_norm in {"llm", "llms", "vlm", "vlms", "lvlm", "lvlms"}:
        return True
    if re.search(r"\b(series|family)\b", model_norm):
        return True

    # Ablation-style aliases without a concrete base model.
    if model_norm in {"direct", "synthesis", "baseline"}:
        return True
    if model_norm.endswith("direct") and len(model_norm) <= 20:
        return True

    # Common non-model terms that frequently appear in experiment sections.
    if re.search(r"\b(benchmark|bench|dataset|corpus|metric|score|accuracy|rouge|bleu|f1)\b", model_norm):
        return True
    if model_norm.endswith("parser"):
        return True

    # Explicit non-model entities (training methods / metrics / evaluation assets).
    if compact in {
        "lora",
        "dpo",
        "kto",
        "grpo",
        "rlhf",
        "mmbench",
        "mmbenchdev",
        "clipscore",
        "cometkiwi",
        "fleur",
    }:
        return True

    # OCR garbage patterns: many short fragments like "bas eu ien".
    if re.fullmatch(r"(?:[a-z0-9]{1,3}\s+){2,}[a-z0-9]{1,3}", model_norm):
        return True

    # Empty/very short names are usually parser noise.
    if len(model_token) <= 1:
        return True

    # Parameter only with no usable model name.
    if not model_token and param_norm:
        return True

    return False


def entities_match(pred_entity: Dict[str, str], gold_entity: Dict[str, str], match_type: str) -> bool:
    pred_param = normalize_param(pred_entity.get("parameters", ""))
    gold_param = normalize_param(gold_entity.get("parameters", ""))
    pred_model = normalize_model(pred_entity.get("model", ""), pred_entity.get("parameters", ""))
    gold_model = normalize_model(gold_entity.get("model", ""), gold_entity.get("parameters", ""))
    pred_family = normalize_model_family(pred_entity.get("model", ""), pred_entity.get("parameters", ""))
    gold_family = normalize_model_family(gold_entity.get("model", ""), gold_entity.get("parameters", ""))

    if match_type == "model_only":
        return (pred_model == gold_model) or (pred_family == gold_family)
    if match_type == "parameter_only":
        return pred_param == gold_param
    return ((pred_model == gold_model) or (pred_family == gold_family)) and (pred_param == gold_param)


class NERMetrics:
    def __init__(self, match_type: str = "exact"):
        self.match_type = match_type
        self.reset()

    def reset(self) -> None:
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0

    def update(self, pred_list: List[Dict[str, str]], gold_list: List[Dict[str, str]]) -> None:
        matched_gold = [False] * len(gold_list)

        for pred_entity in pred_list:
            matched = False
            for i, gold_entity in enumerate(gold_list):
                if not matched_gold[i] and entities_match(pred_entity, gold_entity, self.match_type):
                    matched_gold[i] = True
                    matched = True
                    self.total_tp += 1
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


def load_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    if yaml is None:
        raise RuntimeError("Missing dependency: pyyaml. Install with `pip install pyyaml`.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def resolve_api_key(config: Dict[str, Any], cli_api_key: str) -> str:
    if cli_api_key:
        return cli_api_key

    llm_cfg = config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {}
    key_from_cfg = str(llm_cfg.get("deepseek_api_key", "")).strip()

    if key_from_cfg.startswith("${") and key_from_cfg.endswith("}"):
        env_name = key_from_cfg[2:-1].strip()
        if env_name:
            return os.getenv(env_name, "")
    if key_from_cfg:
        return key_from_cfg
    return os.getenv("DEEPSEEK_API_KEY", "")


def build_tagger(config: Dict[str, Any], api_key: str):
    try:
        import dspy
    except ImportError as exc:
        raise RuntimeError("Missing dependency: dspy. Install with `pip install dspy`.") from exc

    llm_cfg = config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {}
    model = str(llm_cfg.get("model", "deepseek/deepseek-chat"))
    api_base = str(llm_cfg.get("api_base", "https://api.deepseek.com"))
    deepseek_api_key = api_key
    if not deepseek_api_key:
        raise RuntimeError("Missing DeepSeek API key. Set llm.deepseek_api_key in config or DEEPSEEK_API_KEY env.")

    lm = dspy.LM(
        model=model,
        api_base=api_base,
        api_key=deepseek_api_key,
        max_tokens=1024,
    )
    dspy.settings.configure(lm=lm)

    class ExtractModelParameter(dspy.Signature):
        instruction = dspy.InputField(desc="Task description")
        input = dspy.InputField(desc="Text block")
        output = dspy.OutputField(desc="Extracted model-parameter list")

    class ExtractModelParameterTagger(dspy.Module):
        def __init__(self, instruction_template: str):
            super().__init__()
            self.instruction_template = instruction_template
            self.predictor = dspy.Predict(ExtractModelParameter)

        def forward(self, input: str) -> str:
            full_instruction = f"{self.instruction_template}\n\n### Examples:\n\n{DEMOS_TEXT}"
            result = self.predictor(instruction=full_instruction, input=input)
            return result.output

    return ExtractModelParameterTagger(INSTRUCTION)


def predict_for_doc(
    doc: Dict[str, Any], tagger, block_items: List[Dict[str, str]] | None = None, contents: List[str] | None = None
) -> Dict[str, Any]:
    if block_items is None:
        if contents is None:
            block_items = get_content_blocks_with_sections(doc)
        else:
            block_items = [{"sec_num": "", "sec_content": c} for c in contents if isinstance(c, str) and c.strip()]

    pred_list: List[Dict[str, str]] = []
    pred_sec: List[str] = []

    for block in block_items:
        sec_num = str(block.get("sec_num", "")).strip()
        content = block.get("sec_content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        pred_result = parse_prediction(tagger(content))
        if pred_result:
            pred_list.extend(pred_result)
            if sec_num and sec_num not in pred_sec:
                pred_sec.append(sec_num)

    pred_list = deduplicate_entities(pred_list)
    pred_list = prune_generic_specific_conflicts(pred_list)
    pred_list = [
        item
        for item in pred_list
        if not is_likely_noise_entity(item.get("model", ""), item.get("parameters", ""))
    ]
    return {"sec": pred_sec, "pred": pred_list}


def run_predict(input_dir: str, overwrite: bool, config: Dict[str, Any], api_key: str) -> None:
    tagger = build_tagger(config=config, api_key=api_key)

    for file_name in sorted(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        if not file_name.endswith(".json"):
            continue

        print(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        if not overwrite and "pred_model_parameters" in doc:
            print("skip")
            continue

        block_items = get_content_blocks_with_sections(doc)
        if not block_items:
            block_items = get_fallback_content_blocks_with_sections(
                doc=doc, file_path=file_path, input_dir=input_dir, config=config
            )

        doc["pred_model_parameters"] = predict_for_doc(doc, tagger, block_items=block_items)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)


def run_evaluate(input_dir: str, match_type: str, save_predictions: str, config: Dict[str, Any], api_key: str) -> None:
    tagger = build_tagger(config=config, api_key=api_key)
    metrics = NERMetrics(match_type=match_type)
    predictions = []
    empty_input_docs: List[str] = []
    fallback_filled_docs: List[str] = []

    for file_name in sorted(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        if not file_name.endswith(".json"):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        block_items = get_content_blocks_with_sections(doc)
        if not block_items:
            fallback_block_items = get_fallback_content_blocks_with_sections(
                doc=doc, file_path=file_path, input_dir=input_dir, config=config
            )
            if fallback_block_items:
                block_items = fallback_block_items
                fallback_filled_docs.append(file_path)
            else:
                empty_input_docs.append(file_path)

        pred_result = predict_for_doc(doc, tagger, block_items=block_items)
        predicted_raw = pred_result["pred"]
        predicted = clean_entities_for_eval(predicted_raw, match_type=match_type, dedup=True, drop_noise=True)
        gold = doc.get("model_parameters", [])
        gold_cleaned = clean_entities_for_eval(gold, match_type=match_type, dedup=True, drop_noise=False)
        metrics.update(predicted, gold_cleaned)

        predictions.append(
            {
                "source": file_path,
                "predicted": predicted,
                "gold": gold_cleaned,
            }
        )
        print(file_path)

    result = metrics.compute()
    result["empty_input_docs"] = len(empty_input_docs)
    result["fallback_filled_docs"] = len(fallback_filled_docs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if fallback_filled_docs:
        print("info: fallback section content used for docs:")
        for p in fallback_filled_docs:
            print(p)
    if empty_input_docs:
        print("warning: empty gpu_content docs:")
        for p in empty_input_docs:
            print(p)

    if save_predictions:
        with open(save_predictions, "w", encoding="utf-8") as f:
            json.dump(
                {"metrics": result, "empty_input_docs": empty_input_docs, "predictions": predictions},
                f,
                ensure_ascii=False,
                indent=2,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract model parameter scales from papers with DSPy")
    parser.add_argument("--mode", choices=["predict", "evaluate"], default="evaluate")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input-dir", default="paper_section_gpu_test/emnlp2025")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing pred_model_parameters in predict mode")
    parser.add_argument("--api-key", default="", help="Optional direct DeepSeek API key; highest priority")
    parser.add_argument(
        "--match-type",
        choices=["model_only", "parameter_only", "exact"],
        default="exact",
        help="Evaluation matching mode",
    )
    parser.add_argument("--save-predictions", default="./predict.json", help="Save prediction details to a JSON file in evaluate mode")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_model_aliases(config)
    api_key = resolve_api_key(config=config, cli_api_key=args.api_key)

    if args.mode == "predict":
        run_predict(input_dir=args.input_dir, overwrite=args.overwrite, config=config, api_key=api_key)
    else:
        run_evaluate(
            input_dir=args.input_dir,
            match_type=args.match_type,
            save_predictions=args.save_predictions,
            config=config,
            api_key=api_key,
        )


if __name__ == "__main__":
    main()
