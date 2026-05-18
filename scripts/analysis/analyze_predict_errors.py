import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

try:
    import yaml
except ImportError:
    yaml = None


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
    token = token.replace("7ob", "70b")
    token = token.replace("gpt40", "gpt4o")
    token = re.sub(r"(instruction|instruct)\d+(?:\.\d+)?[bkmg]$", r"\1", token)
    token = re.sub(r"(instruction|instruct|chat|it)$", "", token)
    return token


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
    return ACTIVE_MODEL_ALIASES.get(token, token)


def normalize_model_family(text: str, parameters: str = "") -> str:
    token = normalize_model(text, parameters)
    for suffix in FAMILY_VARIANT_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix):
            return token[: -len(suffix)]
    return token


def normalize_model_surface(text: str) -> str:
    model = str(text).strip()
    model = re.sub(r"(?<=\.)\s+", "", model)
    model = re.sub(r"\s{2,}", " ", model)
    return model


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

    if text in {"none", "null", "na", "n/a"}:
        return ""
    if text == "api" or text.startswith("api("):
        return ""
    if text in {"base", "large", "small", "medium", "mini"}:
        return ""

    if any(ch.isdigit() for ch in text):
        text = text.replace("o", "0")

    text = re.sub(r"(\d+(?:\.\d+)?)\s*(million|mn)\b", r"\1m", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(billion|bn)\b", r"\1b", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(thousand|k)\b", r"\1k", text)
    return text


def canonical_key(entity: Dict[str, Any], match_type: str) -> Tuple[str, str]:
    param = normalize_param(str(entity.get("parameters", "")))
    model = normalize_model(str(entity.get("model", "")), str(entity.get("parameters", "")))
    if match_type == "model_only":
        return model, ""
    if match_type == "parameter_only":
        return "", param
    return model, param


def is_likely_noise_entity(model: str, parameters: str) -> bool:
    model_norm = normalize_text(model)
    param_norm = normalize_param(parameters)
    model_token = normalize_model(model)
    compact = normalize_model_token(model)

    if model_norm in {"model", "full model", "base model", "our model"}:
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?[bkmg]\s*model", model_norm):
        return True
    if model_norm in {"llm", "llms", "vlm", "vlms", "lvlm", "lvlms"}:
        return True
    if re.search(r"\b(series|family)\b", model_norm):
        return True
    if model_norm in {"direct", "synthesis", "baseline"}:
        return True
    if model_norm.endswith("direct") and len(model_norm) <= 20:
        return True
    if re.search(r"\b(benchmark|bench|dataset|corpus|metric|score|accuracy|rouge|bleu|f1)\b", model_norm):
        return True
    if model_norm.endswith("parser"):
        return True
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
    if re.fullmatch(r"(?:[a-z0-9]{1,3}\s+){2,}[a-z0-9]{1,3}", model_norm):
        return True
    if len(model_token) <= 1:
        return True
    if not model_token and param_norm:
        return True
    return False


def clean_entities(items: Any, match_type: str, dedup: bool, drop_noise: bool) -> List[Dict[str, str]]:
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


def entities_match(pred: Dict[str, str], gold: Dict[str, str], match_type: str) -> bool:
    pred_param = normalize_param(str(pred.get("parameters", "")))
    gold_param = normalize_param(str(gold.get("parameters", "")))
    pred_model = normalize_model(str(pred.get("model", "")), str(pred.get("parameters", "")))
    gold_model = normalize_model(str(gold.get("model", "")), str(gold.get("parameters", "")))
    pred_family = normalize_model_family(str(pred.get("model", "")), str(pred.get("parameters", "")))
    gold_family = normalize_model_family(str(gold.get("model", "")), str(gold.get("parameters", "")))

    if match_type == "model_only":
        return (pred_model == gold_model) or (pred_family == gold_family)
    if match_type == "parameter_only":
        return pred_param == gold_param
    return ((pred_model == gold_model) or (pred_family == gold_family)) and (pred_param == gold_param)


def short_source(path: str) -> str:
    if not path:
        return ""
    norm = path.replace("\\", "/")
    return os.path.basename(norm)


def run_analysis(
    payload: Dict[str, Any],
    match_type: str,
    dedup: bool,
    top_k: int,
    paper_top_k: int,
    pred_filter: bool,
) -> Dict[str, Any]:
    predictions = payload.get("predictions", [])
    if not isinstance(predictions, list):
        raise ValueError("JSON is missing a valid `predictions` list.")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    fp_counter: Counter = Counter()
    fn_counter: Counter = Counter()
    model_param_mismatch_counter: Counter = Counter()
    fp_example: Dict[Tuple[str, str], Dict[str, str]] = {}
    fn_example: Dict[Tuple[str, str], Dict[str, str]] = {}
    mismatch_example: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    paper_rows: List[Dict[str, Any]] = []

    for row in predictions:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", ""))
        pred_list = clean_entities(row.get("predicted", []), match_type, dedup, drop_noise=pred_filter)
        gold_list = clean_entities(row.get("gold", []), match_type, dedup, drop_noise=False)

        matched_gold = [False] * len(gold_list)
        unmatched_pred: List[Dict[str, str]] = []

        tp = 0
        for pred in pred_list:
            hit = False
            for i, gold in enumerate(gold_list):
                if not matched_gold[i] and entities_match(pred, gold, match_type):
                    matched_gold[i] = True
                    hit = True
                    tp += 1
                    break
            if not hit:
                unmatched_pred.append(pred)

        unmatched_gold = [gold_list[i] for i in range(len(gold_list)) if not matched_gold[i]]
        fp = len(unmatched_pred)
        fn = len(unmatched_gold)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        for pred in unmatched_pred:
            key = (
                normalize_model(pred.get("model", ""), pred.get("parameters", "")),
                normalize_param(pred.get("parameters", "")),
            )
            fp_counter[key] += 1
            fp_example.setdefault(
                key,
                {"model": str(pred.get("model", "")), "parameters": str(pred.get("parameters", ""))},
            )
        for gold in unmatched_gold:
            key = (
                normalize_model(gold.get("model", ""), gold.get("parameters", "")),
                normalize_param(gold.get("parameters", "")),
            )
            fn_counter[key] += 1
            fn_example.setdefault(
                key,
                {"model": str(gold.get("model", "")), "parameters": str(gold.get("parameters", ""))},
            )

        if match_type == "exact":
            used_gold = set()
            for pred in unmatched_pred:
                pred_model = normalize_model(pred.get("model", ""), pred.get("parameters", ""))
                pred_param = normalize_param(pred.get("parameters", ""))
                for i, gold in enumerate(unmatched_gold):
                    if i in used_gold:
                        continue
                    gold_model = normalize_model(gold.get("model", ""), gold.get("parameters", ""))
                    gold_param = normalize_param(gold.get("parameters", ""))
                    if pred_model and pred_model == gold_model and pred_param != gold_param:
                        mismatch_key = (pred_model, pred_param, gold_param)
                        model_param_mismatch_counter[mismatch_key] += 1
                        mismatch_example.setdefault(
                            mismatch_key,
                            {
                                "model": pred.get("model", ""),
                                "pred_parameters": pred.get("parameters", ""),
                                "gold_parameters": gold.get("parameters", ""),
                            },
                        )
                        used_gold.add(i)
                        break

        paper_rows.append(
            {
                "source": source,
                "source_file": short_source(source),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "pred_count": len(pred_list),
                "gold_count": len(gold_list),
                "error_count": fp + fn,
                "fp_examples": unmatched_pred[:3],
                "fn_examples": unmatched_gold[:3],
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    worst_papers = sorted(paper_rows, key=lambda x: (-x["error_count"], -x["fn"], -x["fp"]))[:paper_top_k]

    top_fp = [
        {
            "model": fp_example.get((model, param), {}).get("model", model),
            "parameters": fp_example.get((model, param), {}).get("parameters", param),
            "normalized_model": model,
            "normalized_parameters": param,
            "count": count,
        }
        for (model, param), count in fp_counter.most_common(top_k)
    ]
    top_fn = [
        {
            "model": fn_example.get((model, param), {}).get("model", model),
            "parameters": fn_example.get((model, param), {}).get("parameters", param),
            "normalized_model": model,
            "normalized_parameters": param,
            "count": count,
        }
        for (model, param), count in fn_counter.most_common(top_k)
    ]
    top_mismatch = [
        {
            "model": mismatch_example.get((model, pred_param, gold_param), {}).get("model", model),
            "pred_parameters": mismatch_example.get((model, pred_param, gold_param), {}).get(
                "pred_parameters", pred_param
            ),
            "gold_parameters": mismatch_example.get((model, pred_param, gold_param), {}).get(
                "gold_parameters", gold_param
            ),
            "normalized_model": model,
            "normalized_pred_parameters": pred_param,
            "normalized_gold_parameters": gold_param,
            "count": count,
        }
        for (model, pred_param, gold_param), count in model_param_mismatch_counter.most_common(top_k)
    ]

    return {
        "summary": {
            "match_type": match_type,
            "deduplicate": dedup,
            "pred_filter": pred_filter,
            "num_samples": len(paper_rows),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "support": total_tp + total_fn,
        },
        "worst_papers": worst_papers,
        "top_false_positives": top_fp,
        "top_false_negatives": top_fn,
        "top_model_param_mismatches": top_mismatch,
    }


def print_report(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    print("=== Summary ===")
    print(
        "match_type={match_type} dedup={dedup} pred_filter={pred_filter} samples={samples}".format(
            match_type=summary["match_type"],
            dedup=summary["deduplicate"],
            pred_filter=summary["pred_filter"],
            samples=summary["num_samples"],
        )
    )
    print(
        "precision={:.4f} recall={:.4f} f1={:.4f} tp={} fp={} fn={} support={}".format(
            summary["precision"],
            summary["recall"],
            summary["f1"],
            summary["tp"],
            summary["fp"],
            summary["fn"],
            summary["support"],
        )
    )

    print("\n=== Worst Papers (by fp+fn) ===")
    for r in report["worst_papers"]:
        print(
            "- {} | tp={} fp={} fn={} pred={} gold={} error={}".format(
                r["source_file"] or r["source"],
                r["tp"],
                r["fp"],
                r["fn"],
                r["pred_count"],
                r["gold_count"],
                r["error_count"],
            )
        )

    print("\n=== Top False Positives ===")
    for r in report["top_false_positives"]:
        print("- model={} parameters={} count={}".format(r["model"], r["parameters"], r["count"]))

    print("\n=== Top False Negatives ===")
    for r in report["top_false_negatives"]:
        print("- model={} parameters={} count={}".format(r["model"], r["parameters"], r["count"]))

    if report["top_model_param_mismatches"]:
        print("\n=== Top Model-Parameter Mismatches ===")
        for r in report["top_model_param_mismatches"]:
            print(
                "- model={} pred_param={} gold_param={} count={}".format(
                    r["model"],
                    r["pred_parameters"],
                    r["gold_parameters"],
                    r["count"],
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prediction errors between predict.json results and gold labels.")
    parser.add_argument("--input", default="predict_model_only.json", help="Prediction detail JSON file path")
    parser.add_argument("--config", default="config.yaml", help="Optional config file matching the evaluation script, used for model aliases")
    parser.add_argument(
        "--match-type",
        choices=["exact", "model_only", "parameter_only"],
        default="model_only",
        help="Match type: exact=model and parameter must match; model_only=model only; parameter_only=parameter only",
    )
    parser.add_argument("--no-dedup", action="store_true", help="Do not deduplicate entities within each sample")
    parser.add_argument("--no-pred-filter", action="store_true", help="Do not apply noise filtering to predicted entities")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top error entities")
    parser.add_argument("--paper-top-k", type=int, default=20, help="Number of top error papers")
    parser.add_argument("--output-json", default="./error_report_model_only.json", help="Optional path to save the analysis result as JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_model_aliases(config)

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    report = run_analysis(
        payload=payload,
        match_type=args.match_type,
        dedup=not args.no_dedup,
        top_k=args.top_k,
        paper_top_k=args.paper_top_k,
        pred_filter=not args.no_pred_filter,
    )
    print_report(report)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved analysis result: {args.output_json}")


if __name__ == "__main__":
    main()
