import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple


HIGH_CONF_DROP_EXACT = {
    "alignment-handbook",
    "evol instruct",
}

# Method-like terms that frequently appear in papers but are not always model checkpoints.
METHOD_REVIEW_PATTERNS = [
    r"\bprefix tuning\b",
    r"\bprompt tuning\b",
    r"\bbottleneck adapter\b",
    r"\bdouble bottleneck adapter\b",
    r"\bcompacter\b",
    r"\bpissa\b",
    r"\bhdpissa\b",
    r"\bhd-pissa\b",
    r"\bmora\b",
    r"\bhira\b",
    r"\bdora\b",
    r"^longlora$",
]

# Concrete checkpoint-like patterns that should be kept.
KEEP_PATTERNS = [
    r"\brlhflow-prm-",
    r"\b(instruct|preview|verifier|distill|distilled|coder|chat)\b",
]


def normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def classify_model_name(name: str) -> Tuple[str, str]:
    n = normalize(name)
    if n in HIGH_CONF_DROP_EXACT:
        return "drop", "high_conf_exact"

    for pat in KEEP_PATTERNS:
        if re.search(pat, n):
            return "keep", "checkpoint_like"

    for pat in METHOD_REVIEW_PATTERNS:
        if re.search(pat, n):
            return "review", "method_like"

    return "keep", "default"


def process_file(path: str, include_review: bool) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("model_parameters", [])
    if not isinstance(items, list):
        return {
            "file": os.path.basename(path),
            "path": path,
            "total": 0,
            "drop": [],
            "review": [],
            "keep": [],
            "updated": False,
        }

    keep_items: List[Dict[str, Any]] = []
    drop_items: List[Dict[str, Any]] = []
    review_items: List[Dict[str, Any]] = []

    for it in items:
        if not isinstance(it, dict):
            continue
        model = str(it.get("model", "")).strip()
        cls, reason = classify_model_name(model)
        record = {"model": model, "parameters": str(it.get("parameters", "")), "reason": reason}

        if cls == "drop":
            drop_items.append(record)
            continue
        if cls == "review":
            review_items.append(record)
            if include_review:
                drop_items.append(record)
            else:
                keep_items.append(it)
            continue
        keep_items.append(it)

    updated = len(keep_items) != len(items)
    if updated:
        data["model_parameters"] = keep_items
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "file": os.path.basename(path),
        "path": path,
        "total": len(items),
        "drop": drop_items,
        "review": review_items,
        "keep_count": len(keep_items),
        "updated": updated,
    }


def build_report(input_dir: str, apply: bool, include_review: bool) -> Dict[str, Any]:
    files = sorted([p for p in os.listdir(input_dir) if p.endswith(".json")])
    per_file = []
    for fn in files:
        full = os.path.join(input_dir, fn)
        if not os.path.isfile(full):
            continue
        if apply:
            rec = process_file(full, include_review=include_review)
        else:
            # report-only path: do not write files
            with open(full, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("model_parameters", [])
            if not isinstance(items, list):
                continue
            drop_items = []
            review_items = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                model = str(it.get("model", "")).strip()
                cls, reason = classify_model_name(model)
                rec_item = {"model": model, "parameters": str(it.get("parameters", "")), "reason": reason}
                if cls == "drop":
                    drop_items.append(rec_item)
                elif cls == "review":
                    review_items.append(rec_item)
            rec = {
                "file": fn,
                "path": full,
                "total": len(items),
                "drop": drop_items,
                "review": review_items,
                "keep_count": len(items) - len(drop_items),
                "updated": False,
            }
        if rec["drop"] or rec["review"]:
            per_file.append(rec)

    return {
        "input_dir": input_dir,
        "mode": "apply" if apply else "report",
        "include_review": include_review,
        "files_with_flags": len(per_file),
        "total_drop_candidates": sum(len(x["drop"]) for x in per_file),
        "total_review_candidates": sum(len(x["review"]) for x in per_file),
        "files": per_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit/Clean model_parameters labels")
    parser.add_argument("--input-dir", default="paper_section_gpu_test/emnlp2025")
    parser.add_argument("--mode", choices=["report", "apply"], default="report")
    parser.add_argument("--include-review", action="store_true", help="In apply mode, also remove method-like review items")
    parser.add_argument("--output", default="method_label_audit_v1.json")
    args = parser.parse_args()

    result = build_report(
        input_dir=args.input_dir,
        apply=args.mode == "apply",
        include_review=args.include_review,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: result[k] for k in ["input_dir", "mode", "files_with_flags", "total_drop_candidates", "total_review_candidates"]}, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
