import argparse
import concurrent.futures
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from computing_resource.config import PROJECT_ROOT
from computing_resource.extraction.gpu import (
    PROMPT_MODE_FEW_SHOT,
    PROMPT_MODES,
    build_tagger,
    parse_prediction_result,
    split_predicted_item,
)


DEFAULT_GOLD_JSONL = PROJECT_ROOT / "annotation" / "gold_export" / "emnlp2025_gpu_gold_evaluation_400.jsonl"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_WORKERS = 1
MATCH_NORMALIZATION_STRICT = "strict"
MATCH_NORMALIZATION_LIGHT = "light"
MATCH_NORMALIZATIONS = {MATCH_NORMALIZATION_STRICT, MATCH_NORMALIZATION_LIGHT}
DEFAULT_MATCH_NORMALIZATION = MATCH_NORMALIZATION_LIGHT


def _split_semicolon_text(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(";")]


def _parse_count(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "null", "nan", "n/a"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return None


def parse_gold_entities(row: dict[str, Any]) -> list[dict[str, Any]]:
    names = [name for name in _split_semicolon_text(row.get("final_hardware_name")) if name]
    counts = _split_semicolon_text(row.get("final_count"))
    entities = []
    for index, name in enumerate(names):
        count = counts[index] if index < len(counts) else ""
        entities.append({"name": name, "num": _parse_count(count)})
    return entities


def load_gold_rows(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _prepare_predictions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    for item in items:
        for raw_name, count in split_predicted_item(item):
            predictions.append({"name": raw_name, "num": _parse_count(count)})
    return predictions


def normalize_light_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("o", "0")
    text = text.replace("gib", "gb")
    text = re.sub(r"(\d+)\s*g\b", r"\1gb", text)
    text = re.sub(r"\bgb\s+memory\b", "gb", text)
    text = re.sub(r"\bmemory\b", "", text)
    text = re.sub(r"\bgpus?\b", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _match_name(predicted_name: Any, gold_name: Any, match_normalization: str) -> bool:
    if match_normalization == MATCH_NORMALIZATION_STRICT:
        return predicted_name == gold_name
    if match_normalization == MATCH_NORMALIZATION_LIGHT:
        return normalize_light_name(predicted_name) == normalize_light_name(gold_name)
    raise ValueError(f"Unsupported match_normalization: {match_normalization}")


def _entities_match(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    match_type: str,
    match_normalization: str,
) -> bool:
    if not _match_name(predicted.get("name"), gold.get("name"), match_normalization):
        return False
    if match_type == "name_only":
        return True
    return predicted.get("num") == gold.get("num")


class GoldNERMetrics:
    def __init__(self, match_type: str, match_normalization: str = DEFAULT_MATCH_NORMALIZATION) -> None:
        if match_normalization not in MATCH_NORMALIZATIONS:
            raise ValueError(f"Unsupported match_normalization: {match_normalization}")
        self.match_type = match_type
        self.match_normalization = match_normalization
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0

    def update(self, predictions: list[dict[str, Any]], gold: list[dict[str, Any]]) -> None:
        matched_gold = [False] * len(gold)

        for predicted_entity in predictions:
            matched = False
            for index, gold_entity in enumerate(gold):
                if not matched_gold[index] and _entities_match(
                    predicted_entity,
                    gold_entity,
                    self.match_type,
                    self.match_normalization,
                ):
                    self.total_tp += 1
                    matched_gold[index] = True
                    matched = True
                    break
            if not matched:
                self.total_fp += 1

        for matched in matched_gold:
            if not matched:
                self.total_fn += 1

    def compute(self) -> dict[str, float | int]:
        tp = self.total_tp
        fp = self.total_fp
        fn = self.total_fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn,
        }


def evaluate_rows(
    rows: list[dict[str, Any]],
    tagger: Callable[[str], str],
    workers: int = DEFAULT_WORKERS,
    match_normalization: str = DEFAULT_MATCH_NORMALIZATION,
) -> dict[str, Any]:
    if match_normalization not in MATCH_NORMALIZATIONS:
        raise ValueError(f"Unsupported match_normalization: {match_normalization}")
    metrics = {
        "name_only": GoldNERMetrics(match_type="name_only", match_normalization=match_normalization),
        "exact": GoldNERMetrics(match_type="exact", match_normalization=match_normalization),
    }
    predictions = []
    failures = []

    def evaluate_one(index_and_row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, row = index_and_row
        annotation_unit_id = str(row.get("annotation_unit_id", ""))
        paper_id = str(row.get("paper_id", ""))
        gold = parse_gold_entities(row)
        raw_output = ""
        predicted = []
        row_failures = []
        window_text = str(row.get("window_text", ""))

        if not window_text.strip():
            raw_output = "[]"
        else:
            try:
                raw_output = tagger(window_text)
                parsed, parse_error = parse_prediction_result(raw_output)
            except Exception as exc:
                parse_error = None
                parsed = []
                row_failures.append(
                    {
                        "annotation_unit_id": annotation_unit_id,
                        "paper_id": paper_id,
                        "failure_type": "request_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_output": raw_output,
                    }
                )
            else:
                if parse_error:
                    row_failures.append(
                        {
                            "annotation_unit_id": annotation_unit_id,
                            "paper_id": paper_id,
                            "failure_type": "parse_error",
                            "error": parse_error,
                            "raw_output": raw_output,
                        }
                    )
                else:
                    predicted = _prepare_predictions(parsed)

        return {
            "index": index,
            "annotation_unit_id": annotation_unit_id,
            "paper_id": paper_id,
            "predicted": predicted,
            "gold": gold,
            "raw_output": raw_output,
            "failures": row_failures,
        }

    total = len(rows)
    indexed_rows = list(enumerate(rows, start=1))
    if workers <= 1:
        evaluated = [evaluate_one(item) for item in indexed_rows]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            evaluated = list(executor.map(evaluate_one, indexed_rows))

    for result in evaluated:
        for scorer in metrics.values():
            scorer.update(result["predicted"], result["gold"])
        predictions.append(
            {
                "annotation_unit_id": result["annotation_unit_id"],
                "paper_id": result["paper_id"],
                "predicted": result["predicted"],
                "gold": result["gold"],
                "raw_output": result["raw_output"],
            }
        )
        failures.extend(result["failures"])
        print(f"[gpu-gold] {result['index']}/{total} {result['annotation_unit_id']}")

    return {
        "metrics": {name: scorer.compute() for name, scorer in metrics.items()},
        "predictions": predictions,
        "failures": failures,
    }


def _default_output_dir(model: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model)
    return PROJECT_ROOT / "artifacts" / "analysis" / f"gpu_gold_eval_{safe_model}_{timestamp}"


def save_evaluation_artifacts(
    result: dict[str, Any],
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    metrics_payload = {"metadata": metadata, "metrics": result["metrics"]}
    (target / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (target / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["predictions"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (target / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["failures"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    name_only = result["metrics"]["name_only"]
    exact = result["metrics"]["exact"]
    summary = "\n".join(
        [
            "GPU gold LLM evaluation",
            f"model: {metadata['model']}",
            f"gold_rows: {metadata['gold_rows']}",
            f"failures: {len(result['failures'])}",
            "",
            "name_only:",
            f"  precision: {name_only['precision']:.6f}",
            f"  recall: {name_only['recall']:.6f}",
            f"  f1: {name_only['f1']:.6f}",
            f"  tp/fp/fn/support: {name_only['tp']}/{name_only['fp']}/{name_only['fn']}/{name_only['support']}",
            "",
            "exact:",
            f"  precision: {exact['precision']:.6f}",
            f"  recall: {exact['recall']:.6f}",
            f"  f1: {exact['f1']:.6f}",
            f"  tp/fp/fn/support: {exact['tp']}/{exact['fp']}/{exact['fn']}/{exact['support']}",
            "",
        ]
    )
    (target / "summary.txt").write_text(summary, encoding="utf-8")
    return target


def run_gold_evaluation(
    gold_jsonl: str | Path,
    output_dir: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    limit: int = 0,
    prompt_mode: str = PROMPT_MODE_FEW_SHOT,
    workers: int = DEFAULT_WORKERS,
    api_base: str | None = None,
    api_key_env: str | None = None,
    match_normalization: str = DEFAULT_MATCH_NORMALIZATION,
    tagger: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
    if match_normalization not in MATCH_NORMALIZATIONS:
        raise ValueError(f"Unsupported match_normalization: {match_normalization}")

    rows = load_gold_rows(gold_jsonl, limit=limit)
    active_tagger = tagger or build_tagger(
        model=model,
        prompt_mode=prompt_mode,
        api_base=api_base,
        api_key_env=api_key_env,
    )
    result = evaluate_rows(rows, active_tagger, workers=workers, match_normalization=match_normalization)
    target_dir = Path(output_dir) if output_dir else _default_output_dir(model)
    metadata = {
        "gold_jsonl": str(Path(gold_jsonl)),
        "gold_rows": len(rows),
        "model": model,
        "prompt_mode": prompt_mode,
        "workers": workers,
        "match_normalization": match_normalization,
        "api_base": api_base,
        "api_key_env": api_key_env,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    artifact_dir = save_evaluation_artifacts(result, target_dir, metadata)
    result["artifact_dir"] = str(artifact_dir)
    result["metadata"] = metadata
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GPU LLM extraction on gold JSONL windows.")
    parser.add_argument("--gold-jsonl", default=str(DEFAULT_GOLD_JSONL))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-mode", choices=sorted(PROMPT_MODES), default=PROMPT_MODE_FEW_SHOT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--match-normalization",
        choices=sorted(MATCH_NORMALIZATIONS),
        default=DEFAULT_MATCH_NORMALIZATION,
    )
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key-env", default="")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_gold_evaluation(
        gold_jsonl=args.gold_jsonl,
        output_dir=args.output_dir or None,
        model=args.model,
        limit=args.limit,
        prompt_mode=args.prompt_mode,
        workers=args.workers,
        match_normalization=args.match_normalization,
        api_base=args.api_base or None,
        api_key_env=args.api_key_env or None,
    )
    print(json.dumps({"artifact_dir": result["artifact_dir"], "metrics": result["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
