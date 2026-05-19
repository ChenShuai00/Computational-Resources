import argparse
import json
import os
from typing import Any, Dict, List


def normalize_paper_id(file_name: str) -> str:
    if file_name.endswith("_gpu.json"):
        return file_name[: -len("_gpu.json")]
    return os.path.splitext(file_name)[0]


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def build_links(
    gpu_preds: List[Dict[str, Any]],
    model_preds: List[Dict[str, Any]],
    gpu_secs: List[str],
    model_secs: List[str],
) -> List[Dict[str, Any]]:
    if not gpu_preds or not model_preds:
        return []

    gpu_sec_set = {str(x).strip() for x in gpu_secs if str(x).strip()}
    model_sec_set = {str(x).strip() for x in model_secs if str(x).strip()}
    overlap = sorted(gpu_sec_set & model_sec_set)
    match_level = "section" if overlap else "document"

    links: List[Dict[str, Any]] = []
    for gpu in gpu_preds:
        gpu_name = str(gpu.get("name", "")).strip()
        gpu_num = gpu.get("num")
        if not gpu_name:
            continue

        for model in model_preds:
            model_name = str(model.get("model", "")).strip()
            parameters = str(model.get("parameters", "")).strip()
            if not model_name:
                continue
            links.append(
                {
                    "gpu_name": gpu_name,
                    "gpu_num": gpu_num,
                    "model": model_name,
                    "parameters": parameters,
                    "match_level": match_level,
                    "shared_sections": overlap,
                }
            )
    return links


def build_pred_resources(doc: Dict[str, Any], file_name: str) -> Dict[str, Any]:
    paper_id = normalize_paper_id(file_name)

    pred_result = doc.get("pred_result", {}) if isinstance(doc.get("pred_result", {}), dict) else {}
    pred_model = (
        doc.get("pred_model_parameters", {})
        if isinstance(doc.get("pred_model_parameters", {}), dict)
        else {}
    )

    gpu_preds = [x for x in safe_list(pred_result.get("pred", [])) if isinstance(x, dict)]
    model_preds = [x for x in safe_list(pred_model.get("pred", [])) if isinstance(x, dict)]
    gpu_secs = [str(x).strip() for x in safe_list(pred_result.get("sec", [])) if str(x).strip()]
    model_secs = [str(x).strip() for x in safe_list(pred_model.get("sec", [])) if str(x).strip()]

    return {
        "paper_id": paper_id,
        "source_file": file_name,
        "gpu": gpu_preds,
        "models": model_preds,
        "sections": {
            "gpu_sec": gpu_secs,
            "model_sec": model_secs,
        },
        "links": build_links(gpu_preds=gpu_preds, model_preds=model_preds, gpu_secs=gpu_secs, model_secs=model_secs),
        "version": "v1",
    }


def run_merge(input_dir: str, overwrite: bool) -> None:
    for file_name in sorted(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        if not file_name.endswith(".json"):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        if not overwrite and "pred_resources" in doc:
            print(f"skip: {file_path}")
            continue

        doc["pred_resources"] = build_pred_resources(doc=doc, file_name=file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        print(f"updated: {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge pred_result and pred_model_parameters into pred_resources")
    parser.add_argument("--input-dir", default="paper_section_gpu/emnlp2025")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing pred_resources")
    args = parser.parse_args()
    run_merge(input_dir=args.input_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
