from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from computing_resource.extraction.gpu_catalog import HardwareCatalog


def _default_cache_path(model_path: str | Path) -> Path:
    model_name = Path(model_path).name
    return Path("artifacts") / "cache" / f"gpu_catalog_embeddings_{model_name}.pkl"


def build_embedding_cache_key(catalog: HardwareCatalog, settings: dict, embedding_dim: int | None = None) -> str:
    source_stat = catalog.source_path.stat()
    raw_key = "|".join(
        [
            str(catalog.source_path.resolve()),
            str(source_stat.st_mtime_ns),
            str(len(catalog.row_map)),
            str(settings.get("embedding_model_path", "")),
            str(embedding_dim or ""),
        ]
    )
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _encode_texts(texts: list[str], model_path: str | Path) -> list[list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)
        pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return normalized.cpu().tolist()


def build_or_load_catalog_embeddings(
    catalog: HardwareCatalog,
    settings: dict,
    encoder=None,
) -> dict:
    model_path = settings["embedding_model_path"]
    cache_path = Path(settings.get("embedding_cache_path") or _default_cache_path(model_path))
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(catalog.row_map.keys())
    encoder_fn = encoder or _encode_texts
    if cache_path.exists():
        cached = pickle.loads(cache_path.read_bytes())
        expected_key = build_embedding_cache_key(catalog, settings, cached.get("embedding_dim"))
        if cached.get("cache_key") == expected_key:
            return cached

    vectors = encoder_fn(names, model_path)
    embedding_dim = len(vectors[0]) if vectors else 0
    payload = {
        "hardware_names": names,
        "vectors": vectors,
        "embedding_dim": embedding_dim,
        "cache_key": build_embedding_cache_key(catalog, settings, embedding_dim),
    }
    cache_path.write_bytes(pickle.dumps(payload))
    return payload


def retrieve_embedding_candidates(
    cleaned_hardware_name: str,
    catalog: HardwareCatalog,
    settings: dict,
    encoder=None,
) -> list[dict]:
    cache = build_or_load_catalog_embeddings(catalog, settings, encoder=encoder)
    encoder_fn = encoder or _encode_texts
    query_vector = encoder_fn([cleaned_hardware_name], settings["embedding_model_path"])[0]

    scored = []
    for hardware_name, candidate_vector in zip(cache["hardware_names"], cache["vectors"]):
        score = sum(float(q) * float(c) for q, c in zip(query_vector, candidate_vector))
        scored.append(
            {
                "hardware_name": hardware_name,
                "score": score,
                "hardware_type": catalog.row_map[hardware_name].get("Type", ""),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[: int(settings.get("embedding_top_k", 10))]
