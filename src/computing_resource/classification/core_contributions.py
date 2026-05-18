import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTRIBUTION_LABELS = [
    "artifact_dataset",
    "artifact_method",
    "artifact_task",
    "knowledge_dataset",
    "knowledge_language",
    "knowledge_method",
    "knowledge_people",
    "knowledge_task",
]


@dataclass(frozen=True)
class ContributionCandidate:
    sentence_index: int
    text: str
    labels: tuple[str, ...]
    label_scores: dict[str, float]
    max_score: float
    avg_score: float
    meets_threshold: bool


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _label_score(sentence: dict[str, Any], label: str) -> float:
    value = sentence.get("labels", {}).get(label, {}).get("pos_probability", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _candidate_from_threshold(
    sentence: dict[str, Any],
    threshold: float,
) -> ContributionCandidate | None:
    label_scores = {
        label: _label_score(sentence, label)
        for label in CONTRIBUTION_LABELS
        if _label_score(sentence, label) >= threshold
    }
    if not label_scores:
        return None
    scores = list(label_scores.values())
    return ContributionCandidate(
        sentence_index=int(sentence.get("sentence_index", -1)),
        text=str(sentence.get("text", "")),
        labels=tuple(label_scores),
        label_scores=label_scores,
        max_score=max(scores),
        avg_score=sum(scores) / len(scores),
        meets_threshold=True,
    )


def _fallback_candidate(paper: dict[str, Any]) -> ContributionCandidate | None:
    best_sentence: dict[str, Any] | None = None
    best_label = ""
    best_score = -1.0
    for sentence in paper.get("sentences", []):
        for label in CONTRIBUTION_LABELS:
            score = _label_score(sentence, label)
            if score > best_score:
                best_sentence = sentence
                best_label = label
                best_score = score
    if best_sentence is None or not best_label:
        return None
    return ContributionCandidate(
        sentence_index=int(best_sentence.get("sentence_index", -1)),
        text=str(best_sentence.get("text", "")),
        labels=(best_label,),
        label_scores={best_label: best_score},
        max_score=best_score,
        avg_score=best_score,
        meets_threshold=False,
    )


def select_core_contributions(
    paper: dict[str, Any],
    threshold: float = 0.8,
    max_core_sentences: int = 3,
) -> dict[str, Any]:
    candidates = [
        candidate
        for sentence in paper.get("sentences", [])
        if (candidate := _candidate_from_threshold(sentence, threshold)) is not None
    ]
    labels_at_threshold = sorted({label for candidate in candidates for label in candidate.labels})

    fallback_used = False
    selection_status = "threshold"
    if not candidates:
        fallback = _fallback_candidate(paper)
        candidates = [] if fallback is None else [fallback]
        fallback_used = fallback is not None
        selection_status = "fallback" if fallback_used else "empty"

    selected: list[ContributionCandidate] = []
    uncovered = {label for candidate in candidates for label in candidate.labels}
    remaining = list(candidates)
    while remaining and len(selected) < max_core_sentences:
        best = max(
            remaining,
            key=lambda candidate: (
                len(set(candidate.labels) & uncovered),
                candidate.max_score,
                candidate.avg_score,
                -candidate.sentence_index,
            ),
        )
        selected.append(best)
        uncovered -= set(best.labels)
        remaining.remove(best)
        if not uncovered and len(selected) >= 1:
            break

    core_labels = sorted({label for candidate in selected for label in candidate.labels})
    core_contributions = [
        {
            "sentence_index": candidate.sentence_index,
            "text": candidate.text,
            "labels": list(candidate.labels),
            "label_scores": {
                label: round(score, 6)
                for label, score in sorted(candidate.label_scores.items())
            },
            "max_score": round(candidate.max_score, 6),
            "avg_score": round(candidate.avg_score, 6),
            "meets_threshold": candidate.meets_threshold,
        }
        for candidate in selected
    ]

    return {
        "source_path": paper.get("source_path", ""),
        "anthology_id": paper.get("anthology_id", ""),
        "title": paper.get("title", ""),
        "year": paper.get("year", ""),
        "venue": paper.get("venue", ""),
        "url": paper.get("url", ""),
        "pdf_url": paper.get("pdf_url", ""),
        "threshold": threshold,
        "fallback_used": fallback_used,
        "selection_status": selection_status,
        "num_sentences": len(paper.get("sentences", [])),
        "labels_at_threshold": labels_at_threshold,
        "core_labels": core_labels,
        "core_contributions": core_contributions,
    }


def format_core_contributions(contributions: list[dict[str, Any]]) -> str:
    formatted = []
    for contribution in contributions:
        labels = ",".join(contribution.get("labels", []))
        score = contribution.get("max_score", 0.0)
        sentence_index = contribution.get("sentence_index", -1)
        text = " ".join(str(contribution.get("text", "")).split())
        formatted.append(f"s{sentence_index} p={score:.3f} [{labels}] {text}")
    return " || ".join(formatted)
