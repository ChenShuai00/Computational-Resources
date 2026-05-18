from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def _suggest_action(papers_with_title: int, papers_with_hits: int, hit_rate: float) -> str:
    if papers_with_hits >= 2 and hit_rate >= 0.6:
        return "promote_to_strong_keep"
    if papers_with_hits >= 1 and hit_rate >= 0.2:
        return "conditional_keep"
    return "ignore"


def export_rule_analysis(docs: list[dict], output_dir: str | Path, conference: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    title_papers: defaultdict[str, set[str]] = defaultdict(set)
    title_hit_papers: defaultdict[str, set[str]] = defaultdict(set)

    for doc in docs:
        paper_id = doc.get("paper_id", "")
        for window in doc.get("candidate_windows", []):
            title = window.get("normalized_section_title", "")
            if title:
                title_papers[title].add(paper_id)
        for row in doc.get("normalized_extractions", []):
            title = row.get("normalized_section_title", "")
            if title:
                title_hit_papers[title].add(paper_id)

    stats_rows = []
    for title in sorted(title_papers):
        papers_with_title = len(title_papers[title])
        papers_with_hits = len(title_hit_papers.get(title, set()))
        hit_rate = papers_with_hits / papers_with_title if papers_with_title else 0.0
        stats_rows.append(
            {
                "normalized_section_title": title,
                "papers_with_title": papers_with_title,
                "papers_with_hardware_hits": papers_with_hits,
                "hit_rate": f"{hit_rate:.4f}",
                "suggested_action": _suggest_action(papers_with_title, papers_with_hits, hit_rate),
            }
        )

    stats_path = output_path / f"gpu_rule_stats_{conference}.csv"
    candidates_path = output_path / f"gpu_rule_candidates_{conference}.csv"
    for path in (stats_path, candidates_path):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "normalized_section_title",
                    "papers_with_title",
                    "papers_with_hardware_hits",
                    "hit_rate",
                    "suggested_action",
                ],
            )
            writer.writeheader()
            writer.writerows(stats_rows)
