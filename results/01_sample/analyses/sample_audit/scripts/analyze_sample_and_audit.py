"""Reproduce the corpus flow and the 240-paper consumption-visibility audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def find_repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data" / "analysis_ready" / "manifest.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not locate data/analysis_ready/manifest.csv")


ROOT = find_repository_root()
DEFAULT_OUTPUT = ROOT / "results" / "01_sample" / "analyses" / "sample_audit" / "reproduced"
OUTPUT = Path(os.environ.get("REPRO_OUTPUT_DIR", DEFAULT_OUTPUT)).resolve()


def main() -> None:
    source_data = OUTPUT / "source_data"
    tables = OUTPUT / "tables"
    reports = OUTPUT / "reports"
    for directory in (source_data, tables, reports):
        directory.mkdir(parents=True, exist_ok=True)

    membership = pd.read_csv(ROOT / "data" / "analysis_ready" / "paper_sample_membership.csv")
    audit = pd.read_csv(ROOT / "data" / "analysis_ready" / "consumption_audit_labels.csv")

    full_n = int(len(membership))
    model_n = int(membership["model_reported"].sum())
    strict_n = int(membership["strict_reported"].sum())
    assert (full_n, model_n, strict_n) == (13_921, 6_900, 5_360)

    flow = pd.DataFrame(
        [
            ("Full ACL/EMNLP/NAACL corpus, 2020-2025", full_n, 100.0),
            ("At least one GPU model reported", model_n, 100.0 * model_n / full_n),
            ("GPU model and count reported (strict sample)", strict_n, 100.0 * strict_n / full_n),
        ],
        columns=["sample", "papers", "share_of_full_corpus_pct"],
    )
    flow.to_csv(source_data / "sample_flow.csv", index=False)

    labels = audit["consumption_signal_visible"].astype(int)
    label_counts = (
        labels.value_counts()
        .rename_axis("consumption_signal_visible")
        .reset_index(name="papers")
        .sort_values("consumption_signal_visible")
    )
    label_counts["label"] = label_counts["consumption_signal_visible"].map(
        {0: "not_visible", 1: "visible"}
    )
    label_counts["share_pct"] = 100.0 * label_counts["papers"] / len(audit)
    label_counts = label_counts[["consumption_signal_visible", "label", "papers", "share_pct"]]
    assert len(audit) == 240
    assert dict(zip(label_counts["consumption_signal_visible"], label_counts["papers"])) == {0: 148, 1: 92}
    label_counts.to_csv(tables / "consumption_visibility_audit.csv", index=False)

    summary = {
        "full_corpus_n": full_n,
        "model_reported_n": model_n,
        "strict_reported_n": strict_n,
        "model_reported_share_pct": round(100.0 * model_n / full_n, 1),
        "strict_reported_share_pct": round(100.0 * strict_n / full_n, 1),
        "consumption_audit_n": int(len(audit)),
        "consumption_visible_n": int(labels.sum()),
        "consumption_not_visible_n": int((1 - labels).sum()),
    }
    (reports / "sample_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Sample and consumption-visibility audit

The 2020--2025 ACL/EMNLP/NAACL corpus contains **{full_n:,} papers**. At least one
GPU model is reported for **{model_n:,} papers ({summary['model_reported_share_pct']:.1f}%)**,
and both GPU model and count are reported for **{strict_n:,} papers
({summary['strict_reported_share_pct']:.1f}%)**.

In the frozen 240-paper manual audit, an explicit consumption signal is visible
for **{summary['consumption_visible_n']} papers** and is not visible for
**{summary['consumption_not_visible_n']} papers**. This audit supports the
measurement boundary: the released GPU variables describe text-reported
hardware evidence, not verified real-world energy or compute consumption.
"""
    (reports / "sample_audit.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
