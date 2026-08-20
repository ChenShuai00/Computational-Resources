"""Verify released inputs and independently regenerated paper-result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


MODULES = {
    "sample_audit": "results/01_sample/analyses/sample_audit",
    "reporting": "results/02_reporting/analyses/reporting",
    "gpu_counts": "results/03_gpu_scale/analyses/gpu_counts",
    "gpu_generation": "results/03_gpu_scale/analyses/gpu_generation",
    "gpu_memory": "results/03_gpu_scale/analyses/gpu_memory",
    "reported_peak": "results/03_gpu_scale/analyses/reported_peak",
    "top_gpu_models": "results/03_gpu_scale/analyses/top_gpu_models",
    "combined_evolution": "results/03_gpu_scale/analyses/combined_evolution",
    "country": "results/04_contexts/analyses/country",
    "institution": "results/04_contexts/analyses/institution",
    "topics": "results/04_contexts/analyses/topics",
    "impact": "results/05_scholarly_impact/analyses/impact",
}
QUICK_MODULES = {"sample_audit", "reporting"}


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(root: Path) -> int:
    manifest = pd.read_csv(root / "data" / "analysis_ready" / "manifest.csv")
    checked = 0
    for row in manifest.itertuples(index=False):
        path = root / "data" / "analysis_ready" / row.file
        if not path.is_file():
            raise AssertionError(f"Missing analysis-ready input: {path}")
        if sha256(path) != row.sha256:
            raise AssertionError(f"Input SHA-256 mismatch: {path}")
        frame = pd.read_csv(path)
        if (len(frame), len(frame.columns)) != (int(row.rows), int(row.columns)):
            raise AssertionError(f"Input shape mismatch: {path}")
        checked += 1

    membership = pd.read_csv(root / "data" / "analysis_ready" / "paper_sample_membership.csv")
    counts = (len(membership), int(membership.model_reported.sum()), int(membership.strict_reported.sum()))
    if counts != (13_921, 6_900, 5_360):
        raise AssertionError(f"Frozen sample counts changed: {counts}")
    if membership.paper_id.nunique() != len(membership):
        raise AssertionError("paper_sample_membership.csv contains duplicate paper IDs")
    return checked


def assert_csv_equal(reference: Path, reproduced: Path) -> None:
    left = pd.read_csv(reference)
    right = pd.read_csv(reproduced)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        raise AssertionError(f"CSV schema/shape mismatch: {reference} vs {reproduced}")
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(a, b, rtol=1e-8, atol=1e-10, equal_nan=True):
                raise AssertionError(f"Numeric mismatch in {reference}, column {column}")
        else:
            a = left[column].fillna("<NA>").astype(str)
            b = right[column].fillna("<NA>").astype(str)
            if not a.equals(b):
                raise AssertionError(f"Text mismatch in {reference}, column {column}")


def verify_outputs(root: Path, output_root: Path, quick: bool) -> tuple[int, int]:
    csv_checked = 0
    figures_checked = 0
    selected = QUICK_MODULES if quick else set(MODULES)
    for module_id in sorted(selected):
        module = root / MODULES[module_id]
        reference = module / "reference"
        reproduced = output_root / module_id
        if not reproduced.is_dir():
            raise AssertionError(f"Missing reproduced workflow directory: {reproduced}")
        for ref_csv in reference.rglob("*.csv"):
            if "publication_figures" in ref_csv.parts:
                continue
            candidate = reproduced / ref_csv.relative_to(reference)
            if candidate.is_file():
                assert_csv_equal(ref_csv, candidate)
                csv_checked += 1
        for ref_png in (reference / "figures").glob("*.png"):
            candidate = reproduced / "figures" / ref_png.name
            if not candidate.is_file():
                continue
            with Image.open(ref_png) as expected, Image.open(candidate) as actual:
                if expected.size != actual.size:
                    raise AssertionError(
                        f"Figure dimension mismatch: {ref_png.name}: {expected.size} != {actual.size}"
                    )
                if expected.mode != actual.mode:
                    raise AssertionError(f"Figure color-mode mismatch: {ref_png.name}")
            figures_checked += 1
    return csv_checked, figures_checked


def verify_publication_contracts(root: Path) -> int:
    contracts = pd.read_csv(root / "code" / "figure_contracts.csv")
    for row in contracts.itertuples(index=False):
        path = root / row.path
        if not path.is_file():
            raise AssertionError(f"Missing publication figure: {path}")
        with Image.open(path) as image:
            if image.size != (int(row.width_px), int(row.height_px)):
                raise AssertionError(f"Publication figure dimensions changed: {path}")
        if not str(row.panels).strip() or not str(row.required_labels).strip():
            raise AssertionError(f"Incomplete figure contract: {path}")
    return len(contracts)


def verify_paper_claims(root: Path, output_root: Path) -> None:
    sample = json.loads((output_root / "sample_audit" / "reports" / "sample_audit.json").read_text(encoding="utf-8"))
    expected = {
        "full_corpus_n": 13_921,
        "model_reported_n": 6_900,
        "strict_reported_n": 5_360,
        "model_reported_share_pct": 49.6,
        "strict_reported_share_pct": 38.5,
        "consumption_audit_n": 240,
        "consumption_visible_n": 92,
        "consumption_not_visible_n": 148,
    }
    if sample != expected:
        raise AssertionError(f"Headline sample claims changed: {sample}")
    if (output_root / "impact" / "tables" / "overall_gpu_capability_table.csv").is_file():
        effects = pd.read_csv(output_root / "impact" / "tables" / "overall_gpu_capability_table.csv")
        primary = effects.loc[effects.Outcome.eq("NLP topic-year percentile")].iloc[0]
        if primary["Capacity effect"] != "3.52 pp" or not np.isclose(float(primary["ΔR² capacity"]), 0.004207):
            raise AssertionError("Primary adjusted-impact claim changed")
        if set(effects.N.astype(int)) != {2_194, 5_357}:
            raise AssertionError("Citation/award model sample sizes changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    root = root_dir()
    output_root = (args.output_root or root / "results" / "reproduced").resolve()
    input_count = verify_inputs(root)
    csv_count, figure_count = verify_outputs(root, output_root, args.quick)
    contract_count = verify_publication_contracts(root)
    verify_paper_claims(root, output_root)
    print(
        f"PASS: {input_count} inputs, {csv_count} regenerated tables, "
        f"{figure_count} regenerated figures, and {contract_count} publication figures verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
