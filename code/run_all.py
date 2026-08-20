"""Run every public, offline paper-results workflow in a fixed order."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workflow:
    workflow_id: str
    script: str
    quick: bool = False
    arguments: tuple[str, ...] = ()


WORKFLOWS = (
    Workflow("sample_audit", "results/01_sample/analyses/sample_audit/scripts/analyze_sample_and_audit.py", True),
    Workflow("reporting", "results/02_reporting/analyses/reporting/scripts/plot_gpu_only_reporting.py", True),
    Workflow("gpu_counts", "results/03_gpu_scale/analyses/gpu_counts/scripts/analyze_gpu_count_bins_over_time.py"),
    Workflow("gpu_generation", "results/03_gpu_scale/analyses/gpu_generation/scripts/analyze_gpu_generation_family_over_time.py"),
    Workflow("gpu_memory", "results/03_gpu_scale/analyses/gpu_memory/scripts/analyze_memory.py"),
    Workflow("reported_peak", "results/03_gpu_scale/analyses/reported_peak/scripts/analyze_reported_peak_tflops_distribution.py"),
    Workflow("top_gpu_models", "results/03_gpu_scale/analyses/top_gpu_models/scripts/analyze_top_gpu_models_by_year.py"),
    Workflow("combined_evolution", "results/03_gpu_scale/analyses/combined_evolution/scripts/plot_rq1_gpu_resource_combined_from_data.py"),
    Workflow("country", "results/04_contexts/analyses/country/scripts/analyze_country_compute.py"),
    Workflow("institution", "results/04_contexts/analyses/institution/scripts/analyze_institution_compute_fe.py"),
    Workflow("topics", "results/04_contexts/analyses/topics/scripts/analyze_nlp_topic_compute.py"),
    Workflow(
        "impact",
        "results/05_scholarly_impact/analyses/impact/scripts/analyze_gpu_only_citation_modeling.py",
        arguments=("--quiet",),
    ),
)


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "data" / "analysis_ready" / "manifest.csv").is_file():
        raise FileNotFoundError("Run from a complete release containing data/analysis_ready/manifest.csv")
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run the sample and reporting smoke workflows only.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Generated-output root (default: results/reproduced).",
    )
    args = parser.parse_args()

    root = repository_root()
    output_root = (args.output_root or root / "results" / "reproduced").resolve()
    selected = [workflow for workflow in WORKFLOWS if not args.quick or workflow.quick]
    print(f"Running {len(selected)} workflow(s); outputs: {output_root}")

    for index, workflow in enumerate(selected, start=1):
        script = root / workflow.script
        output = output_root / workflow.workflow_id
        env = os.environ.copy()
        env["REPRO_OUTPUT_DIR"] = str(output)
        command = [sys.executable, str(script), *workflow.arguments]
        if workflow.workflow_id == "impact":
            command.extend(["--input-data-dir", str(root / "data" / "analysis_ready"), "--output-dir", str(output)])
        print(f"[{index}/{len(selected)}] {workflow.workflow_id}")
        subprocess.run(command, cwd=root, env=env, check=True)

    print("All selected workflows completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
