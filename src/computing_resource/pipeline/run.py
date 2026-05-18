import argparse
import subprocess
import sys
from pathlib import Path

from computing_resource.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config


ALL_STEPS = ["download", "split", "parse", "section", "checklist", "gpu"]


def run_cmd(cmd):
    print(">>", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def build_runtime_paths(config: dict) -> dict:
    conference = config["conference"]
    paths = config["paths"]
    conference_id = f'{conference["name"]}{conference["year"]}'
    return {
        "conference_id": conference_id,
        "paper_dir": paths["papers_root"] / conference_id,
        "checklist_dir": paths["checklists_root"] / conference_id,
        "checklist_parse_dir": paths["checklist_parse_root"] / conference_id,
        "parse_out_dir": paths["parses_root"] / conference_id,
        "section_dir": paths["sections_root"] / conference_id,
        "gpu_root": paths["gpu_root"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paper processing pipeline from the default configuration")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--steps",
        nargs="+",
        default=ALL_STEPS,
        choices=ALL_STEPS,
        help="Pipeline steps to run",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    conference = cfg["conference"]
    runtime_paths = build_runtime_paths(cfg)
    py = sys.executable

    if "download" in args.steps:
        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "scripts" / "download_pdf.py"),
                "--conference-name",
                str(conference["name"]),
                "--conference-year",
                str(conference["year"]),
                "--paper-type",
                str(conference["paper_type"]),
                "--total-num",
                str(conference["total_num"]),
                "--paper-folder",
                str(runtime_paths["paper_dir"]),
                "--sleep-seconds",
                str(pipeline.get("sleep_seconds", 1.0)),
            ]
        )

    if "split" in args.steps:
        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "scripts" / "split_pdf.py"),
                "--input-dir",
                str(runtime_paths["paper_dir"]),
                "--output-dir",
                str(runtime_paths["checklist_dir"]),
            ]
        )

    if "parse" in args.steps:
        runtime_paths["parse_out_dir"].mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "scripts" / "parse_pdfs.py"),
                "mineru-hosted-api",
                "--input-dir",
                str(runtime_paths["paper_dir"]),
                "--conference",
                runtime_paths["conference_id"],
                "--output-dir",
                str(runtime_paths["parse_out_dir"]),
            ]
        )

    if "section" in args.steps:
        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "scripts" / "paper_section.py"),
                "--conference",
                runtime_paths["conference_id"],
                "--parse-root",
                str(cfg["paths"]["parses_root"]),
                "--save-root",
                str(cfg["paths"]["sections_root"]),
            ]
        )

    if "checklist" in args.steps:
        runtime_paths["checklist_parse_dir"].mkdir(parents=True, exist_ok=True)
        for pdf_path in sorted(runtime_paths["checklist_dir"].glob("*.pdf")):
            out_path = runtime_paths["checklist_parse_dir"] / f"{pdf_path.stem}.json"
            run_cmd(
                [
                    py,
                    str(PROJECT_ROOT / "scripts" / "parse_acl2023_checklist.py"),
                    str(pdf_path),
                    str(out_path),
                ]
            )

    if "gpu" in args.steps:
        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "scripts" / "extract_gpu.py"),
                "--input-dir",
                str(runtime_paths["parse_out_dir"]),
            ]
        )


if __name__ == "__main__":
    main()
