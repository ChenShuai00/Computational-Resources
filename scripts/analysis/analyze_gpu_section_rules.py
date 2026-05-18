from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from computing_resource.extraction.gpu_rule_analysis import export_rule_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export GPU/TPU section hit statistics and candidate rule tables")
    parser.add_argument("--input-dir", required=True, help="Directory containing *_gpu.json result files")
    parser.add_argument("--conference", required=True, help="Conference identifier, such as emnlp2024")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "analysis"))
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def load_docs(input_dir: str | Path) -> list[dict]:
    docs = []
    for path in sorted(Path(input_dir).glob("*_gpu.json")):
        docs.append(json.loads(path.read_text(encoding="utf-8")))
    return docs


def main(argv=None) -> None:
    args = parse_args(argv)
    docs = load_docs(args.input_dir)
    export_rule_analysis(docs, args.output_dir, args.conference)


if __name__ == "__main__":
    main()
