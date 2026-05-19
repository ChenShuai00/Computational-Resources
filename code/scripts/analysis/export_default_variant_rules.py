from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from computing_resource.extraction.gpu_default_variant_rules import export_default_variant_rules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export filled default benchmark variant selections to a YAML rules file."
    )
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = export_default_variant_rules(
        input_path=args.input_path,
        output_path=args.output_path,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
