import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from computing_resource.metadata import acl, acl_bundle, enrich, merge, openalex, references_enrich, semantic_scholar


COMMANDS = {
    "acl": acl.main,
    "acl-bundle": acl_bundle.main,
    "enrich": enrich.main,
    "openalex": openalex.main,
    "references-enrich": references_enrich.main,
    "semantic-scholar": semantic_scholar.main,
    "merge": merge.main,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified metadata fetch and merge entry point")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv=None):
    parsed = build_parser().parse_args(argv)
    COMMANDS[parsed.command](parsed.args)


if __name__ == "__main__":
    main()
