import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from computing_resource.extraction import mineru_hosted_api


COMMANDS = {
    "mineru-hosted-api": mineru_hosted_api.main,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified PDF parsing entry point")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv=None):
    parsed = build_parser().parse_args(argv)
    return COMMANDS[parsed.command](parsed.args)


if __name__ == "__main__":
    main()
