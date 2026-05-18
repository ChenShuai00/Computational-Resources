import argparse
import subprocess
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent


COMMANDS = {
    "gpu": ROOT / "scripts" / "extract_gpu.py",
    "model-parameters": ROOT / "scripts" / "extract_model_parameters_dspy.py",
    "merge": ROOT / "scripts" / "merge_pred_resources.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified resource extraction entry point")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv=None):
    parsed = build_parser().parse_args(argv)
    cmd = [sys.executable, str(COMMANDS[parsed.command]), *parsed.args]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
