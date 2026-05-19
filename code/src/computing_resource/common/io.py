import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional


def build_logger(log_file: Optional[Path] = None, stream=None) -> Callable[[str], None]:
    stream = sys.stdout if stream is None else stream

    def log(message: str) -> None:
        print(message, file=stream)
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    return log


def maybe_sleep(seconds: float, sleep_fn=time.sleep) -> None:
    if seconds > 0:
        sleep_fn(seconds)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def derive_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / input_path.relative_to(input_root)


def iter_json_files(input_root: Path, conference: str | None = None):
    for path in sorted(input_root.rglob("*.json")):
        if path.name == "index.json":
            continue
        if conference:
            rel = path.relative_to(input_root).as_posix()
            if not rel.startswith(f"{conference}/") and path.stem != conference:
                continue
        yield path
