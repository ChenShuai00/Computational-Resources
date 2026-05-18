from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from computing_resource.extraction.gpu_excel_cli import (
    build_export_and_renormalize_parser as build_parser,
    export_and_renormalize_gpu_excel,
    main_export_and_renormalize as main,
    resolve_export_and_renormalize_paths as resolve_paths,
)


if __name__ == "__main__":
    main()
