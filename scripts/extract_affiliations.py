from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from computing_resource.extraction.affiliations import main  # noqa: E402


if __name__ == "__main__":
    main()
