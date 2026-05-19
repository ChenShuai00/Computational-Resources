from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from computing_resource.metadata.acl_bundle import main


if __name__ == "__main__":
    main()
