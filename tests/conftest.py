from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GCNET_ROOT = REPOSITORY_ROOT / "gcnet"

for path in (REPOSITORY_ROOT, GCNET_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
