"""Assemble the interactive site into ``_site/`` (served by GitHub Pages).

Copies ``web/`` and the bundled map data, so the maps have a single source
of truth in ``src/graph_coloring/apps/data``. Preview locally with:

    python scripts/build_web.py && python -m http.server -d _site
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "_site"
DATA = ROOT / "src" / "graph_coloring" / "apps" / "data"


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(ROOT / "web", SITE)
    (SITE / "data").mkdir()
    for f in sorted(DATA.glob("*.geojson")):
        shutil.copy2(f, SITE / "data" / f.name)
    print(f"built {SITE.relative_to(ROOT)}/ ({len(list(SITE.rglob('*')))} files)")


if __name__ == "__main__":
    main()
