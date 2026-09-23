"""Map coloring.

Regions are nodes and two regions are adjacent when they share a border of
positive length (touching at a single point does not count). By the Four
Color Theorem (Appel & Haken, 1976) every such planar map can be colored
with at most 4 colors, and some maps need all 4: a region surrounded by an
odd number of neighbours that form a ring (an *odd wheel*) forces it.

Bundled maps (built from Natural Earth by ``scripts/build_map_data.py``):

* ``"south_america"``: 12 countries plus French Guiana. Needs 4 colors:
  Argentina, Bolivia, Brazil and Paraguay are pairwise adjacent (a K4).
* ``"chile_regions"``: the 16 regions of Chile. Needs only 3 colors: the
  map is essentially a path, except for the triangle Valparaíso-Santiago-
  O'Higgins.
* ``"us_states"``: the 48 contiguous US states. Needs 4 colors (Nevada is
  surrounded by a 5-cycle of states).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

MAPS = ("south_america", "chile_regions", "us_states")

# Short labels for plotting the Chilean regions.
CHILE_SHORT_NAMES = {
    "Aisén del General Carlos Ibáñez del Campo": "Aysén",
    "Libertador General Bernardo O'Higgins": "O'Higgins",
    "Magallanes y Antártica Chilena": "Magallanes",
    "Región Metropolitana de Santiago": "Metropolitana",
}

# Categorical palette. The first four colors (enough for any map) stay
# distinguishable for every pair, including under color-vision deficiency.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#4a3aa7",  # violet
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#e34948",  # red
]


@dataclass
class MapData:
    """A map: its adjacency graph plus polygon geometries for drawing."""

    name: str
    title: str
    graph: nx.Graph
    geometries: dict[str, dict]
    """GeoJSON geometry (Polygon or MultiPolygon) of each region."""


def load_map(name: str) -> MapData:
    """Load one of the bundled maps (see :data:`MAPS`)."""
    if name not in MAPS:
        raise ValueError(f"unknown map {name!r}; choose from {MAPS}")
    path = resources.files("graph_coloring.apps") / "data" / f"{name}.geojson"
    data = json.loads(path.read_text(encoding="utf-8"))
    geometries = {f["properties"]["name"]: f["geometry"] for f in data["features"]}
    G = nx.Graph()
    G.add_nodes_from(sorted(geometries))
    G.add_edges_from(data["adjacency"])
    return MapData(name=name, title=data["name"], graph=G, geometries=geometries)


def _polygons(geometry: dict) -> list[list[list[float]]]:
    """Exterior rings of a Polygon/MultiPolygon."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [poly[0] for poly in geometry["coordinates"]]
    raise ValueError(f"unsupported geometry {geometry['type']}")


def _label_point(geometry: dict) -> tuple[float, float]:
    """Centroid of the largest ring (good enough for placing a label)."""
    ring = max(_polygons(geometry), key=lambda r: abs(_signed_area(r)))
    a = _signed_area(ring)
    cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:], strict=False):
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    return cx / (6 * a), cy / (6 * a)


def _signed_area(ring: list[list[float]]) -> float:
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:], strict=False))


def plot_map(
    m: MapData,
    coloring: dict[str, int] | None = None,
    ax: plt.Axes | None = None,
    labels: bool | dict[str, str] = False,
    palette: list[str] = PALETTE,
    conflicts: bool = True,
    title: str | None = None,
    min_label_area: float = 0.0,
    callouts: bool = False,
) -> plt.Axes:
    """Draw ``m`` with regions filled by ``coloring``.

    ``labels`` may be ``True`` (region names) or a dict of custom labels.
    Regions smaller than ``min_label_area`` (squared degrees) are left
    unlabelled. With ``callouts=True`` labels go in a column to the right
    of the map, joined to their region by a leader line; this suits long,
    narrow maps such as Chile. With ``conflicts=True`` borders between
    same-colored neighbours are marked with a red line.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    patches, colors = [], []
    for region, geom in m.geometries.items():
        fill = "#dddddd" if coloring is None else palette[coloring[region] % len(palette)]
        for ring in _polygons(geom):
            patches.append(Polygon(ring, closed=True))
            colors.append(fill)
    ax.add_collection(
        PatchCollection(patches, facecolor=colors, edgecolor="#333333", linewidth=0.5)
    )

    points = {region: _label_point(g) for region, g in m.geometries.items()}
    if coloring is not None and conflicts:
        for u, v in m.graph.edges:
            if coloring[u] == coloring[v]:
                (x0, y0), (x1, y1) = points[u], points[v]
                ax.plot([x0, x1], [y0, y1], color="red", linewidth=2, zorder=3)
    if labels:
        names = labels if isinstance(labels, dict) else {}
        box = {"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75}
        shown = [
            r
            for r in points
            if max(abs(_signed_area(ring)) for ring in _polygons(m.geometries[r])) >= min_label_area
        ]
        if callouts:
            ax.autoscale_view()
            x_text = ax.get_xlim()[1] + 1.0
            ymin, ymax = ax.get_ylim()
            # spread the labels evenly, keeping their north-south order
            order = sorted(shown, key=lambda r: -points[r][1])
            step = (ymax - ymin) / max(len(order) - 1, 1)
            for i, region in enumerate(order):
                ax.annotate(
                    names.get(region, region),
                    xy=points[region],
                    xytext=(x_text, ymax - i * step),
                    ha="left",
                    va="center",
                    fontsize=8,
                    zorder=4,
                    arrowprops={"arrowstyle": "-", "color": "#52514e", "linewidth": 0.6},
                )
        else:
            for region in shown:
                ax.annotate(
                    names.get(region, region),
                    points[region],
                    ha="center",
                    va="center",
                    fontsize=7,
                    zorder=4,
                    bbox=box,
                )

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    if title is not None:
        ax.set_title(title, pad=24 if callouts else 6)
    return ax
